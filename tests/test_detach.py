"""Tensor.detach() -- a real view (shares Storage, O(1), no data copy)
with requires_grad=False and no grad_node: cut from the autograd graph
without cutting from the underlying memory. Added specifically to let
nn.BatchNorm1d fold a differentiable batch statistic into its
running_mean/running_var buffers without growing the graph across every
training step -- see its own docstring in python/kansai/nn.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai

# ---------------------------------------------------------------------
# 1. detach() preserves values and drops requires_grad.
# ---------------------------------------------------------------------

x = kansai.from_flat([1, 2, 3, 4], [4], requires_grad=True)
d = x.detach()
assert d.tolist() == [1.0, 2.0, 3.0, 4.0], "detach() must preserve values exactly"
assert d.requires_grad is False, "detach() output must have requires_grad=False"
print("detach() preserves values and drops requires_grad: OK")

# ---------------------------------------------------------------------
# 2. A real view: shares Storage with the original (checked by mutating
#    through the detached handle via add_ and confirming the original
#    sees it too), not a copy.
# ---------------------------------------------------------------------

y = kansai.from_flat([1, 2, 3, 4], [4], requires_grad=True)
dy = y.detach()
dy.add_(kansai.from_flat([10, 10, 10, 10], [4]), 1.0)
assert y.tolist() == [11.0, 12.0, 13.0, 14.0], "detach() must share Storage (a view), not copy"
print("detach() shares Storage with the original (mutating one mutates both): OK")

# ---------------------------------------------------------------------
# 3. Computation built entirely from detached tensors never requires
#    grad, and doesn't attach a grad_node to feed a spurious backward().
# ---------------------------------------------------------------------

a = kansai.from_flat([1, 2, 3], [3], requires_grad=True)
b = kansai.from_flat([4, 5, 6], [3], requires_grad=True)
z = a.detach().mul(b.detach())
assert z.requires_grad is False, "a computation built only from detached tensors must not require grad"
print("computation from detached tensors doesn't require grad: OK")

# ---------------------------------------------------------------------
# 4. Detaching mid-graph stops gradient flow to everything upstream of
#    the detach point, while gradient still flows normally through any
#    OTHER path that doesn't go through it -- the actual reason
#    detach() exists: cutting one path, not the whole graph.
# ---------------------------------------------------------------------

p = kansai.from_flat([2.0], [1], requires_grad=True)
q = kansai.from_flat([3.0], [1], requires_grad=True)
mid = p.mul(q)          # depends on both p and q
cut = mid.detach()      # same value, but no path back to p or q
loss = cut.mul(q).sum()  # depends on q (still attached) and cut (detached, constant as far as autograd is concerned)
loss.backward()

# d(loss)/dq = cut's value (mid.detach() == p*q, held constant) = p*q = 6.0 -- NOT 2*p*q
# (which is what it would be if q's OTHER path through `mid` also contributed).
assert q.grad is not None, "q must still receive a gradient through the undetached path"
check_val = q.grad.tolist()[0]
assert abs(check_val - 6.0) < 1e-5, f"expected d(loss)/dq == 6.0 (p*q, treating the detached path as constant), got {check_val}"
assert p.grad is None, "p must receive NO gradient -- its only path to loss goes through the detached tensor"
print(f"detach() correctly cuts gradient flow through itself while leaving other paths intact "
      f"(q.grad={check_val}, p.grad=None): OK")

print("\ndetach() test passed.")

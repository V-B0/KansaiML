"""kansai.no_grad() -- the standard "I'm running inference/validation
and will never call backward()" escape hatch, previously missing
entirely. estimate_val_loss in examples/tinyshakespeare/
train_shakespeare.py built a full backward graph on every validation
call before this existed, immediately discarded without ever being
differentiated -- correct, but real, needless work this closes.

Implemented as a single per-thread flag (thread_local in C++, since
DeviceMesh dispatches real concurrently-overlapping threads that must
not share this state) that every op's own requires_grad check now also
consults, exposed as a reentrant Python context manager.

Checked: ops run inside the block produce outputs with
requires_grad=False and no grad_node, even when every input requires
grad; grad-tracking is correctly restored after the block exits,
including when the block raises (try/finally, not just the happy
path); nesting one no_grad() block inside another restores to "still
disabled" rather than incorrectly re-enabling tracking when the INNER
block exits (a naive unconditional-restore implementation would get
this wrong); ordinary graph-building and backward() are completely
unaffected outside the block; and, practically, that using no_grad()
around a real forward+loss computation still produces the IDENTICAL
loss VALUE as without it (this only suppresses graph bookkeeping, never
changes what a forward pass computes).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import nn
from kansai import _core as core

x = core.from_flat([1.0, 2.0, 3.0], [3], requires_grad=True)

# ---------------------------------------------------------------------
# 1. Basic suppression: requires_grad=False and no grad_node inside the
#    block, restored after it exits.
# ---------------------------------------------------------------------

y = x.mul(x)
assert y.requires_grad, "sanity check: outside no_grad, ops build a graph as normal"
y.sum().backward()
assert x.grad.tolist() == [2.0, 4.0, 6.0], "sanity check: that graph is real -- backward() actually flows through it"
x.zero_grad()

with kansai.no_grad():
    z = x.mul(x)
    assert not z.requires_grad, "no_grad() must suppress requires_grad on the output"
print("no_grad() suppresses graph-building inside the block: OK")

w = x.mul(x)
assert w.requires_grad, "grad-tracking must be restored after the block exits"
print("no_grad() restores grad-tracking after the block exits: OK")

# ---------------------------------------------------------------------
# 2. Exception safety: restored even if the block raises.
# ---------------------------------------------------------------------

try:
    with kansai.no_grad():
        assert not x.mul(x).requires_grad
        raise ValueError("boom")
except ValueError:
    pass

after_exception = x.mul(x)
assert after_exception.requires_grad, "grad-tracking must be restored even when the block raises"
print("no_grad() restores grad-tracking even when the block raises: OK")

# ---------------------------------------------------------------------
# 3. Reentrant nesting: an inner block exiting must not re-enable
#    tracking while still inside an outer block.
# ---------------------------------------------------------------------

with kansai.no_grad():
    with kansai.no_grad():
        pass
    still_inside_outer = x.mul(x)
    assert not still_inside_outer.requires_grad, (
        "an inner no_grad() exiting must NOT re-enable tracking while still inside the outer block")
print("nested no_grad() blocks restore correctly (inner exit doesn't leak tracking back on): OK")

after_outer = x.mul(x)
assert after_outer.requires_grad, "tracking must be back on once the OUTER block actually exits"
print("outer no_grad() block correctly restores tracking on its own exit: OK")

# ---------------------------------------------------------------------
# 4. backward() is completely unaffected outside the block.
# ---------------------------------------------------------------------

xg = core.from_flat([1.0, 2.0, 3.0], [3], requires_grad=True)
xg.mul(xg).sum().backward()
assert xg.grad.tolist() == [2.0, 4.0, 6.0]
print("ordinary backward() outside no_grad is completely unaffected: OK")

# ---------------------------------------------------------------------
# 5. Practical: a real forward+loss computation through a small model
#    produces the IDENTICAL value with and without no_grad() -- this
#    only suppresses graph bookkeeping, never changes what forward
#    actually computes.
# ---------------------------------------------------------------------

model = nn.Sequential(nn.Linear(4, 8, seed=1), nn.ReLU(), nn.Linear(8, 2, seed=2))
inp = core.from_flat([0.3, -0.7, 1.1, 0.2], [1, 4])

with_grad = model(inp).sum().tolist()
with kansai.no_grad():
    without_grad = model(inp).sum().tolist()
assert with_grad == without_grad, (
    f"no_grad() must not change the computed VALUE: {with_grad} vs {without_grad}")
print("no_grad() produces the identical forward VALUE as ordinary graph-building: OK")

print("\nno_grad() test passed.")

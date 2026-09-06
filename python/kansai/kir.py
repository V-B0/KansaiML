"""
KIR -- Kansai Intermediate Representation (prototype).

Per the project plan, KIR is meant to be the single contract between the
Python frontend and every hardware backend: a typed, SSA-form graph that
the optimizer (fusion, DCE, layout, memory planning) transforms before a
backend ever sees it. Building that as a C++ layer straight away is a lot
to get right blind, so this module is a pure-Python reference
implementation of the schema -- it exists to validate node shape, tracing
mechanics, and pass semantics cheaply before committing to the real
(C++, backend-consumable) version.

Every Node produces exactly one value (SSA), referenced by its id. A
Graph's nodes are always stored in a valid topological order: tracing
follows normal Python program order, so an op's inputs are appended to
the graph before the op itself -- no separate topo-sort is ever needed,
the list order *is* the order a backend must execute in.

Known limitation, deliberately not solved here: `jit()` bakes any
non-traced operand (e.g. a Module's weight/bias Tensor) into the graph as
a constant node holding a *reference* to that Tensor. This happens to
stay correct across a training loop only because kansai's optimizers
mutate parameters in place (SGD's `add_`) rather than rebinding them --
the constant node's captured reference still points at live storage.
Functional/immutable parameter updates would break this silently; making
that safe by construction is later Phase 2 work (weights becoming their
own kind of graph input, not a constant baked in at trace time).
"""

from dataclasses import dataclass, field
from typing import Any

from . import _core as core


# ---------------------------------------------------------------------
# Graph / Node
# ---------------------------------------------------------------------

@dataclass
class Node:
    id: int
    op: str                     # "placeholder" | "constant" | "add" | "matmul" | ...
    inputs: list                # ids of Nodes producing each operand
    shape: list
    dtype: str = "float32"
    attrs: dict = field(default_factory=dict)

    def __repr__(self):
        ins = ", ".join(f"%{i}" for i in self.inputs)
        extra = f"  {self.attrs}" if self.attrs and "value" not in self.attrs else ""
        return f"%{self.id} = {self.op}({ins})  : {self.dtype}{self.shape}{extra}"


class Graph:
    def __init__(self):
        self.nodes: list = []
        self.inputs: list = []       # ids of placeholder nodes, in argument order
        self.output = None           # id of the node whose value the graph returns

    def add(self, op: str, inputs: list, shape: list, dtype: str = "float32", **attrs) -> int:
        node = Node(id=len(self.nodes), op=op, inputs=list(inputs), shape=list(shape),
                    dtype=dtype, attrs=attrs)
        self.nodes.append(node)
        return node.id

    def placeholder(self, shape: list, dtype: str = "float32") -> int:
        nid = self.add("placeholder", [], shape, dtype)
        self.inputs.append(nid)
        return nid

    def __repr__(self):
        lines = [repr(n) for n in self.nodes]
        lines.append(f"return %{self.output}")
        return "\n".join(lines)


# ---------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------

class TraceValue:
    """A placeholder tensor used only during tracing: carries shape/dtype
    metadata and a Node id, never real storage. Its operator surface
    mirrors kansai.Tensor exactly, so traced code needs no special
    syntax -- the same forward() a Module already has is what gets jit'd."""

    __slots__ = ("graph", "node_id", "shape", "dtype")

    def __init__(self, graph: Graph, node_id: int, shape: list, dtype: str = "float32"):
        self.graph = graph
        self.node_id = node_id
        self.shape = list(shape)
        self.dtype = dtype

    def _coerce(self, other) -> "TraceValue":
        if isinstance(other, TraceValue):
            return other
        # A real kansai.Tensor (typically a Module's weight/bias) reached
        # during tracing -- capture it as a constant node. See the module
        # docstring for why this stays sound only under in-place updates.
        nid = self.graph.add("constant", [], list(other.shape), value=other)
        return TraceValue(self.graph, nid, list(other.shape))

    def _binop(self, other: "TraceValue", op: str, out_shape: list) -> "TraceValue":
        nid = self.graph.add(op, [self.node_id, other.node_id], out_shape, self.dtype)
        return TraceValue(self.graph, nid, out_shape, self.dtype)

    def add(self, other):
        other = self._coerce(other)
        bias_broadcast = (len(other.shape) == 1 and len(self.shape) == 2
                           and self.shape[1] == other.shape[0])
        if not bias_broadcast and self.shape != other.shape:
            raise ValueError(f"add: shape mismatch {self.shape} vs {other.shape}")
        return self._binop(other, "add", list(self.shape))

    def sub(self, other):
        other = self._coerce(other)
        if self.shape != other.shape:
            raise ValueError(f"sub: shape mismatch {self.shape} vs {other.shape}")
        return self._binop(other, "sub", list(self.shape))

    def mul(self, other):
        other = self._coerce(other)
        if self.shape != other.shape:
            raise ValueError(f"mul: shape mismatch {self.shape} vs {other.shape}")
        return self._binop(other, "mul", list(self.shape))

    def matmul(self, other):
        other = self._coerce(other)
        if len(self.shape) != 2 or len(other.shape) != 2 or self.shape[1] != other.shape[0]:
            raise ValueError(f"matmul: incompatible shapes {self.shape} vs {other.shape}")
        out_shape = [self.shape[0], other.shape[1]]
        return self._binop(other, "matmul", out_shape)

    def conv2d(self, weight, bias, stride, padding):
        weight = self._coerce(weight)
        bias = self._coerce(bias)
        if len(self.shape) != 4 or len(weight.shape) != 4:
            raise ValueError(f"conv2d: input and weight must be 4D, got {self.shape} and {weight.shape}")
        N, Cin, H, W = self.shape
        Cout, Cin_w, kH, kW = weight.shape
        if Cin != Cin_w:
            raise ValueError(f"conv2d: channel mismatch {Cin} vs {Cin_w}")
        Hout = (H + 2 * padding - kH) // stride + 1
        Wout = (W + 2 * padding - kW) // stride + 1
        out_shape = [N, Cout, Hout, Wout]
        nid = self.graph.add("conv2d", [self.node_id, weight.node_id, bias.node_id], out_shape, self.dtype,
                              stride=stride, padding=padding)
        return TraceValue(self.graph, nid, out_shape, self.dtype)

    def relu(self):
        nid = self.graph.add("relu", [self.node_id], list(self.shape), self.dtype)
        return TraceValue(self.graph, nid, list(self.shape), self.dtype)

    def sum(self):
        nid = self.graph.add("sum", [self.node_id], [1], self.dtype)
        return TraceValue(self.graph, nid, [1], self.dtype)

    def mean(self):
        nid = self.graph.add("mean", [self.node_id], [1], self.dtype)
        return TraceValue(self.graph, nid, [1], self.dtype)

    __add__ = add
    __sub__ = sub
    __mul__ = mul
    __matmul__ = matmul


def trace(fn, *example_inputs) -> Graph:
    """Runs fn once with placeholder values shaped like example_inputs,
    recording every op into a Graph. This is operator-overload tracing --
    the same technique JAX and torch.fx reach for first, before full
    bytecode analysis -- and is the right starting point: it proves the
    IR schema and replay semantics before paying for a real Python-frame
    tracer (which is its own, much larger, later milestone)."""
    graph = Graph()
    trace_args = []
    for inp in example_inputs:
        nid = graph.placeholder(list(inp.shape))
        trace_args.append(TraceValue(graph, nid, list(inp.shape)))
    result = fn(*trace_args)
    graph.output = result.node_id
    return graph


# ---------------------------------------------------------------------
# Interpreter -- the reference executor every optimization pass must
# still agree with (no fusion, no memory reuse: it's a correctness
# baseline, not a fast path).
# ---------------------------------------------------------------------

_OP_TABLE = {
    "add": lambda a, b: a.add(b),
    "sub": lambda a, b: a.sub(b),
    "mul": lambda a, b: a.mul(b),
    "matmul": lambda a, b: a.matmul(b),
    "relu": lambda a: a.relu(),
    "sum": lambda a: a.sum(),
    "mean": lambda a: a.mean(),
    # vjp ops grad() builds backward graphs out of (see the Memory
    # planning section below is where fusion/DCE live; grad() and its
    # vjp rules are further down, in the Autograd section) -- forward-
    # only, same as everything else this table dispatches.
    "matmul_nt": lambda a, b: core.matmul_nt(a, b),
    "matmul_tn": lambda a, b: core.matmul_tn(a, b),
    "relu_backward": lambda x, g: core.relu_backward(x, g),
    "sum_axis0": lambda g: core.sum_axis0(g),
}


def run(graph: Graph, *args) -> "core.Tensor":
    values = {}
    for nid, arg in zip(graph.inputs, args):
        values[nid] = arg

    for node in graph.nodes:
        if node.op == "placeholder":
            continue
        if node.op == "constant":
            values[node.id] = node.attrs["value"]
            continue
        if node.op == "tuple":
            values[node.id] = tuple(values[i] for i in node.inputs)
            continue
        if node.op == "broadcast_scalar":
            values[node.id] = core.broadcast_scalar(
                values[node.inputs[0]], node.shape, node.attrs["scale"]
            )
            continue
        if node.op == "conv2d":
            x, w, b = (values[i] for i in node.inputs)
            values[node.id] = x.conv2d(w, b, node.attrs["stride"], node.attrs["padding"])
            continue
        fn = _OP_TABLE[node.op]
        values[node.id] = fn(*(values[i] for i in node.inputs))

    return values[graph.output]


def jit(fn):
    """Traces fn on first call (shape-specialized) and replays the
    recorded graph on every later call with a matching input shape.
    Not faster than eager yet -- there is no fusion or memory planning
    in this slice of Phase 2 -- but it proves the trace -> IR -> replay
    path is correct, including through backward() and an optimizer loop,
    since `run()` dispatches back to the real (autograd-tracked) Tensor
    ops in traced order."""
    cache: dict = {}

    def wrapped(*args):
        key = tuple(tuple(a.shape) for a in args)
        graph = cache.get(key)
        if graph is None:
            graph = trace(fn, *args)
            cache[key] = graph
        return run(graph, *args)

    wrapped.trace = lambda *args: trace(fn, *args)  # escape hatch for inspection
    return wrapped


# ---------------------------------------------------------------------
# Optimization passes
# ---------------------------------------------------------------------

_UNARY_ELEMENTWISE = {"relu"}
_BINARY_ELEMENTWISE = {"add", "sub", "mul"}


def _fusable(node: Node, by_id: dict) -> bool:
    if node.op in _UNARY_ELEMENTWISE:
        return True
    if node.op in _BINARY_ELEMENTWISE:
        a_shape = by_id[node.inputs[0]].shape
        b_shape = by_id[node.inputs[1]].shape
        if a_shape == b_shape:
            return True
        # the bias-broadcast add case: (batch, features) + (features,)
        if node.op == "add" and len(a_shape) == 2 and len(b_shape) == 1 and a_shape[1] == b_shape[0]:
            return True
        return False
    return False


def _classify_pair(first_node: Node, second_node: Node, by_id: dict):
    """Matches a 2-node segment (first_node's result feeding directly
    into second_node) against the small, explicit set of patterns
    Kansai actually has a specialized fused kernel for. Returns the
    pattern's op name, or None if this pair -- while a valid elementwise
    segment -- isn't one of the ones worth a dedicated kernel yet. See
    elementwise_fusion()'s docstring for why this is a lookup table and
    not a general compiler."""
    if (first_node.op, second_node.op) == ("add", "relu"):
        a_id, b_id = first_node.inputs
        if by_id[a_id].shape != by_id[b_id].shape:  # bias-broadcast add, then relu
            return "fused_bias_relu"
        return None

    if (first_node.op, second_node.op) == ("sub", "mul"):
        if second_node.inputs[0] == first_node.id and second_node.inputs[1] == first_node.id:  # diff.mul(diff)
            return "fused_sub_square"
        return None

    return None


def _segment_group(members: list, by_id: dict):
    """Greedily scans a fusion group's member ids left to right,
    matching the longest known pattern at each position (2 nodes, since
    every pattern here is a pair) and falling back to passing a single
    member through unfused wherever nothing matches. Returns a list of
    (pattern_name_or_None, member_ids) segments, in order -- a group
    longer than 2 nodes (e.g. bias_relu's `add,relu` immediately
    followed by another layer's unactivated `add`, once fusion's greedy
    single-use merging has absorbed both into one group) is *not*
    rejected wholesale the way a single whole-group pattern match would;
    each recognizable pair inside it still gets fused, chained to
    whatever's on either side."""
    segments = []
    i = 0
    n = len(members)
    while i < n:
        if i + 1 < n:
            pattern = _classify_pair(by_id[members[i]], by_id[members[i + 1]], by_id)
            if pattern is not None:
                segments.append((pattern, [members[i], members[i + 1]]))
                i += 2
                continue
        segments.append((None, [members[i]]))
        i += 1
    return segments


def elementwise_fusion(graph: Graph) -> Graph:
    """Fuses maximal linear chains of elementwise ops (add/sub/mul/relu,
    including the bias-broadcast add case) that match one of Kansai's
    known fused-kernel patterns -- currently `bias_add -> relu` (every
    Linear layer's forward epilogue) and `sub -> mul(self)` (the
    diff*diff core of MSE loss) -- replacing each match with a single
    node that runs a dedicated, straight-line C++ kernel in one pass.

    This is deliberately a small lookup table, not a general "compile
    any elementwise chain" pass. An earlier version of this file *was*
    general: it compiled arbitrary chains to a tiny per-element bytecode
    interpreted by a generic C++ stack machine. It was correct but
    measured ~7x *slower* than running the ops unfused -- interpreting a
    bytecode per element can't be auto-vectorized, so for ops this cheap
    (a single flop each) the dispatch overhead swamps the memory
    bandwidth it was supposed to save. Real fusion compilers (XLA, TVM,
    Triton) generate or select specialized code per pattern for exactly
    this reason. This file does that on a small scale: recognize a
    known shape, hand it to a kernel that's just as vectorizable as the
    unfused ops, in one pass instead of two or three.

    A chain is only considered for fusion when each producer in it has
    exactly one consumer overall (`x.mul(x)` counts as one consumer
    using the value twice, not two) -- fusing a value with more than one
    real consumer would mean recomputing it once per consumer. Within a
    chain longer than one recognizable pair -- e.g. a layer's
    `add,relu` immediately followed by the next layer's unactivated
    `add`, absorbed into the same group by the single-use merge above --
    each known 2-node pattern still gets fused (see _segment_group),
    just chained to its neighbors rather than requiring the *whole*
    group to match one shape.

    Forward-only: the emitted fused nodes carry no autograd information
    -- see run_fused()'s docstring for what that means and doesn't mean.
    """
    by_id = {n.id: n for n in graph.nodes}

    consumers: dict = {n.id: set() for n in graph.nodes}
    for n in graph.nodes:
        for i in n.inputs:
            consumers[i].add(n.id)
    if graph.output is not None:
        consumers[graph.output].add("__output__")

    def is_single_use(node_id):
        return len(consumers[node_id]) == 1

    group_id_of: dict = {}
    groups: dict = {}
    next_gid = 0

    for node in graph.nodes:
        if not _fusable(node, by_id):
            continue
        merge_into = None
        for inp_id in node.inputs:
            if _fusable(by_id[inp_id], by_id) and is_single_use(inp_id):
                merge_into = group_id_of[inp_id]
                break
        if merge_into is None:
            merge_into = next_gid
            next_gid += 1
            groups[merge_into] = []
        groups[merge_into].append(node.id)
        group_id_of[node.id] = merge_into

    group_tail = {gid: members[-1] for gid, members in groups.items()}
    group_segments = {
        gid: _segment_group(members, by_id)
        for gid, members in groups.items()
        if len(members) >= 2
    }

    remap: dict = {}
    new_graph = Graph()

    for node in graph.nodes:
        gid = group_id_of.get(node.id)

        if gid is not None and gid in group_segments:
            if node.id != group_tail[gid]:
                continue  # every member of this group is emitted when we reach the tail
            for pattern, member_ids in group_segments[gid]:
                if pattern is not None:
                    first_node = by_id[member_ids[0]]
                    a_id, b_id = first_node.inputs
                    new_inputs = [remap[a_id], remap[b_id]]
                    last_member = by_id[member_ids[-1]]
                    new_id = new_graph.add(pattern, new_inputs, list(last_member.shape), last_member.dtype)
                else:
                    raw_node = by_id[member_ids[0]]
                    new_inputs = [remap[i] for i in raw_node.inputs]
                    new_id = new_graph.add(raw_node.op, new_inputs, list(raw_node.shape),
                                            raw_node.dtype, **raw_node.attrs)
                for mid in member_ids:
                    remap[mid] = new_id
            continue

        new_inputs = [remap[i] for i in node.inputs]
        new_id = new_graph.add(node.op, new_inputs, list(node.shape), node.dtype, **node.attrs)
        remap[node.id] = new_id
        if node.op == "placeholder":
            new_graph.inputs.append(new_id)

    new_graph.output = remap[graph.output]
    return new_graph


def run_fused(graph: Graph, *args) -> "core.Tensor":
    """Like run(), but dispatches the fused pattern nodes
    (fused_bias_relu, fused_sub_square) to their dedicated C++ kernels
    instead of the eager ops that produced them pre-fusion. Every other
    node still goes through the real Tensor ops, same as run().

    Forward-only, deliberately: a fused node's output carries no
    grad_node, so calling backward() through it produces no gradient for
    whatever fed it. That's the right tradeoff for an inference or
    benchmarking path, not (yet) for training -- fusing *through*
    backward() means differentiating each fused kernel in closed form
    (both patterns here have simple, known derivatives, so it's tractable
    future work, just not done in this slice of Phase 2). Use run() --
    unfused -- for anything a training loop needs to call .backward()
    on."""
    values = {}
    for nid, arg in zip(graph.inputs, args):
        values[nid] = arg

    for node in graph.nodes:
        if node.op == "placeholder":
            continue
        if node.op == "constant":
            values[node.id] = node.attrs["value"]
            continue
        if node.op == "tuple":
            values[node.id] = tuple(values[i] for i in node.inputs)
            continue
        if node.op == "broadcast_scalar":
            values[node.id] = core.broadcast_scalar(
                values[node.inputs[0]], node.shape, node.attrs["scale"]
            )
            continue
        if node.op == "fused_bias_relu":
            x, bias = (values[i] for i in node.inputs)
            values[node.id] = core.fused_bias_relu(x, bias)
            continue
        if node.op == "fused_sub_square":
            a, b = (values[i] for i in node.inputs)
            values[node.id] = core.fused_sub_square(a, b)
            continue
        if node.op == "conv2d":
            x, w, b = (values[i] for i in node.inputs)
            values[node.id] = x.conv2d(w, b, node.attrs["stride"], node.attrs["padding"])
            continue
        fn = _OP_TABLE[node.op]
        values[node.id] = fn(*(values[i] for i in node.inputs))

    return values[graph.output]


def dead_code_elimination(graph: Graph) -> Graph:
    """Removes every node not on the path from a placeholder or the
    graph's output. Placeholder nodes are always kept even if unused --
    dropping one would silently change the graph's calling convention,
    which a generic DCE pass has no business doing. Renumbers surviving
    nodes so the result is still a dense, valid SSA graph in topological
    order (a filtered subsequence of an already-topological list stays
    topological)."""
    by_id = {n.id: n for n in graph.nodes}
    keep: set = set()
    stack = list(graph.inputs) + [graph.output]
    while stack:
        nid = stack.pop()
        if nid in keep:
            continue
        keep.add(nid)
        stack.extend(by_id[nid].inputs)

    remap: dict = {}
    new_graph = Graph()
    for node in graph.nodes:
        if node.id not in keep:
            continue
        new_inputs = [remap[i] for i in node.inputs]
        new_id = new_graph.add(node.op, new_inputs, list(node.shape), node.dtype, **node.attrs)
        remap[node.id] = new_id
        if node.op == "placeholder":
            new_graph.inputs.append(new_id)
    new_graph.output = remap[graph.output]
    return new_graph


# ---------------------------------------------------------------------
# Memory planning
# ---------------------------------------------------------------------

@dataclass
class MemoryPlan:
    last_use: dict            # node_id -> index of its last consumer (or len(graph.nodes) for the output)
    slot_of: dict             # node_id -> slot id, only for nodes that own a buffer (not placeholder/constant)
    num_slots: int            # peak concurrent buffers needed -- the plan's actual "footprint" number
    num_computed_nodes: int    # total nodes that ever needed a buffer, for comparison


def plan_memory(graph: Graph) -> MemoryPlan:
    """Liveness analysis plus a greedy slot assignment for a traced
    graph -- the two things a memory planner is actually for, independent
    of how the runtime chooses to act on them (see run_planned()).

    A node's liveness interval is [its own index, last_use[node]]: the
    index of the last consumer that reads it, or the graph's own output
    (kept alive to the very end). Two nodes can only share a slot if
    their intervals don't overlap; since the node list is already in a
    valid execution order (see Graph's own docstring), "doesn't overlap"
    just means one's last use is at or before the other's own index --
    no separate interval-intersection math needed.

    The assignment itself is greedy, in node order: reuse the
    most-recently-freed slot if one exists, else open a new one. Same
    family of algorithm as linear-scan register allocation. placeholder
    and constant nodes never get a slot -- they're externally owned
    (a caller's own tensor, or a Module's weight/bias), and handing one
    of those to a reuse pool would mean silently corrupting someone
    else's data, not a temporary.

    num_slots is the plan's headline number: the peak count of buffers
    ever live at once, almost always well under num_computed_nodes. For
    a graph whose op sequence repeats every call -- a training loop
    across steps, or any KIR graph run more than once, which is the
    actual case this exists for -- the same sizes recur in the same
    order every call, so a size-aware pool converges to reusing exactly
    num_slots distinct buffers after the first call or two, rather than
    mallocing fresh ones forever. See run_planned() for how that
    reuse is actually driven at runtime.
    """
    n = len(graph.nodes)
    by_id = {node.id: node for node in graph.nodes}

    last_use: dict = {}
    for idx, node in enumerate(graph.nodes):
        for inp in node.inputs:
            last_use[inp] = idx  # nodes are visited in order, so the last write wins
    if graph.output is not None:
        last_use[graph.output] = n  # never released during execution

    slot_of: dict = {}
    free_slots: list = []
    slot_busy_until: dict = {}
    next_slot = 0

    for idx, node in enumerate(graph.nodes):
        for sid, busy_until in slot_busy_until.items():
            if busy_until <= idx and sid not in free_slots:
                free_slots.append(sid)

        if node.op in ("placeholder", "constant"):
            continue

        sid = free_slots.pop() if free_slots else next_slot
        if sid == next_slot:
            next_slot += 1

        slot_of[node.id] = sid
        slot_busy_until[sid] = last_use.get(node.id, idx)

    return MemoryPlan(last_use=last_use, slot_of=slot_of, num_slots=next_slot,
                       num_computed_nodes=len(slot_of))


def run_planned(graph: Graph, plan: MemoryPlan, pool, *args) -> "core.Tensor":
    """Like run(), but every computed node's output buffer is drawn from
    `pool` (a core.StoragePool) via core.set_active_pool() instead of
    malloc'd fresh -- so a graph run hundreds of times with the same
    input shapes (a training loop) mallocs once per distinct buffer size
    and reuses those buffers on every call after.

    Forward-only and pool-exclusive, enforced rather than just
    documented: every argument must have requires_grad=False. Autograd's
    GradNode chain keeps its own references to whatever it needs for
    backward() -- if this released a buffer a GradNode still expected to
    read later, reusing that buffer for a different node would silently
    corrupt whatever .backward() computes. That failure mode is much
    worse than a wrong forward value, so it's a hard assertion, not a
    docstring warning.

    Release timing comes straight from plan.last_use: once a node's last
    consumer (per the plan) has run, nothing in this graph reads it
    again, so its buffer goes back to the pool immediately -- except
    placeholder/constant nodes, which are never released regardless of
    what last_use says, because they're externally owned (the caller's
    own tensors), not temporaries this function is allowed to recycle.

    set_active_pool/clear_active_pool are paired in a try/finally, so a
    pool never stays "active" past this call even if an op raises.

    One buffer is deliberately *not* released here: the graph's own
    output. This function can't know how long the caller wants to keep
    it -- reading its value, using it as the next call's input, whatever
    -- so ownership passes to the caller. If nothing calls
    core.release_to_pool(pool, result) once that value is no longer
    needed, the pool mallocs a fresh output buffer on every call, which
    quietly throws away most of the point of pooling in a tight loop
    that discards each iteration's result after reading it.

    Does not support a grad()-produced (multi-output "tuple") graph, on
    purpose rather than by oversight: last_use is computed per node, and
    a tuple node's inputs all get last_use == the tuple's own index, the
    same as any other consumer -- there's no equivalent yet of the
    single-output case's "kept alive to the very end" special-casing for
    each of a tuple's members. Run such a graph through this and a
    returned gradient's buffer could be handed to a later node before
    the caller ever reads it. There's no silent breakage from this,
    though: "tuple" and "broadcast_scalar" aren't in this function's
    dispatch, so it raises KeyError immediately rather than executing
    unsafely. Generalizing the protection graph.output already gets to
    every member of a tuple output is the fix; not done in this slice.
    """
    for a in args:
        assert not a.requires_grad, (
            "run_planned is forward-only: pass tensors with requires_grad=False -- "
            "a pooled buffer can't safely coexist with a live backward graph"
        )

    by_id = {node.id: node for node in graph.nodes}
    values: dict = {}
    for nid, arg in zip(graph.inputs, args):
        values[nid] = arg

    core.set_active_pool(pool)
    try:
        for idx, node in enumerate(graph.nodes):
            if node.op == "placeholder":
                pass
            elif node.op == "constant":
                values[node.id] = node.attrs["value"]
            elif node.op == "fused_bias_relu":
                x, bias = (values[i] for i in node.inputs)
                values[node.id] = core.fused_bias_relu(x, bias)
            elif node.op == "fused_sub_square":
                fa, fb = (values[i] for i in node.inputs)
                values[node.id] = core.fused_sub_square(fa, fb)
            elif node.op == "conv2d":
                cx, cw, cb = (values[i] for i in node.inputs)
                values[node.id] = cx.conv2d(cw, cb, node.attrs["stride"], node.attrs["padding"])
            else:
                fn = _OP_TABLE[node.op]
                values[node.id] = fn(*(values[i] for i in node.inputs))

            for inp in node.inputs:
                if by_id[inp].op in ("placeholder", "constant"):
                    continue
                if plan.last_use.get(inp) == idx and inp in values:
                    core.release_to_pool(pool, values[inp])
                    del values[inp]
    finally:
        core.clear_active_pool()

    return values[graph.output]


# ---------------------------------------------------------------------
# Autograd, by source transformation
# ---------------------------------------------------------------------
#
# grad(graph, wrt) returns a NEW Graph computing gradients -- not a tape
# replayed at runtime. Every op in the forward graph is re-embedded into
# the new graph first (so vjp rules have primal values to work with --
# relu's needs its input's sign, mul's needs the other operand, etc.),
# then walked in reverse, applying one vjp rule per op and accumulating
# cotangents into any node with more than one consumer -- exactly what
# Tensor::backward() does over the eager tape (core/src/Tensor.cpp),
# just building graph nodes here instead of executing real ops. The
# result is compiled and optimizable exactly like any other KIR graph
# (run it, or DCE it, or -- for the single-output case -- fuse or
# memory-plan it): this is what "the derivative is a program, not a
# tape" (the project's design decision) actually cashes out to.

def _vjp_add(bwd, node, primal_id, g_out, by_id):
    a_id, b_id = node.inputs
    grad_a = g_out
    if by_id[a_id].shape == by_id[b_id].shape:
        grad_b = g_out
    else:
        # the bias-broadcast case: b is (features,), g_out is
        # (batch, features) -- sum the batch dim back down to match.
        grad_b = bwd.add("sum_axis0", [g_out], by_id[b_id].shape, node.dtype)
    return [grad_a, grad_b]


def _vjp_sub(bwd, node, primal_id, g_out, by_id):
    shape = node.shape
    zero_id = bwd.add("constant", [], shape, node.dtype, value=core.zeros(shape))
    neg_g = bwd.add("sub", [zero_id, g_out], shape, node.dtype)
    return [g_out, neg_g]


def _vjp_mul(bwd, node, primal_id, g_out, by_id):
    a_id, b_id = node.inputs
    grad_a = bwd.add("mul", [g_out, primal_id[b_id]], node.shape, node.dtype)
    grad_b = bwd.add("mul", [g_out, primal_id[a_id]], node.shape, node.dtype)
    return [grad_a, grad_b]


def _vjp_matmul(bwd, node, primal_id, g_out, by_id):
    a_id, b_id = node.inputs
    grad_a = bwd.add("matmul_nt", [g_out, primal_id[b_id]], by_id[a_id].shape, node.dtype)
    grad_b = bwd.add("matmul_tn", [primal_id[a_id], g_out], by_id[b_id].shape, node.dtype)
    return [grad_a, grad_b]


def _vjp_relu(bwd, node, primal_id, g_out, by_id):
    x_id = node.inputs[0]
    grad_x = bwd.add("relu_backward", [primal_id[x_id], g_out], node.shape, node.dtype)
    return [grad_x]


def _vjp_sum(bwd, node, primal_id, g_out, by_id):
    x_id = node.inputs[0]
    x_shape = by_id[x_id].shape
    grad_x = bwd.add("broadcast_scalar", [g_out], x_shape, node.dtype, scale=1.0)
    return [grad_x]


def _vjp_mean(bwd, node, primal_id, g_out, by_id):
    x_id = node.inputs[0]
    x_shape = by_id[x_id].shape
    n = 1
    for d in x_shape:
        n *= d
    grad_x = bwd.add("broadcast_scalar", [g_out], x_shape, node.dtype, scale=1.0 / n)
    return [grad_x]


_VJP_RULES = {
    "add": _vjp_add,
    "sub": _vjp_sub,
    "mul": _vjp_mul,
    "matmul": _vjp_matmul,
    "relu": _vjp_relu,
    "sum": _vjp_sum,
    "mean": _vjp_mean,
}


def find_constant(graph: Graph, tensor) -> int:
    """Finds the id of the constant node holding this exact Tensor
    object (by identity -- e.g. a Module's weight or bias, as captured
    by TraceValue._coerce during tracing). For building a `wrt` list to
    pass to grad()."""
    for node in graph.nodes:
        if node.op == "constant" and node.attrs["value"] is tensor:
            return node.id
    raise ValueError("no constant node in this graph holds this tensor -- was it traced?")


def grad(graph: Graph, wrt: list) -> Graph:
    """Reverse-mode AD by source transformation: returns a new Graph
    computing d(graph.output)/d(w) for each node id w in `wrt` (typically
    placeholder or constant node ids -- find_constant() locates a
    parameter's). graph.output must be scalar (numel 1), same
    restriction as Tensor.backward().

    Must run on an *unfused*, *unplanned* graph -- apply
    elementwise_fusion or plan_memory to the result afterward if wanted,
    never before. Differentiate first, optimize the derivative program
    second: same order any AD-then-compile pipeline uses, and the only
    order that makes sense here, since the vjp rules above are written
    against the *unfused* op vocabulary (add/sub/mul/matmul/relu/...),
    not fused_bias_relu/fused_sub_square.

    A wrt node with no path back to the output gets an exact zero
    gradient of its own shape, not an error: asking for the gradient of
    something that provably didn't affect the result is a legitimate
    question with a well-defined answer, not a mistake to reject.

    Returns a single-output graph if len(wrt) == 1, or a graph whose
    output is a "tuple" node (unpacked into a Python tuple by run()/
    run_fused() -- see their dispatch loops) if len(wrt) > 1.
    """
    by_id = {n.id: n for n in graph.nodes}
    out_node = by_id[graph.output]
    out_numel = 1
    for d in out_node.shape:
        out_numel *= d
    if out_numel != 1:
        raise ValueError("grad() requires a scalar (numel==1) output, same as Tensor.backward()")

    bwd = Graph()

    # Re-embed every forward node so vjp rules have primal values to
    # read (relu's sign, mul's other operand, ...) -- this necessarily
    # recomputes the whole forward pass a second time inside bwd; see
    # the module README for the cost of that and what would avoid it.
    primal_id = {}
    for node in graph.nodes:
        if node.op == "placeholder":
            primal_id[node.id] = bwd.placeholder(node.shape, node.dtype)
        elif node.op == "constant":
            primal_id[node.id] = bwd.add("constant", [], node.shape, node.dtype, value=node.attrs["value"])
        else:
            remapped_inputs = [primal_id[i] for i in node.inputs]
            primal_id[node.id] = bwd.add(node.op, remapped_inputs, node.shape, node.dtype, **node.attrs)

    cot = {}  # forward node id -> bwd graph node id holding its accumulated cotangent

    def accumulate(node_id, new_grad_id):
        shape, dtype = by_id[node_id].shape, by_id[node_id].dtype
        if node_id in cot:
            cot[node_id] = bwd.add("add", [cot[node_id], new_grad_id], shape, dtype)
        else:
            cot[node_id] = new_grad_id

    accumulate(graph.output, bwd.add("constant", [], out_node.shape, out_node.dtype, value=core.ones(out_node.shape)))

    for node in reversed(graph.nodes):
        if node.id not in cot or node.op in ("placeholder", "constant"):
            continue
        g_out = cot[node.id]
        input_grads = _VJP_RULES[node.op](bwd, node, primal_id, g_out, by_id)
        for inp_id, grad_id in zip(node.inputs, input_grads):
            accumulate(inp_id, grad_id)

    result_ids = []
    for w in wrt:
        if w in cot:
            result_ids.append(cot[w])
        else:
            shape = by_id[w].shape
            result_ids.append(bwd.add("constant", [], shape, by_id[w].dtype, value=core.zeros(shape)))

    if len(result_ids) == 1:
        bwd.output = result_ids[0]
    else:
        bwd.output = bwd.add("tuple", result_ids, [], "float32")

    return bwd


# ---------------------------------------------------------------------
# Metal backend (Phase 3)
# ---------------------------------------------------------------------

def _metal_elementwise_kind(node, by_id):
    """Returns "bias_relu"/"add_bias" if `node` is one of the two Metal
    elementwise kernels run_metal batches, else None. For an "add" node
    this means bias-broadcast specifically (a plain same-shape add is
    left on CPU, same as run_metal's non-batched ops)."""
    if node.op == "fused_bias_relu":
        return "bias_relu"
    if node.op == "add":
        a_id, b_id = node.inputs
        if by_id[a_id].shape != by_id[b_id].shape:
            return "add_bias"
    return None


def run_metal(graph: Graph, *args) -> "core.Tensor":
    """Like run_fused(), but matmul and the fused bias+activation chain
    dispatch to the Metal backend (backend/metal) instead of Accelerate
    -- real GPU compute, same KIR graph, same op vocabulary, a different
    backend underneath. This is the architecture's central claim (KIR is
    the contract between frontend and backend; swapping backends means
    writing a new codegen/dispatch target, not touching the graph or the
    ops that produced it) demonstrated concretely rather than asserted.

    Expects an already-fused graph (elementwise_fusion's output): matmul
    dispatches to metal_matmul_mps (MPSMatrixMultiplication, Apple's own
    GEMM -- see the README for why this replaced the hand-tiled kernel
    as the default here: it's a strict upgrade in every case measured
    except a wash at the smallest sizes, and it's what actually gets
    Metal to parity with, or past, Accelerate at large problem sizes,
    which hand tiling alone never reached); fused_bias_relu and
    fused_sub_square go to their dedicated Metal kernels; a plain
    bias-broadcast add (a layer's final, unactivated output --
    elementwise_fusion only fuses add+relu *pairs*, so a solo add stays
    a solo add) goes to metal_add_bias. Every other op this graph
    vocabulary has -- sub, mul, relu, sum, mean -- now has its own Metal
    kernel too (see backend/metal/MetalOps.mm), completing what used to
    be a real gap: the loss computation (sub -> mul -> mean, or the
    fused_sub_square shape of it) used to fall back to CPU inside
    run_metal even when everything upstream of it was GPU-resident.
    conv2d goes to metal_conv2d: im2col on CPU (a memory-layout unfold,
    not FLOP-heavy) followed by metal_matmul_mps per batch item for the
    actual GEMM, plus the NCHW bias broadcast on Metal -- same
    im2col+matmul structure as the CPU Conv2d, just with the GEMM (the
    part that actually dominates runtime at any real channel count) on
    the GPU instead of Accelerate.

    tuple and broadcast_scalar are handled the same way run()/run_fused()
    handle them (unpack a multi-wrt() result / broadcast a reduction's
    cotangent back out to its input shape) rather than going to Metal --
    these only ever appear in a grad()-produced backward graph, never in
    ordinary forward tracing, and exist here purely so a distributed
    backward pass (see distributed.py's dtensor_grad) can run its
    backward graph on the "metal" device at all; without these two
    cases this function would KeyError on any such graph. The
    backward-only ops grad() itself emits -- matmul_nt, matmul_tn,
    relu_backward, sum_axis0 -- have no dedicated Metal kernel and fall
    through to the CPU _OP_TABLE below, same as any other unrecognized
    op; writing Metal kernels for those is real, unattempted future
    work, not a gap this function hides.

    Consecutive bias_relu/add_bias nodes -- wherever one's only
    non-bias input is the immediately preceding one, e.g. a second
    layer's unactivated output feeding straight off the first layer's
    fused_bias_relu with no matmul in between -- are batched into one
    core.metal_elementwise_chain call instead of dispatched one at a
    time. Measured on a synthetic chain: batching cuts 8-10x off the
    wall time at typical MLP-layer sizes, because each individual
    metal_bias_relu/metal_add_bias call pays its own command-buffer
    round trip (encode, commit, block on waitUntilCompleted, copy the
    result back to host) -- batching N steps pays for exactly one round
    trip no matter how many steps it covers.

    Forward-only, same scope as run_fused/run_planned: no grad_node is
    attached to anything computed here.
    """
    if not core.metal_available():
        raise RuntimeError("run_metal: no Metal device available on this system")

    by_id = {n.id: n for n in graph.nodes}
    values = {}
    for nid, arg in zip(graph.inputs, args):
        values[nid] = arg

    chain_input_id = None
    chain_kinds: list = []
    chain_biases: list = []
    chain_node_ids: list = []

    def flush_chain():
        nonlocal chain_input_id, chain_kinds, chain_biases, chain_node_ids
        if not chain_node_ids:
            return
        result = core.metal_elementwise_chain(values[chain_input_id], chain_kinds, chain_biases)
        values[chain_node_ids[-1]] = result
        chain_input_id, chain_kinds, chain_biases, chain_node_ids = None, [], [], []

    for node in graph.nodes:
        if node.op == "placeholder":
            continue
        if node.op == "constant":
            values[node.id] = node.attrs["value"]
            continue
        if node.op == "tuple":
            flush_chain()
            values[node.id] = tuple(values[i] for i in node.inputs)
            continue
        if node.op == "broadcast_scalar":
            flush_chain()
            values[node.id] = core.broadcast_scalar(
                values[node.inputs[0]], node.shape, node.attrs["scale"]
            )
            continue

        kind = _metal_elementwise_kind(node, by_id)
        if kind is not None:
            primary_id, bias_id = node.inputs
            if chain_node_ids and primary_id == chain_node_ids[-1]:
                chain_kinds.append(kind)
                chain_biases.append(values[bias_id])
                chain_node_ids.append(node.id)
            else:
                flush_chain()
                chain_input_id = primary_id
                chain_kinds = [kind]
                chain_biases = [values[bias_id]]
                chain_node_ids = [node.id]
            continue

        flush_chain()  # this node isn't chainable -- dispatch whatever was pending first

        if node.op == "matmul":
            a, b = (values[i] for i in node.inputs)
            values[node.id] = core.metal_matmul_mps(a, b)
            continue
        if node.op == "fused_sub_square":
            a, b = (values[i] for i in node.inputs)
            values[node.id] = core.metal_fused_sub_square(a, b)
            continue
        if node.op == "add":
            # Only a same-shape add ever reaches here -- a bias-broadcast
            # one was already caught by _metal_elementwise_kind above and
            # routed through the batched chain instead.
            a, b = (values[i] for i in node.inputs)
            values[node.id] = core.metal_add(a, b)
            continue
        if node.op == "sub":
            a, b = (values[i] for i in node.inputs)
            values[node.id] = core.metal_sub(a, b)
            continue
        if node.op == "mul":
            a, b = (values[i] for i in node.inputs)
            values[node.id] = core.metal_mul(a, b)
            continue
        if node.op == "relu":
            (x,) = (values[i] for i in node.inputs)
            values[node.id] = core.metal_relu(x)
            continue
        if node.op == "sum":
            (x,) = (values[i] for i in node.inputs)
            values[node.id] = core.metal_sum(x)
            continue
        if node.op == "mean":
            (x,) = (values[i] for i in node.inputs)
            values[node.id] = core.metal_mean(x)
            continue
        if node.op == "conv2d":
            x, w, b = (values[i] for i in node.inputs)
            values[node.id] = core.metal_conv2d(x, w, b, node.attrs["stride"], node.attrs["padding"])
            continue
        fn = _OP_TABLE[node.op]
        values[node.id] = fn(*(values[i] for i in node.inputs))

    flush_chain()
    return values[graph.output]

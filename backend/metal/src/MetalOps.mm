#include "kansai/backend/metal/MetalOps.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cstring>
#include <stdexcept>

namespace {

// Must match kTileSize in matmul() below -- the kernel's threadgroup
// arrays and the C++ dispatch's threadsPerThreadgroup both have to
// agree on the tile's edge length. Interpolated into the MSL source
// text below (a raw string literal is never macro-expanded, so a
// #define here would leave the *literal text* "KANSAI_MATMUL_TILE"
// inside the shader source for Metal's own compiler to choke on as an
// undefined identifier -- this has to be a real substitution).
constexpr int kMatmulTile = 16;

NSString* const kShaderSource = [NSString stringWithFormat:@R"MSL(
#include <metal_stdlib>
using namespace metal;

// Tiled matmul: the naive version (one GPU thread per output element,
// reading full rows/columns straight from device memory every time)
// made every thread in a threadgroup re-fetch the *same* rows of A and
// columns of B its neighbors were already fetching -- no reuse at all,
// entirely global-memory-bandwidth bound. This stages one
// TILE x TILE tile of A and one of B into threadgroup (on-chip shared)
// memory per step, synchronizes once so every thread has finished
// writing before any thread reads, then has all TILE*TILE threads in
// the group reuse those two tiles for TILE*TILE multiply-adds each --
// cutting global memory traffic by roughly a factor of TILE compared
// to the naive kernel. Boundary tiles (M, K, or N not a multiple of
// TILE) are handled by zero-padding an out-of-range read and masking
// an out-of-range write, so correctness doesn't depend on the problem
// size being tile-aligned.
constant uint TILE = %d;

kernel void matmul_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* out     [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]])
{
    threadgroup float Asub[%d][%d];
    threadgroup float Bsub[%d][%d];

    uint row = gid.y;
    uint col = gid.x;
    float acc = 0.0;

    uint numTiles = (K + TILE - 1) / TILE;
    for (uint t = 0; t < numTiles; ++t) {
        uint aCol = t * TILE + tid.x;
        uint bRow = t * TILE + tid.y;

        Asub[tid.y][tid.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0;
        Bsub[tid.y][tid.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; ++k) {
            acc += Asub[tid.y][k] * Bsub[k][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        out[row * N + col] = acc;
    }
}

kernel void bias_relu_kernel(
    device const float* x    [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* out        [[buffer(2)]],
    constant uint& features [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    float v = x[gid] + bias[gid %% features];
    out[gid] = v > 0.0 ? v : 0.0;
}

kernel void add_bias_kernel(
    device const float* x    [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* out        [[buffer(2)]],
    constant uint& features [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    out[gid] = x[gid] + bias[gid %% features];
}

// Conv2d's bias broadcast: bias varies per output *channel* -- the
// middle dimension of a flattened (N, C, HW) index -- not the last
// dimension the way Linear's (batch, features) bias does, so this needs
// its own kernel rather than reusing add_bias_kernel above.
kernel void add_bias_nchw_kernel(
    device const float* x    [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* out        [[buffer(2)]],
    constant uint& C [[buffer(3)]],
    constant uint& HW [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint c = (gid / HW) %% C;
    out[gid] = x[gid] + bias[c];
}

// Plain same-shape elementwise ops -- completing Metal's op coverage
// alongside the bias-broadcast kernels above (Linear's own epilogue
// needed those; loss computation, and anything shaped like a residual
// connection, needs these). One thread per element, same as
// bias_relu/add_bias.
kernel void add_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out     [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    out[gid] = a[gid] + b[gid];
}

kernel void sub_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out     [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    out[gid] = a[gid] - b[gid];
}

kernel void mul_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out     [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    out[gid] = a[gid] * b[gid];
}

kernel void relu_kernel(
    device const float* x [[buffer(0)]],
    device float* out     [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    out[gid] = max(x[gid], 0.0);
}

// (a-b)^2 in one pass -- the Metal-side twin of the CPU backend's
// fused_sub_square, the diff*diff core of MSE loss.
kernel void fused_sub_square_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out     [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    float d = a[gid] - b[gid];
    out[gid] = d * d;
}

// sum(x)*scale in one dispatch -- scale=1 for sum(), scale=1/n for
// mean(). A single threadgroup handles the whole reduction: each thread
// first grid-strides over the input accumulating a partial sum (so this
// is correct for any n, not just n <= threadgroup size), then a
// standard tree reduction in threadgroup memory combines the 256
// partials down to one value. Not the fastest possible reduction
// (multiple threadgroups with a second combining pass would scale
// better for very large n), but every reduction in this codebase's
// actual use (loss values) is small -- correctness and simplicity over
// squeezing out a multi-pass reduction for inputs this size.
kernel void reduce_sum_kernel(
    device const float* x [[buffer(0)]],
    device float* out     [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]])
{
    threadgroup float shared[256];
    float local_sum = 0.0;
    for (uint i = tid; i < n; i += tgSize) {
        local_sum += x[i];
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[0] = shared[0] * scale;
}
)MSL", kMatmulTile, kMatmulTile, kMatmulTile, kMatmulTile, kMatmulTile];

struct MetalState {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> matmul_pipeline = nil;
    id<MTLComputePipelineState> bias_relu_pipeline = nil;
    id<MTLComputePipelineState> add_bias_pipeline = nil;
    id<MTLComputePipelineState> add_bias_nchw_pipeline = nil;
    id<MTLComputePipelineState> add_pipeline = nil;
    id<MTLComputePipelineState> sub_pipeline = nil;
    id<MTLComputePipelineState> mul_pipeline = nil;
    id<MTLComputePipelineState> relu_pipeline = nil;
    id<MTLComputePipelineState> fused_sub_square_pipeline = nil;
    id<MTLComputePipelineState> reduce_sum_pipeline = nil;
    bool ok = false;
};

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString* name) {
    id<MTLFunction> fn = [library newFunctionWithName:name];
    if (!fn) return nil;
    NSError* error = nil;
    return [device newComputePipelineStateWithFunction:fn error:&error];
}

MetalState& state() {
    static MetalState s = [] {
        MetalState st;
        @autoreleasepool {
            st.device = MTLCreateSystemDefaultDevice();
            if (!st.device) return st;
            st.queue = [st.device newCommandQueue];

            NSError* error = nil;
            id<MTLLibrary> library = [st.device newLibraryWithSource:kShaderSource options:nil error:&error];
            if (!library) return st;

            st.matmul_pipeline = make_pipeline(st.device, library, @"matmul_kernel");
            st.bias_relu_pipeline = make_pipeline(st.device, library, @"bias_relu_kernel");
            st.add_bias_pipeline = make_pipeline(st.device, library, @"add_bias_kernel");
            st.add_bias_nchw_pipeline = make_pipeline(st.device, library, @"add_bias_nchw_kernel");
            st.add_pipeline = make_pipeline(st.device, library, @"add_kernel");
            st.sub_pipeline = make_pipeline(st.device, library, @"sub_kernel");
            st.mul_pipeline = make_pipeline(st.device, library, @"mul_kernel");
            st.relu_pipeline = make_pipeline(st.device, library, @"relu_kernel");
            st.fused_sub_square_pipeline = make_pipeline(st.device, library, @"fused_sub_square_kernel");
            st.reduce_sum_pipeline = make_pipeline(st.device, library, @"reduce_sum_kernel");
            if (!st.matmul_pipeline || !st.bias_relu_pipeline || !st.add_bias_pipeline
                || !st.add_bias_nchw_pipeline || !st.add_pipeline || !st.sub_pipeline || !st.mul_pipeline
                || !st.relu_pipeline || !st.fused_sub_square_pipeline || !st.reduce_sum_pipeline) {
                return st;
            }

            st.ok = true;
        }
        return st;
    }();
    return s;
}

void require_available() {
    if (!state().ok) throw std::runtime_error("kan::metal: no Metal device/pipeline available on this system");
}

// Wraps `ptr` directly as a GPU-visible buffer instead of copying it
// into a separate Metal-owned one -- the whole point of NoCopy. Safe
// only because kan::Storage (core/Storage.cpp) always allocates
// page-aligned memory rounded *up* to a full page, specifically so
// every Tensor's buffer qualifies for this; `length` here only needs to
// be the logical byte count actually used (<=  the real allocation),
// never the full page-rounded size. `deallocator:nil` means Metal never
// frees this memory itself -- its lifetime stays owned by the caller's
// Storage, which must outlive the command buffer this gets used in;
// true everywhere here since every metal:: function blocks on
// waitUntilCompleted before returning, so the underlying Tensor is
// always still alive (the caller holds it) when the GPU touches it.
// Input buffers are declared `device const float*` in every kernel
// here, so the GPU-side compiler -- not the C++ type system -- is what
// actually enforces read-only access to a `const float*` wrapped this
// way; the const_cast just satisfies the Objective-C API, which has no
// const-correct overload.
id<MTLBuffer> wrap_no_copy(id<MTLDevice> device, const void* ptr, NSUInteger length) {
    id<MTLBuffer> buf = [device newBufferWithBytesNoCopy:const_cast<void*>(ptr)
                                                    length:length
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
    if (!buf) throw std::runtime_error("kan::metal: newBufferWithBytesNoCopy failed (pointer not page-aligned?)");
    return buf;
}

// Shared dispatch shape for the two elementwise kernels (bias_relu,
// add_bias): both take (x, bias, out, features) and run one thread per
// output element.
void dispatch_elementwise_bias_op(id<MTLComputePipelineState> pipeline, const float* x, const float* bias,
                                   float* out, int64_t batch, int64_t features) {
    require_available();
    @autoreleasepool {
        MetalState& s = state();
        int64_t n = batch * features;

        id<MTLBuffer> buf_x = wrap_no_copy(s.device, x, static_cast<NSUInteger>(n * sizeof(float)));
        id<MTLBuffer> buf_bias = wrap_no_copy(s.device, bias, static_cast<NSUInteger>(features * sizeof(float)));
        id<MTLBuffer> buf_out = wrap_no_copy(s.device, out, static_cast<NSUInteger>(n * sizeof(float)));
        uint32_t uF = static_cast<uint32_t>(features);

        id<MTLCommandBuffer> cmd = [s.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_x offset:0 atIndex:0];
        [enc setBuffer:buf_bias offset:0 atIndex:1];
        [enc setBuffer:buf_out offset:0 atIndex:2];
        [enc setBytes:&uF length:sizeof(uint32_t) atIndex:3];

        NSUInteger tw = MIN(static_cast<NSUInteger>(n), pipeline.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(n), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        // No memcpy: buf_out wraps `out`'s own memory directly, and the
        // GPU's writes are guaranteed visible on the CPU the moment
        // waitUntilCompleted returns.
    }
}

// Same-shape elementwise binary op: one thread per element, no bias
// broadcast (that's dispatch_elementwise_bias_op, above).
void dispatch_elementwise_binary_op(id<MTLComputePipelineState> pipeline, const float* a, const float* b,
                                     float* out, int64_t n) {
    require_available();
    @autoreleasepool {
        MetalState& s = state();
        NSUInteger nbytes = static_cast<NSUInteger>(n * sizeof(float));

        id<MTLBuffer> buf_a = wrap_no_copy(s.device, a, nbytes);
        id<MTLBuffer> buf_b = wrap_no_copy(s.device, b, nbytes);
        id<MTLBuffer> buf_out = wrap_no_copy(s.device, out, nbytes);

        id<MTLCommandBuffer> cmd = [s.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_a offset:0 atIndex:0];
        [enc setBuffer:buf_b offset:0 atIndex:1];
        [enc setBuffer:buf_out offset:0 atIndex:2];

        NSUInteger tw = MIN(static_cast<NSUInteger>(n), pipeline.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(n), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void dispatch_elementwise_unary_op(id<MTLComputePipelineState> pipeline, const float* x, float* out, int64_t n) {
    require_available();
    @autoreleasepool {
        MetalState& s = state();
        NSUInteger nbytes = static_cast<NSUInteger>(n * sizeof(float));

        id<MTLBuffer> buf_x = wrap_no_copy(s.device, x, nbytes);
        id<MTLBuffer> buf_out = wrap_no_copy(s.device, out, nbytes);

        id<MTLCommandBuffer> cmd = [s.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_x offset:0 atIndex:0];
        [enc setBuffer:buf_out offset:0 atIndex:1];

        NSUInteger tw = MIN(static_cast<NSUInteger>(n), pipeline.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(n), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace

namespace kan::metal {

bool available() { return state().ok; }

void matmul(const float* a, const float* b, float* out, int64_t M, int64_t K, int64_t N) {
    require_available();
    @autoreleasepool {
        MetalState& s = state();

        id<MTLBuffer> buf_a = wrap_no_copy(s.device, a, static_cast<NSUInteger>(M * K * sizeof(float)));
        id<MTLBuffer> buf_b = wrap_no_copy(s.device, b, static_cast<NSUInteger>(K * N * sizeof(float)));
        id<MTLBuffer> buf_out = wrap_no_copy(s.device, out, static_cast<NSUInteger>(M * N * sizeof(float)));

        uint32_t uM = static_cast<uint32_t>(M), uK = static_cast<uint32_t>(K), uN = static_cast<uint32_t>(N);

        id<MTLCommandBuffer> cmd = [s.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:s.matmul_pipeline];
        [enc setBuffer:buf_a offset:0 atIndex:0];
        [enc setBuffer:buf_b offset:0 atIndex:1];
        [enc setBuffer:buf_out offset:0 atIndex:2];
        [enc setBytes:&uM length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&uK length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&uN length:sizeof(uint32_t) atIndex:5];

        // dispatchThreadgroups (not dispatchThreads) deliberately: the
        // tiled kernel needs every threadgroup to be the FULL
        // kMatmulTile x kMatmulTile, even at the M/N boundary, since
        // every thread in the group must participate in loading Asub/
        // Bsub (its own output can still be masked out via the row/col
        // bounds check) -- dispatchThreads's non-uniform threadgroup
        // sizing at boundaries would hand some boundary threadgroups
        // fewer threads than kMatmulTile*kMatmulTile, leaving parts of
        // the shared tile unwritten for whichever threads never got
        // scheduled.
        MTLSize threadsPerThreadgroup = MTLSizeMake(kMatmulTile, kMatmulTile, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake(
            (static_cast<NSUInteger>(N) + kMatmulTile - 1) / kMatmulTile,
            (static_cast<NSUInteger>(M) + kMatmulTile - 1) / kMatmulTile,
            1);
        [enc dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        // No memcpy: buf_out wraps `out` directly.
    }
}

void matmul_mps(const float* a, const float* b, float* out, int64_t M, int64_t K, int64_t N) {
    require_available();
    @autoreleasepool {
        MetalState& s = state();

        NSUInteger rowBytesA = static_cast<NSUInteger>(K) * sizeof(float);
        NSUInteger rowBytesB = static_cast<NSUInteger>(N) * sizeof(float);
        NSUInteger rowBytesC = static_cast<NSUInteger>(N) * sizeof(float);

        id<MTLBuffer> buf_a = wrap_no_copy(s.device, a, static_cast<NSUInteger>(M) * rowBytesA);
        id<MTLBuffer> buf_b = wrap_no_copy(s.device, b, static_cast<NSUInteger>(K) * rowBytesB);
        id<MTLBuffer> buf_out = wrap_no_copy(s.device, out, static_cast<NSUInteger>(M) * rowBytesC);

        MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(M)
                                                                            columns:static_cast<NSUInteger>(K)
                                                                           rowBytes:rowBytesA
                                                                           dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(K)
                                                                            columns:static_cast<NSUInteger>(N)
                                                                           rowBytes:rowBytesB
                                                                           dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(M)
                                                                            columns:static_cast<NSUInteger>(N)
                                                                           rowBytes:rowBytesC
                                                                           dataType:MPSDataTypeFloat32];

        MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:buf_a descriptor:descA];
        MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:buf_b descriptor:descB];
        MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:buf_out descriptor:descC];

        MPSMatrixMultiplication* gemm = [[MPSMatrixMultiplication alloc] initWithDevice:s.device
                                                                           transposeLeft:NO
                                                                          transposeRight:NO
                                                                              resultRows:static_cast<NSUInteger>(M)
                                                                           resultColumns:static_cast<NSUInteger>(N)
                                                                         interiorColumns:static_cast<NSUInteger>(K)
                                                                                   alpha:1.0
                                                                                    beta:0.0];

        id<MTLCommandBuffer> cmd = [s.queue commandBuffer];
        [gemm encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        [cmd commit];
        [cmd waitUntilCompleted];
        // No memcpy: buf_out wraps `out` directly.
    }
}

void bias_relu(const float* x, const float* bias, float* out, int64_t batch, int64_t features) {
    dispatch_elementwise_bias_op(state().bias_relu_pipeline, x, bias, out, batch, features);
}

void add_bias(const float* x, const float* bias, float* out, int64_t batch, int64_t features) {
    dispatch_elementwise_bias_op(state().add_bias_pipeline, x, bias, out, batch, features);
}

void add_bias_nchw(const float* x, const float* bias, float* out, int64_t N, int64_t C, int64_t HW) {
    require_available();
    @autoreleasepool {
        MetalState& s = state();
        int64_t n = N * C * HW;

        id<MTLBuffer> buf_x = wrap_no_copy(s.device, x, static_cast<NSUInteger>(n * sizeof(float)));
        id<MTLBuffer> buf_bias = wrap_no_copy(s.device, bias, static_cast<NSUInteger>(C * sizeof(float)));
        id<MTLBuffer> buf_out = wrap_no_copy(s.device, out, static_cast<NSUInteger>(n * sizeof(float)));
        uint32_t uC = static_cast<uint32_t>(C), uHW = static_cast<uint32_t>(HW);

        id<MTLCommandBuffer> cmd = [s.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:s.add_bias_nchw_pipeline];
        [enc setBuffer:buf_x offset:0 atIndex:0];
        [enc setBuffer:buf_bias offset:0 atIndex:1];
        [enc setBuffer:buf_out offset:0 atIndex:2];
        [enc setBytes:&uC length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&uHW length:sizeof(uint32_t) atIndex:4];

        NSUInteger tw = MIN(static_cast<NSUInteger>(n), s.add_bias_nchw_pipeline.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(n), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void add(const float* a, const float* b, float* out, int64_t n) {
    dispatch_elementwise_binary_op(state().add_pipeline, a, b, out, n);
}

void sub(const float* a, const float* b, float* out, int64_t n) {
    dispatch_elementwise_binary_op(state().sub_pipeline, a, b, out, n);
}

void mul(const float* a, const float* b, float* out, int64_t n) {
    dispatch_elementwise_binary_op(state().mul_pipeline, a, b, out, n);
}

void relu(const float* x, float* out, int64_t n) {
    dispatch_elementwise_unary_op(state().relu_pipeline, x, out, n);
}

void fused_sub_square(const float* a, const float* b, float* out, int64_t n) {
    dispatch_elementwise_binary_op(state().fused_sub_square_pipeline, a, b, out, n);
}

void reduce_sum(const float* x, float* out, int64_t n, float scale) {
    require_available();
    @autoreleasepool {
        MetalState& s = state();
        id<MTLBuffer> buf_x = wrap_no_copy(s.device, x, static_cast<NSUInteger>(n * sizeof(float)));
        id<MTLBuffer> buf_out = wrap_no_copy(s.device, out, sizeof(float));
        uint32_t uN = static_cast<uint32_t>(n);

        id<MTLCommandBuffer> cmd = [s.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:s.reduce_sum_pipeline];
        [enc setBuffer:buf_x offset:0 atIndex:0];
        [enc setBuffer:buf_out offset:0 atIndex:1];
        [enc setBytes:&uN length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&scale length:sizeof(float) atIndex:3];

        // Fixed at 256 to match the kernel's `threadgroup float
        // shared[256]` -- one threadgroup handles the entire reduction
        // (see reduce_sum_kernel's own comment for why that's the right
        // tradeoff for this codebase's actual reduction sizes).
        constexpr NSUInteger kReduceThreads = 256;
        [enc dispatchThreads:MTLSizeMake(kReduceThreads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kReduceThreads, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void run_elementwise_chain(const float* x0, int64_t batch, int64_t features,
                            const std::vector<ElemStep>& steps, float* out) {
    require_available();
    @autoreleasepool {
        MetalState& s = state();
        int64_t n = batch * features;
        NSUInteger nbytes = static_cast<NSUInteger>(n * sizeof(float));
        uint32_t uF = static_cast<uint32_t>(features);

        id<MTLCommandBuffer> cmd = [s.queue commandBuffer];

        id<MTLBuffer> cur = wrap_no_copy(s.device, x0, nbytes);

        for (size_t i = 0; i < steps.size(); ++i) {
            const ElemStep& step = steps[i];
            bool is_last = (i + 1 == steps.size());

            id<MTLBuffer> bias_buf = wrap_no_copy(s.device, step.bias, static_cast<NSUInteger>(features * sizeof(float)));
            // Every step but the last writes into a pure GPU scratch
            // buffer (no corresponding host tensor exists for an
            // intermediate value, so there's nothing to NoCopy-wrap);
            // the last step writes directly into the caller's `out`,
            // which -- like every kan::Tensor's storage -- is already
            // page-aligned, so wrapping it skips the final copy-back
            // entirely.
            id<MTLBuffer> next = is_last
                ? wrap_no_copy(s.device, out, nbytes)
                : [s.device newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
            id<MTLComputePipelineState> pipeline =
                step.kernel == ElemKernel::BiasRelu ? s.bias_relu_pipeline : s.add_bias_pipeline;

            // A new encoder per step (ended before the next begins) is
            // the standard way to chain several dispatches into one
            // command buffer -- Metal's automatic hazard tracking makes
            // this step's writes to `next` visible to the following
            // step's read of it as `cur`, with no explicit fence needed.
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:cur offset:0 atIndex:0];
            [enc setBuffer:bias_buf offset:0 atIndex:1];
            [enc setBuffer:next offset:0 atIndex:2];
            [enc setBytes:&uF length:sizeof(uint32_t) atIndex:3];

            NSUInteger tw = MIN(static_cast<NSUInteger>(n), pipeline.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(n), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
            [enc endEncoding];

            cur = next;
        }

        [cmd commit];
        [cmd waitUntilCompleted];
        // No memcpy: the last step's buffer wraps `out` directly.
    }
}

} // namespace kan::metal

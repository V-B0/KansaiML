#include "kansai/backend/metal/MetalOps.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstring>
#include <stdexcept>

namespace {

NSString* const kShaderSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void matmul_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out     [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= N || gid.y >= M) return;
    float acc = 0.0;
    for (uint k = 0; k < K; ++k) {
        acc += a[gid.y * K + k] * b[k * N + gid.x];
    }
    out[gid.y * N + gid.x] = acc;
}

kernel void bias_relu_kernel(
    device const float* x    [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* out        [[buffer(2)]],
    constant uint& features [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    float v = x[gid] + bias[gid % features];
    out[gid] = v > 0.0 ? v : 0.0;
}

kernel void add_bias_kernel(
    device const float* x    [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* out        [[buffer(2)]],
    constant uint& features [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    out[gid] = x[gid] + bias[gid % features];
}
)MSL";

struct MetalState {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> matmul_pipeline = nil;
    id<MTLComputePipelineState> bias_relu_pipeline = nil;
    id<MTLComputePipelineState> add_bias_pipeline = nil;
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
            if (!st.matmul_pipeline || !st.bias_relu_pipeline || !st.add_bias_pipeline) return st;

            st.ok = true;
        }
        return st;
    }();
    return s;
}

void require_available() {
    if (!state().ok) throw std::runtime_error("kan::metal: no Metal device/pipeline available on this system");
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

        id<MTLBuffer> buf_x = [s.device newBufferWithBytes:x
                                                      length:static_cast<NSUInteger>(n * sizeof(float))
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_bias = [s.device newBufferWithBytes:bias
                                                         length:static_cast<NSUInteger>(features * sizeof(float))
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_out = [s.device newBufferWithLength:static_cast<NSUInteger>(n * sizeof(float))
                                                       options:MTLResourceStorageModeShared];
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

        std::memcpy(out, [buf_out contents], static_cast<size_t>(n) * sizeof(float));
    }
}

} // namespace

namespace kan::metal {

bool available() { return state().ok; }

void matmul(const float* a, const float* b, float* out, int64_t M, int64_t K, int64_t N) {
    require_available();
    @autoreleasepool {
        MetalState& s = state();

        id<MTLBuffer> buf_a = [s.device newBufferWithBytes:a
                                                      length:static_cast<NSUInteger>(M * K * sizeof(float))
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_b = [s.device newBufferWithBytes:b
                                                      length:static_cast<NSUInteger>(K * N * sizeof(float))
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_out = [s.device newBufferWithLength:static_cast<NSUInteger>(M * N * sizeof(float))
                                                       options:MTLResourceStorageModeShared];

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

        NSUInteger w = s.matmul_pipeline.threadExecutionWidth;
        NSUInteger h = s.matmul_pipeline.maxTotalThreadsPerThreadgroup / w;
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(N), static_cast<NSUInteger>(M), 1)
        threadsPerThreadgroup:MTLSizeMake(w, h, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        std::memcpy(out, [buf_out contents], static_cast<size_t>(M * N) * sizeof(float));
    }
}

void bias_relu(const float* x, const float* bias, float* out, int64_t batch, int64_t features) {
    dispatch_elementwise_bias_op(state().bias_relu_pipeline, x, bias, out, batch, features);
}

void add_bias(const float* x, const float* bias, float* out, int64_t batch, int64_t features) {
    dispatch_elementwise_bias_op(state().add_bias_pipeline, x, bias, out, batch, features);
}

} // namespace kan::metal

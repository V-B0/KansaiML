#include "kansai/backend/cpu/Fused.hpp"

namespace kan::cpu {

void fused_bias_relu(const float* x, const float* bias, float* out, int64_t batch, int64_t features) {
    for (int64_t i = 0; i < batch; ++i) {
        const float* xrow = x + i * features;
        float* orow = out + i * features;
        for (int64_t j = 0; j < features; ++j) {
            float v = xrow[j] + bias[j];
            orow[j] = v > 0.0f ? v : 0.0f;
        }
    }
}

void fused_sub_square(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        float d = a[i] - b[i];
        out[i] = d * d;
    }
}

} // namespace kan::cpu

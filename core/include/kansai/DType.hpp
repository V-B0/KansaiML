#pragma once
#include <cstddef>

namespace kan {

enum class DType {
    Float32,
};

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::Float32: return sizeof(float);
    }
    return 0;
}

} // namespace kan

#include "kansai/Storage.hpp"
#include <cstdlib>
#include <new>

namespace kan {

Storage::Storage(size_t nbytes) : data_(nullptr), nbytes_(nbytes) {
    if (nbytes_ > 0) {
        data_ = std::malloc(nbytes_);
        if (!data_) throw std::bad_alloc();
    }
}

Storage::~Storage() {
    std::free(data_);
}

} // namespace kan

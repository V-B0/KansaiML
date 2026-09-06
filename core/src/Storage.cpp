#include "kansai/Storage.hpp"
#include <cstdlib>
#include <new>
#include <unistd.h>

namespace kan {

namespace {

size_t page_size() {
    static const size_t sz = static_cast<size_t>(getpagesize());
    return sz;
}

size_t round_up_to_page(size_t n) {
    size_t p = page_size();
    return ((n + p - 1) / p) * p;
}

} // namespace

Storage::Storage(size_t nbytes) : data_(nullptr), nbytes_(0) {
    // Always allocate at least one page, even for a zero-byte request,
    // so a Storage never wraps a null/zero-length region -- Metal's
    // NoCopy buffer wrapping (backend/metal) needs a real, non-empty,
    // page-aligned block to point at regardless of the tensor's logical
    // size.
    nbytes_ = round_up_to_page(nbytes > 0 ? nbytes : 1);
    if (posix_memalign(&data_, page_size(), nbytes_) != 0) {
        throw std::bad_alloc();
    }
}

Storage::~Storage() {
    std::free(data_);
}

} // namespace kan

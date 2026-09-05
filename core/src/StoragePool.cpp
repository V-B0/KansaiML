#include "kansai/StoragePool.hpp"

namespace kan {

StoragePtr StoragePool::acquire(size_t nbytes) {
    // First-fit: any free buffer big enough is reused as-is, even if
    // it's larger than requested. For a graph whose op sequence repeats
    // every call (the case this pool exists for), the same sizes recur
    // in the same order every time, so this converges to a stable reuse
    // pattern after the first call or two rather than fragmenting.
    for (size_t i = 0; i < free_list_.size(); ++i) {
        if (free_list_[i]->nbytes() >= nbytes) {
            StoragePtr found = std::move(free_list_[i]);
            free_list_.erase(free_list_.begin() + static_cast<long>(i));
            return found;
        }
    }
    ++total_allocated_;
    return std::make_shared<Storage>(nbytes);
}

void StoragePool::release(StoragePtr storage) {
    free_list_.push_back(std::move(storage));
}

} // namespace kan

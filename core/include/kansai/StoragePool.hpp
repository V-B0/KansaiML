#pragma once
#include "kansai/Storage.hpp"
#include <cstddef>
#include <vector>

namespace kan {

// A first-fit free-list pool of Storage buffers, meant to be reused
// across repeated executions of the same KIR graph (see kir.run_planned
// in kir.py) instead of malloc'ing a fresh buffer for every intermediate
// tensor on every call. Correct release timing -- never handing out a
// buffer someone still holds a live reference to -- is the caller's
// responsibility; see run_planned()'s docstring for how that safety is
// actually maintained.
class StoragePool {
public:
    StoragePtr acquire(size_t nbytes);
    void release(StoragePtr storage);

    size_t num_free() const { return free_list_.size(); }
    size_t num_allocated() const { return total_allocated_; }

private:
    std::vector<StoragePtr> free_list_;
    size_t total_allocated_ = 0;
};

} // namespace kan

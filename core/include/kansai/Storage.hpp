#pragma once
#include <cstddef>
#include <memory>

namespace kan {

// Owns a single raw, ref-counted heap allocation. No dtype/shape knowledge —
// that lives one layer up, in TensorData.
class Storage {
public:
    explicit Storage(size_t nbytes);
    ~Storage();

    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t nbytes() const { return nbytes_; }

private:
    void* data_;
    size_t nbytes_;
};

using StoragePtr = std::shared_ptr<Storage>;

} // namespace kan

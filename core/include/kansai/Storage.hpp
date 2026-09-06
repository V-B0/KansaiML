#pragma once
#include <cstddef>
#include <memory>

namespace kan {

// Owns a single raw, ref-counted heap allocation. No dtype/shape knowledge —
// that lives one layer up, in TensorData.
//
// Always page-aligned (the constructor rounds the requested size up to
// the system page size and allocates via posix_memalign) -- not for any
// CPU-side reason, but so backend/metal can wrap this exact memory with
// MTLDevice::newBufferWithBytesNoCopy instead of copying it into a
// separate GPU-visible buffer first. Apple Silicon's unified memory
// means CPU and GPU already share the same physical RAM; a page-aligned
// CPU allocation is genuinely GPU-visible with zero copying, not just
// in principle. nbytes() reports the actual (rounded-up) allocation
// size, not the originally requested one -- Metal's NoCopy wrapping
// needs the real size, and the only other consumer (StoragePool's
// free-list size check) only becomes more permissive by seeing the
// true, possibly-larger figure.
//
// The cost: every Storage now consumes at least one full page (16KB on
// Apple Silicon), even for a handful of floats. Negligible for
// realistic tensor sizes; a real (if small in absolute terms) floor for
// code that allocates many tiny tensors.
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

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "kansai/Fused.hpp"
#include "kansai/GradOps.hpp"
#include "kansai/StoragePool.hpp"
#include "kansai/Tensor.hpp"
#ifdef KANSAI_HAS_METAL
#include "kansai/MetalOps.hpp"
#endif

namespace nb = nanobind;
using namespace kan;

NB_MODULE(_core, m) {
    m.doc() = "Kansai core tensor engine (Milestone 0 — CPU only)";

    nb::class_<Tensor>(m, "Tensor")
        .def("__repr__", [](const Tensor& t) {
            std::string s = "Tensor(shape=[";
            for (size_t i = 0; i < t.shape().size(); ++i) {
                if (i) s += ", ";
                s += std::to_string(t.shape()[i]);
            }
            s += "], requires_grad=";
            s += (t.requires_grad() ? "True" : "False");
            s += ")";
            return s;
        })
        .def_prop_ro("shape", [](const Tensor& t) { return t.shape(); })
        .def_prop_ro("requires_grad", &Tensor::requires_grad)
        .def_prop_ro("grad", [](const Tensor& t) -> std::optional<Tensor> { return t.grad(); })
        .def("numel", &Tensor::numel)
        .def("tolist", &Tensor::to_vector)
        .def("backward", &Tensor::backward)
        .def("zero_grad", &Tensor::zero_grad)
        // call_guard<gil_scoped_release> on every compute-heavy method
        // below: each is pure C++ number-crunching on its own Tensor's
        // buffers (Accelerate calls or hand-written loops), touching no
        // Python object once inside, so releasing the GIL for the
        // duration is safe -- and is exactly what lets kir.run's "cpu"
        // dispatch and kir.run_metal's "metal" dispatch actually overlap
        // when driven from separate Python threads (see
        // distributed.py's dtensor_run/dtensor_grad, the reason this
        // was added). Left off zero_grad/tolist/etc.: trivial, not on
        // any hot path this matters for.
        .def("add_", &Tensor::add_, nb::arg("other"), nb::arg("alpha") = 1.0f,
             nb::call_guard<nb::gil_scoped_release>())
        .def("add", &Tensor::add, nb::arg("other"), nb::call_guard<nb::gil_scoped_release>())
        .def("sub", &Tensor::sub, nb::arg("other"), nb::call_guard<nb::gil_scoped_release>())
        .def("mul", &Tensor::mul, nb::arg("other"), nb::call_guard<nb::gil_scoped_release>())
        .def("matmul", &Tensor::matmul, nb::arg("other"), nb::call_guard<nb::gil_scoped_release>())
        .def("relu", &Tensor::relu, nb::call_guard<nb::gil_scoped_release>())
        .def("sum", &Tensor::sum, nb::call_guard<nb::gil_scoped_release>())
        .def("mean", &Tensor::mean, nb::call_guard<nb::gil_scoped_release>())
        .def("conv2d", &Tensor::conv2d, nb::arg("weight"), nb::arg("bias"), nb::arg("stride"), nb::arg("padding"),
             nb::call_guard<nb::gil_scoped_release>())
        .def("reshape", &Tensor::reshape, nb::arg("shape"), nb::call_guard<nb::gil_scoped_release>())
        .def("transpose", &Tensor::transpose, nb::arg("dim0"), nb::arg("dim1"),
             nb::call_guard<nb::gil_scoped_release>())
        .def("slice", &Tensor::slice, nb::arg("dim"), nb::arg("start"), nb::arg("stop"),
             nb::call_guard<nb::gil_scoped_release>())
        .def("sqrt", &Tensor::sqrt, nb::call_guard<nb::gil_scoped_release>())
        .def("reciprocal", &Tensor::reciprocal, nb::call_guard<nb::gil_scoped_release>())
        .def("div", &Tensor::div, nb::arg("other"), nb::call_guard<nb::gil_scoped_release>())
        .def("__add__", &Tensor::add)
        .def("__sub__", &Tensor::sub)
        .def("__mul__", &Tensor::mul)
        .def("__matmul__", &Tensor::matmul)
        .def("__truediv__", &Tensor::div);

    m.def("zeros", &Tensor::zeros, nb::arg("shape"), nb::arg("requires_grad") = false);
    m.def("ones", &Tensor::ones, nb::arg("shape"), nb::arg("requires_grad") = false);
    m.def("randn", &Tensor::randn, nb::arg("shape"), nb::arg("std") = 1.0f,
          nb::arg("requires_grad") = false, nb::arg("seed") = 0);
    m.def("from_flat", &Tensor::from_flat, nb::arg("data"), nb::arg("shape"),
          nb::arg("requires_grad") = false);
    m.def("cat", &Tensor::cat, nb::arg("tensors"), nb::arg("dim"), nb::call_guard<nb::gil_scoped_release>());

    m.def("fused_bias_relu", &fused_bias_relu, nb::arg("x"), nb::arg("bias"),
          nb::call_guard<nb::gil_scoped_release>());
    m.def("fused_sub_square", &fused_sub_square, nb::arg("a"), nb::arg("b"),
          nb::call_guard<nb::gil_scoped_release>());

    // kir.grad()'s own backward-only vocabulary -- these run on the
    // "cpu" side of a distributed backward pass (dtensor_grad), same
    // reasoning for releasing the GIL as the Tensor methods above.
    m.def("relu_backward", &relu_backward, nb::arg("input"), nb::arg("grad_output"),
          nb::call_guard<nb::gil_scoped_release>());
    m.def("sum_axis0", &sum_axis0, nb::arg("grad_output"), nb::call_guard<nb::gil_scoped_release>());
    m.def("broadcast_scalar", &broadcast_scalar, nb::arg("grad_output"), nb::arg("shape"), nb::arg("scale"),
          nb::call_guard<nb::gil_scoped_release>());
    m.def("matmul_nt", &matmul_nt, nb::arg("a"), nb::arg("b"), nb::call_guard<nb::gil_scoped_release>());
    m.def("matmul_tn", &matmul_tn, nb::arg("a"), nb::arg("b"), nb::call_guard<nb::gil_scoped_release>());
    m.def("reduce_to_shape", &reduce_to_shape, nb::arg("grad"), nb::arg("target_shape"),
          nb::call_guard<nb::gil_scoped_release>());

    nb::class_<StoragePool>(m, "StoragePool")
        .def(nb::init<>())
        .def("num_free", &StoragePool::num_free)
        .def("num_allocated", &StoragePool::num_allocated);

    // set_active_pool takes the pool as a raw C++ pointer for
    // Tensor::zeros() to consult (see Tensor.cpp) -- so this module also
    // holds a real Python reference (`g_active_pool_pyref`) for as long
    // as a pool is active, or the pointer could dangle the moment
    // Python's own last reference to the pool object went away. Always
    // goes through kir.pooled() (a context manager) on the Python side,
    // which guarantees clear_active_pool() runs even if the wrapped
    // code raises.
    static nb::object g_active_pool_pyref;

    m.def("set_active_pool", [](nb::object pool_obj) {
        set_active_pool(nb::cast<StoragePool*>(pool_obj));
        g_active_pool_pyref = pool_obj;
    });
    m.def("clear_active_pool", []() {
        set_active_pool(nullptr);
        g_active_pool_pyref = nb::object();
    });
    m.def("release_to_pool", [](StoragePool& pool, const Tensor& t) {
        pool.release(t.storage_ptr());
    });

#ifdef KANSAI_HAS_METAL
    // Every one of these blocks on waitUntilCompleted internally (see
    // backend/metal/MetalOps.mm) -- without releasing the GIL here,
    // that wait would hold the GIL for its entire duration, and a
    // concurrent "cpu"-device Python thread could never even resume its
    // own bytecode, let alone overlap its own Accelerate call, during
    // that time. Metal's command queue is documented thread-safe for
    // concurrent command-buffer creation/submission (Apple's Metal Best
    // Practices Guide), so releasing the GIL here doesn't introduce a
    // new race -- state() (backend/metal/MetalOps.mm) is a function-
    // local static, whose first-call initialization C++11 already
    // guarantees is thread-safe.
    m.def("metal_available", &metal_available);
    m.def("metal_matmul", &metal_matmul, nb::arg("a"), nb::arg("b"), nb::call_guard<nb::gil_scoped_release>());
    m.def("metal_matmul_mps", &metal_matmul_mps, nb::arg("a"), nb::arg("b"),
          nb::call_guard<nb::gil_scoped_release>());
    m.def("metal_bias_relu", &metal_bias_relu, nb::arg("x"), nb::arg("bias"),
          nb::call_guard<nb::gil_scoped_release>());
    m.def("metal_add_bias", &metal_add_bias, nb::arg("x"), nb::arg("bias"),
          nb::call_guard<nb::gil_scoped_release>());
    m.def("metal_add", &metal_add, nb::arg("a"), nb::arg("b"), nb::call_guard<nb::gil_scoped_release>());
    m.def("metal_sub", &metal_sub, nb::arg("a"), nb::arg("b"), nb::call_guard<nb::gil_scoped_release>());
    m.def("metal_mul", &metal_mul, nb::arg("a"), nb::arg("b"), nb::call_guard<nb::gil_scoped_release>());
    m.def("metal_relu", &metal_relu, nb::arg("x"), nb::call_guard<nb::gil_scoped_release>());
    m.def("metal_fused_sub_square", &metal_fused_sub_square, nb::arg("a"), nb::arg("b"),
          nb::call_guard<nb::gil_scoped_release>());
    m.def("metal_sum", &metal_sum, nb::arg("x"), nb::call_guard<nb::gil_scoped_release>());
    m.def("metal_mean", &metal_mean, nb::arg("x"), nb::call_guard<nb::gil_scoped_release>());
    m.def("metal_elementwise_chain", &metal_elementwise_chain, nb::arg("x"), nb::arg("kinds"), nb::arg("biases"),
          nb::call_guard<nb::gil_scoped_release>());
    m.def("metal_conv2d", &metal_conv2d, nb::arg("x"), nb::arg("weight"), nb::arg("bias"),
          nb::arg("stride"), nb::arg("padding"), nb::call_guard<nb::gil_scoped_release>());
#else
    m.def("metal_available", []() { return false; });
#endif
}

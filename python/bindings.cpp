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
        .def("add_", &Tensor::add_, nb::arg("other"), nb::arg("alpha") = 1.0f)
        .def("add", &Tensor::add, nb::arg("other"))
        .def("sub", &Tensor::sub, nb::arg("other"))
        .def("mul", &Tensor::mul, nb::arg("other"))
        .def("matmul", &Tensor::matmul, nb::arg("other"))
        .def("relu", &Tensor::relu)
        .def("sum", &Tensor::sum)
        .def("mean", &Tensor::mean)
        .def("__add__", &Tensor::add)
        .def("__sub__", &Tensor::sub)
        .def("__mul__", &Tensor::mul)
        .def("__matmul__", &Tensor::matmul);

    m.def("zeros", &Tensor::zeros, nb::arg("shape"), nb::arg("requires_grad") = false);
    m.def("ones", &Tensor::ones, nb::arg("shape"), nb::arg("requires_grad") = false);
    m.def("randn", &Tensor::randn, nb::arg("shape"), nb::arg("std") = 1.0f,
          nb::arg("requires_grad") = false, nb::arg("seed") = 0);
    m.def("from_flat", &Tensor::from_flat, nb::arg("data"), nb::arg("shape"),
          nb::arg("requires_grad") = false);

    m.def("fused_bias_relu", &fused_bias_relu, nb::arg("x"), nb::arg("bias"));
    m.def("fused_sub_square", &fused_sub_square, nb::arg("a"), nb::arg("b"));

    m.def("relu_backward", &relu_backward, nb::arg("input"), nb::arg("grad_output"));
    m.def("sum_axis0", &sum_axis0, nb::arg("grad_output"));
    m.def("broadcast_scalar", &broadcast_scalar, nb::arg("grad_output"), nb::arg("shape"), nb::arg("scale"));
    m.def("matmul_nt", &matmul_nt, nb::arg("a"), nb::arg("b"));
    m.def("matmul_tn", &matmul_tn, nb::arg("a"), nb::arg("b"));

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
    m.def("metal_available", &metal_available);
    m.def("metal_matmul", &metal_matmul, nb::arg("a"), nb::arg("b"));
    m.def("metal_bias_relu", &metal_bias_relu, nb::arg("x"), nb::arg("bias"));
    m.def("metal_add_bias", &metal_add_bias, nb::arg("x"), nb::arg("bias"));
    m.def("metal_elementwise_chain", &metal_elementwise_chain, nb::arg("x"), nb::arg("kinds"), nb::arg("biases"));
#else
    m.def("metal_available", []() { return false; });
#endif
}

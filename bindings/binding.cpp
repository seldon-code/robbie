#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "network.hpp"
#include "loss_functions.hpp"

#include "add.hpp"
#include "fc_layer.hpp"

PYBIND11_MODULE(robbielib, m) {
    m.doc() = "pybind11 plugin for neural networks"; // optional module docstring

    // Network class 
    // Instantiated templates 
    py::class_<Robbie::Network<double, Robbie::LossFunctions::MeanSquareError>>(m, "Network")
        .def(py::init<>())
        .def_readwrite("loss_tol", &Robbie::Network<double, Robbie::LossFunctions::MeanSquareError>::loss_tol);

    m.def("add", &Robbie::add, "A function that adds two numbers",
        py::arg("i"), py::arg("j"));

        py::class_<Robbie::FCLayer<double>>(m, "FCLayer")
        .def(py::init< size_t, size_t >())
        .def("name", &Robbie::FCLayer<double>::name);

}

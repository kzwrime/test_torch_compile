#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <vector>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace extension_cpp {

// SiLU activation: out = a * sigmoid(a)
void mysilu_out_cpu(const at::Tensor& a, at::Tensor& out) {
  TORCH_CHECK(a.sizes() == out.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(out.dtype() == at::kFloat);
  TORCH_CHECK(out.is_contiguous());
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CPU);
  at::Tensor a_contig = a.contiguous();
  const float* a_ptr = a_contig.data_ptr<float>();
  float* out_ptr = out.data_ptr<float>();

  for (int64_t i = 0; i < out.numel(); i++) {
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    float x = a_ptr[i];
    float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
    out_ptr[i] = x * sigmoid_x;
  }
}

// Defines the operators
TORCH_LIBRARY(extension_cpp, m) {
  m.def("mysilu_out(Tensor a, Tensor(a!) out) -> ()");
}

// Registers CPU implementations for mysilu_out
TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
  m.impl("mysilu_out", &mysilu_out_cpu);
}

}

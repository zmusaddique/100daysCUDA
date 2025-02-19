#include <Python.h>

extern "C" {
PyObjects *PyInit__C(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT, "_C", NULL, -1, NULL,
  };
  return PyModule_Create(&module_def);
}
}

namespace flash_attention_2 {

}

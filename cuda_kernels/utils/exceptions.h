#include <pybind11/pybind11.h>
namespace py = pybind11;

#define PY_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            throw py::value_error(std::string("Invalid input: condition '") + #cond + "' was not satisfied."); \
        } \
    } while (0)

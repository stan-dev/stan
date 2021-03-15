#ifndef STAN_IO_VALIDATE_DIMS_HPP
#define STAN_IO_VALIDATE_DIMS_HPP

#include <stan/io/var_context.hpp>
#include <string>
#include <vector>

namespace stan {
namespace io {

/**
 * Check variable dimensions against variable declaration.
 *
 * @param context The var context to check.
 * @param stage stan program processing stage
 * @param name variable name
 * @param base_type declared stan variable type
 * @param dims_declared variable dimensions
 * @throw std::runtime_error if mismatch between declared
 *        dimensions and dimensions found in context.
 */
inline void validate_dims(const stan::io::var_context& context,
                          const std::string& stage, const std::string& name,
                          const std::string& base_type,
                          const std::vector<size_t>& dims_declared) {
  bool is_int_type = base_type == "int";
  if (is_int_type) {
    if (!context.contains_i(name)) {
      std::stringstream msg;
      msg << (context.contains_r(name) ? "int variable contained non-int values"
                                       : "variable does not exist")
          << "; processing stage=" << stage << "; variable name=" << name
          << "; base type=" << base_type;
      throw std::runtime_error(msg.str());
    }
  } else {
    if (!context.contains_r(name)) {
      std::stringstream msg;
      msg << "variable does not exist"
          << "; processing stage=" << stage << "; variable name=" << name
          << "; base type=" << base_type;
      throw std::runtime_error(msg.str());
    }
  }
  std::vector<size_t> dims = context.dims_r(name);
  if (dims.size() != dims_declared.size()) {
    std::stringstream msg;
    msg << "mismatch in number dimensions declared and found in context"
        << "; processing stage=" << stage << "; variable name=" << name
        << "; dims declared=";
    context.dims_msg(msg, dims_declared);
    msg << "; dims found=";
    context.dims_msg(msg, dims);
    throw std::runtime_error(msg.str());
  }
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims_declared[i] != dims[i]) {
      std::stringstream msg;
      msg << "mismatch in dimension declared and found in context"
          << "; processing stage=" << stage << "; variable name=" << name
          << "; position=" << i << "; dims declared=";
      context.dims_msg(msg, dims_declared);
      msg << "; dims found=";
      context.dims_msg(msg, dims);
      throw std::runtime_error(msg.str());
    }
  }
}

}  // namespace io
}  // namespace stan
#endif

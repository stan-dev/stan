#ifndef STAN_IO_EMPTY_VAR_CONTEXT_HPP
#define STAN_IO_EMPTY_VAR_CONTEXT_HPP

#include <stan/io/var_context.hpp>
#include <stan/io/validate_dims.hpp>
#include <string>
#include <vector>

namespace stan {
namespace io {

/**
 * This is an implementation of a var_context that doesn't contain
 * any variables.
 */
class empty_var_context : public var_context {
 public:
  /**
   * Destructor
   */
  virtual ~empty_var_context() {}

  /**
   * Return <code>true</code> if the specified variable name is
   * defined. Always returns <code>false</code>.
   *
   * @param name Name of variable.
   * @return <code>false</code>
   */
  bool contains_r(const std::string& name) const { return false; }

  /**
   * Always returns an empty vector.
   *
   * @param name Name of variable.
   * @return empty vector
   */
  std::vector<double> vals_r(const std::string& name) const {
    return std::vector<double>();
  }

  /**
   * Always returns an empty vector.
   *
   * @param name Name of variable.
   * @return empty vector
   */
  std::vector<size_t> dims_r(const std::string& name) const {
    return std::vector<size_t>();
  }

  /**
   * Return <code>true</code> if the specified variable name has
   * integer values. Always returns <code>false</code>.
   *
   * @param name Name of variable.
   * @return false
   */
  bool contains_i(const std::string& name) const { return false; }

  /**
   * Returns an empty vector.
   *
   * @param name Name of variable.
   * @return empty vector
   */
  std::vector<int> vals_i(const std::string& name) const {
    return std::vector<int>();
  }

  /**
   * Return the dimensions of the specified floating point variable.
   * Returns an empty vector.
   *
   * @param name Name of variable.
   * @return empty vector
   */
  std::vector<size_t> dims_i(const std::string& name) const {
    return std::vector<size_t>();
  }

  /**
   * Check variable dimensions against variable declaration.
   * This context has no variables.
   *
   * @param stage stan program processing stage
   * @param name variable name
   * @param base_type declared stan variable type
   * @param dims_declared variable dimensions
   * @throw std::runtime_error if mismatch between declared
   *        dimensions and dimensions found in context.
   */
  void validate_dims(const std::string& stage, const std::string& name,
                     const std::string& base_type,
                     const std::vector<size_t>& dims_declared) const {
    stan::io::validate_dims(*this, stage, name, base_type, dims_declared);
  }

  /**
   * Fill a list of the names of the floating point variables in
   * the context. This context has no variables.
   *
   * @param names Vector to store the list of names in.
   */
  void names_r(std::vector<std::string>& names) const { names.clear(); }

  /**
   * Fill a list of the names of the integer variables in
   * the context. This context has no variables.
   *
   * @param names Vector to store the list of names in.
   */
  void names_i(std::vector<std::string>& names) const { names.clear(); }
};

}  // namespace io
}  // namespace stan
#endif

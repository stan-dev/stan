#ifndef STAN_IO_VAR_CONTEXT_HPP
#define STAN_IO_VAR_CONTEXT_HPP

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace stan {

namespace io {

/**
 * A <code>var_context</code> reads array variables of integer and
 * floating point type by name and dimension.
 *
 * <p>An array's dimensionality is described by a sequence of
 * (unsigned) integers.  For instance, <code>(7, 2, 3)</code> picks
 * out a 7 by 2 by 3 array.  The empty dimensionality sequence
 * <code>()</code> is used for scalars.
 *
 * <p>Multidimensional arrays are stored in column-major order,
 * meaning the first index changes the most quickly.
 *
 * <p>If a variable has integer variables, it should return
 * those integer values cast to floating point values when
 * accessed through the floating-point methods.
 */
class var_context {
 public:
  virtual ~var_context() {}

  /**
   * Return <code>true</code> if the specified variable name is
   * defined.  This method should return <code>true</code> even
   * if the values are all integers.
   *
   * @param name Name of variable.
   * @return <code>true</code> if the variable exists with real
   * values.
   */
  virtual bool contains_r(const std::string& name) const = 0;

  /**
   * Return the floating point values for the variable of the
   * specified variable name in last-index-major order.  This
   * method should cast integers to floating point values if the
   * values of the named variable are all integers.
   *
   * <p>If there is no variable of the specified name, the empty
   * vector is returned.
   *
   * @param name Name of variable.
   * @return Sequence of values for the named variable.
   */
  virtual std::vector<double> vals_r(const std::string& name) const = 0;

  /**
   * Return the dimensions for the specified floating point variable.
   * If the variable doesn't exist or if it is a scalar, the
   * return result should be the empty vector.
   *
   * @param name Name of variable.
   * @return Sequence of dimensions for the variable.
   */
  virtual std::vector<size_t> dims_r(const std::string& name) const = 0;

  /**
   * Return <code>true</code> if the specified variable name has
   * integer values.
   *
   * @param name Name of variable.
   * @return <code>true</code> if an integer variable of the specified
   * name is defined.
   */
  virtual bool contains_i(const std::string& name) const = 0;

  /**
   * Return the integer values for the variable of the specified
   * name in last-index-major order or the empty sequence if the
   * variable is not defined.
   *
   * @param name Name of variable.
   * @return Sequence of integer values.
   */
  virtual std::vector<int> vals_i(const std::string& name) const = 0;

  /**
   * Return the dimensions of the specified floating point variable.
   * If the variable doesn't exist (or if it is a scalar), the
   * return result should be the empty vector.
   *
   * @param name Name of variable.
   * @return Sequence of dimensions for the variable.
   */
  virtual std::vector<size_t> dims_i(const std::string& name) const = 0;

  /**
   * Fill a list of the names of the floating point variables in
   * the context.
   *
   * @param names Vector to store the list of names in.
   */
  virtual void names_r(std::vector<std::string>& names) const = 0;

  /**
   * Fill a list of the names of the integer variables in
   * the context.
   *
   * @param names Vector to store the list of names in.
   */
  virtual void names_i(std::vector<std::string>& names) const = 0;

  /**
   * Check variable dimensions against variable declaration.
   *
   * @param stage stan program processing stage
   * @param name variable name
   * @param base_type declared stan variable type
   * @param dims_declared variable dimensions
   * @throw std::runtime_error if mismatch between declared
   *        dimensions and dimensions found in context.
   */
  virtual void validate_dims(
      const std::string& stage, const std::string& name,
      const std::string& base_type,
      const std::vector<size_t>& dims_declared) const = 0;

  /**
   * Append vector of dimensions to message string.
   *
   * @param msg message string
   * @param dims array of dimension sizes
   */
  void dims_msg(std::stringstream& msg, const std::vector<size_t>& dims) const {
    msg << '(';
    for (size_t i = 0; i < dims.size(); ++i) {
      if (i > 0)
        msg << ',';
      msg << dims[i];
    }
    msg << ')';
  }

  static std::vector<size_t> to_vec() { return std::vector<size_t>(); }
  static std::vector<size_t> to_vec(size_t n1) {
    std::vector<size_t> v(1);
    v[0] = n1;
    return v;
  }
  static std::vector<size_t> to_vec(size_t n1, size_t n2) {
    std::vector<size_t> v(2);
    v[0] = n1;
    v[1] = n2;
    return v;
  }
  static std::vector<size_t> to_vec(size_t n1, size_t n2, size_t n3) {
    std::vector<size_t> v(3);
    v[0] = n1;
    v[1] = n2;
    v[2] = n3;
    return v;
  }
  static std::vector<size_t> to_vec(size_t n1, size_t n2, size_t n3,
                                    size_t n4) {
    std::vector<size_t> v(4);
    v[0] = n1;
    v[1] = n2;
    v[2] = n3;
    v[3] = n4;
    return v;
  }
  static std::vector<size_t> to_vec(size_t n1, size_t n2, size_t n3, size_t n4,
                                    size_t n5) {
    std::vector<size_t> v(5);
    v[0] = n1;
    v[1] = n2;
    v[2] = n3;
    v[3] = n4;
    v[4] = n5;
    return v;
  }
  static std::vector<size_t> to_vec(size_t n1, size_t n2, size_t n3, size_t n4,
                                    size_t n5, size_t n6) {
    std::vector<size_t> v(6);
    v[0] = n1;
    v[1] = n2;
    v[2] = n3;
    v[3] = n4;
    v[4] = n5;
    v[5] = n6;
    return v;
  }
  static std::vector<size_t> to_vec(size_t n1, size_t n2, size_t n3, size_t n4,
                                    size_t n5, size_t n6, size_t n7) {
    std::vector<size_t> v(7);
    v[0] = n1;
    v[1] = n2;
    v[2] = n3;
    v[3] = n4;
    v[4] = n5;
    v[5] = n6;
    v[6] = n7;
    return v;
  }
  static std::vector<size_t> to_vec(size_t n1, size_t n2, size_t n3, size_t n4,
                                    size_t n5, size_t n6, size_t n7,
                                    size_t n8) {
    std::vector<size_t> v(8);
    v[0] = n1;
    v[1] = n2;
    v[2] = n3;
    v[3] = n4;
    v[4] = n5;
    v[5] = n6;
    v[6] = n7;
    v[7] = n8;
    return v;
  }
};

}  // namespace io

}  // namespace stan

#endif

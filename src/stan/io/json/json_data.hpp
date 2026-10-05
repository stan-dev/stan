#ifndef STAN_IO_JSON_JSON_DATA_HPP
#define STAN_IO_JSON_JSON_DATA_HPP

#include <stan/io/json/json_data_handler.hpp>
#include <stan/io/json/json_error.hpp>
#include <stan/io/json/rapidjson_parser.hpp>
#include <stan/io/var_context.hpp>
#include <stan/io/validate_dims.hpp>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <complex>

namespace stan {

namespace json {

/**
 * A <code>json_data</code> is a <code>var_context</code> object
 * that represents a set of named values which are typed to either
 * <code>double</code> or <code>int</code> and can be either scalar
 * value or an array of values of any dimensionality.
 * Arrays must be rectangular and the values of an array are all of
 * the same type, either double or int.
 *
 * <p>The dimensions and values of variables are accessed by variable name.
 * The values of a variable are stored as a vector of values and
 * a vector of array dimensions, where a scalar value consists of
 * a single value and an empty vector for the dimensionality.
 * Multidimensional arrays are stored in column-major order,
 * meaning the first index changes the most quickly.
 * If all the values of an array are int values, the array will be
 * stored as a vector of ints, else the array will be stored
 * as a vector of type double.
 *
 * <p>A variable nested inside one or more arrays of tuples is named by the
 * dotted path to its innermost tuple slot.  Its dimensions are the dimensions
 * of the enclosing arrays followed by the dimensions of the variable itself,
 * and its values hold one contiguous block per enclosing tuple element, laid
 * out back to back.  The enclosing array dimensions are in element (row-major)
 * order, while the values within each block are column-major as above; this
 * mixed layout is the convention assumed by
 * <code>stan::io::read_from_context</code>.  The dimension vector alone does
 * not say where the enclosing dimensions end and the variable's own begin,
 * so <code>var_entry::num_outer_dims</code> records that split.
 *
 * <p><code>json_data</code> objects are created by using the
 * <code>json_parser</code> and a <code>json_data_handler</code>
 * to read a single JSON text from an input stream.
 */
class json_data : public stan::io::var_context {
 private:
  vars_map_r vars_r_;
  vars_map_i vars_i_;

  std::vector<double> const empty_vec_r_;
  std::vector<int> const empty_vec_i_;
  std::vector<size_t> const empty_vec_ui_;

  /**
   * Return <code>true</code> if this json_data contains the specified
   * variable name defined as a real-valued variable. This method
   * returns <code>false</code> if the values are all integers.
   *
   * @param name Variable name to test.
   * @return <code>true</code> if the variable exists in the
   * real values of the json_data.
   */
  bool contains_r_only(const std::string &name) const {
    return vars_r_.find(name) != vars_r_.end();
  }

  /**
   * Decode complex components within each innermost array block.
   *
   * Values are stored as one block per enclosing tuple element, laid out
   * back to back.  Within a block the real components precede the imaginary
   * ones, so the component of a value is a half block away from its real part.
   *
   * @tparam T Stored scalar type, either int or double.
   * @param name Variable name.
   * @param var Stored values and dimensions for the variable.
   * @return Complex values in the order of the input blocks.
   * @throw json_error if nonempty data has no trailing component dimension 2.
   */
  template <typename T>
  std::vector<std::complex<double>> vals_c_impl(const std::string &name,
                                                const var_entry<T> &var) const {
    const auto &values = var.values;
    if (values.empty())
      return {};
    // The trailing 2 must be the variable's own, not an enclosing array dim.
    if (!var.has_own_dims() || var.dims.back() != 2) {
      throw json_error("Variable: " + name
                       + ", expected a trailing dimension of 2 for complex "
                         "values.");
    }
    const size_t block_size = var.block_size();
    const size_t half_block = block_size / 2;
    std::vector<std::complex<double>> result;
    result.reserve(values.size() / 2);
    for (size_t start = 0; start < values.size(); start += block_size) {
      for (size_t i = 0; i < half_block; ++i) {
        result.emplace_back(values[start + i], values[start + half_block + i]);
      }
    }
    return result;
  }

 public:
  /**
   * Construct a json_data object from the specified input stream.
   *
   * <b>Warning:</b> This method does not close the input stream.
   *
   * @param in Input stream from which to read.
   * @throws json_exception if data is not well-formed stan data declaration
   */
  explicit json_data(std::istream &in) : vars_r_(), vars_i_() {
    json_data_handler handler(vars_r_, vars_i_);
    rapidjson_parse(in, handler);
  }

  /**
   * Return <code>true</code> if this json_data contains the specified
   * variable name. This method returns <code>true</code>
   * even if the values are all integers.
   *
   * @param name Variable name to test.
   * @return <code>true</code> if the variable exists.
   */
  bool contains_r(const std::string &name) const {
    return contains_r_only(name) || contains_i(name);
  }

  /**
   * Return <code>true</code> if this json_data contains an integer
   * valued array with the specified name.
   *
   * @param name Variable name to test.
   * @return <code>true</code> if the variable name has an integer
   * array value.
   */
  bool contains_i(const std::string &name) const {
    return vars_i_.find(name) != vars_i_.end();
  }

  /**
   * Return the double values for the variable with the specified
   * name or null.
   *
   * @param name Name of variable.
   * @return Values of variable.
   */
  std::vector<double> vals_r(const std::string &name) const {
    if (contains_r_only(name)) {
      return vars_r_.find(name)->second.values;
    } else if (contains_i(name)) {
      const std::vector<int> &vec_int = vars_i_.find(name)->second.values;
      return std::vector<double>(vec_int.begin(), vec_int.end());
    }
    return empty_vec_r_;
  }

  /**
   * Read out the complex values for the variable with the specified
   * name and return a flat vector of complex values.
   *
   * @param name Name of Variable of type string.
   * @return Vector of complex numbers with values equal to the read input.
   * @throw json_error if nonempty data has no trailing component dimension 2.
   */
  std::vector<std::complex<double>> vals_c(const std::string &name) const {
    if (contains_r_only(name)) {
      return vals_c_impl(name, vars_r_.find(name)->second);
    } else if (contains_i(name)) {
      return vals_c_impl(name, vars_i_.find(name)->second);
    }
    return std::vector<std::complex<double>>{};
  }

  /**
   * Return the dimensions for the variable with the specified
   * name.
   *
   * @param name Name of variable.
   * @return Dimensions of variable.
   */
  std::vector<size_t> dims_r(const std::string &name) const {
    if (contains_r_only(name)) {
      return vars_r_.find(name)->second.dims;
    } else if (contains_i(name)) {
      return vars_i_.find(name)->second.dims;
    }
    return empty_vec_ui_;
  }

  /**
   * Return the integer values for the variable with the specified
   * name.
   *
   * @param name Name of variable.
   * @return Values.
   */
  std::vector<int> vals_i(const std::string &name) const {
    if (contains_i(name)) {
      return vars_i_.find(name)->second.values;
    }
    return empty_vec_i_;
  }

  /**
   * Return the dimensions for the integer variable with the specified
   * name.
   *
   * @param name Name of variable.
   * @return Dimensions of variable.
   */
  std::vector<size_t> dims_i(const std::string &name) const {
    if (contains_i(name)) {
      return vars_i_.find(name)->second.dims;
    }
    return empty_vec_ui_;
  }

  /**
   * Return a list of the names of the floating point variables in
   * the json_data.
   *
   * @param names Vector to store the list of names in.
   */
  virtual void names_r(std::vector<std::string> &names) const {
    names.resize(0);
    for (vars_map_r::const_iterator it = vars_r_.begin(); it != vars_r_.end();
         ++it)
      names.push_back((*it).first);
  }

  /**
   * Return a list of the names of the integer variables in
   * the json_data.
   *
   * @param names Vector to store the list of names in.
   */
  virtual void names_i(std::vector<std::string> &names) const {
    names.resize(0);
    for (vars_map_i::const_iterator it = vars_i_.begin(); it != vars_i_.end();
         ++it)
      names.push_back((*it).first);
  }

  /**
   * Remove variable from the object.
   *
   * @param name Name of the variable to remove.
   * @return If variable is removed returns <code>true</code>, else
   *   returns <code>false</code>.
   */
  bool remove(const std::string &name) {
    return (vars_i_.erase(name) > 0) || (vars_r_.erase(name) > 0);
  }

  void validate_dims(const std::string &stage, const std::string &name,
                     const std::string &base_type,
                     const std::vector<size_t> &dims_declared) const {
    std::vector<size_t> dims = dims_r(name);

    // JSON '[ ]' is ambiguous - any multi-dim variable with len 0 dim
    // treat non-existent variables as size-0 objects
    size_t num_values = dims.empty() ? 0 : size_from_dims(dims);
    if (num_values == 0 && size_from_dims(dims_declared) == 0)
      return;

    stan::io::validate_dims(*this, stage, name, base_type, dims_declared);
  }
};

}  // namespace json

}  // namespace stan
#endif

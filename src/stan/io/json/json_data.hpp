#ifndef STAN_IO_JSON_JSON_DATA_HPP
#define STAN_IO_JSON_JSON_DATA_HPP

#include <stan/io/json/json_data_handler.hpp>
#include <stan/io/json/json_error.hpp>
#include <stan/io/json/rapidjson_parser.hpp>
#include <stan/io/var_context.hpp>
#include <boost/unordered/unordered_flat_map.hpp>
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
 * <p><code>json_data</code> objects are created by using the
 * <code>json_parser</code> and a <code>json_data_handler</code>
 * to read a single JSON text from an input stream.
 */
class json_data : public stan::io::var_context {
 private:
  vars_map_r vars_r_;
  vars_map_i vars_i_;
  boost::unordered_flat_map<std::string, size_t> array_block_sizes_;

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
   * Decode complex components within each tuple-free array block.
   *
   * @tparam T Stored scalar type, either int or double.
   * @param name Variable name.
   * @param data Flat values and combined dimensions for the variable.
   * @return Complex values in the order of the input blocks.
   * @throw json_error if nonempty data has no trailing component dimension 2.
   */
  template <typename T>
  std::vector<std::complex<double>> vals_c_impl(
      const std::string &name,
      const std::pair<std::vector<T>, std::vector<size_t>> &data) const {
    const auto &values = data.first;
    if (values.empty())
      return {};
    const auto block = array_block_sizes_.find(name);
    if (block == array_block_sizes_.end() || data.second.empty()
        || data.second.back() != 2) {
      throw json_error("Variable: " + name
                       + ", expected a trailing dimension of 2 for complex "
                         "values.");
    }
    const size_t block_size = block->second;
    const size_t offset = block_size / 2;
    std::vector<std::complex<double>> result(values.size() / 2);
    for (size_t start = 0; start < values.size(); start += block_size) {
      for (size_t i = 0; i < offset; ++i) {
        result[start / 2 + i]
            = {static_cast<double>(values[start + i]),
               static_cast<double>(values[start + offset + i])};
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
    array_block_sizes_ = handler.array_block_sizes();
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
      return (vars_r_.find(name)->second).first;
    } else if (contains_i(name)) {
      std::vector<int> vec_int = (vars_i_.find(name)->second).first;
      std::vector<double> vec_r(vec_int.size());
      for (size_t ii = 0; ii < vec_int.size(); ii++) {
        vec_r[ii] = vec_int[ii];
      }
      return vec_r;
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
      return (vars_r_.find(name)->second).second;
    } else if (contains_i(name)) {
      return (vars_i_.find(name)->second).second;
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
      return (vars_i_.find(name)->second).first;
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
      return (vars_i_.find(name)->second).second;
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
    array_block_sizes_.erase(name);
    return (vars_i_.erase(name) > 0) || (vars_r_.erase(name) > 0);
  }

  void validate_dims(const std::string &stage, const std::string &name,
                     const std::string &base_type,
                     const std::vector<size_t> &dims_declared) const {
    std::vector<size_t> dims = dims_r(name);

    // JSON '[ ]' is ambiguous - any multi-dim variable with len 0 dim
    size_t num_elements = 1;
    if (dims.size() == 0) {
      // treat non-existent variables as size-0 objects
      num_elements = 0;
    } else {
      for (size_t i = 0; i < dims.size(); ++i) {
        num_elements *= dims[i];
      }
    }

    size_t num_elements_expected = 1;
    for (size_t i = 0; i < dims_declared.size(); ++i) {
      num_elements_expected *= dims_declared[i];
    }

    if (num_elements == 0 && num_elements_expected == 0)
      return;

    if (dims.size() != dims_declared.size()) {
      std::stringstream msg;
      msg << "mismatch in number dimensions declared and found in context"
          << "; processing stage=" << stage << "; variable name=" << name
          << "; dims declared=";
      dims_msg(msg, dims_declared);
      msg << "; dims found=";
      dims_msg(msg, dims);
      throw std::runtime_error(msg.str());
    }
    for (size_t i = 0; i < dims.size(); ++i) {
      if (dims_declared[i] != dims[i]) {
        std::stringstream msg;
        msg << "mismatch in dimension declared and found in context"
            << "; processing stage=" << stage << "; variable name=" << name
            << "; position=" << i << "; dims declared=";
        dims_msg(msg, dims_declared);
        msg << "; dims found=";
        dims_msg(msg, dims);
        throw std::runtime_error(msg.str());
      }
    }

    bool is_int_type = base_type == "int";
    if (is_int_type) {
      if (!contains_i(name)) {
        std::stringstream msg;
        msg << (contains_r(name) ? "int variable contained non-int values"
                                 : "variable does not exist")
            << "; processing stage=" << stage << "; variable name=" << name
            << "; base type=" << base_type;
        throw std::runtime_error(msg.str());
      }
    } else {
      if (!contains_r(name)) {
        std::stringstream msg;
        msg << "variable does not exist"
            << "; processing stage=" << stage << "; variable name=" << name
            << "; base type=" << base_type;
        throw std::runtime_error(msg.str());
      }
    }
  }
};

}  // namespace json

}  // namespace stan
#endif

#ifndef STAN_IO_ARRAY_VAR_CONTEXT_HPP
#define STAN_IO_ARRAY_VAR_CONTEXT_HPP

#include <stan/io/var_context.hpp>
#include <stan/io/validate_dims.hpp>
#include <stan/math.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <map>
#include <algorithm>
#include <functional>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include <utility>

namespace stan {

namespace io {

/**
 * An array_var_context object represents a named arrays
 * with dimensions constructed from an array, a vector
 * of names, and a vector of all dimensions for each element.
 */
class array_var_context : public var_context {
 private:
  // Pair used in data maps
  template <typename T>
  using data_pair_t = std::pair<std::vector<T>, std::vector<size_t>>;

  std::map<std::string, data_pair_t<double>> vars_r_;  // Holds data for reals
  std::map<std::string, data_pair_t<int>> vars_i_;     // Holds data for doubles
  // When search for variable name fails, return one these
  const std::vector<double> empty_vec_r_;
  const std::vector<int> empty_vec_i_;
  const std::vector<size_t> empty_vec_ui_;

  /**
   * Search over the real variables to check if a name is in the map
   * @param name The name of the variable to search for
   * @return logical indicating if the variable was found in the map of reals.
   */
  bool contains_r_only(const std::string& name) const {
    return vars_r_.find(name) != vars_r_.end();
  }

  /**
   * Check (1) if the vector size of dimensions is no smaller
   * than the name vector size; (2) if the size of the input
   * array is large enough for what is needed.
   *
   * @param names The names for each variable
   * @param array_size The total size of the vector holding the values we want
   * to access.
   * @param dims Vector holding the dimensions for each variable.
   * @return If the array size is equal to the number of dimensions,
   * a vector of the cumulative sum of the dimensions of each inner element of
   * dims. The return of this function is used in the add_* methods to get the
   * sequence of values For each variable.
   * @throw std::invalid_argument when size of dimensions is less
   *  then array size or array is not long enough to hold
   *  the dimensions of the data.
   */
  template <typename T>
  inline std::vector<size_t> validate_dims(
      const std::vector<std::string>& names, const T array_size,
      const std::vector<std::vector<size_t>>& dims) {
    const size_t num_par = names.size();
    stan::math::check_less_or_equal("validate_dims", "array_var_context",
                                    dims.size(), num_par);
    std::vector<size_t> elem_dims_total(dims.size() + 1);
    for (int i = 0; i < dims.size(); i++) {
      elem_dims_total[i + 1] = std::accumulate(dims[i].begin(), dims[i].end(),
                                               1, std::multiplies<T>())
                               + elem_dims_total[i];
    }
    stan::math::check_less_or_equal("validate_dims", "array_var_context",
                                    elem_dims_total[dims.size()], array_size);
    return elem_dims_total;
  }

  /**
   * Adds a set of floating point variables to the floating point map.
   * @param names Names of each variable.
   * @param values The real values of variable in a contiguous
   * column major order container.
   * @param dims the dimensions for each variable.
   * @throw std::invalid_argument when size of dimensions is less
   *  then array size or array is not long enough to hold
   *  the dimensions of the data.
   */
  void add_r(const std::vector<std::string>& names,
             const std::vector<double>& values,
             const std::vector<std::vector<size_t>>& dims) {
    std::vector<size_t> dim_vec = validate_dims(names, values.size(), dims);
    for (size_t i = 0; i < names.size(); i++) {
      vars_r_.emplace(names[i],
                      data_pair_t<double>{{values.data() + dim_vec[i],
                                           values.data() + dim_vec[i + 1]},
                                          dims[i]});
    }
  }

  /**
   * Adds a set of floating point variables to the floating point map.
   * @param names Names of each variable.
   * @param values The real values of variable in an Eigen column vector.
   * @param dims the dimensions for each variable.
   * @throw std::invalid_argument when size of dimensions is less
   *  then array size or array is not long enough to hold
   *  the dimensions of the data.
   */
  void add_r(const std::vector<std::string>& names,
             const Eigen::VectorXd& values,
             const std::vector<std::vector<size_t>>& dims) {
    std::vector<size_t> dim_vec = validate_dims(names, values.size(), dims);
    for (size_t i = 0; i < names.size(); i++) {
      vars_r_.emplace(names[i],
                      data_pair_t<double>{{values.data() + dim_vec[i],
                                           values.data() + dim_vec[i + 1]},
                                          dims[i]});
    }
  }

  /**
   * Adds a set of integer variables to the integer map.
   * @param names Names of each variable.
   * @param values The integer values of variable in a vector.
   * @param dims the dimensions for each variable.
   * @throw std::invalid_argument when size of dimensions is less
   *  then array size or array is not long enough to hold
   *  the dimensions of the data.
   */
  void add_i(const std::vector<std::string>& names,
             const std::vector<int>& values,
             const std::vector<std::vector<size_t>>& dims) {
    std::vector<size_t> dim_vec = validate_dims(names, values.size(), dims);
    for (size_t i = 0; i < names.size(); i++) {
      vars_i_.emplace(names[i],
                      data_pair_t<int>{{values.data() + dim_vec[i],
                                        values.data() + dim_vec[i + 1]},
                                       dims[i]});
    }
  }

 public:
  /**
   * Construct an array_var_context from only real value arrays.
   *
   * @param names_r  names for each element
   * @param values_r a vector of double values for all elements
   * @param dim_r   a vector of dimensions
   * @throw std::invalid_argument when size of dimensions is less
   *  then array size or array is not long enough to hold
   *  the dimensions of the data.
   */
  array_var_context(const std::vector<std::string>& names_r,
                    const std::vector<double>& values_r,
                    const std::vector<std::vector<size_t>>& dim_r) {
    add_r(names_r, values_r, dim_r);
  }

  /**
   * Construct an array_var_context from an Eigen column vector.
   *
   * @param names_r  names for each element
   * @param values_r a vector of double values for all elements
   * @param dim_r   a vector of dimensions
   * @throw std::invalid_argument when size of dimensions is less
   *  then array size or array is not long enough to hold
   *  the dimensions of the data.
   */
  array_var_context(const std::vector<std::string>& names_r,
                    const Eigen::VectorXd& values_r,
                    const std::vector<std::vector<size_t>>& dim_r) {
    add_r(names_r, values_r, dim_r);
  }

  /**
   * Construct an array_var_context from only integer value arrays.
   *
   * @param names_i  names for each element
   * @param values_i a vector of integer values for all elements
   * @param dim_i   a vector of dimensions
   * @throw std::invalid_argument when size of dimensions is less
   *  then array size or array is not long enough to hold
   *  the dimensions of the data.
   */
  array_var_context(const std::vector<std::string>& names_i,
                    const std::vector<int>& values_i,
                    const std::vector<std::vector<size_t>>& dim_i) {
    add_i(names_i, values_i, dim_i);
  }

  /**
   * Construct an array_var_context from arrays of both double
   * and integer separately
   *
   * @param names_r  names for each element
   * @param values_r a vector of double values for all elements
   * @param dim_r   a vector of dimensions
   * @param names_i  names for each element
   * @param values_i a vector of integer values for all elements
   * @param dim_i   a vector of dimensions
   * @throw std::invalid_argument when size of dimensions is less
   *  then array size or array is not long enough to hold
   *  the dimensions of the data.
   */
  array_var_context(const std::vector<std::string>& names_r,
                    const std::vector<double>& values_r,
                    const std::vector<std::vector<size_t>>& dim_r,
                    const std::vector<std::string>& names_i,
                    const std::vector<int>& values_i,
                    const std::vector<std::vector<size_t>>& dim_i) {
    add_i(names_i, values_i, dim_i);
    add_r(names_r, values_r, dim_r);
  }

  /**
   * Construct an array_var_context from arrays of both double
   * and integer separately
   *
   * @param names_r  names for each element
   * @param values_r Eigen column vector of double elements.
   * @param dim_r   a vector of dimensions
   * @param names_i  names for each element
   * @param values_i a vector of integer values for all elements
   * @param dim_i   a vector of dimensions
   * @throw std::invalid_argument when size of dimensions is less
   *  then array size or array is not long enough to hold
   *  the dimensions of the data.
   */
  array_var_context(const std::vector<std::string>& names_r,
                    const Eigen::VectorXd& values_r,
                    const std::vector<std::vector<size_t>>& dim_r,
                    const std::vector<std::string>& names_i,
                    const std::vector<int>& values_i,
                    const std::vector<std::vector<size_t>>& dim_i) {
    add_i(names_i, values_i, dim_i);
    add_r(names_r, values_r, dim_r);
  }

  /**
   * Return <code>true</code> if this dump contains the specified
   * variable name is defined. This method returns <code>true</code>
   * even if the values are all integers.
   *
   * @param name Variable name to test.
   * @return <code>true</code> if the variable exists.
   */
  bool contains_r(const std::string& name) const {
    return contains_r_only(name) || contains_i(name);
  }

  /**
   * Return <code>true</code> if this dump contains an integer
   * valued array with the specified name.
   *
   * @param name Variable name to test.
   * @return <code>true</code> if the variable name has an integer
   * array value.
   */
  bool contains_i(const std::string& name) const {
    return vars_i_.find(name) != vars_i_.end();
  }

  /**
   * Return the double values for the variable with the specified
   * name or null.
   *
   * @param name Name of variable.
   * @return Values of variable.
   *
   */
  std::vector<double> vals_r(const std::string& name) const {
    const auto ret_val_r = vars_r_.find(name);
    if (ret_val_r != vars_r_.end()) {
      return ret_val_r->second.first;
    } else {
      const auto ret_val_i = vars_i_.find(name);
      if (ret_val_i != vars_i_.end()) {
        return {ret_val_i->second.first.begin(), ret_val_i->second.first.end()};
      }
    }
    return empty_vec_r_;
  }

  /**
   * Return the double values for the variable with the specified
   * name or null.
   *
   * @param name Name of variable.
   * @return Values of variable.
   *
   */
  std::vector<std::complex<double>> vals_c(const std::string& name) const {
    const auto val_r = vars_r_.find(name);
    if (val_r != vars_r_.end()) {
      std::vector<std::complex<double>> ret_c(val_r->second.first.size() / 2);
      int comp_iter;
      int real_iter;
      for (comp_iter = 0, real_iter = 0; real_iter < val_r->second.first.size();
           comp_iter += 1, real_iter += 2) {
        ret_c[comp_iter] = std::complex<double>{
            val_r->second.first[real_iter], val_r->second.first[real_iter + 1]};
      }
      return ret_c;
    } else {
      const auto val_i = vars_i_.find(name);
      if (val_i != vars_i_.end()) {
        std::vector<std::complex<double>> ret_c(val_i->second.first.size() / 2);
        int comp_iter;
        int real_iter;
        for (comp_iter = 0, real_iter = 0;
             real_iter < val_i->second.first.size();
             comp_iter += 1, real_iter += 2) {
          ret_c[comp_iter] = std::complex<double>{
              static_cast<double>(val_i->second.first[real_iter]),
              static_cast<double>(val_i->second.first[real_iter + 1])};
        }
        return ret_c;
      }
    }
    return std::vector<std::complex<double>>{};
  }

  /**
   * Return the dimensions for the double variable with the specified
   * name.
   *
   * @param name Name of variable.
   * @return Dimensions of variable.
   */
  std::vector<size_t> dims_r(const std::string& name) const {
    const auto ret_val_r = vars_r_.find(name);
    if (ret_val_r != vars_r_.end()) {
      return ret_val_r->second.second;
    } else {
      const auto ret_val_i = vars_i_.find(name);
      if (ret_val_i != vars_i_.end()) {
        return ret_val_i->second.second;
      }
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
  std::vector<int> vals_i(const std::string& name) const {
    auto ret_val_i = vars_i_.find(name);
    if (ret_val_i != vars_i_.end()) {
      return ret_val_i->second.first;
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
  std::vector<size_t> dims_i(const std::string& name) const {
    auto ret_val_i = vars_i_.find(name);
    if (ret_val_i != vars_i_.end()) {
      return ret_val_i->second.second;
    }
    return empty_vec_ui_;
  }

  /**
   * Check variable dimensions against variable declaration.
   * Only used for data read in from file.
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
   * Return a list of the names of the floating point variables in
   * the dump.
   *
   * @param names Vector to store the list of names in.
   */
  virtual void names_r(std::vector<std::string>& names) const {
    names.clear();
    names.reserve(vars_r_.size());
    for (const auto& vars_r_iter : vars_r_) {
      names.push_back(vars_r_iter.first);
    }
  }

  /**
   * Return a list of the names of the integer variables in
   * the dump.
   *
   * @param names Vector to store the list of names in.
   */
  virtual void names_i(std::vector<std::string>& names) const {
    names.clear();
    names.reserve(vars_i_.size());
    for (const auto& vars_i_iter : vars_r_) {
      names.push_back(vars_i_iter.first);
    }
  }

  /**
   * Remove variable from the object.
   *
   * @param name Name of the variable to remove.
   * @return If variable is removed returns <code>true</code>, else
   *   returns <code>false</code>.
   */
  bool remove(const std::string& name) {
    return (vars_i_.erase(name) > 0) || (vars_r_.erase(name) > 0);
  }
};
}  // namespace io
}  // namespace stan
#endif

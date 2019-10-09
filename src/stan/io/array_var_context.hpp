#ifndef STAN_IO_ARRAY_VAR_CONTEXT_HPP
#define STAN_IO_ARRAY_VAR_CONTEXT_HPP

#include <stan/io/var_context.hpp>
#include <stan/math.hpp>
#include <boost/throw_exception.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
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
   * @return logical indiciating if the variable was found in the map of reals.
   */
  template <typename T, require_convertible_t<T, std::string>...>
  bool contains_r_only(T&& name) const {
    return vars_r_.find(std::forward<T>(name)) != vars_r_.end();
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
  template <typename NameVec, typename ArrSize, typename DimVec,
            require_vector_vt<is_string_convertible, NameVec>...,
            require_arithmetic_t<ArrSize>...,
            require_vector_vt<is_vector, DimVec>...,
            require_vector_st<is_index, DimVec>...>
  inline auto validate_dims(NameVec&& names, const ArrSize array_size,
                            DimVec&& dims) {
    const size_t num_par = names.size();
    stan::math::check_less_or_equal("validate_dims", "array_var_context",
                                    dims.size(), num_par);
    std::vector<size_t> elem_dims_total(dims.size() + 1);
    for (int i = 0; i < dims.size(); i++) {
      elem_dims_total[i + 1] = std::accumulate(dims[i].begin(), dims[i].end(),
                                               1, std::multiplies<ArrSize>())
                               + elem_dims_total[i];
    }
    stan::math::check_less_or_equal("validate_dims", "array_var_context",
                                    elem_dims_total[dims.size()], array_size);
    return elem_dims_total;
  }

  /**
   * Adds a set of integer variables to the integer map.
   * @tparam NameVec Vector holding strings of names
   * @tparam ValueVec Vector holding the numeric values of the variable
   * @tparam DimVec Vector holding dimensions of variables
   * @param names Names of each variable.
   * @param values The integer values of variable in a vector.
   * @param dims the dimensions for each variable.
   * @throw std::invalid_argument when size of dimensions is less
   *  then array size or array is not long enough to hold
   *  the dimensions of the data.
   */
  template <typename NameVec, typename ValueVec, typename DimVec,
            require_vector_vt<is_string_convertible, NameVec>...,
            require_vector_vt<std::is_arithmetic, ValueVec>...,
            require_vector_vt<is_vector, DimVec>...,
            require_vector_st<is_index, DimVec>...>
  void add_vals(NameVec&& names, ValueVec&& values, DimVec&& dims) {
    auto dim_vec = validate_dims(names, values.size(), dims);
    for (size_t i = 0; i < names.size(); i++) {
      if (std::is_floating_point<value_type_t<ValueVec>>::value) {
        vars_r_.emplace(std::piecewise_construct,
                        std::forward_as_tuple(
                            std::forward<value_type_t<NameVec>>(names[i])),
                        std::forward_as_tuple(
                            std::vector<double>(values.data() + dim_vec[i],
                                                values.data() + dim_vec[i + 1]),
                            dims[i]));
      } else {
        vars_i_.emplace(std::piecewise_construct,
                        std::forward_as_tuple(
                            std::forward<value_type_t<NameVec>>(names[i])),
                        std::forward_as_tuple(
                            std::vector<int>(values.data() + dim_vec[i],
                                             values.data() + dim_vec[i + 1]),
                            dims[i]));
      }
    }
  }

 public:
  /**
   * Construct an array_var_context from only real value arrays.
   * @tparam NameVec Vector holding strings of names
   * @tparam ValueVec Vector holding the numeric values of the variable
   * @tparam DimVec Vector holding dimensions of variables
   * @param names_r  names for each element
   * @param values_r a vector of double values for all elements
   * @param dim_r   a vector of dimensions
   * @throw std::invalid_argument when size of dimensions is less
   *  then array size or array is not long enough to hold
   *  the dimensions of the data.
   */
  template <typename NameVec, typename ValueVec, typename DimVec,
            require_vector_vt<is_string_convertible, NameVec>...,
            require_vector_vt<std::is_arithmetic, ValueVec>...,
            require_vector_vt<is_vector, DimVec>...,
            require_vector_st<is_index, DimVec>...>
  array_var_context(NameVec&& names, ValueVec&& values, DimVec&& dims) {
    add_vals(std::forward<NameVec>(names), std::forward<ValueVec>(values),
             std::forward<DimVec>(dims));
  }

  /**
   * Construct an array_var_context from arrays of both double
   * and integer separately
   * @tparam NameVec Vector holding strings of names
   * @tparam FloatingVec Vector holding the floating point values of the
   * variable
   * @tparam IntVec Vector holding the integer values of the variable
   * @tparam DimVec Vector holding dimensions of variables
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
  template <typename NameVec, typename FloatingVec, typename IntVec,
            typename DimVec,
            require_vector_vt<is_string_convertible, NameVec>...,
            require_vector_vt<std::is_floating_point, FloatingVec>...,
            require_vector_vt<is_index, IntVec>...,
            require_vector_vt<is_vector, DimVec>...,
            require_vector_st<is_index, DimVec>...>
  array_var_context(NameVec&& names_r, FloatingVec&& values_r, DimVec&& dim_r,
                    NameVec&& names_i, IntVec&& values_i, DimVec&& dim_i) {
    add_vals(std::forward<NameVec>(names_r),
             std::forward<FloatingVec>(values_r), std::forward<DimVec>(dim_r));
    add_vals(std::forward<NameVec>(names_i), std::forward<IntVec>(values_i),
             std::forward<DimVec>(dim_i));
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

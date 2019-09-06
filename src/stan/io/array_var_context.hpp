#ifndef STAN_IO_ARRAY_VAR_CONTEXT_HPP
#define STAN_IO_ARRAY_VAR_CONTEXT_HPP

#include <stan/io/var_context.hpp>
#include <stan/math/prim/meta.hpp>
#include <boost/throw_exception.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
#include <numeric>
#include <unordered_map>

namespace stan {

namespace io {


/**
 * An array_var_context object represents a named arrays
 * with dimensions constructed from an array, a vector
 * of names, and a vector of all dimensions for each element.
 */
class array_var_context : public var_context {
 private:
  // Map holding reals
  using pair_r_ = std::pair<std::vector<double>, std::vector<size_t>>;
  using map_r_ = std::unordered_map<std::string, pair_r_>;
  map_r_ vars_r_;
  // Map holding integers
  using pair_i_ =std::pair<std::vector<int>, std::vector<size_t>>;
  using map_i_ = std::unordered_map<std::string, pair_i_>;
  map_i_ vars_i_;
  // When search for variable name fails, return one these
  std::vector<double> const empty_vec_r_;
  std::vector<int> const empty_vec_i_;
  std::vector<size_t> const empty_vec_ui_;

  bool contains_r_only(const std::string& name) const {
    return vars_r_.find(name) != vars_r_.end();
  }

  /**
   * Check (1) if the vector size of dimensions is no smaller
   * than the name vector size; (2) if the size of the input
   * array is large enough for what is needed.
   * @param names The names for each variable
   * @param array_size The total size of the vector holding the values we want to access.
   * @param dims Vector holding the dimensions for each variable.
   * @return If the array size is equal to the number of dimensions, 
   * a vector of the cumulative sum of the dimensions of each inner element of dims.
   * The return of this function is used in the add_* methods to get the sequence of values
   * For each variable.
   */
  template <typename T>
  std::vector<size_t> validate_dims(const std::vector<std::string>& names, const T array_size,
                const std::vector<std::vector<size_t>>& dims) {
    const size_t num_par = names.size();
    if (num_par > dims.size()) {
      std::stringstream msg;
      msg << "size of vector of dimensions (found " << dims.size() << ") "
          << "should be no smaller than number of parameters (found " << num_par
          << ").";
      BOOST_THROW_EXCEPTION(std::invalid_argument(msg.str()));
    }
    std::vector<size_t> elem_dims_total(dims.size() + 1);
    elem_dims_total[0] = 0;
    std::transform(dims.begin(), dims.end(), elem_dims_total.begin() + 1, [](auto&& x) {
      return std::accumulate(x.begin(), x.end(), 1, std::multiplies<T>());
    });
    auto total = std::accumulate(elem_dims_total.begin(), elem_dims_total.end(), 0);
    if (total > array_size) {
      std::stringstream msg;
      msg << "array is not long enough for all elements: " << array_size
          << " is found, but " << total << " is needed.";
      BOOST_THROW_EXCEPTION(std::invalid_argument(msg.str()));
    }
    std::vector<size_t> array_end_vec(elem_dims_total.size());
    std::partial_sum(elem_dims_total.begin(), elem_dims_total.end(), array_end_vec.begin());
    return array_end_vec;
  }

  // This is just here till the next math submodule is updated
  template <typename T>
  using is_vector_floating_point = std::integral_constant<bool,
     is_vector<std::decay_t<T>>::value && 
     std::is_floating_point<typename scalar_type<std::decay_t<T>>::type>::value>;

  template<typename T, std::enable_if_t<is_vector_floating_point<T>::value>...>
  void add_r(const std::vector<std::string>& names,
             T&& values,
             const std::vector<std::vector<size_t>>& dims) {
    std::vector<size_t> dim_vec = validate_dims(names, values.size(), dims);
    for (size_t i = 0; i < names.size(); i++) {
      vars_r_[names[i]] = {{std::forward<decltype(values.data())>(values.data()) + dim_vec[i],
                            std::forward<decltype(values.data())>(values.data()) + dim_vec[i + 1]}, dims[i]};
    }
  }

  void add_i(const std::vector<std::string>& names,
             const std::vector<int>& values,
             const std::vector<std::vector<size_t>>& dims) {
    std::vector<size_t> dim_vec = validate_dims(names, values.size(), dims);
    for (size_t i = 0; i < names.size(); i++) {
      vars_i_[names[i]] = {{values.data() + dim_vec[i],
                            values.data() + dim_vec[i + 1]}, dims[i]};
    }
  }

 public:
  /**
   * Construct an array_var_context from only real value arrays.
   *
   * @param names_r  names for each element
   * @param values_r a vector of double values for all elements
   * @param dim_r   a vector of dimensions
   */
  array_var_context(const std::vector<std::string>& names_r,
                    const std::vector<double>& values_r,
                    const std::vector<std::vector<size_t>>& dim_r) {
    add_r(names_r, values_r, dim_r);
  }

  /**
   * Construct an array_var_context from an Eigen::RowVectorXd.
   *
   * @param names_r  names for each element
   * @param values_r an Eigen RowVector double values for all elements
   * @param dim_r   a vector of dimensions
   */
  array_var_context(const std::vector<std::string>& names_r,
                    const Eigen::RowVectorXd& values_r,
                    const std::vector<std::vector<size_t>>& dim_r) {
    add_r(names_r, values_r, dim_r);
  }

  /**
   * Construct an array_var_context from only integer value arrays.
   *
   * @param names_i  names for each element
   * @param values_i a vector of integer values for all elements
   * @param dim_i   a vector of dimensions
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
    if (contains_r_only(name)) {
      return (vars_r_.find(name)->second).first;
    } else if (contains_i(name)) {
      std::vector<int> vec_int = (vars_i_.find(name)->second).first;
      return {vec_int.begin(), vec_int.end()};
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
  std::vector<int> vals_i(const std::string& name) const {
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
  std::vector<size_t> dims_i(const std::string& name) const {
    if (contains_i(name)) {
      return (vars_i_.find(name)->second).second;
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
    for (auto& vars_r_iter : vars_r_) {
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
    for (auto& vars_i_iter : vars_r_) {
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

#ifndef STAN_MODEL_INDEXING_ACCESS_HELPERS_HPP
#define STAN_MODEL_INDEXING_ACCESS_HELPERS_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/fun/to_arena.hpp>

namespace stan {

namespace model {

namespace internal {
// Internal helpers so we can reuse min_max index assign for Eigen/var<Eigen>
template <typename T, require_var_matrix_t<T>* = nullptr>
inline auto rowwise_reverse(const T& x) {
  return x.rowwise_reverse();
}

template <typename T, require_eigen_t<T>* = nullptr>
inline auto rowwise_reverse(const T& x) {
  return x.rowwise().reverse();
}

template <typename T, require_var_matrix_t<T>* = nullptr>
inline auto colwise_reverse(const T& x) {
  return x.colwise_reverse();
}

template <typename T, require_eigen_t<T>* = nullptr>
inline auto colwise_reverse(const T& x) {
  return x.colwise().reverse();
}

/**
 * Base case of assignment
 * @tparam T1 Any type that's not a var matrix.
 * @tparam T2 Any type that's not a var matrix.
 * @param x The value to assign to
 * @param y The value to assign from.
 */
template <typename T1, typename T2,
          require_all_t<is_stan_scalar<T1>, is_stan_scalar<T2>>* = nullptr>
void assign_impl(T1&& x, T2&& y, const char* name) {
  x = std::forward<T2>(y);
}

/**
 * Base case of assignment
 * @tparam T1 Any type that's not a var matrix.
 * @tparam T2 Any type that's not a var matrix.
 * @param x The value to assign to
 * @param y The value to assign from.
 */
template <typename T1, typename T2,
          require_any_not_t<is_var_matrix<T1>, is_eigen<T2>>* = nullptr,
          require_all_t<is_matrix<T1>, is_matrix<T2>>* = nullptr>
void assign_impl(T1&& x, T2&& y, const char* name) {
  // We are allowed to assign to fully uninitialized matrix
  if (x.size() != 0) {
    static constexpr const char* obj_type
        = is_vector<T1>::value ? "vector" : "matrix";
    stan::math::check_size_match(
        (std::string(obj_type) + " assign columns").c_str(), name, x.cols(),
        "right hand side columns", y.cols());
    stan::math::check_size_match(
        (std::string(obj_type) + " assign rows").c_str(), name, x.rows(),
        "right hand side rows", y.rows());
  }
  x = std::forward<T2>(y);
}

/**
 * Base case of assignment
 * @tparam T1 Any type that's not a var matrix.
 * @tparam T2 Any type that's not a var matrix.
 * @param x The value to assign to
 * @param y The value to assign from.
 */
template <typename T1, typename T2,
          require_all_t<is_std_vector<T1>, is_std_vector<T2>>* = nullptr>
void assign_impl(T1&& x, T2&& y, const char* name) {
  // We are allowed to assign to fully uninitialized matrix
  if (unlikely(x.size() != 0)) {
    stan::math::check_size_match("assign array size", name, x.size(),
                                 "right hand side", y.size());
  }
  x = std::forward<T2>(y);
}

/**
 * Assigning an `Eigen::Matrix<double>` to a `var<Matrix>`
 * In this case we need to
 * 1. Store the previous values from `x`
 * 2. Assign the values from `y` to the values of `x`
 * 3. Setup a reverse pass callback that sets the `x` values to it's previous
 *  values and then zero's out the adjoints.
 *
 * @tparam Mat1 A `var_value` with inner type derived from `EigenBase`
 * @tparam Mat2 A type derived from `EigenBase` with an arithmetic scalar.
 * @param x The var matrix to assign to
 * @param y The eigen matrix to assign from.
 */
template <typename Mat1, typename Mat2, require_var_matrix_t<Mat1>* = nullptr,
          require_eigen_st<std::is_arithmetic, Mat2>* = nullptr>
void assign_impl(Mat1&& x, Mat2&& y, const char* name) {
  if (x.size() != 0) {
    static constexpr const char* obj_type
        = is_vector<Mat1>::value ? "vector" : "matrix";
    stan::math::check_size_match(
        (std::string(obj_type) + " assign columns").c_str(), name, x.cols(),
        "right hand side columns", y.cols());
    stan::math::check_size_match(
        (std::string(obj_type) + " assign rows").c_str(), name, x.rows(),
        "right hand side rows", y.rows());
  }
  auto prev_vals = stan::math::to_arena(x.val());
  x.vi_->val_ = std::forward<Mat2>(y);
  stan::math::reverse_pass_callback([x, prev_vals]() mutable {
    x.vi_->val_ = prev_vals;
    x.vi_->adj_.setZero();
  });
}
}  // namespace internal
}  // namespace model
}  // namespace stan
#endif

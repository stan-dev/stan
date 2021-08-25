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
          require_any_not_t<is_var_matrix<T1>, is_eigen<T2>>* = nullptr>
void assign_impl(T1&& x, T2&& y) {
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
void assign_impl(Mat1&& x, Mat2&& y) {
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

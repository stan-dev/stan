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
 * Assign two Stan scalars
 * @tparam T1 A scalar
 * @tparam T2 A scalar
 * @param x The value to assign to
 * @param y The value to assign from.
 */
template <typename T1, typename T2,
          require_all_t<is_stan_scalar<T1>, is_stan_scalar<T2>>* = nullptr>
void assign_impl(T1&& x, T2&& y, const char* name) {
  x = std::forward<T2>(y);
}

/**
 * Assign two var matrices
 * @tparam T1 A `var_value<T>` with inner type derived from `Eigen::EigenBase`
 * @tparam T2 A `var_value<T>` with inner type derived from `Eigen::EigenBase`
 * @param x The value to assign to
 * @param y The value to assign from.
 */
template <typename T1, typename T2, require_all_var_matrix_t<T1, T2>* = nullptr>
void assign_impl(T1&& x, T2&& y, const char* name) {
  // We are allowed to assign to fully uninitialized matrix
  if (!x.is_uninitialized()) {
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
 * Assign an Eigen object to another Eigen object.
 * @tparam T1 A type derived from `Eigen::EigenBase`
 * @tparam T2 A type derived from `Eigen::EigenBase`
 * @param x The value to assign to
 * @param y The value to assign from.
 */
template <typename T1, typename T2, require_all_eigen_t<T1, T2>* = nullptr>
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
 * Assign one standard vector to another
 * @tparam T1 A standard vector
 * @tparam T2 A standard vector
 * @param x The value to assign to
 * @param y The value to assign from.
 */
template <typename T1, typename T2, require_all_std_vector_t<T1, T2>* = nullptr>
void assign_impl(T1&& x, T2&& y, const char* name) {
  // We are allowed to assign to fully uninitialized matrix
  if (x.size() != 0) {
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
  if (!x.is_uninitialized()) {
    static constexpr const char* obj_type
        = is_vector<Mat1>::value ? "vector" : "matrix";
    stan::math::check_size_match(
        (std::string(obj_type) + " assign columns").c_str(), name, x.cols(),
        "right hand side columns", y.cols());
    stan::math::check_size_match(
        (std::string(obj_type) + " assign rows").c_str(), name, x.rows(),
        "right hand side rows", y.rows());
    auto prev_vals = stan::math::to_arena(x.val());
    x.vi_->val_ = std::forward<Mat2>(y);
    stan::math::reverse_pass_callback([x, prev_vals]() mutable {
      x.vi_->val_ = prev_vals;
      x.vi_->adj_.setZero();
    });
  } else {
    x = stan::math::var_value<plain_type_t<Mat2>>(std::forward<Mat2>(y));
  }
}

template <typename... Types>
struct is_tuple_impl : std::false_type {};
template <typename... Types>
struct is_tuple_impl<std::tuple<Types...>> : std::true_type {};

template <typename T>
struct is_tuple : is_tuple_impl<std::decay_t<T>> {};

template <typename Tuple1, typename Tuple2,
          require_all_t<internal::is_tuple<Tuple1>,
                        internal::is_tuple<Tuple2>>* = nullptr>
inline void assign_impl(Tuple1&& x, Tuple2&& y, const char* name) {
  x = std::forward<Tuple2>(y);
}

}  // namespace internal
}  // namespace model
}  // namespace stan
#endif

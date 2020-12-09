#ifndef STAN_MODEL_INDEXING_ACCESS_HELPERS_HPP
#define STAN_MODEL_INDEXING_ACCESS_HELPERS_HPP

#include <stan/math/prim.hpp>
#include <stan/math/rev/meta.hpp>

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

}  // namespace internal
}  // namespace model
}  // namespace stan
#endif

#ifndef STAN_MODEL_INDEXING_ACCESS_HELPERS_HPP
#define STAN_MODEL_INDEXING_ACCESS_HELPERS_HPP

#include <stan/math/prim.hpp>

namespace stan {

namespace model {

namespace internal {
// Internal helpers so we can reuse min_max index assign for Eigen/var<Eigen>
template <typename T, require_var_matrix_t<T>* = nullptr>
auto rowwise_reverse(T&& x) {
  return std::forward<T>(x).rowwise_reverse();
}

template <typename T, require_eigen_t<T>* = nullptr>
auto rowwise_reverse(T&& x) {
  return std::forward<T>(x).rowwise().reverse();
}

template <typename T, require_var_matrix_t<T>* = nullptr>
auto colwise_reverse(T&& x) {
  return std::forward<T>(x).colwise_reverse();
}

template <typename T, require_eigen_t<T>* = nullptr>
auto colwise_reverse(T&& x) {
  return std::forward<T>(x).colwise().reverse();
}
}  // namespace internal
}
}
#endif

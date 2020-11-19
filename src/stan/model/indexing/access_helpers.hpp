#ifndef STAN_MODEL_INDEXING_ACCESS_HELPERS_HPP
#define STAN_MODEL_INDEXING_ACCESS_HELPERS_HPP

#include <stan/math/prim.hpp>
#include <stan/math/rev/meta.hpp>

namespace stan {

namespace model {

namespace internal {
// Internal helpers so we can reuse min_max index assign for Eigen/var<Eigen>
template <typename T, require_var_matrix_t<T>* = nullptr>
inline auto rowwise_reverse(T&& x) {
  return std::forward<T>(x).rowwise_reverse();
}

template <typename T, require_eigen_t<T>* = nullptr>
inline auto rowwise_reverse(T&& x) {
  return std::forward<T>(x).rowwise().reverse();
}

template <typename T, require_var_matrix_t<T>* = nullptr>
inline auto colwise_reverse(T&& x) {
  return std::forward<T>(x).colwise_reverse();
}

template <typename T, require_eigen_t<T>* = nullptr>
inline auto colwise_reverse(T&& x) {
  return std::forward<T>(x).colwise().reverse();
}

inline bool check_duplicate(const arena_t<std::vector<std::array<int, 2>>>& x_idx,
                     int i, int j) {
  for (size_t k = 0; k < x_idx.size(); ++k) {
    if (x_idx[k][0] == i && x_idx[k][1] == j) {
      return true;
    }
  }
  return false;
}

inline bool check_duplicate(const arena_t<std::vector<int>>& x_idx, int i) {
  for (size_t k = 0; k < x_idx.size(); ++k) {
    if (x_idx[k] == i) {
      return true;
    }
  }
  return false;
}
}  // namespace internal
}
}
#endif

#ifndef STAN_MODEL_INDEXING_RVALUE_AT_HPP
#define STAN_MODEL_INDEXING_RVALUE_AT_HPP

#include <stan/model/indexing/index.hpp>

namespace stan {

namespace internal {

template <typename T>
using is_eigen_dense_dynamic = stan::internal::is_eigen_matrix_dynamic_impl<
    std::decay_t<T>,
    stan::is_eigen_dense_base<T>::value && stan::is_eigen<T>::value>;

template <typename T>
using require_eigen_dense_dynamic_t = require_t<is_eigen_dense_dynamic<T>>;

template <typename... Types>
using require_all_eigen_dense_dynamic_t
    = require_all_t<is_eigen_dense_dynamic<Types>...>;

}  // namespace internal

namespace model {

// relative indexing from 0; multi-indexing and return from 1
// no error checking from these methods, just indexing

/**
 * Return the index in the underlying array corresponding to the
 * specified position in the specified multi-index.
 *
 * @param[in] n Relative index position (from 0).
 * @param[in] idx Index (from 1).
 * @return Underlying index position (from 1).
 */
inline int rvalue_at(int n, const index_multi& idx) { return idx.ns_[n]; }

/**
 * Return the index in the underlying array corresponding to the
 * specified position in the specified omni-index.
 *
 * @param[in] n Relative index position (from 0).
 * @param[in] idx Index (from 1).
 * @return Underlying index position (from 1).
 */
inline constexpr int rvalue_at(int n, const index_omni& idx) { return n + 1; }

/**
 * Return the index in the underlying array corresponding to the
 * specified position in the specified min-index.
 *
 * All indexing begins from 1.
 *
 * @param[in] n Relative index position (from 0).
 * @param[in] idx Index (from 1)
 * @return Underlying index position (from 1).
 */
inline constexpr int rvalue_at(int n, const index_min& idx) {
  return idx.min_ + n;
}

/**
 * Return the index in the underlying array corresponding to the
 * specified position in the specified max-index.
 *
 * All indexing begins from 1.
 *
 * @param[in] n Relative index position (from 0).
 * @param[in] idx Index (from 1).
 * @return Underlying index position (from 1).
 */
inline constexpr int rvalue_at(int n, const index_max& idx) { return n + 1; }

/**
 * Return the index in the underlying array corresponding to the
 * specified position in the specified min-max-index.
 *
 * All indexing begins from 1.
 *
 * @param[in] n Relative index position (from 0).
 * @param[in] idx Index (from 1).
 * @return Underlying index position (from 1).
 */
inline constexpr int rvalue_at(int n, const index_min_max& idx) {
  return idx.min_ + n;
}

}  // namespace model
}  // namespace stan
#endif

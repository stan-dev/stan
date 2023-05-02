#ifndef STAN_MODEL_INDEXING_RVALUE_INDEX_SIZE_HPP
#define STAN_MODEL_INDEXING_RVALUE_INDEX_SIZE_HPP

#include <stan/model/indexing/index.hpp>

#ifdef STAN_OPENCL
#include <stan/math/opencl/indexing_rev.hpp>
#endif

namespace stan {

namespace model {

// no error checking

/**
 * Return size of specified multi-index.
 *
 * @param[in] idx Input index (from 1).
 * @param[in] size Size of container (ignored here).
 * @return Size of result.
 */
inline int rvalue_index_size(const index_multi& idx, int size) noexcept {
  return idx.ns_.size();
}

inline constexpr int rvalue_index_size(const index_uni& idx,
                                       int size) noexcept {
  return 1;
}

/**
 * Return size of specified omni-index for specified size of
 * input.
 *
 * @param[in] idx Input index (from 1).
 * @param[in] size Size of container.
 * @return Size of result.
 */
inline int rvalue_index_size(const index_omni& idx, int size) noexcept {
  return size;
}

/**
 * Return size of specified min index for specified size of
 * input.
 *
 * @param[in] idx Input index (from 1).
 * @param[in] size Size of container.
 * @return Size of result.
 */
inline int rvalue_index_size(const index_min& idx, int size) noexcept {
  return size - idx.min_ + 1;
}

/**
 * Return size of specified max index.
 *
 * @param[in] idx Input index (from 1).
 * @param[in] size Size of container (ignored).
 * @return Size of result.
 */
inline int rvalue_index_size(const index_max& idx, int size) noexcept {
  if (idx.max_ > 0) {
    return idx.max_;
  } else {
    return 0;
  }
}

/**
 * Return size of specified min - max index.
 *
 * @param[in] idx Input index (from 1).
 * @param[in] size Size of container (ignored).
 * @return Size of result.
 */
inline int rvalue_index_size(const index_min_max& idx, int size) noexcept {
  return (idx.max_ < idx.min_) ? 0 : (idx.max_ - idx.min_ + 1);
}

#ifdef STAN_OPENCL
inline int rvalue_index_size(const stan::math::matrix_cl<int>& idx,
                             int size) noexcept {
  return idx.size();
}
#endif

}  // namespace model
}  // namespace stan
#endif

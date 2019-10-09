#ifndef STAN_MODEL_INDEXING_RVALUE_INDEX_SIZE_HPP
#define STAN_MODEL_INDEXING_RVALUE_INDEX_SIZE_HPP

#include <stan/model/indexing/index.hpp>

namespace stan {

namespace model {

/**
 * Return size of specified multi-index.
 *
 * @param[in] idx Input index (from 1).
 * @param[in] size Size of container (ignored here).
 * @return Size of result.
 */
template <typename Ind>
inline int rvalue_index_size(const index_multi& idx, Ind size) {
  return idx.ns_.size();
}

/**
 * Return size of specified omni-index for specified size of
 * input.
 *
 * @param[in] idx Input index (from 1).
 * @param[in] size Size of container.
 * @return Size of result.
 */
template <typename Ind>
inline int rvalue_index_size(const index_omni& idx, Ind size) {
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
template <typename Ind>
inline int rvalue_index_size(const index_min& idx, Ind size) {
  return size - idx.min_ + 1;
}

/**
 * Return size of specified max index.
 *
 * @param[in] idx Input index (from 1).
 * @param[in] size Size of container (ignored).
 * @return Size of result.
 */
template <typename Ind>
inline int rvalue_index_size(const index_max& idx, Ind size) {
  return idx.max_;
}

/**
 * Return size of specified min - max index.  If the maximum value
 * index is less than the minimun index, the size will be zero.
 *
 * @param[in] idx Input index (from 1).
 * @param[in] size Size of container (ignored).
 * @return Size of result.
 */
template <typename Ind>
inline int rvalue_index_size(const index_min_max& idx, Ind size) {
  return (idx.max_ < idx.min_) ? 0 : (idx.max_ - idx.min_ + 1);
}

}  // namespace model
}  // namespace stan
#endif

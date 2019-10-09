#ifndef STAN_MODEL_INDEXING_INDEX_LIST_HPP
#define STAN_MODEL_INDEXING_INDEX_LIST_HPP

#include <stan/model/indexing/index.hpp>

namespace stan {

namespace model {

/**
 * Structure for an empty (size zero) index list.
 */
struct nil_index_list {};

/**
 * Template structure for an index list consisting of a head and
 * tail index.
 *
 * @tparam H type of index stored as the head of the list.
 * @tparam T type of index list stored as the tail of the list.
 */
template <typename H, typename T>
struct cons_index_list {
  const H head_;
  const T tail_;

  /**
   * Construct a non-empty index list with the specified index for
   * a head and specified index list for a tail.
   *
   * @param head Index for head.
   * @param tail Index list for tail.
   */
  explicit cons_index_list(const H& head, const T& tail)
      : head_(head), tail_(tail) {}
};

// factory-like function does type inference for I and T
template <typename I, typename T>
inline auto cons_list(const I& idx1, const T& t) {
  return cons_index_list<I, T>(idx1, t);
}

inline auto index_list() { return nil_index_list(); }

template <typename I>
inline auto index_list(const I& idx) {
  return cons_list(idx, index_list());
}

template <typename I1, typename... Idx2>
inline auto index_list(const I1& idx1, const Idx2&... idx2) {
  return cons_list(idx1, index_list(idx2...));
}

template <typename I, typename L>
using generic_index = cons_index_list<I, L>;

using single_index = cons_index_list<index_uni, nil_index_list>;

template <typename I>
using multiple_index = cons_index_list<I, nil_index_list>;

template <typename I1, typename I2>
using variadic_multiple_index = cons_index_list<I1, multiple_index<I2>>;

template <typename I>
using variadic_single_index = cons_index_list<I, single_index>;

using uni_single_index = cons_index_list<index_uni, single_index>;

template <typename L>
using uni_variadic_index = cons_index_list<index_uni, L>;

template <typename L>
using uni_multiple_index = cons_index_list<index_uni, multiple_index<L>>;

}  // namespace model
}  // namespace stan
#endif

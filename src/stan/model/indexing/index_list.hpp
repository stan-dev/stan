#ifndef STAN_MODEL_INDEXING_INDEX_LIST_HPP
#define STAN_MODEL_INDEXING_INDEX_LIST_HPP

#include <stan/math/prim/meta.hpp>
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
  template <typename HH, typename TT,
   require_same_t<H, HH>..., require_same_t<T, TT>...>
  explicit cons_index_list(HH&& head, TT&& tail)
      : head_(std::forward<HH>(head)), tail_(std::forward<TT>(tail)) {}
};

// factory-like function does type inference for I and T
template <typename I, typename T>
inline auto cons_list(I&& idx1, T&& t) {
  return cons_index_list<I, T>(std::forward<I>(idx1), std::forward<T>(t));
}

inline auto index_list() { return nil_index_list(); }

template <typename I>
inline auto index_list(I&& idx) {
  return cons_list(std::forward<I>(idx), index_list());
}

template <typename I1, typename... Idx2>
inline auto index_list(I1&& idx1, Idx2&&... idx2) {
  return cons_list(std::forward<I1>(idx1), index_list(std::forward<Idx2>(idx2)...));
}

template <typename I, typename L>
using generic_index = cons_index_list<I, L>;

using single_index = cons_index_list<index_uni, nil_index_list>;

using uni_single_index = cons_index_list<index_uni, single_index>;

template <typename I>
using multiple_index = cons_index_list<I, nil_index_list>;

template <typename L>
using uni_variadic_index = cons_index_list<index_uni, L>;

template <typename L>
using uni_multiple_index = cons_index_list<index_uni, multiple_index<L>>;

template <typename I>
using variadic_single_index = cons_index_list<I, single_index>;

template <typename I1, typename I2>
using variadic_multiple_index = cons_index_list<I1, multiple_index<I2>>;


}  // namespace model
}  // namespace stan
#endif

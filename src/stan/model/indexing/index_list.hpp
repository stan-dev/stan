#ifndef STAN_MODEL_INDEXING_INDEX_LIST_HPP
#define STAN_MODEL_INDEXING_INDEX_LIST_HPP

#include <utility>
#include <type_traits>

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
  std::decay_t<H> head_;
  std::decay_t<T> tail_;

  /**
   * Construct a non-empty index list with the specified index for
   * a head and specified index list for a tail.
   *
   * @param head Index for head.
   * @param tail Index list for tail.
   */
  template <typename Head, typename Tail>
  explicit cons_index_list(Head&& head, Tail&& tail)
      : head_(std::forward<Head>(head)), tail_(std::forward<Tail>(tail)) {}
};

/**
 * Construct a pack of indices.
 * @tparam T1 The first index type.
 * @tparam T2 The second index type.
 * @param idx1 first index placed in `head_`
 * @param idx2 second index placed in `tail_`
 */
template <typename T1, typename T2>
inline auto cons_list(T1&& idx1, T2&& idx2) {
  return cons_index_list<std::decay_t<T1>, std::decay_t<T2>>(
      std::forward<T1>(idx1), std::forward<T2>(idx2));
}

/**
 * Expansion stop for index_list returning back a `nul_index_list`
 */
inline auto index_list() { return nil_index_list(); }

/**
 * Factory-like function to construct a `cons_index_list` of `cons_index_list`s
 * @tparam I1 First index type
 * @tparam I2 Parameter pack of index types.
 * @param idx1 First index to construct the cons_index_list.
 * @param idx2 A parameter pack expanded and recursivly called into
 *  `index_list()`
 */
template <typename T, typename... Types>
inline auto index_list(T&& idx1, Types&&... idx2) {
  return cons_list(std::forward<T>(idx1),
                   index_list(std::forward<Types>(idx2)...));
}

}  // namespace model
}  // namespace stan
#endif

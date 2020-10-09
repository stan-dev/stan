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
  explicit constexpr cons_index_list(Head&& head, Tail&& tail)
      : head_(std::forward<Head>(head)), tail_(std::forward<Tail>(tail)) {}
};

// factory-like function does type inference for I and T
template <typename I, typename T>
inline constexpr auto cons_list(I&& idx1, T&& t) {
  return cons_index_list<std::decay_t<I>, std::decay_t<T>>(
      std::forward<I>(idx1), std::forward<T>(t));
}

/**
 * Expansion stop for index_list returning back a `nul_index_list`
 */
inline constexpr auto index_list() { return nil_index_list(); }

/**
 * Factory-like function to construct a `cons_index_list` of `cons_index_list`s
 * @tparam I1 First index type
 * @tparam I2 Parameter pack of index types.
 * @param idx1 First index to construct the cons_index_list.
 * @param idx2 A parameter pack expanded and recursivly called into
 *  `index_list()`
 */
template <typename I1, typename... I2>
inline constexpr auto index_list(I1&& idx1, I2&&... idx2) {
  return cons_list(std::forward<I1>(idx1),
                   index_list(std::forward<I2>(idx2)...));
}

}  // namespace model
}  // namespace stan
#endif

#ifndef STAN_MODEL_INDEXING_INDEX_HPP
#define STAN_MODEL_INDEXING_INDEX_HPP

#include <stan/math/prim/meta.hpp>
#include <vector>

namespace stan {

namespace model {

// SINGLE INDEXING (reduces dimensionality)

/**
 * Structure for an indexing consisting of a single index.
 * Applying this index reduces the dimensionality of the container
 * to which it is applied by one.
 */
struct index_uni {
  int n_;

  /**
   * Construct a single indexing from the specified index.
   *
   * @param n single index.
   */
  explicit index_uni(int n) noexcept : n_(n) {}
};

// MULTIPLE INDEXING (does not reduce dimensionality)

/**
 * Structure for an indexing consisting of multiple indexes.  The
 * indexes do not need to be unique or in order.
 */
struct index_multi {
  std::vector<int> ns_;

  /**
   * Construct a multiple indexing from the specified indexes.
   *
   * @param ns multiple indexes.
   */
  template <typename T, require_std_vector_vt<std::is_integral, T>* = nullptr>
  explicit index_multi(T&& ns) noexcept : ns_(std::forward<T>(ns)) {}
};

/**
 * Structure for an indexing that consists of all indexes for a
 * container.  Applying this index is a no-op.
 */
struct index_omni {};

/**
 * Structure for an indexing from a minimum index (inclusive) to
 * the end of a container.
 */
struct index_min {
  int min_;

  /**
   * Construct an indexing from the specified minimum index (inclusive).
   *
   * @param min minimum index (inclusive).
   */
  explicit index_min(int min) noexcept : min_(min) {}
};

/**
 * Structure for an indexing from the start of a container to a
 * specified maximum index (inclusive).
 */
struct index_max {
  int max_;

  /**
   * Construct an indexing from the start of the container up to
   * the specified maximum index (inclusive).
   *
   * @param max maximum index (inclusive).
   */
  explicit index_max(int max) noexcept : max_(max) {}
};

/**
 * Structure for an indexing from a minimum index (inclusive) to a
 * maximum index (inclusive).
 */
struct index_min_max {
  int min_;
  int max_;
  /**
   * Return whether the index is positive or negative
   */
  inline bool is_ascending() const noexcept { return min_ <= max_; }
  /**
   * Construct an indexing from the specified minimum index
   * (inclusive) and maximum index (inclusive).
   *
   * @param min minimum index (inclusive).
   * @param max maximum index (inclusive).
   */
  index_min_max(int min, int max) noexcept : min_(min), max_(max) {}
};

}  // namespace model
}  // namespace stan
#endif

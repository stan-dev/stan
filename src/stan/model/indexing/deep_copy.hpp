#ifndef STAN_MODEL_INDEXING_DEEP_COPY_HPP
#define STAN_MODEL_INDEXING_DEEP_COPY_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/meta.hpp>
#include <vector>

namespace stan {

namespace model {

/**
 * Return the specified argument as a constant reference.
 *
 * <p>Warning: because of the usage pattern of this class, this
 * function only needs to return value references, not actual
 * copies.  The functions that call this overload recursively will
 * be doing the actual copies with assignment.
 *
 * @tparam T Any type.
 * @param x Input value.
 * @return Copy of input.
 */
template <typename T>
inline plain_type_t<T> deep_copy(T&& x) {
  return std::forward<T>(x);
}

}  // namespace model
}  // namespace stan
#endif

#ifndef STAN_MODEL_INDEXING_DEEP_COPY_HPP
#define STAN_MODEL_INDEXING_DEEP_COPY_HPP

#include <stan/math/prim/meta.hpp>
#include <Eigen/Dense>
#include <vector>

namespace stan {

// TODO(Steve): Put these is Stan math
template <typename Container>
using require_not_container_t = require_not_t<
    math::disjunction<is_eigen<Container>, is_std_vector<Container>>>;

template <typename Container>
using require_container_t = require_t<
    math::disjunction<is_eigen<Container>, is_std_vector<Container>>>;

namespace model {

/**
 * Return the specified argument as a constant reference.
 *
 * <p>Warning: because of the usage pattern of this class, this
 * function only needs to return value references, not actual
 * copies.  The functions that call this overload recursively will
 * be doing the actual copies with assignment.
 *
 * @tparam T Type of scalar.
 * @param x Input value.
 * @return Constant reference to input.
 */
template <typename Container, typename = require_not_container_t<Container>>
inline const Container& deep_copy(Container&& x) {
  return std::forward<Container>(x);
}

/**
 * Return a copy of the specified matrix, vector, or row
 * vector.  The return value is a copy in the sense that modifying
 * its contents will not affect the original matrix.
 *
 * <p>Warning:  This function assumes that the elements of the
 * matrix deep copy under assignment.
 *
 * @tparam T Scalar type.
 * @tparam R Row type specificiation.
 * @tparam C Column type specificiation.
 * @param a Input matrix, vector, or row vector.
 * @return Deep copy of input.
 */
template <typename Container, typename = require_container_t<Container>>
inline Container deep_copy(Container&& a) {
  Container result(std::forward<Container>(a));
  return result;
}

}  // namespace model
}  // namespace stan
#endif

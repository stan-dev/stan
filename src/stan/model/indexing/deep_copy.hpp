#ifndef STAN_MODEL_INDEXING_DEEP_COPY_HPP
#define STAN_MODEL_INDEXING_DEEP_COPY_HPP

#include <stan/math/prim.hpp>
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
 * @tparam T Type of scalar.
 * @param x Input value.
 * @return Constant reference to input.
 */
template <typename T, require_stan_scalar_t<T>* = nullptr>
inline const T& deep_copy(const T& x) {
  return x;
}

template <typename T, require_stan_scalar_t<T>* = nullptr>
inline T& deep_copy(T& x) {
  return x;
}

template <typename T, require_stan_scalar_t<T>* = nullptr>
inline T deep_copy(T&& x) {
  return std::forward<T>(x);
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
 * @tparam R Row type specification.
 * @tparam C Column type specification.
 * @param a Input matrix, vector, or row vector.
 * @return Deep copy of input.
 */
template <typename EigMat, require_eigen_t<EigMat>* = nullptr>
inline plain_type_t<EigMat> deep_copy(EigMat&& a) {
  return plain_type_t<EigMat>(std::forward<EigMat>(a));
}

/**
 * Return a deep copy of the specified standard vector.  The
 * return value is a copy in the sense that modifying its contents
 * will not affect the original vector.
 *
 * <p>Warning:  This function assumes that the elements of the
 * vector deep copy under assignment.
 *
 * @tparam T Scalar type.
 * @param v Input vector.
 * @return Deep copy of input.
 */
template <typename StdVec, require_std_vector_t<StdVec>* = nullptr>
inline std::decay_t<StdVec> deep_copy(StdVec&& v) {
  return {std::forward<StdVec>(v)};
}

}  // namespace model
}  // namespace stan
#endif

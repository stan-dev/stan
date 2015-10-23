#ifndef STAN_MODEL_INDEXING_DEEP_COPY_HPP
#define STAN_MODEL_INDEXING_DEEP_COPY_HPP

#include <Eigen/Dense>
#include <vector>

namespace stan {
  namespace model {

    // template <typename T>
    // inline void deep_copy(const T& x, T& x_copy) {
    //   x_copy = x;
    // }

    // template <typename T, int R, int C>
    // inline void deep_copy(const Eigen::Matrix<T, R, C>& m,
    //                       Eigen::Matrix<T, R, C>& m_copy) {
    //   m_copy.resize(m.rows(), m.cols());
    //   for (int j = 0; j < m.cols(); ++j)
    //     for (int i = 0 ; i < m.rows(); ++i)
    //       deep_copy(m(i,j), m_copy(i,j));
    // }

    // template <typename T>
    // inline void deep_copy(const std::vector<T>& v, 
    //                       std::vector<T>& v_copy) {
    //   v_copy.resize(v.size());
    //   for (size_t i = 0; i < v.size(); ++i)
    //     deep_copy(v[i], v_copy[i]);
    // }

    // /**
    //  * Return a copy of the specified argument.
    //  *
    //  * <p>The argument is input as a constant reference, but returned
    //  * as a regular type in order to make the copy.
    //  *
    //  * @tparam T Type of scalar.
    //  * @param x Input value.
    //  * @return Deep copy of input.
    //  */
    // template <typename T>
    // inline T deep_copy(const T& x, int n) {
    //   return x;
    // }

    // /**
    //  * Return a deep copy of the specified matrix, vector, or row
    //  * vector.  The copy is deep in the sense that
    //  * <code>deep_copy</code> is used recursively to copy values.
    //  *
    //  * @tparam T Scalar type.
    //  * @tparam R Row specificiation.
    //  * @tparam C Column specificiation.
    //  * @param a Input matrix, vector, or row vector.
    //  * @return Deep copy of input.
    //  */
    // template <typename T, int R, int C>
    // inline Eigen::Matrix<T, R, C> deep_copy(const Eigen::Matrix<T, R, C>& a) {
    //   Eigen::Matrix<T, R, C> result;
    //   for (int i = 0; i < a.size(); ++i)
    //     result(i) = deep_copy(a(i));
    //   return result;
    // }

    // /**
    //  * Return a deep copy of the specified standard vector.  The copy
    //  * is deep in the sense that <code>deep_copy</code> is used
    //  * recursively to copy values.
    //  *
    //  * @tparam T Scalar type.
    //  * @param v Input vector.
    //  * @return Deep copy of input.
    //  */
    // template <typename T>
    // inline std::vector<T> deep_copy(const std::vector<T>& v) {
    //   std::vector<T> v_copy;
    //   deep_copy(v, v_copy);
    //   return v_copy;
    // }

  }
}
#endif

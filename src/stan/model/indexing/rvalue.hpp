#ifndef STAN_MODEL_INDEXING_RVALUE_HPP
#define STAN_MODEL_INDEXING_RVALUE_HPP

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <stan/math/prim.hpp>
#include <stan/model/indexing/index.hpp>
#include <stan/model/indexing/index_list.hpp>
#include <stan/model/indexing/rvalue_at.hpp>
#include <stan/model/indexing/rvalue_index_size.hpp>
#include <stan/model/indexing/rvalue_return.hpp>
#include <vector>

namespace stan {

namespace model {

// all indexing from 1

/**
 * Return the result of indexing a specified scalar type with
 * a nil index list, which just returns the scalar.
 *
 * Types:  T[] : T
 *
 * @tparam T Scalar type.
 * @param[in] c Value to index.
 * @return Input value.
 */
template <typename T, typename = require_stan_scalar_t<T>>
inline T rvalue(T c, const nil_index_list& /*idx*/, const char* /*name*/ = "",
                int /*depth*/ = 0) {
  return c;
}

/**
 * Return the result of indexing a specified non-scalar value with
 * a nil index list, which just returns the value.
 *
 * Types:  T[] : T
 *
 * @tparam T Scalar type.
 * @param[in] c Value to index.
 * @return Input value.
 */
template <typename T, typename = require_not_stan_scalar_t<T>>
inline decltype(auto) rvalue(T&& c, const nil_index_list& /*idx*/,
                             const char* /*name*/ = "", int /*depth*/ = 0) {
  return std::forward<T>(c);
}

/**
 * Return the result of indexing the specified Eigen vector with a
 * sequence containing one single index, returning a scalar.
 *
 * Types:  vec[single] : scal
 *
 * @tparam EigVec Type of the Eigen Vector.
 * @param[in] v Vector being indexed.
 * @param[in] idx One single index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing vector.
 */
template <typename EigVec, typename = require_eigen_vector_t<EigVec>>
inline auto rvalue(const EigVec& v,
                   const cons_index_list<index_uni, nil_index_list>& idx,
                   const char* name = "ANON", int depth = 0) {
  int ones_idx = idx.head_.n_;
  const Eigen::Ref<const typename EigVec::PlainObject>& vec = v;
  math::check_range("vector[single] indexing", name, v.size(), ones_idx);
  return vec.coeff(ones_idx - 1);
}

/**
 * Return the result of indexing the specified Eigen vector with a
 * sequence containing one multiple index, returning a vector.
 *
 * Types: vec[multiple] : vec
 *
 * @tparam EigVec Type of the Eigen Vector.
 * @tparam I Multi-index type.
 * @param[in] v Eigen vector.
 * @param[in] idx Index consisting of one multi-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing vector.
 */
template <typename EigVec, typename I,
          typename = require_not_same_t<index_uni, I>,
          typename = require_eigen_vector_t<EigVec>>
inline auto rvalue(const EigVec& v,
                   const cons_index_list<I, nil_index_list>& idx,
                   const char* name = "ANON", int depth = 0) {
  int size = rvalue_index_size(idx.head_, v.size());
  const Eigen::Ref<const typename EigVec::PlainObject>& vec = v;
  typename EigVec::PlainObject a(size);
  for (int i = 0; i < size; ++i) {
    int n = rvalue_at(i, idx.head_);
    math::check_range("vector[multi] indexing", name, v.size(), n);
    a.coeffRef(i) = vec.coeff(n - 1);
  }
  return a;
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * sequence consisting of one single index, returning a row vector.
 *
 * Types:  mat[single] : rowvec
 *
 * @tparam EigMat The type of the Eigen Matrix.
 * @param[in] a Eigen matrix.
 * @param[in] idx Index consisting of one uni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat, typename = require_eigen_t<EigMat>,
          typename = require_not_eigen_vector_t<EigMat>>
inline auto rvalue(const EigMat& a,
                   const cons_index_list<index_uni, nil_index_list>& idx,
                   const char* name = "ANON", int depth = 0) {
  int n = idx.head_.n_;
  math::check_range("matrix[uni] indexing", name, a.rows(), n);
  return a.row(n - 1).eval();
}

/**
 * Return the result of indexing the specified Eigen matrix with
 * an omni and single index, returning a vector.
 *
 * Types:  mat[omni, single] : vec
 *
 * @tparam EigMat The type of the Eigen Matrix.
 * @param[in] a Eigen matrix.
 * @param[in] idx Index consisting of one uni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat, typename = require_eigen_t<EigMat>,
          typename = require_not_eigen_vector_t<EigMat>>
inline auto rvalue(
    const EigMat& a,
    const cons_index_list<index_omni,
                          cons_index_list<index_uni, nil_index_list>>& idx,
    const char* name = "ANON", int depth = 0) {
  int n = idx.tail_.head_.n_;
  math::check_range("matrix[uni] indexing", name, a.rows(), n);
  return a.col(n - 1).eval();
}

/**
 * Return the result of indexing the specified Eigen matrix with
 * a single index and an omni index, returning a vector.
 *
 * Types:  mat[single, omni] : vec
 *
 * @tparam EigMat The type of the Eigen Matrix.
 * @param[in] a Eigen matrix.
 * @param[in] idx Index consisting of one uni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat, typename = require_eigen_t<EigMat>,
          typename = require_not_eigen_vector_t<EigMat>>
inline auto rvalue(
    const EigMat& a,
    const cons_index_list<index_uni,
                          cons_index_list<index_omni, nil_index_list>>& idx,
    const char* name = "ANON", int depth = 0) {
  int n = idx.head_.n_;
  math::check_range("matrix[uni] indexing", name, a.rows(), n);
  return a.row(n - 1).eval();
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * sequence consisting of a one multiple index, returning a matrix.
 *
 * Types:  mat[multiple] : mat
 *
 * @tparam EigMat The type of the Eigen Matrix.
 * @tparam I Type of multiple index.
 * @param[in] a Matrix to index.
 * @param[in] idx Index consisting of single multiple index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat, typename I,
          typename = require_not_same_t<index_uni, I>,
          typename = require_eigen_t<EigMat>,
          typename = require_not_eigen_vector_t<EigMat>>
inline auto rvalue(const EigMat& a,
                   const cons_index_list<I, nil_index_list>& idx,
                   const char* name = "ANON", int depth = 0) {
  int n_rows = rvalue_index_size(idx.head_, a.rows());
  typename EigMat::PlainObject b(n_rows, a.cols());
  const Eigen::Ref<const typename EigMat::PlainObject>& mat = a;
  for (int i = 0; i < n_rows; ++i) {
    int n = rvalue_at(i, idx.head_);
    math::check_range("matrix[multi] indexing", name, mat.rows(), n);
    b.row(i) = mat.row(n - 1);
  }
  return b;
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * sequence consisting of two single indexes, returning a scalar.
 *
 * Types:  mat[single,single] : scalar
 *
 * @tparam EigMat The type of the Eigen Matrix.
 * @param[in] a Matrix to index.
 * @param[in] idx Pair of single indexes.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat, typename = require_eigen_t<EigMat>,
          typename = require_not_eigen_vector_t<EigMat>>
inline auto rvalue(
    const EigMat& a,
    const cons_index_list<index_uni,
                          cons_index_list<index_uni, nil_index_list>>& idx,
    const char* name = "ANON", int depth = 0) {
  int m = idx.head_.n_;
  int n = idx.tail_.head_.n_;
  const Eigen::Ref<const typename EigMat::PlainObject>& mat = a;
  math::check_range("matrix[uni,uni] indexing, row", name, a.rows(), m);
  math::check_range("matrix[uni,uni] indexing, col", name, a.cols(), n);
  return mat.coeff(m - 1, n - 1);
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * sequence consisting of a single index and multiple index,
 * returning a row vector.
 *
 * Types:  mat[single,multiple] : row vector
 *
 * @tparam EigMat The type of the Eigen Matrix.
 * @tparam I Type of multiple index.
 * @param[in] a Matrix to index.
 * @param[in] idx Pair of single index and multiple index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat, typename I,
          typename = require_not_same_t<index_uni, I>,
          typename = require_eigen_t<EigMat>,
          typename = require_not_eigen_vector_t<EigMat>>
inline auto rvalue(
    const EigMat& a,
    const cons_index_list<index_uni, cons_index_list<I, nil_index_list>>& idx,
    const char* name = "ANON", int depth = 0) {
  int m = idx.head_.n_;
  math::check_range("matrix[uni,multi] indexing, row", name, a.rows(), m);
  return rvalue(a.row(m - 1), idx.tail_);
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * sequence consisting of a multiple index and a single index,
 * returning a vector.
 *
 * Types:  mat[multiple,single] : vector
 *
 * @tparam EigMat The type of the Eigen Matrix.
 * @tparam I Type of multiple index.
 * @param[in] a Matrix to index.
 * @param[in] idx Pair multiple index and single index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat, typename I,
          typename = require_not_same_t<index_uni, I>,
          typename = require_eigen_t<EigMat>,
          typename = require_not_eigen_vector_t<EigMat>>
inline auto rvalue(
    const EigMat& a,
    const cons_index_list<I, cons_index_list<index_uni, nil_index_list>>& idx,
    const char* name = "ANON", int depth = 0) {
  int rows = rvalue_index_size(idx.head_, a.rows());
  Eigen::Matrix<scalar_type_t<EigMat>, EigMat::RowsAtCompileTime, 1> c(rows);
  const Eigen::Ref<const typename EigMat::PlainObject>& mat = a;
  for (int i = 0; i < rows; ++i) {
    int m = rvalue_at(i, idx.head_);
    int n = idx.tail_.head_.n_;
    math::check_range("matrix[multi,uni] index row", name, a.rows(), m);
    math::check_range("matrix[multi,uni] index col", name, a.cols(), n);
    c.coeffRef(i) = mat.coeff(m - 1, n - 1);
  }
  return c;
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * sequence consisting of a pair o multiple indexes, returning a
 * a matrix.
 *
 * Types:  mat[multiple,multiple] : mat
 *
 * @tparam EigMat The type of the Eigen Matrix.
 * @tparam I Type of multiple index.
 * @param[in] a Matrix to index.
 * @param[in] idx Pair of multiple indexes.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat, typename I1, typename I2,
          typename = require_any_not_same_t<index_uni, I1, I2>,
          typename = require_eigen_t<EigMat>,
          typename = require_not_eigen_vector_t<EigMat>>
inline auto rvalue(
    const EigMat& a,
    const cons_index_list<I1, cons_index_list<I2, nil_index_list>>& idx,
    const char* name = "ANON", int depth = 0) {
  int rows = rvalue_index_size(idx.head_, a.rows());
  int cols = rvalue_index_size(idx.tail_.head_, a.cols());
  Eigen::Matrix<scalar_type_t<EigMat>, EigMat::RowsAtCompileTime,
                EigMat::ColsAtCompileTime>
      c(rows, cols);
  const Eigen::Ref<const typename EigMat::PlainObject>& mat = a;
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      int m = rvalue_at(i, idx.head_);
      int n = rvalue_at(j, idx.tail_.head_);
      math::check_range("matrix[multi,multi] row index", name, mat.rows(), m);
      math::check_range("matrix[multi,multi] col index", name, mat.cols(), n);
      c.coeffRef(i, j) = mat.coeff(m - 1, n - 1);
    }
  }
  return c;
}

/**
 * Return the result of indexing the specified array with
 * a list of indexes beginning with a single index;  the result is
 * determined recursively.  Note that arrays are represented as
 * standard library vectors.
 *
 * Types:  std::vector<T>[single | L] : T[L]
 *
 * @tparam T Type of list elements.
 * @tparam L Index list type for indexes after first index.
 * @param[in] c Container of list elements.
 * @param[in] idx Index list beginning with single index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing array.
 */
template <typename T, typename L>
inline auto rvalue(const std::vector<T>& c,
                   const cons_index_list<index_uni, L>& idx,
                   const char* name = "ANON", int depth = 0) {
  int n = idx.head_.n_;
  math::check_range("array[uni,...] index", name, c.size(), n);
  return rvalue(c[n - 1], idx.tail_, name, depth + 1);
}

/**
 * Return the result of indexing the specified array with
 * a list of indexes beginning with a multiple index;  the result is
 * determined recursively.  Note that arrays are represented as
 * standard library vectors.
 *
 * Types:  std::vector<T>[multiple | L] : std::vector<T[L]>
 *
 * @tparam T Type of list elements.
 * @tparam L Index list type for indexes after first index.
 * @param[in] c Container of list elements.
 * @param[in] idx Index list beginning with multiple index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing array.
 */
template <typename T, typename I, typename L>
inline auto rvalue(const std::vector<T>& c, const cons_index_list<I, L>& idx,
                   const char* name = "ANON", int depth = 0) {
  typename rvalue_return<std::vector<T>, cons_index_list<I, L>>::type result;
  for (int i = 0; i < rvalue_index_size(idx.head_, c.size()); ++i) {
    int n = rvalue_at(i, idx.head_);
    math::check_range("array[multi,...] index", name, c.size(), n);
    result.push_back(rvalue(c[n - 1], idx.tail_, name, depth + 1));
  }
  return result;
}

}  // namespace model
}  // namespace stan
#endif

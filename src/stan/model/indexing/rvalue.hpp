#ifndef STAN_MODEL_INDEXING_RVALUE_HPP
#define STAN_MODEL_INDEXING_RVALUE_HPP

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <stan/math/prim/mat.hpp>
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
 * Return the result of indexing a specified value with
 * a nil index list, which just returns the value.
 *
 * Types:  T[] : T
 *
 * @tparam T Scalar type.
 * @param[in] c Value to index.
 * @return Input value.
 */
template <typename T>
inline T& rvalue(T&& c, const nil_index_list& /*idx*/,
                 const char* /*name*/ = "", int /*depth*/ = 0) {
  return std::forward<T>(c);
}

// TODO(Steve): Put this in math and fix row/col (and make one for not eigen
// vector)
template <typename T>
using require_eigen_row_vector = require_t<is_eigen_row_vector<T>>;
template <typename T>
using require_eigen_col_vector = require_t<is_eigen_col_vector<T>>;
/**
 * Return the result of indexing the specified Eigen vector with a
 * sequence containing one single index, returning a scalar.
 *
 * Types:  vec[single] : scal
 *
 * @tparam T Scalar type.
 * @param[in] v Vector being indexed.
 * @param[in] idx One single index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing vector.
 */
template <typename Vec, require_eigen_vector_t<Vec>...>
inline auto rvalue(Vec&& v, const single_index& idx, const char* name = "ANON",
                   int depth = 0) {
  int ones_idx = idx.head_.n_;
  if (is_eigen_row_vector<Vec>::value) {
    math::check_range("row_vector[single] indexing", name, v.size(), ones_idx);
  } else {
    math::check_range("vector[single] indexing", name, v.size(), ones_idx);
  }
  return v.coeffRef(ones_idx - 1);
}

/**
 * Return the result of indexing the specified Eigen vector with a
 * sequence containing one multiple index, returning a vector.
 *
 * Types: vec[multiple] : vec
 *
 * @tparam T Scalar type.
 * @tparam I Multi-index type.
 * @param[in] v Eigen vector.
 * @param[in] idx Index consisting of one multi-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing vector.
 */
template <typename Vec, typename I, require_eigen_vector_t<Vec>...,
          require_not_same_t<I, index_uni>...>
inline auto rvalue(Vec&& v, const multiple_index<I>& idx,
                   const char* name = "ANON", int depth = 0) {
  int size = rvalue_index_size(idx.head_, v.size());
  std::decay_t<decltype(v.eval())> a(size);
  for (int i = 0; i < size; ++i) {
    int n = rvalue_at(i, idx.head_);
    if (is_eigen_row_vector<Vec>::value) {
      math::check_range("row_vector[multi] indexing", name, v.size(), n);
    } else {
      math::check_range("vector[multi] indexing", name, v.size(), n);
    }
    a.coeffRef(i) = v.coeffRef(n - 1);
  }
  return a;
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * sequence consisting of one single index, returning a row vector.
 *
 * Types:  mat[single] : rowvec
 *
 * @tparam T Scalar type.
 * @param[in] a Eigen matrix.
 * @param[in] idx Index consisting of one uni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename Mat, require_eigen_t<Mat>...,
          require_not_eigen_vector_t<Mat>...>
inline auto rvalue(Mat&& a, const single_index& idx, const char* name = "ANON",
                   int depth = 0) {
  int n = idx.head_.n_;
  math::check_range("matrix[uni] indexing", name, a.rows(), n);
  return a.row(n - 1);
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * sequence consisting of a one multiple index, returning a matrix.
 *
 * Types:  mat[multiple] : mat
 *
 * @tparam T Scalar type.
 * @tparam I Type of multiple index.
 * @param[in] a Matrix to index.
 * @param[in] idx Index consisting of single multiple index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename Mat, typename I, require_eigen_t<Mat>...,
          require_not_eigen_vector_t<Mat>...,
          require_not_same_t<I, index_uni>...>
inline auto rvalue(Mat&& a, const multiple_index<I>& idx,
                   const char* name = "ANON", int depth = 0) {
  int n_rows = rvalue_index_size(idx.head_, a.rows());
  std::decay_t<Mat> b(n_rows, a.cols());
  for (int i = 0; i < n_rows; ++i) {
    int n = rvalue_at(i, idx.head_);
    math::check_range("matrix[multi] indexing", name, a.rows(), n);
    b.row(i) = a.row(n - 1);
  }
  return b;
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * sequence consisting of two single indexes, returning a scalar.
 *
 * Types:  mat[single,single] : scalar
 *
 * @tparam T Scalar type.
 * @param[in] a Matrix to index.
 * @param[in] idx Pair of single indexes.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename Mat, require_eigen_t<Mat>...,
          require_not_eigen_vector_t<Mat>...>
inline auto rvalue(Mat&& a, const uni_single_index& idx,
                   const char* name = "ANON", int depth = 0) {
  int m = idx.head_.n_;
  int n = idx.tail_.head_.n_;
  math::check_range("matrix[uni,uni] indexing, row", name, a.rows(), m);
  math::check_range("matrix[uni,uni] indexing, col", name, a.cols(), n);
  return a.coeffRef(m - 1, n - 1);
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * sequence consisting of a single index and multiple index,
 * returning a row vector.
 *
 * Types:  mat[single,multiple] : row vector
 *
 * @tparam T Scalar type.
 * @tparam I Type of multiple index.
 * @param[in] a Matrix to index.
 * @param[in] idx Pair of single index and multiple index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename Mat, typename I, require_not_same_t<I, index_uni>...,
          require_eigen_t<Mat>..., require_not_eigen_vector_t<Mat>...>
inline auto rvalue(Mat&& a, const uni_multiple_index<I>& idx,
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
 * @tparam T Scalar type.
 * @tparam I Type of multiple index.
 * @param[in] a Matrix to index.
 * @param[in] idx Pair multiple index and single index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename Mat, typename I, require_not_same_t<I, index_uni>...,
          require_eigen_t<Mat>..., require_not_eigen_vector_t<Mat>...>
inline auto rvalue(Mat&& a, const variadic_single_index<I>& idx,
                   const char* name = "ANON", int depth = 0) {
  int rows = rvalue_index_size(idx.head_, a.rows());
  Eigen::VectorXd c(rows);
  for (int i = 0; i < rows; ++i) {
    int m = rvalue_at(i, idx.head_);
    int n = idx.tail_.head_.n_;
    math::check_range("matrix[multi,uni] index row", name, a.rows(), m);
    math::check_range("matrix[multi,uni] index col", name, a.cols(), n);
    c.coeffRef(i) = a.coeffRef(m - 1, n - 1);
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
 * @tparam T Scalar type.
 * @tparam I Type of multiple index.
 * @param[in] a Matrix to index.
 * @param[in] idx Pair of multiple indexes.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename Mat, typename I1, typename I2,
          require_not_same_t<index_uni, I1>...,
          require_not_same_t<index_uni, I2>..., require_eigen_t<Mat>...,
          require_not_eigen_vector_t<Mat>...>
inline auto rvalue(
    Mat&& a,
    const cons_index_list<I1, cons_index_list<I2, nil_index_list>>& idx,
    const char* name = "ANON", int depth = 0) {
  int rows = rvalue_index_size(idx.head_, a.rows());
  int cols = rvalue_index_size(idx.tail_.head_, a.cols());
  std::decay_t<decltype(a.eval())> c(rows, cols);
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      int m = rvalue_at(i, idx.head_);
      int n = rvalue_at(j, idx.tail_.head_);
      math::check_range("matrix[multi,multi] row index", name, a.rows(), m);
      math::check_range("matrix[multi,multi] col index", name, a.cols(), n);
      c.coeffRef(i, j) = a.coeffRef(m - 1, n - 1);
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
template <typename Vec, typename L, require_std_vector_t<Vec>...>
inline auto rvalue(Vec&& c, const uni_variadic_index<L>& idx,
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
template <typename Vec, typename I, typename L, require_std_vector_t<Vec>...>
inline auto rvalue(Vec&& c, const generic_index<I, L>& idx,
                   const char* name = "ANON", int depth = 0) {
  rvalue_return_t<std::decay_t<Vec>, generic_index<I, L>> result;
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

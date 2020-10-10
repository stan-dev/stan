#ifndef STAN_MODEL_INDEXING_RVALUE_HPP
#define STAN_MODEL_INDEXING_RVALUE_HPP

#include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>
#include <stan/model/indexing/index.hpp>
#include <stan/model/indexing/index_list.hpp>
#include <stan/model/indexing/rvalue_at.hpp>
#include <stan/model/indexing/rvalue_index_size.hpp>
#include <stan/model/indexing/rvalue_return.hpp>
#include <type_traits>
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
inline T rvalue(T&& c, const nil_index_list& /*idx*/, const char* /*name*/ = "",
                int /*depth*/ = 0) {
  return std::forward<T>(c);
}

/**
 * Return the result of indexing a type without taking a subset. Mostly used as
 * an intermediary rvalue function when doing multiple subsets.
 *
 * Types:  type[omni] : type
 *
 * @tparam T Any type.
 * @param[in] an object.
 * @param[in] idx Index consisting of one omni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename T>
inline T rvalue(T&& a, const cons_index_list<index_omni, nil_index_list>& idx,
                const char* name = "ANON", int depth = 0) {
  return std::forward<T>(a);
}

/**
 * Return the result of indexing a type without taking a subset
 *
 * Types:  type[omni, omni] : type
 *
 * @tparam T Any type.
 * @param[in] an object.
 * @param[in] idx Index consisting of two omni-indices.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename T>
inline T rvalue(
    T&& a,
    const cons_index_list<index_omni,
                          cons_index_list<index_omni, nil_index_list>>& idx,
    const char* name = "ANON", int depth = 0) {
  return std::forward<T>(a);
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * sequence consisting of one single index, returning a row vector.
 *
 * Types:  mat[single,] : rowvec
 *
 * @tparam EigMat An eigen matrix
 * @param[in] a Eigen matrix.
 * @param[in] idx Index consisting of one uni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(
    EigMat&& a,
    const cons_index_list<index_uni,
                          cons_index_list<index_omni, nil_index_list>>& idx,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[uni] indexing", name, a.rows(), idx.head_.n_);
  return a.row(idx.head_.n_ - 1).eval();
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * sequence consisting of one single index, returning a vector.
 *
 * Types:  mat[,single] : vec
 *
 * @tparam EigMat An eigen matrix
 * @param[in] a Eigen matrix.
 * @param[in] idx Index consisting of one uni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(
    EigMat&& a,
    const cons_index_list<index_omni,
                          cons_index_list<index_uni, nil_index_list>>& idx,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[uni] indexing", name, a.cols(), idx.tail_.head_.n_);
  return a.col(idx.tail_.head_.n_ - 1).eval();
}

/**
 * Return the result of indexing the specified Eigen vector with a
 * sequence containing one single index, returning a scalar.
 *
 * Types:  vec[single] : scal
 *
 * @tparam EigVec An eigen vector
 * @param[in] v Vector being indexed.
 * @param[in] idx One single index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing vector.
 */
template <typename EigVec, require_eigen_vector_t<EigVec>* = nullptr>
inline auto rvalue(EigVec&& v,
                   const cons_index_list<index_uni, nil_index_list>& idx,
                   const char* name = "ANON", int depth = 0) {
  using stan::math::to_ref;
  math::check_range("vector[single] indexing", name, v.size(), idx.head_.n_);
  return to_ref(v).coeffRef(idx.head_.n_ - 1);
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * sequence consisting of one single index, returning a row vector.
 *
 * Types:  mat[single] : rowvec
 *
 * @tparam EigMat An eigen matrix
 * @param[in] a Eigen matrix.
 * @param[in] idx Index consisting of one uni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(EigMat&& a,
                   const cons_index_list<index_uni, nil_index_list>& idx,
                   const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[uni] indexing", name, a.rows(), idx.head_.n_);
  return a.row(idx.head_.n_ - 1).eval();
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * sequence consisting of two single indexes, returning a scalar.
 *
 * Types:  mat[single,single] : scalar
 *
 * @tparam EigMat An eigen type
 * @param[in] a Matrix to index.
 * @param[in] idx Pair of single indexes.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat, require_eigen_t<EigMat>* = nullptr>
inline auto rvalue(
    EigMat&& a,
    const cons_index_list<index_uni,
                          cons_index_list<index_uni, nil_index_list>>& idx,
    const char* name = "ANON", int depth = 0) {
  using stan::math::to_ref;
  math::check_range("matrix[uni,uni] indexing, row", name, a.rows(),
                    idx.head_.n_);
  math::check_range("matrix[uni,uni] indexing, col", name, a.cols(),
                    idx.tail_.head_.n_);
  return to_ref(a).coeffRef(idx.head_.n_ - 1, idx.tail_.head_.n_ - 1);
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * sequence consisting of a single index and multiple index,
 * returning a row vector.
 *
 * Types:  mat[single, multiple] : row vector
 *
 * @tparam EigMat An eigen matrix
 * @tparam I Type of multiple index.
 * @param[in] a Matrix to index.
 * @param[in] idx Pair of single index and multiple index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat, typename I,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr,
          require_same_t<std::decay_t<I>, I>* = nullptr>
inline auto rvalue(
    EigMat&& a,
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
 * Types:  mat[multiple, single] : vector
 *
 * @tparam EigMat An eigen matrix
 * @tparam I Type of multiple index.
 * @param[in] a Matrix to index.
 * @param[in] idx Pair multiple index and single index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat, typename I,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(
    EigMat&& a,
    const cons_index_list<I, cons_index_list<index_uni, nil_index_list>>& idx,
    const char* name = "ANON", int depth = 0) {
  const int m = idx.tail_.head_.n_;
  math::check_range("matrix[multi,uni] index col", name, a.cols(), m);
  return rvalue(a.col(m - 1), index_list(idx.head_));
}

/**
 * Return the result of indexing the specified Eigen vector min_max index,
 *  returning a vector
 *
 * Types:  vec[min_max] : vector
 *
 * @tparam EigVec An eigen vector
 * @param[in] v Vector being indexed.
 * @param[in] idx One single index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing vector.
 */
template <typename EigVec, require_eigen_vector_t<EigVec>* = nullptr>
inline auto rvalue(EigVec&& v,
                   const cons_index_list<index_min_max, nil_index_list>& idx,
                   const char* name = "ANON", int depth = 0) {
  math::check_range("vector[min_max] min indexing", name, v.size(),
                    idx.head_.min_);
  math::check_range("vector[min_max] max indexing", name, v.size(),
                    idx.head_.max_);
  if (idx.head_.min_ <= idx.head_.max_) {
    return v.segment(idx.head_.min_ - 1, idx.head_.max_ - (idx.head_.min_ - 1))
        .eval();
  } else {
    return v.segment(idx.head_.max_ - 1, idx.head_.min_ - (idx.head_.max_ - 1))
        .reverse()
        .eval();
  }
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * min_max_index returning a block from max to min.
 *
 * Types:  mat[min_max] : matrix
 *
 * @tparam EigMat An eigen matrix
 * @tparam I Type of multiple index.
 * @param[in] a Matrix to index.
 * @param[in] idx Pair multiple index and single index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline plain_type_t<EigMat> rvalue(
    EigMat&& a, const cons_index_list<index_min_max, nil_index_list>& idx,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[multi] indexing", name, a.rows(),
                    idx.head_.min_ - 1);
  math::check_range("matrix[multi] indexing", name, a.rows(),
                    idx.head_.max_ - 1);
  if (idx.head_.min_ <= idx.head_.max_) {
    return a.block(idx.head_.min_ - 1, 0, idx.head_.max_ - (idx.head_.min_ - 1),
                   a.cols());
  } else {
    return a
        .block(idx.head_.max_ - 1, 0, idx.head_.min_ - (idx.head_.max_ - 1),
               a.cols())
        .rowwise()
        .reverse();
  }
}

/**
 * Return the result of indexing the specified Eigen matrix with two min_max
 * indices, returning back a block of the Eigen matrix.
 *
 * Types:  mat[min_max, min_max] : matrix
 *
 * @tparam EigMat An eigen matrix
 * @param[in] a Eigen matrix.
 * @param[in] idx Index consisting of one uni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(
    EigMat&& mat,
    const cons_index_list<index_min_max,
                          cons_index_list<index_min_max, nil_index_list>>& idx,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[min_max, min_max] min row indexing", name,
                    mat.rows(), idx.head_.min_);
  math::check_range("matrix[min_max, min_max] max row indexing", name,
                    mat.rows(), idx.head_.min_);
  math::check_range("matrix[min_max, min_max] min column indexing", name,
                    mat.cols(), idx.tail_.head_.min_);
  math::check_range("matrix[min_max, min_max] max column indexing", name,
                    mat.cols(), idx.tail_.head_.min_);
  if (idx.head_.min_ <= idx.head_.max_) {
    if (idx.tail_.head_.min_ <= idx.tail_.head_.max_) {
      return mat
          .block(idx.head_.min_ - 1, idx.tail_.head_.min_ - 1,
                 idx.head_.max_ - (idx.head_.min_ - 1),
                 idx.tail_.head_.max_ - (idx.tail_.head_.min_ - 1))
          .eval();
    } else {
      return mat
          .block(idx.head_.min_ - 1, idx.tail_.head_.max_ - 1,
                 idx.head_.max_ - (idx.head_.min_ - 1),
                 idx.tail_.head_.min_ - (idx.tail_.head_.max_ - 1))
          .rowwise()
          .reverse()
          .eval();
    }
  } else {
    if (idx.tail_.head_.min_ <= idx.tail_.head_.max_) {
      return mat
          .block(idx.head_.max_ - 1, idx.tail_.head_.min_ - 1,
                 idx.head_.min_ - (idx.head_.max_ - 1),
                 idx.tail_.head_.max_ - (idx.tail_.head_.min_ - 1))
          .colwise()
          .reverse()
          .eval();
    } else {
      return mat
          .block(idx.head_.max_ - 1, idx.tail_.head_.max_ - 1,
                 idx.head_.min_ - (idx.head_.max_ - 1),
                 idx.tail_.head_.min_ - (idx.tail_.head_.max_ - 1))
          .reverse()
          .eval();
    }
  }
}

/**
 * Return the result of indexing the specified Eigen matrix with a min index
 * returning back a block of rows min:N and all cols
 *
 * Types:  mat[min:] : matrix
 *
 * @tparam EigMat An eigen matrix
 * @param[in] a Eigen matrix.
 * @param[in] idx Index consisting of one uni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(EigMat&& a,
                   const cons_index_list<index_min, nil_index_list>& idx,
                   const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[multi] indexing", name, a.rows(),
                    idx.head_.min_ - 1);
  return a
      .block(idx.head_.min_ - 1, 0, a.rows() - (idx.head_.min_ - 1), a.cols())
      .eval();
}

/**
 * Return the result of indexing the specified Eigen matrix with a max index
 * returning back a block of rows 1:max and all columns
 *
 * Types:  mat[:max] : matrix
 *
 * @tparam EigMat An eigen matrix
 * @param[in] a Eigen matrix.
 * @param[in] idx Index consisting of one uni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(EigMat&& a,
                   const cons_index_list<index_max, nil_index_list>& idx,
                   const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[multi] indexing", name, a.rows(),
                    idx.head_.max_ - 1);
  return a.block(0, 0, idx.head_.max_, a.cols()).eval();
}

/**
 * Return the result of indexing the specified Eigen vector with a
 * sequence containing one multiple index, returning a vector.
 *
 * Types: vec[multiple] : vec
 *
 * @tparam EigMat An eigen vector
 * @tparam I Multi-index type.
 * @param[in] v Eigen vector.
 * @param[in] idx Index consisting of one multi-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing vector.
 */
template <typename EigVec, typename I,
          require_eigen_vector_t<EigVec>* = nullptr,
          require_same_t<std::decay_t<I>, I>* = nullptr>
inline plain_type_t<EigVec> rvalue(
    EigVec&& v, const cons_index_list<I, nil_index_list>& idx,
    const char* name = "ANON", int depth = 0) {
  const int size = rvalue_index_size(idx.head_, v.size());
  const auto& v_ref = stan::math::to_ref(v);
  plain_type_t<EigVec> a(size);
  for (int i = 0; i < size; ++i) {
    int n = rvalue_at(i, idx.head_);
    math::check_range("vector[multi] indexing", name, v.size(), n);
    a.coeffRef(i) = v_ref.coeff(n - 1);
  }
  return a;
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * sequence consisting of a one multiple index, returning a matrix.
 *
 * Types:  mat[multiple] : mat
 *
 * @tparam EigMat An eigen matrix
 * @tparam I Type of multiple index.
 * @param[in] a Matrix to index.
 * @param[in] idx Index consisting of single multiple index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat, typename I,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr,
          require_same_t<std::decay_t<I>, I>* = nullptr>
inline auto rvalue(EigMat&& a, const cons_index_list<I, nil_index_list>& idx,
                   const char* name = "ANON", int depth = 0) {
  const int n_rows = rvalue_index_size(idx.head_, a.rows());
  const auto& a_ref = stan::math::to_ref(a);
  plain_type_t<EigMat> b(n_rows, a_ref.cols());
  for (int i = 0; i < n_rows; ++i) {
    const int n = rvalue_at(i, idx.head_);
    math::check_range("matrix[multi] indexing", name, a_ref.rows(), n);
    b.row(i) = a_ref.row(n - 1);
  }
  return b;
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * sequence consisting of a pair o multiple indexes, returning a
 * a matrix.
 *
 * Types:  mat[multiple,multiple] : mat
 *
 * @tparam EigMat An eigen matrix
 * @tparam I Type of multiple index.
 * @param[in] a Matrix to index.
 * @param[in] idx Pair of multiple indexes.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat, typename I1, typename I2,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr,
          require_same_t<std::decay_t<I1>, I1>* = nullptr,
          require_same_t<std::decay_t<I2>, I2>* = nullptr>
inline plain_type_t<EigMat> rvalue(
    EigMat&& a,
    const cons_index_list<I1, cons_index_list<I2, nil_index_list>>& idx,
    const char* name = "ANON", int depth = 0) {
  const auto& a_ref = stan::math::to_ref(a);
  const int rows = rvalue_index_size(idx.head_, a_ref.rows());
  const int cols = rvalue_index_size(idx.tail_.head_, a_ref.cols());
  plain_type_t<EigMat> c(rows, cols);
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      int m = rvalue_at(i, idx.head_);
      int n = rvalue_at(j, idx.tail_.head_);
      math::check_range("matrix[multi,multi] row index", name, a_ref.rows(), m);
      math::check_range("matrix[multi,multi] col index", name, a_ref.cols(), n);
      c.coeffRef(i, j) = a_ref.coeffRef(m - 1, n - 1);
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
template <typename StdVec, typename L, require_std_vector_t<StdVec>* = nullptr,
          require_same_t<std::decay_t<L>, L>* = nullptr>
inline auto rvalue(StdVec&& c, const cons_index_list<index_uni, L>& idx,
                   const char* name = "ANON", int depth = 0) {
  const int n = idx.head_.n_;
  math::check_range("array[uni,...] index", name, c.size(), n);
  return rvalue(c[n - 1], idx.tail_, name, depth + 1);
}

/**
 * Return the result of indexing the specified array with
 * a single index.
 *
 * Types:  std::vector<T>[single] : T
 *
 * @tparam StdVec a standard vector
 * @param[in] c Container of list elements.
 * @param[in] idx Index list beginning with single index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing array.
 */
template <typename StdVec, typename L, require_std_vector_t<StdVec>* = nullptr,
          require_same_t<std::decay_t<L>, L>* = nullptr>
inline auto rvalue(StdVec&& c,
                   const cons_index_list<index_uni, nil_index_list>& idx,
                   const char* name = "ANON", int depth = 0) {
  const int n = idx.head_.n_;
  math::check_range("array[uni,...] index", name, c.size(), n);
  return c[n - 1];
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
template <typename StdVec, typename I, typename L,
          require_std_vector_t<StdVec>* = nullptr,
          require_same_t<std::decay_t<I>, I>* = nullptr,
          require_same_t<std::decay_t<L>, L>* = nullptr>
inline auto rvalue(StdVec&& c, const cons_index_list<I, L>& idx,
                   const char* name = "ANON", int depth = 0) {
  rvalue_return_t<std::decay_t<StdVec>, cons_index_list<I, L>> result;
  const int index_size = rvalue_index_size(idx.head_, c.size());
  if (index_size > 0) {
    result.reserve(index_size);
  }
  for (int i = 0; i < index_size; ++i) {
    int n = rvalue_at(i, idx.head_);
    math::check_range("array[multi,...] index", name, c.size(), n);
    result.push_back(rvalue(c[n - 1], idx.tail_, name, depth + 1));
  }
  return result;
}

}  // namespace model
}  // namespace stan
#endif

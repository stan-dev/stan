#ifndef STAN_MODEL_INDEXING_RVALUE_HPP
#define STAN_MODEL_INDEXING_RVALUE_HPP

#include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>
#include <stan/model/indexing/deep_copy.hpp>
#include <stan/model/indexing/index.hpp>
#include <stan/model/indexing/index_list.hpp>
#include <stan/model/indexing/rvalue_at.hpp>
#include <stan/model/indexing/rvalue_index_size.hpp>
#include <type_traits>
#include <vector>

namespace stan {

namespace model {

/**
 * Indexing Notes:
 * The different index types:
 * index_uni - A single cell.
 * index_multi - Access multiple cells.
 * index_omni - A no-op for all indices along a dimension.
 * index_min - index from min:N
 * index_max - index from 1:max
 * index_min_max - index from min:max
 * nil_index_list - no-op
 * The order of the overloads are
 * vector / row_vector:
 *  - all index overloads
 * matrix:
 *  - all row index overloads
 *    - Return a subset of rows.
 *  - column/row overloads
 *    - overload on both the row and column indices.
 *  - column overloads
 *    - These take a subset of columns and then call the row slice rvalue
 *       over the column subset.
 * Std vector:
 *  - single element and elementwise overloads
 *  - General overload for nested std vectors.
 */


/**
 * Return the result of indexing a specified value with
 * a nil index list, which just returns the value.
 *
 * Types:  T[] : T
 *
 * @tparam T Scalar type.
 * @param[in] x Value to index.
 * @return Input value.
 */
template <typename T>
inline T rvalue(T&& x, const nil_index_list& /*idx*/, const char* /*name*/ = "",
                int /*depth*/ = 0) {
  return std::forward<T>(x);
}

/**
 * Return the result of indexing a type without taking a subset. Mostly used as
 * an intermediary rvalue function when doing multiple subsets.
 *
 * Types:  type[omni] : type
 *
 * @tparam T Any type.
 * @param[in] x an object.
 * @param[in] idxs Index consisting of one omni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename T>
inline T rvalue(T&& x, const cons_index_list<index_omni, nil_index_list>& idxs,
                const char* name = "ANON", int depth = 0) {
  return std::forward<T>(x);
}

/**
 * Return the result of indexing a type without taking a subset
 *
 * Types:  type[omni, omni] : type
 *
 * @tparam T Any type.
 * @param[in] x an object.
 * @param[in] idxs Index consisting of two omni-indices.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename T>
inline T rvalue(T&& x,
    const cons_index_list<index_omni,
                          cons_index_list<index_omni, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  return std::forward<T>(x);
}

/**
 * Return a single element of an Eigen Vector.
 *
 * Types:  vec[uni] : scal
 *
 * @tparam EigVec An eigen vector
 * @param[in] v Vector being indexed.
 * @param[in] idxs One single index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing vector.
 */
template <typename EigVec, require_eigen_vector_t<EigVec>* = nullptr>
inline auto rvalue(EigVec&& v,
                   const cons_index_list<index_uni, nil_index_list>& idxs,
                   const char* name = "ANON", int depth = 0) {
  using stan::math::to_ref;
  math::check_range("vector[uni] indexing", name, v.size(), idxs.head_.n_);
  return v.coeff(idxs.head_.n_ - 1);
}

/**
 * Return a non-contiguous subset of elements in a vector.
 *
 * Types:  vec[multi] : vec
 *
 * @tparam Vec Eigen type with either dynamic rows or columns, but not both.
 * @param[in] x Eigen vector type.
 * @param[in] idxs Sequence of integers.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the value size isn't the same as
 * the indexed size.
 */
template <typename Vec, require_eigen_vector_t<Vec>* = nullptr>
inline plain_type_t<Vec> rvalue(Vec&& v, const cons_index_list<index_multi, nil_index_list>& idxs,
    const char* name = "ANON", int depth = 0) {
  const auto v_size = v.size();
  const auto& v_ref = stan::math::to_ref(v);
  plain_type_t<Vec> ret_v(idxs.head_.ns_.size());
  for (int i = 0; i < idxs.head_.ns_.size(); ++i) {
    math::check_range("vector[multi] indexing", name, v_ref.size(),
                      idxs.head_.ns_[i]);
    ret_v.coeffRef(i) = v_ref.coeff(idxs.head_.ns_[i] - 1);
  }
  return ret_v;
}

/**
 * Return a range of an Eigen vector
 *
 * Types:  vec[min_max] : vector
 *
 * @tparam EigVec An eigen vector
 * @param[in] v Vector being indexed.
 * @param[in] idxs One single index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing vector.
 */
template <typename EigVec, require_eigen_vector_t<EigVec>* = nullptr>
inline auto rvalue(EigVec&& v,
                   const cons_index_list<index_min_max, nil_index_list>& idxs,
                   const char* name = "ANON", int depth = 0) {
  math::check_range("vector[min_max] min indexing", name, v.size(),
                    idxs.head_.min_);
  math::check_range("vector[min_max] max indexing", name, v.size(),
                    idxs.head_.max_);
  if (idxs.head_.is_ascending()) {
    return v.segment(idxs.head_.min_ - 1, idxs.head_.max_ - (idxs.head_.min_ - 1))
        .eval();
  } else {
    return v.segment(idxs.head_.max_ - 1, idxs.head_.min_ - (idxs.head_.max_ - 1))
        .reverse()
        .eval();
  }
}

/**
 * Return a tail slice of a vector
 *
 * Types:  vector[min:N] : vector
 *
 * @tparam Vec Eigen type with either dynamic rows or columns, but not both.
 * @param[in] x vector
 * @param[in] idxs An index.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename Vec, require_eigen_vector_t<Vec>* = nullptr>
inline auto rvalue(Vec&& x,
                   const cons_index_list<index_min, nil_index_list>& idxs,
                   const char* name = "ANON", int depth = 0) {
  stan::math::check_range("vector[min] indexing range", name, x.size(),
                          idxs.head_.min_);
  return x.tail(x.size() - idxs.head_.min_ + 1).eval();
}

/**
 * Return a head slice of a vector
 *
 * Types:  vector[1:max] <- vector
 *
 * @tparam Vec Eigen type with either dynamic rows or columns, but not both.
 * @param[in] x Eigen vector type.
 * @param[in] idxs An index.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename Vec,
          require_all_eigen_vector_t<Vec>* = nullptr>
inline auto rvalue(Vec&& x,
                   const cons_index_list<index_max, nil_index_list>& idxs,
                   const char* name = "ANON", int depth = 0) {
  stan::math::check_range("vector[min] indexing range", name, x.size(),
                          idxs.head_.max_);
  return x.head(idxs.head_.max_).eval();
}

/**
 * Return the result of indexing the Eigen matrix with a
 * sequence consisting of one single index, returning a row vector.
 *
 * Types:  mat[uni] : rowvec
 *
 * @tparam EigMat An eigen matrix
 * @param[in] x Eigen matrix.
 * @param[in] idxs Index consisting of one uni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(EigMat&& x,
                   const cons_index_list<index_uni, nil_index_list>& idxs,
                   const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[uni] indexing", name, x.rows(), idxs.head_.n_);
  return x.row(idxs.head_.n_ - 1).eval();
}

/**
 * Return the specified Eigen matrix at the specified multi index.
 *
 * Types:  mat[multi] = mat
 *
 * @tparam EigMat Eigen type with dynamic rows and columns.
 * @param[in] x Eigen type
 * @param[in] idxs An indexing from the start of the container up to
 * the specified maximum index (inclusive).
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename EigMat,
          stan::internal::require_all_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline plain_type_t<EigMat> rvalue(
    EigMat&& x, const cons_index_list<index_multi, nil_index_list>& idxs,
    const char* name = "ANON", int depth = 0) {
  const auto& x_ref = stan::math::to_ref(x);
  plain_type_t<EigMat> x_ret(idxs.head_.ns_.size(), x.cols());
  for (int i = 0; i < idxs.head_.ns_.size(); ++i) {
    const int n = idxs.head_.ns_[i];
    math::check_range("matrix[multi] subset range", name, x_ref.rows(), n);
    x_ret.row(i) = x_ref.row(n - 1);
  }
  return x_ret;
}

/**
 * Return the result of indexing the Eigen matrix with a min index
 * returning back a block of rows min:N and all cols
 *
 * Types:  mat[min:N] : matrix
 *
 * @tparam EigMat An eigen matrix
 * @param[in] x Eigen matrix.
 * @param[in] idxs Index consisting of one uni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(EigMat&& x,
                   const cons_index_list<index_min, nil_index_list>& idxs,
                   const char* name = "ANON", int depth = 0) {
 const auto row_size = x.rows() - (idxs.head_.min_ - 1);
  math::check_range("matrix[min] indexing", name, x.rows(), idxs.head_.min_);
  return x.bottomRows(row_size).eval();
}

/**
 * Return the Eigen matrix at the specified max index
 *
 * Types:  mat[:max] = mat
 *
 * @tparam EigMat Eigen type with dynamic rows and columns.
 * @param[in] x Eigen type
 * @param[in] idxs An indexing from the start of the container up to
 * the specified maximum index (inclusive).
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename EigMat,
          stan::internal::require_all_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(EigMat&& x,
                   const cons_index_list<index_max, nil_index_list>& idxs,
                   const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[max] indexing range", name, x.cols(),
                    idxs.head_.max_);
  return x.topRows(idxs.head_.max_).eval();
}

/**
 * Return the Eigen matrix at the specified min_max index.
 *
 * Types:  mat[min_max] = mat
 *
 * @tparam EigMat Eigen type with dynamic rows and columns.
 * @param[in] x Eigen type
 * @param[in] idxs An indexing from the start of the container up to
 * the specified maximum index (inclusive).
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename EigMat,
          stan::internal::require_all_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(EigMat&& x,
                   const cons_index_list<index_min_max, nil_index_list>& idxs,
                   const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[min_max] max row indexing", name, x.rows(),
                    idxs.head_.max_);
  math::check_range("matrix[min_max] min row indexing", name, x.rows(),
                    idxs.head_.min_);
  if (idxs.head_.is_ascending()) {
    return x.middleRows(idxs.head_.min_ - 1, idxs.head_.max_ - 1).eval();
  } else {
    return x.middleRows(idxs.head_.max_ - 1, idxs.head_.min_ - 1).colwise().reverse().eval();
  }
}

/**
 * Return the result of indexing an Eigen matrix with two min_max
 * indices, returning back a block of the Eigen matrix.
 *
 * Types:  mat[min_max, min_max] : matrix
 *
 * @tparam EigMat An eigen matrix
 * @param[in] x Eigen matrix.
 * @param[in] idxs Index consisting of one uni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(
    EigMat&& x,
    const cons_index_list<index_min_max,
                          cons_index_list<index_min_max, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[min_max, min_max] min row indexing", name,
                    x.rows(), idxs.head_.min_);
  math::check_range("matrix[min_max, min_max] max row indexing", name,
                    x.rows(), idxs.head_.max_);
  math::check_range("matrix[min_max, min_max] min column indexing", name,
                    x.cols(), idxs.tail_.head_.min_);
  math::check_range("matrix[min_max, min_max] max column indexing", name,
                    x.cols(), idxs.tail_.head_.max_);
  if (idxs.head_.is_ascending()) {
    if (idxs.tail_.head_.is_ascending()) {
      return x.block(idxs.head_.min_ - 1, idxs.tail_.head_.min_ - 1,
                 idxs.head_.max_ - (idxs.head_.min_ - 1),
                 idxs.tail_.head_.max_ - (idxs.tail_.head_.min_ - 1))
          .eval();
    } else {
      return x.block(idxs.head_.min_ - 1, idxs.tail_.head_.max_ - 1,
                 idxs.head_.max_ - (idxs.head_.min_ - 1),
                 idxs.tail_.head_.min_ - (idxs.tail_.head_.max_ - 1))
          .rowwise()
          .reverse()
          .eval();
    }
  } else {
    if (idxs.tail_.head_.is_ascending()) {
      return x.block(idxs.head_.max_ - 1, idxs.tail_.head_.min_ - 1,
                 idxs.head_.min_ - (idxs.head_.max_ - 1),
                 idxs.tail_.head_.max_ - (idxs.tail_.head_.min_ - 1))
          .colwise()
          .reverse()
          .eval();
    } else {
      return x.block(idxs.head_.max_ - 1, idxs.tail_.head_.max_ - 1,
                 idxs.head_.min_ - (idxs.head_.max_ - 1),
                 idxs.tail_.head_.min_ - (idxs.tail_.head_.max_ - 1))
          .reverse()
          .eval();
    }
  }
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * sequence consisting of two single indexes, returning a scalar.
 *
 * Types:  mat[single,single] : scalar
 *
 * @tparam EigMat An eigen type
 * @param[in] x Matrix to index.
 * @param[in] idxs Pair of single indexes.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat, require_eigen_t<EigMat>* = nullptr>
inline auto rvalue(
    EigMat&& x,
    const cons_index_list<index_uni,
                          cons_index_list<index_uni, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[uni,uni] indexing, row", name, x.rows(),
                    idxs.head_.n_);
  math::check_range("matrix[uni,uni] indexing, col", name, x.cols(),
                    idxs.tail_.head_.n_);
  return x.coeff(idxs.head_.n_ - 1, idxs.tail_.head_.n_ - 1);
}

/**
 * Random access to a vector's cells to a row of an eigen matrix.
 *
 * Types:  mat[uni, multi] = vector
 *
 * @tparam EigMat Eigen type with dynamic rows and columns.
 * @param[in] x Matrix to index.
 * @param[in] idxs Pair of multiple indexes (from 1).
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename EigMat,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline Eigen::Matrix<value_type_t<EigMat>, 1, -1> rvalue(
    EigMat&& x,
    const cons_index_list<index_uni,
                          cons_index_list<index_multi, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[uni, multi] index range", name, x.cols(),
                    idxs.head_.n_);
  const auto& x_ref = stan::math::to_ref(x);
  Eigen::Matrix<value_type_t<EigMat>, 1, -1> x_ret(1,
                                                   idxs.tail_.head_.ns_.size());
  for (int i = 0; i < idxs.tail_.head_.ns_.size(); ++i) {
    math::check_range("matrix[uni, multi] index range", name, x.cols(),
                      idxs.tail_.head_.ns_[i]);
    x_ret.coeffRef(i)
        = x_ref.coeff(idxs.head_.n_ - 1, idxs.tail_.head_.ns_[i] - 1);
  }
  return x_ret;
}

/**
 * Random access of a matrix column with a multi index on the row.
 * Types:  mat[multi, uni] = vector
 *
 * @tparam EigMat Eigen type with dynamic rows and columns.
 * @param[in] x Matrix to index.
 * @param[in] idxs Pair of multiple indexes (from 1).
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename EigMat,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline Eigen::Matrix<value_type_t<EigMat>, -1, 1> rvalue(
    EigMat&& x,
    const cons_index_list<index_multi,
                          cons_index_list<index_uni, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[multi, uni] rvalue range", name, x.cols(),
                    idxs.tail_.head_.n_);
  const auto& x_ref = stan::math::to_ref(x);
  Eigen::Matrix<value_type_t<EigMat>, -1, 1> x_ret(idxs.head_.ns_.size(), 1);
  for (int i = 0; i < idxs.head_.ns_.size(); ++i) {
    math::check_range("matrix[multi, uni] rvalue range", name, x_ref.rows(),
                      idxs.head_.ns_[i]);
    x_ret.coeffRef(i)
        = x_ref.coeffRef(idxs.head_.ns_[i] - 1, idxs.tail_.head_.n_ - 1);
  }
  return x_ret;
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * sequence consisting of a pair of multiple indexes, returning a
 * a matrix.
 *
 * Types:  mat[multi, multi] : mat
 *
 * @tparam EigMat An eigen matrix
 * @param[in] x Matrix to index.
 * @param[in] idxs Pair of multiple indexes.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline plain_type_t<EigMat> rvalue(
    EigMat&& x,
    const cons_index_list<index_multi,
                          cons_index_list<index_multi, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  const auto& x_ref = stan::math::to_ref(x);
  const int rows = idxs.head_.ns_.size();
  const int cols = idxs.tail_.head_.ns_.size();
  plain_type_t<EigMat> x_ret(rows, cols);
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      const int m = idxs.head_.ns_[i];
      const int n = idxs.tail_.head_.ns_[j];
      math::check_range("matrix[multi,multi] row index", name, x_ref.rows(), m);
      math::check_range("matrix[multi,multi] col index", name, x_ref.cols(), n);
      x_ret.coeffRef(i, j) = x_ref.coeff(m - 1, n - 1);
    }
  }
  return x_ret;
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * sequence consisting of one single index.
 *
 * Types:  mat[Idx, uni] : vec
 *
 * @tparam EigMat Eigen type with dynamic rows and columns.
 * @param[in] x Eigen matrix.
 * @param[in] idxs Index consisting of one uni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 */
template <typename EigMat, typename Idx,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(
    EigMat&& x,
    const cons_index_list<Idx, cons_index_list<index_uni, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[..., uni] indexing", name, x.cols(),
                    idxs.tail_.head_.n_);
  return deep_copy(rvalue(x.col(idxs.tail_.head_.n_ - 1), index_list(idxs.head_),
                          name, depth + 1));
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * sequence consisting of multi index.
 *
 * Types:  mat[Idx, multi] : vec
 *
 * @tparam EigMat An eigen matrix
 * @param[in] x Eigen matrix.
 * @param[in] idxs Index consisting of one uni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat, typename Idx,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr,
          require_not_same_t<std::decay_t<Idx>, index_uni>* = nullptr>
inline plain_type_t<EigMat> rvalue(
    EigMat&& x,
    const cons_index_list<Idx, cons_index_list<index_multi, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  const auto& x_ref = stan::math::to_ref(x);
  const int rows = rvalue_index_size(idxs.head_, x_ref.rows());
  const int cols = rvalue_index_size(idxs.tail_.head_, x_ref.cols());
  plain_type_t<EigMat> x_ret(rows, idxs.tail_.head_.ns_.size());
  for (int j = 0; j < idxs.tail_.head_.ns_.size(); ++j) {
    const int n = idxs.tail_.head_.ns_[j];
    math::check_range("matrix[..., multi] col index", name, x_ref.cols(), n);
    x_ret.col(j)
        = rvalue(x_ref.col(n - 1), index_list(idxs.head_), name, depth + 1);
  }
  return x_ret;
}

/**
 * Return the Eigen matrix with all columns and a slice of rows.
 *
 * Types:  mat[Idx, omni] = mat
 *
 * @tparam EigMat Eigen type with dynamic rows and columns.
 * @param[in] x Eigen type
 * @param[in] idxs Pair of multiple indexes (from 1).
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename EigMat, typename Idx,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(
    EigMat&& x,
    const cons_index_list<Idx, cons_index_list<index_omni, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  return deep_copy(rvalue(std::forward<EigMat>(x), index_list(idxs.head_), name, depth + 1));
}

/**
 * Return the Eigen matrix at the specified min
 * index to the specified matrix value.
 *
 * Types:  mat[Idx, min] = mat
 *
 * @tparam EigMat Eigen type with dynamic rows and columns.
 * @tparam Idx An index.
 * @param[in] x Eigen type
 * @param[in] idxs Container holding a row index and an index from a minimum
 * index (inclusive) to the end of a container.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename EigMat, typename Idx,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(
    EigMat&& x,
    const cons_index_list<Idx, cons_index_list<index_min, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  const auto col_size = x.cols() - (idxs.tail_.head_.min_ - 1);
  math::check_range("matrix[..., min] indexing", name, x.cols(),
                    idxs.tail_.head_.min_);
  return deep_copy(rvalue(x.rightCols(col_size),
                          index_list(idxs.head_), name, depth + 1));
}

/**
 * Return the Eigen matrix at the specified max index
 *
 * Types:  mat[Idx, max] = mat
 *
 * @tparam EigMat Eigen type with dynamic rows and columns.
 * @tparam Idx An index.
 * @param[in] x Eigen type
 * @param[in] idxs Index holding a row index and an index from the start of the
 * container up to the specified maximum index (inclusive).
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename EigMat, typename Idx,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(
    EigMat&& x,
    const cons_index_list<Idx, cons_index_list<index_max, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[..., max] indexing", name, x.cols(),
                    idxs.tail_.head_.max_);
  return deep_copy(rvalue(x.leftCols(idxs.tail_.head_.max_),
                          index_list(idxs.head_), name, depth + 1));
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * min_max_index returning a block from min to max.
 *
 * Types:  mat[Idx, min_max] : matrix
 *
 * @tparam EigMat An eigen matrix
 * @tparam Idx Type of index.
 * @param[in] x Matrix to index.
 * @param[in] idxs Pair multiple index and single index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat, typename Idx,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(
    EigMat&& x,
    const cons_index_list<Idx, cons_index_list<index_min_max, nil_index_list>>&
        idxs,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[..., min_max] indexing", name, x.rows(),
                    idxs.tail_.head_.min_);
  math::check_range("matrix[..., min_max] indexing", name, x.rows(),
                    idxs.tail_.head_.max_);
  if (idxs.tail_.head_.is_ascending()) {
    const auto col_start = idxs.tail_.head_.min_ - 1;
    return deep_copy(
        rvalue(x.middleCols(col_start, idxs.tail_.head_.max_ - col_start),
               index_list(idxs.head_), name, depth + 1));
  } else {
    const auto col_start = idxs.tail_.head_.max_ - 1;
    return deep_copy(
        rvalue(x.middleCols(col_start, idxs.tail_.head_.min_ - col_start).rowwise().reverse(),
               index_list(idxs.head_), name, depth + 1));
  }
}


/**
 * Return the result of indexing the specified array with
 * a list of indexes beginning with a single index;  the result is
 * determined recursively.  Note that arrays are represented as
 * standard library vectors.
 *
 * Types:  std::vector<T>[single | Idx] : T[Idx]
 *
 * @tparam T Type of list elements.
 * @tparam Idx Index list type for indexes after first index.
 * @param[in] v Container of list elements.
 * @param[in] idxs Index list beginning with single index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing array.
 */
template <typename StdVec, typename Idx, require_std_vector_t<StdVec>* = nullptr>
inline auto rvalue(StdVec&& v, const cons_index_list<index_uni, Idx>& idxs,
                   const char* name = "ANON", int depth = 0) {
  math::check_range("array[uni, ...] index", name, v.size(), idxs.head_.n_);
  return rvalue(v[idxs.head_.n_ - 1], idxs.tail_, name, depth + 1);
}

/**
 * Return the result of indexing the specified array with
 * a single index.
 *
 * Types:  std::vector<T>[single] : T
 *
 * @tparam StdVec a standard vector
 * @param[in] c Container of list elements.
 * @param[in] idxs Index list beginning with single index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing array.
 */
template <typename StdVec, require_std_vector_t<StdVec>* = nullptr>
inline auto rvalue(StdVec&& v,
                   const cons_index_list<index_uni, nil_index_list>& idxs,
                   const char* name = "ANON", int depth = 0) {
  math::check_range("array[uni, ...] index", name, v.size(), idxs.head_.n_);
  return v[idxs.head_.n_ - 1];
}

/**
 * Return the result of indexing the specified array with
 * a list of indexes beginning with a multiple index;  the result is
 * determined recursively.  Note that arrays are represented as
 * standard library vectors.
 *
 * Types:  std::vector<T>[Idx1, Idx2] : std::vector<T>[Idx2]
 *
 * @tparam T Type of list elements.
 * @tparam Idx1 Index list type for first index.
 * @tparam Idx2 Index list type for second index index.
 * @param[in] v Container of list elements.
 * @param[in] idxs Index list beginning with multiple index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing array.
 */
template <typename StdVec, typename Idx1, typename Idx2,
          require_std_vector_t<StdVec>* = nullptr>
inline auto rvalue(StdVec&& v, const cons_index_list<Idx1, Idx2>& idxs,
                   const char* name = "ANON", int depth = 0) {
  using inner_type = plain_type_t<decltype(
      rvalue(v[rvalue_at(0, idxs.head_) - 1], idxs.tail_))>;
  std::vector<inner_type> result;
  const int index_size = rvalue_index_size(idxs.head_, v.size());
  if (index_size > 0) {
    result.reserve(index_size);
  }
  for (int i = 0; i < index_size; ++i) {
    const int n = rvalue_at(i, idxs.head_);
    math::check_range("array[..., ...] index", name, v.size(), n);
    result.emplace_back(rvalue(v[n - 1], idxs.tail_, name, depth + 1));
  }
  return result;
}

}  // namespace model
}  // namespace stan
#endif

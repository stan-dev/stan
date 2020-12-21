#ifndef STAN_MODEL_INDEXING_RVALUE_HPP
#define STAN_MODEL_INDEXING_RVALUE_HPP

#include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>
#include <stan/model/indexing/index.hpp>
#include <stan/model/indexing/index_list.hpp>
#include <stan/model/indexing/rvalue_at.hpp>
#include <stan/model/indexing/rvalue_index_size.hpp>
#include <stan/model/indexing/access_helpers.hpp>
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
 * Types:  plain_type[omni] : plain_type
 *
 * @tparam T A type that is a plain object.
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
inline T rvalue(
    T&& x,
    const cons_index_list<index_omni,
                          cons_index_list<index_omni, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  return std::forward<T>(x);
}

/**
 * Return a single element of a Vector.
 *
 * Types:  vector[uni] : scaler
 *
 * @tparam Vec An Eigen vector or `var_value<T>` where `T` is an eigen vector.
 * @param[in] v Vector being indexed.
 * @param[in] idxs One single index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing vector.
 */
template <typename Vec, require_vector_t<Vec>* = nullptr,
          require_not_std_vector_t<Vec>* = nullptr>
inline auto rvalue(Vec&& v,
                   const cons_index_list<index_uni, nil_index_list>& idxs,
                   const char* name = "ANON", int depth = 0) {
  using stan::math::to_ref;
  math::check_range("vector[uni] indexing", name, v.size(), idxs.head_.n_);
  return v.coeff(idxs.head_.n_ - 1);
}

/**
 * Return a non-contiguous subset of elements in a vector.
 *
 * Types:  vector[multi] = vector
 *
 * @tparam EigVec Eigen type with either dynamic rows or columns, but not both.
 * @param[in] v Eigen vector type.
 * @param[in] idxs Sequence of integers.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the value size isn't the same as
 * the indexed size.
 */
template <typename EigVec, require_eigen_vector_t<EigVec>* = nullptr>
inline plain_type_t<EigVec> rvalue(
    EigVec&& v, const cons_index_list<index_multi, nil_index_list>& idxs,
    const char* name = "ANON", int depth = 0) {
  const auto v_size = v.size();
  const auto& v_ref = stan::math::to_ref(v);
  plain_type_t<EigVec> ret_v(idxs.head_.ns_.size());
  for (int i = 0; i < idxs.head_.ns_.size(); ++i) {
    math::check_range("vector[multi] indexing", name, v_ref.size(),
                      idxs.head_.ns_[i]);
    ret_v.coeffRef(i) = v_ref.coeff(idxs.head_.ns_[i] - 1);
  }
  return ret_v;
}

/**
 * Return a range of a vector
 *
 * Types:  vector[min_max] = vector
 *
 * @tparam Vec An Eigen vector or `var_value<T>` where `T` is an eigen vector.
 * @param[in] v Vector being indexed.
 * @param[in] idxs An index to select from a minimum to maximum range.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing vector.
 */
template <typename Vec, require_vector_t<Vec>* = nullptr,
          require_not_std_vector_t<Vec>* = nullptr>
inline auto rvalue(Vec&& v,
                   const cons_index_list<index_min_max, nil_index_list>& idxs,
                   const char* name = "ANON", int depth = 0) {
  math::check_range("vector[min_max] min indexing", name, v.size(),
                    idxs.head_.min_);
  math::check_range("vector[min_max] max indexing", name, v.size(),
                    idxs.head_.max_);
  if (idxs.head_.is_ascending()) {
    const auto slice_start = idxs.head_.min_ - 1;
    const auto slice_size = idxs.head_.max_ - slice_start;
    return v.segment(slice_start, slice_size).eval();
  } else {
    const auto slice_start = idxs.head_.max_ - 1;
    const auto slice_size = idxs.head_.min_ - slice_start;
    return v.segment(slice_start, slice_size).reverse().eval();
  }
}

/**
 * Return a tail slice of a vector
 *
 * Types:  vector[min:N] = vector
 *
 * @tparam Vec An Eigen vector or `var_value<T>` where `T` is an eigen vector.
 * @param[in] x vector
 * @param[in] idxs An indexing from a specific minimum index to the end out
 *  of a bottom row of a matrix
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename Vec, require_vector_t<Vec>* = nullptr,
          require_not_std_vector_t<Vec>* = nullptr>
inline auto rvalue(Vec&& x,
                   const cons_index_list<index_min, nil_index_list>& idxs,
                   const char* name = "ANON", int depth = 0) {
  stan::math::check_range("vector[min] indexing", name, x.size(),
                          idxs.head_.min_);
  return x.tail(x.size() - idxs.head_.min_ + 1);
}

/**
 * Return a head slice of a vector
 *
 * Types:  vector[1:max] <- vector
 *
 * @tparam Vec An Eigen vector or `var_value<T>` where `T` is an eigen vector.
 * @param[in] x vector.
 * @param[in] idxs An indexing from the start of the container up to
 * the specified maximum index (inclusive).
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename Vec, require_vector_t<Vec>* = nullptr,
          require_not_std_vector_t<Vec>* = nullptr>
inline auto rvalue(Vec&& x,
                   const cons_index_list<index_max, nil_index_list>& idxs,
                   const char* name = "ANON", int depth = 0) {
  stan::math::check_range("vector[max] indexing", name, x.size(),
                          idxs.head_.max_);
  return x.head(idxs.head_.max_);
}

/**
 * Return the result of indexing the matrix with a
 * sequence consisting of one single index, returning a row vector.
 *
 * Types:  matrix[uni] : row vector
 *
 * @tparam Mat An eigen matrix or `var_value<T>` whose inner type is an Eigen
 * matrix.
 * @param[in] x matrix.
 * @param[in] idxs Index consisting of one uni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename Mat, require_dense_dynamic_t<Mat>* = nullptr>
inline auto rvalue(Mat&& x,
                   const cons_index_list<index_uni, nil_index_list>& idxs,
                   const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[uni] indexing", name, x.rows(), idxs.head_.n_);
  return x.row(idxs.head_.n_ - 1);
}

/**
 * Return the specified Eigen matrix at the specified multi index.
 *
 * Types:  matrix[multi] = matrix
 *
 * @tparam EigMat Eigen type with dynamic rows and columns.
 * @param[in] x Eigen type
 * @param[in] idxs A multi index for selecting a set of rows.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename EigMat, require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline plain_type_t<EigMat> rvalue(
    EigMat&& x, const cons_index_list<index_multi, nil_index_list>& idxs,
    const char* name = "ANON", int depth = 0) {
  const auto& x_ref = stan::math::to_ref(x);
  plain_type_t<EigMat> x_ret(idxs.head_.ns_.size(), x.cols());
  for (int i = 0; i < idxs.head_.ns_.size(); ++i) {
    const int n = idxs.head_.ns_[i];
    math::check_range("matrix[multi] row indexing", name, x_ref.rows(), n);
    x_ret.row(i) = x_ref.row(n - 1);
  }
  return x_ret;
}

/**
 * Return the result of indexing the matrix with a min index
 * returning back a block of rows min:N and all cols
 *
 * Types:  matrix[min:N] = matrix
 *
 * @tparam Mat An eigen matrix or `var_value<T>` whose inner type is an Eigen
 * matrix.
 * @param[in] x matrix.
 * @param[in] idxs Index consisting of one uni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename Mat, require_dense_dynamic_t<Mat>* = nullptr>
inline auto rvalue(Mat&& x,
                   const cons_index_list<index_min, nil_index_list>& idxs,
                   const char* name = "ANON", int depth = 0) {
  const auto row_size = x.rows() - (idxs.head_.min_ - 1);
  math::check_range("matrix[min] row indexing", name, x.rows(),
                    idxs.head_.min_);
  return x.bottomRows(row_size);
}

/**
 * Return the 1:max rows of a matrix.
 *
 * Types:  matrix[:max] = matrix
 *
 * @tparam Mat An eigen matrix or `var_value<T>` whose inner type is an Eigen
 * matrix.
 * @param[in] x matrix
 * @param[in] idxs An indexing from the start of the container up to
 * the specified maximum index (inclusive).
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename Mat, require_dense_dynamic_t<Mat>* = nullptr>
inline auto rvalue(Mat&& x,
                   const cons_index_list<index_max, nil_index_list>& idxs,
                   const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[max] row indexing", name, x.rows(),
                    idxs.head_.max_);
  return x.topRows(idxs.head_.max_);
}

/**
 * Return a range of rows for a matrix.
 *
 * Types:  matrix[min_max] = matrix
 *
 * @tparam Mat An eigen matrix or `var_value<T>` whose inner type is an Eigen
 * matrix.
 * @param[in] x Dense matrix
 * @param[in] idxs A min max index to select a range of rows.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename Mat, require_dense_dynamic_t<Mat>* = nullptr>
inline auto rvalue(Mat&& x,
                   const cons_index_list<index_min_max, nil_index_list>& idxs,
                   const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[min_max] max row indexing", name, x.rows(),
                    idxs.head_.max_);
  math::check_range("matrix[min_max] min row indexing", name, x.rows(),
                    idxs.head_.min_);
  if (idxs.head_.is_ascending()) {
    const auto row_size = idxs.head_.max_ - idxs.head_.min_ + 1;
    return x.middleRows(idxs.head_.min_ - 1, row_size).eval();
  } else {
    const auto row_size = idxs.head_.min_ - idxs.head_.max_ + 1;
    return internal::colwise_reverse(
               x.middleRows(idxs.head_.max_ - 1, row_size))
        .eval();
  }
}

/**
 * Return the result of indexing a matrix with two min_max
 * indices, returning back a block of a matrix.
 *
 * Types:  matrix[min_max, min_max] = matrix
 *
 * @tparam Mat An eigen matrix or `var_value<T>` whose inner type is an Eigen
 * matrix.
 * @param[in] x Eigen matrix.
 * @param[in] idxs Two min max indices for selecting a block of the matrix.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename Mat, require_dense_dynamic_t<Mat>* = nullptr>
inline auto rvalue(
    Mat&& x,
    const cons_index_list<index_min_max,
                          cons_index_list<index_min_max, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[min_max, min_max] min row indexing", name, x.rows(),
                    idxs.head_.min_);
  math::check_range("matrix[min_max, min_max] max row indexing", name, x.rows(),
                    idxs.head_.max_);
  math::check_range("matrix[min_max, min_max] min column indexing", name,
                    x.cols(), idxs.tail_.head_.min_);
  math::check_range("matrix[min_max, min_max] max column indexing", name,
                    x.cols(), idxs.tail_.head_.max_);
  if (idxs.head_.is_ascending()) {
    if (idxs.tail_.head_.is_ascending()) {
      return x
          .block(idxs.head_.min_ - 1, idxs.tail_.head_.min_ - 1,
                 idxs.head_.max_ - (idxs.head_.min_ - 1),
                 idxs.tail_.head_.max_ - (idxs.tail_.head_.min_ - 1))
          .eval();
    } else {
      return internal::rowwise_reverse(
                 x.block(idxs.head_.min_ - 1, idxs.tail_.head_.max_ - 1,
                         idxs.head_.max_ - (idxs.head_.min_ - 1),
                         idxs.tail_.head_.min_ - (idxs.tail_.head_.max_ - 1)))
          .eval();
    }
  } else {
    if (idxs.tail_.head_.is_ascending()) {
      return internal::colwise_reverse(
                 x.block(idxs.head_.max_ - 1, idxs.tail_.head_.min_ - 1,
                         idxs.head_.min_ - (idxs.head_.max_ - 1),
                         idxs.tail_.head_.max_ - (idxs.tail_.head_.min_ - 1)))
          .eval();
    } else {
      return x
          .block(idxs.head_.max_ - 1, idxs.tail_.head_.max_ - 1,
                 idxs.head_.min_ - (idxs.head_.max_ - 1),
                 idxs.tail_.head_.min_ - (idxs.tail_.head_.max_ - 1))
          .reverse()
          .eval();
    }
  }
}

/**
 * Return a scalar from a matrix
 *
 * Types:  matrix[uni,uni] : scalar
 *
 * @tparam Mat An eigen matrix or `var_value<T>` whose inner type is an Eigen
 * matrix.
 * @param[in] x Matrix to index.
 * @param[in] idxs Pair of single indexes.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename Mat, require_dense_dynamic_t<Mat>* = nullptr>
inline auto rvalue(
    Mat&& x,
    const cons_index_list<index_uni,
                          cons_index_list<index_uni, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[uni,uni] row indexing", name, x.rows(),
                    idxs.head_.n_);
  math::check_range("matrix[uni,uni] column indexing", name, x.cols(),
                    idxs.tail_.head_.n_);
  return x.coeff(idxs.head_.n_ - 1, idxs.tail_.head_.n_ - 1);
}

/**
 * Return a row of an Eigen matrix with possibly unordered cells.
 *
 * Types:  matrix[uni, multi] = row vector
 *
 * @tparam EigMat Eigen type with dynamic rows and columns.
 * @param[in] x Matrix to index.
 * @param[in] idxs A uni index for the row and multi index for the columns
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename EigMat, require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline Eigen::Matrix<value_type_t<EigMat>, 1, Eigen::Dynamic> rvalue(
    EigMat&& x,
    const cons_index_list<index_uni,
                          cons_index_list<index_multi, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[uni, multi] row indexing", name, x.rows(),
                    idxs.head_.n_);
  const auto& x_ref = stan::math::to_ref(x);
  Eigen::Matrix<value_type_t<EigMat>, 1, Eigen::Dynamic> x_ret(
      1, idxs.tail_.head_.ns_.size());
  for (int i = 0; i < idxs.tail_.head_.ns_.size(); ++i) {
    math::check_range("matrix[uni, multi] column indexing", name, x.cols(),
                      idxs.tail_.head_.ns_[i]);
    x_ret.coeffRef(i)
        = x_ref.coeff(idxs.head_.n_ - 1, idxs.tail_.head_.ns_[i] - 1);
  }
  return x_ret;
}

/**
 * Return a column of an Eigen matrix that is a possibly non-contiguous subset
 *  of the input Eigen matrix.
 *
 * Types:  matrix[multi, uni] = vector
 *
 * @tparam EigMat Eigen type with dynamic rows and columns.
 * @param[in] x Matrix to index.
 * @param[in] idxs A multi index for the rows and a uni index for the column
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename EigMat, require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline Eigen::Matrix<value_type_t<EigMat>, Eigen::Dynamic, 1> rvalue(
    EigMat&& x,
    const cons_index_list<index_multi,
                          cons_index_list<index_uni, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[multi, uni] column indexing", name, x.cols(),
                    idxs.tail_.head_.n_);
  const auto& x_ref = stan::math::to_ref(x);
  Eigen::Matrix<value_type_t<EigMat>, Eigen::Dynamic, 1> x_ret(
      idxs.head_.ns_.size());
  for (int i = 0; i < idxs.head_.ns_.size(); ++i) {
    math::check_range("matrix[multi, uni] row indexing", name, x_ref.rows(),
                      idxs.head_.ns_[i]);
    x_ret.coeffRef(i)
        = x_ref.coeff(idxs.head_.ns_[i] - 1, idxs.tail_.head_.n_ - 1);
  }
  return x_ret;
}

/**
 * Return an Eigen matrix that is a possibly non-contiguous subset of the input
 *  Eigen matrix.
 *
 * Types:  matrix[multi, multi] = matrix
 *
 * @tparam EigMat An eigen matrix
 * @param[in] x Matrix to index.
 * @param[in] idxs Pair of multiple indexes.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat, require_eigen_dense_dynamic_t<EigMat>* = nullptr>
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
      math::check_range("matrix[multi,multi] row indexing", name, x_ref.rows(),
                        m);
      math::check_range("matrix[multi,multi] column indexing", name,
                        x_ref.cols(), n);
      x_ret.coeffRef(i, j) = x_ref.coeff(m - 1, n - 1);
    }
  }
  return x_ret;
}

/**
 * Return a column of a matrix with the range of the column specificed
 *  by another index.
 *
 * Types:  matrix[Idx, uni] = vector
 *
 * @tparam Mat An eigen matrix or `var_value<T>` whose inner type is an Eigen
 * matrix.
 * @param[in] x matrix.
 * @param[in] idxs Any index for the rows and a uni index for the columns
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 */
template <typename Mat, typename Idx, require_dense_dynamic_t<Mat>* = nullptr>
inline auto rvalue(
    Mat&& x,
    const cons_index_list<Idx, cons_index_list<index_uni, nil_index_list>>&
        idxs,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[..., uni] column indexing", name, x.cols(),
                    idxs.tail_.head_.n_);
  return rvalue(x.col(idxs.tail_.head_.n_ - 1), index_list(idxs.head_), name,
                depth + 1);
}

/**
 * Return an Eigen matrix of possibly unordered columns with each column
 *  range specified by another index.
 *
 * Types:  matrix[Idx, multi] = matrix
 *
 * @tparam EigMat An eigen matrix
 * @param[in] x Eigen matrix.
 * @param[in] idxs Any index for the rows and a multi index for the columns
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat, typename Idx,
          require_eigen_dense_dynamic_t<EigMat>* = nullptr,
          require_not_same_t<std::decay_t<Idx>, index_uni>* = nullptr>
inline plain_type_t<EigMat> rvalue(
    EigMat&& x,
    const cons_index_list<Idx, cons_index_list<index_multi, nil_index_list>>&
        idxs,
    const char* name = "ANON", int depth = 0) {
  const auto& x_ref = stan::math::to_ref(x);
  const int rows = rvalue_index_size(idxs.head_, x_ref.rows());
  const int cols = rvalue_index_size(idxs.tail_.head_, x_ref.cols());
  plain_type_t<EigMat> x_ret(rows, idxs.tail_.head_.ns_.size());
  for (int j = 0; j < idxs.tail_.head_.ns_.size(); ++j) {
    const int n = idxs.tail_.head_.ns_[j];
    math::check_range("matrix[..., multi] column indexing", name, x_ref.cols(),
                      n);
    x_ret.col(j)
        = rvalue(x_ref.col(n - 1), index_list(idxs.head_), name, depth + 1);
  }
  return x_ret;
}

/**
 * Return the matrix with all columns and a slice of rows.
 *
 * Types:  matrix[Idx, omni] = matrix
 *
 * @tparam Mat An eigen matrix or `var_value<T>` whose inner type is an Eigen
 * matrix.
 * @param[in] x type
 * @param[in] idxs Any index for the rows and omni index for the columns
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename Mat, typename Idx, require_dense_dynamic_t<Mat>* = nullptr>
inline auto rvalue(
    Mat&& x,
    const cons_index_list<Idx, cons_index_list<index_omni, nil_index_list>>&
        idxs,
    const char* name = "ANON", int depth = 0) {
  return rvalue(std::forward<Mat>(x), index_list(idxs.head_), name, depth + 1);
}

/**
 * Return columns min:N of the matrix with the range of the columns
 *  defined by another index.
 *
 * Types:  matrix[Idx, min] = matrix
 *
 * @tparam Mat An eigen matrix or `var_value<T>` whose inner type is an Eigen
 * matrix.
 * @tparam Idx An index.
 * @param[in] x type
 * @param[in] idxs Container holding a row index and an index from a minimum
 * index (inclusive) to the end of a container.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename Mat, typename Idx, require_dense_dynamic_t<Mat>* = nullptr>
inline auto rvalue(
    Mat&& x,
    const cons_index_list<Idx, cons_index_list<index_min, nil_index_list>>&
        idxs,
    const char* name = "ANON", int depth = 0) {
  const auto col_size = x.cols() - (idxs.tail_.head_.min_ - 1);
  math::check_range("matrix[..., min] column indexing", name, x.cols(),
                    idxs.tail_.head_.min_);
  return rvalue(x.rightCols(col_size), index_list(idxs.head_), name, depth + 1);
}

/**
 * Return columns 1:max of input matrix with the range of the columns
 *  defined by another index.
 *
 * Types:  matrix[Idx, max] = matrix
 *
 * @tparam Mat An eigen matrix or `var_value<T>` whose inner type is an Eigen
 * matrix.
 * @tparam Idx An index.
 * @param[in] x Eigen type
 * @param[in] idxs Index holding a row index and an index from the start of the
 * container up to the specified maximum index (inclusive).
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename Mat, typename Idx, require_dense_dynamic_t<Mat>* = nullptr>
inline auto rvalue(
    Mat&& x,
    const cons_index_list<Idx, cons_index_list<index_max, nil_index_list>>&
        idxs,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[..., max] column indexing", name, x.cols(),
                    idxs.tail_.head_.max_);
  return rvalue(x.leftCols(idxs.tail_.head_.max_), index_list(idxs.head_), name,
                depth + 1);
}

/**
 * Return the result of indexing the specified matrix with a
 * min_max_index returning a block from min to max.
 *
 * Types:  matrix[Idx, min_max] = matrix
 *
 * @tparam Mat An eigen matrix or `var_value<T>` whose inner type is an Eigen
 * matrix.
 * @tparam Idx Type of index.
 * @param[in] x Matrix to index.
 * @param[in] idxs Any index for the rows and a min max index for the columns
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename Mat, typename Idx, require_dense_dynamic_t<Mat>* = nullptr>
inline auto rvalue(
    Mat&& x,
    const cons_index_list<Idx, cons_index_list<index_min_max, nil_index_list>>&
        idxs,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[..., min_max] min column indexing", name, x.cols(),
                    idxs.tail_.head_.min_);
  math::check_range("matrix[..., min_max] max column indexing", name, x.cols(),
                    idxs.tail_.head_.max_);
  if (idxs.tail_.head_.is_ascending()) {
    const auto col_start = idxs.tail_.head_.min_ - 1;
    return rvalue(x.middleCols(col_start, idxs.tail_.head_.max_ - col_start),
                  index_list(idxs.head_), name, depth + 1)
        .eval();
  } else {
    const auto col_start = idxs.tail_.head_.max_ - 1;
    return rvalue(internal::rowwise_reverse(x.middleCols(
                      col_start, idxs.tail_.head_.min_ - col_start)),
                  index_list(idxs.head_), name, depth + 1)
        .eval();
  }
}

/**
 * Return the result of indexing the specified array with
 * a list of indexes beginning with a single index;  the result is
 * determined recursively.  Note that arrays are represented as
 * standard library vectors.
 *
 * Types:  std::vector<T>[uni | Idx] : T[Idx]
 *
 * @tparam T Type of list elements.
 * @tparam Idx Index list type for indexes after first index.
 * @param[in] v Container of list elements.
 * @param[in] idxs Index list beginning with single index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing array.
 */
template <typename StdVec, typename Idx,
          require_std_vector_t<StdVec>* = nullptr>
inline auto rvalue(StdVec&& v, const cons_index_list<index_uni, Idx>& idxs,
                   const char* name = "ANON", int depth = 0) {
  math::check_range("array[uni, ...] index", name, v.size(), idxs.head_.n_);
  if (std::is_rvalue_reference<StdVec>::value) {
    return rvalue(std::move(v[idxs.head_.n_ - 1]), idxs.tail_, name, depth + 1);
  } else {
    return rvalue(v[idxs.head_.n_ - 1], idxs.tail_, name, depth + 1);
  }
}

/**
 * Return the result of indexing the specified array with
 * a single index.
 *
 * Types:  std::vector<T>[uni] : T
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
    if (std::is_rvalue_reference<StdVec>::value) {
      result.emplace_back(
          rvalue(std::move(v[n - 1]), idxs.tail_, name, depth + 1));
    } else {
      result.emplace_back(rvalue(v[n - 1], idxs.tail_, name, depth + 1));
    }
  }
  return result;
}

}  // namespace model
}  // namespace stan
#endif

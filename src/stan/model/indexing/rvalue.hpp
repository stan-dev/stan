#ifndef STAN_MODEL_INDEXING_RVALUE_HPP
#define STAN_MODEL_INDEXING_RVALUE_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/to_ref.hpp>
#include <stan/model/indexing/index.hpp>
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
inline T rvalue(T&& x, const char* /*name*/) {
  return std::forward<T>(x);
}

template <typename T>
inline T& rvalue(T& x, const char* /*name*/) {
  return x;
}

template <typename T>
inline const T& rvalue(const T& x, const char* /*name*/) {
  return x;
}

/**
 * Return the result of indexing a type without taking a subset. Mostly used as
 * an intermediary rvalue function when doing multiple subsets.
 *
 * Types:  plain_type[omni] : plain_type
 *
 * @tparam T A type that is a plain object.
 * @param[in] x an object.
 * @return Result of indexing matrix.
 */
template <typename T>
inline T rvalue(T&& x, const char* /*name*/, index_omni /*idx*/) {
  return std::forward<T>(x);
}

template <typename T>
inline T& rvalue(T& x, const char* /*name*/, index_omni /*idx*/) {
  return x;
}

template <typename T>
inline const T& rvalue(const T& x, const char* /*name*/, index_omni /*idx*/) {
  return x;
}

/**
 * Return the result of indexing a type without taking a subset
 *
 * Types:  type[omni, omni] : type
 *
 * @tparam T Any type.
 * @param[in] x an object.
 * @param[in] name String form of expression being evaluated.
 * @return Result of indexing matrix.
 */
template <typename T>
inline T rvalue(T&& x, const char* name, index_omni /*idx1*/,
                index_omni /*idx2*/) {
  return std::forward<T>(x);
}

template <typename T>
inline T& rvalue(T& x, const char* name, index_omni /*idx1*/,
                 index_omni /*idx2*/) {
  return x;
}

template <typename T>
inline const T& rvalue(const T& x, const char* name, index_omni /*idx1*/,
                       index_omni /*idx2*/) {
  return x;
}

/**
 * Return a single element of a Vector.
 *
 * Types:  vector[uni] : scaler
 *
 * @tparam Vec An Eigen vector or `var_value<T>` where `T` is an eigen vector.
 * @param[in] v Vector being indexed.
 * @param[in] name String form of expression being evaluated.
 * @param[in] idx One single index.
 * @return Result of indexing vector.
 */
template <typename Vec, require_vector_t<Vec>* = nullptr,
          require_not_std_vector_t<Vec>* = nullptr>
inline auto rvalue(Vec&& v, const char* name, index_uni idx) {
  using stan::math::to_ref;
  math::check_range("vector[uni] indexing", name, v.size(), idx.n_);
  return v.coeff(idx.n_ - 1);
}

/**
 * Return a non-contiguous subset of elements in a vector.
 *
 * Types:  vector[multi] = vector
 *
 * @tparam EigVec Eigen type with either dynamic rows or columns, but not both.
 * @param[in] v Eigen vector type.
 * @param[in] name Name of variable
 * @param[in] idx Sequence of integers.
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the value size isn't the same as
 * the indexed size.
 */
template <typename EigVec, require_eigen_vector_t<EigVec>* = nullptr>
inline auto rvalue(EigVec&& v, const char* name, const index_multi& idx) {
  return stan::math::make_holder(
      [name, &idx](auto& v_ref) {
        return plain_type_t<EigVec>::NullaryExpr(
            idx.ns_.size(), [name, &idx, &v_ref](Eigen::Index i) {
              math::check_range("vector[multi] indexing", name, v_ref.size(),
                                idx.ns_[i]);
              return v_ref.coeff(idx.ns_[i] - 1);
            });
      },
      stan::math::to_ref(v));
}

/**
 * Return a range of a vector
 *
 * Types:  vector[min_max] = vector
 *
 * @tparam Vec An Eigen vector or `var_value<T>` where `T` is an eigen vector.
 * @param[in] v Vector being indexed.
 * @param[in] name String form of expression being evaluated.
 * @param[in] idx An index to select from a minimum to maximum range.
 * @return Result of indexing vector.
 */
template <typename Vec, require_vector_t<Vec>* = nullptr,
          require_not_std_vector_t<Vec>* = nullptr>
inline auto rvalue(Vec&& v, const char* name, index_min_max idx) {
  math::check_range("vector[min_max] min indexing", name, v.size(), idx.min_);
  math::check_range("vector[min_max] max indexing", name, v.size(), idx.max_);
  if (idx.is_ascending()) {
    const auto slice_start = idx.min_ - 1;
    const auto slice_size = idx.max_ - slice_start;
    return v.segment(slice_start, slice_size).eval();
  } else {
    const auto slice_start = idx.max_ - 1;
    const auto slice_size = idx.min_ - slice_start;
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
 * @param[in] name Name of variable
 * @param[in] idx An indexing from a specific minimum index to the end out
 *  of a bottom row of a matrix
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename Vec, require_vector_t<Vec>* = nullptr,
          require_not_std_vector_t<Vec>* = nullptr>
inline auto rvalue(Vec&& x, const char* name, index_min idx) {
  stan::math::check_range("vector[min] indexing", name, x.size(), idx.min_);
  return x.tail(x.size() - idx.min_ + 1);
}

/**
 * Return a head slice of a vector
 *
 * Types:  vector[1:max] <- vector
 *
 * @tparam Vec An Eigen vector or `var_value<T>` where `T` is an eigen vector.
 * @param[in] x vector.
 * @param[in] name Name of variable
 * @param[in] idx An indexing from the start of the container up to
 * the specified maximum index (inclusive).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename Vec, require_vector_t<Vec>* = nullptr,
          require_not_std_vector_t<Vec>* = nullptr>
inline auto rvalue(Vec&& x, const char* name, index_max idx) {
  stan::math::check_range("vector[max] indexing", name, x.size(), idx.max_);
  return x.head(idx.max_);
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
 * @param[in] name String form of expression being evaluated.
 * @param[in] idx uni-index
 * @return Result of indexing matrix.
 */
template <typename Mat, require_dense_dynamic_t<Mat>* = nullptr>
inline auto rvalue(Mat&& x, const char* name, index_uni idx) {
  math::check_range("matrix[uni] indexing", name, x.rows(), idx.n_);
  return x.row(idx.n_ - 1);
}

/**
 * Return the specified Eigen matrix at the specified multi index.
 *
 * Types:  matrix[multi] = matrix
 *
 * @tparam EigMat Eigen type with dynamic rows and columns.
 * @param[in] x Eigen type
 * @param[in] name Name of variable
 * @param[in] idx A multi index for selecting a set of rows.
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename EigMat, require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline plain_type_t<EigMat> rvalue(EigMat&& x, const char* name,
                                   const index_multi& idx) {
  for (int i = 0; i < idx.ns_.size(); ++i) {
    math::check_range("matrix[multi] row indexing", name, x.rows(), idx.ns_[i]);
  }
  return stan::math::make_holder(
      [&idx](auto& x_ref) {
        return plain_type_t<EigMat>::NullaryExpr(
            idx.ns_.size(), x_ref.cols(),
            [&idx, &x_ref](Eigen::Index i, Eigen::Index j) {
              return x_ref.coeff(idx.ns_[i] - 1, j);
            });
      },
      stan::math::to_ref(x));
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
 * @param[in] name String form of expression being evaluated.
 * @param[in] idx index_min
 * @return Result of indexing matrix.
 */
template <typename Mat, require_dense_dynamic_t<Mat>* = nullptr>
inline auto rvalue(Mat&& x, const char* name, index_min idx) {
  const auto row_size = x.rows() - (idx.min_ - 1);
  math::check_range("matrix[min] row indexing", name, x.rows(), idx.min_);
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
 * @param[in] name Name of variable
 * @param[in] idx An indexing from the start of the container up to
 * the specified maximum index (inclusive).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename Mat, require_dense_dynamic_t<Mat>* = nullptr>
inline auto rvalue(Mat&& x, const char* name, index_max idx) {
  math::check_range("matrix[max] row indexing", name, x.rows(), idx.max_);
  return x.topRows(idx.max_);
}

/**
 * Return a range of rows for a matrix.
 *
 * Types:  matrix[min_max] = matrix
 *
 * @tparam Mat An eigen matrix or `var_value<T>` whose inner type is an Eigen
 * matrix.
 * @param[in] x Dense matrix
 * @param[in] name Name of variable
 * @param[in] idx A min max index to select a range of rows.
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename Mat, require_dense_dynamic_t<Mat>* = nullptr>
inline auto rvalue(Mat&& x, const char* name, index_min_max idx) {
  math::check_range("matrix[min_max] max row indexing", name, x.rows(),
                    idx.max_);
  math::check_range("matrix[min_max] min row indexing", name, x.rows(),
                    idx.min_);
  if (idx.is_ascending()) {
    const auto row_size = idx.max_ - idx.min_ + 1;
    return x.middleRows(idx.min_ - 1, row_size).eval();
  } else {
    const auto row_size = idx.min_ - idx.max_ + 1;
    return internal::colwise_reverse(x.middleRows(idx.max_ - 1, row_size))
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
 * @param[in] name String form of expression being evaluated.
 * @param[in] row_idx Min max index for selecting rows.
 * @param[in] col_idx Min max index for selecting cols.
 * @return Result of indexing matrix.
 */
template <typename Mat, require_dense_dynamic_t<Mat>* = nullptr>
inline auto rvalue(Mat&& x, const char* name, index_min_max row_idx,
                   index_min_max col_idx) {
  math::check_range("matrix[min_max, min_max] min row indexing", name, x.rows(),
                    row_idx.min_);
  math::check_range("matrix[min_max, min_max] max row indexing", name, x.rows(),
                    row_idx.max_);
  math::check_range("matrix[min_max, min_max] min column indexing", name,
                    x.cols(), col_idx.min_);
  math::check_range("matrix[min_max, min_max] max column indexing", name,
                    x.cols(), col_idx.max_);
  if (row_idx.is_ascending()) {
    if (col_idx.is_ascending()) {
      return x
          .block(row_idx.min_ - 1, col_idx.min_ - 1,
                 row_idx.max_ - (row_idx.min_ - 1),
                 col_idx.max_ - (col_idx.min_ - 1))
          .eval();
    } else {
      return internal::rowwise_reverse(
                 x.block(row_idx.min_ - 1, col_idx.max_ - 1,
                         row_idx.max_ - (row_idx.min_ - 1),
                         col_idx.min_ - (col_idx.max_ - 1)))
          .eval();
    }
  } else {
    if (col_idx.is_ascending()) {
      return internal::colwise_reverse(
                 x.block(row_idx.max_ - 1, col_idx.min_ - 1,
                         row_idx.min_ - (row_idx.max_ - 1),
                         col_idx.max_ - (col_idx.min_ - 1)))
          .eval();
    } else {
      return x
          .block(row_idx.max_ - 1, col_idx.max_ - 1,
                 row_idx.min_ - (row_idx.max_ - 1),
                 col_idx.min_ - (col_idx.max_ - 1))
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
 * @param[in] name String form of expression being evaluated.
 * @param[in] row_idx uni index for selecting rows.
 * @param[in] col_idx uni index for selecting cols.
 * @return Result of indexing matrix.
 */
template <typename Mat, require_dense_dynamic_t<Mat>* = nullptr>
inline auto rvalue(Mat&& x, const char* name, index_uni row_idx,
                   index_uni col_idx) {
  math::check_range("matrix[uni,uni] row indexing", name, x.rows(), row_idx.n_);
  math::check_range("matrix[uni,uni] column indexing", name, x.cols(),
                    col_idx.n_);
  return x.coeff(row_idx.n_ - 1, col_idx.n_ - 1);
}

/**
 * Return a row of an Eigen matrix with possibly unordered cells.
 *
 * Types:  matrix[uni, multi] = row vector
 *
 * @tparam EigMat Eigen type with dynamic rows and columns.
 * @param[in] x Matrix to index.
 * @param[in] name Name of variable
 * @param[in] row_idx uni index for selecting rows.
 * @param[in] col_idx multi index for selecting cols.
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename EigMat, require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline Eigen::Matrix<value_type_t<EigMat>, 1, Eigen::Dynamic> rvalue(
    EigMat&& x, const char* name, index_uni row_idx,
    const index_multi& col_idx) {
  math::check_range("matrix[uni, multi] row indexing", name, x.rows(),
                    row_idx.n_);

  return stan::math::make_holder(
      [name, row_idx, &col_idx](auto& x_ref) {
        return Eigen::Matrix<value_type_t<EigMat>, 1, Eigen::Dynamic>::
            NullaryExpr(col_idx.ns_.size(), [name, row_i = row_idx.n_ - 1,
                                             &col_idx, &x_ref](Eigen::Index i) {
              math::check_range("matrix[uni, multi] column indexing", name,
                                x_ref.cols(), col_idx.ns_[i]);
              return x_ref.coeff(row_i, col_idx.ns_[i] - 1);
            });
      },
      stan::math::to_ref(x));
}

/**
 * Return a column of an Eigen matrix that is a possibly non-contiguous subset
 *  of the input Eigen matrix.
 *
 * Types:  matrix[multi, uni] = vector
 *
 * @tparam EigMat Eigen type with dynamic rows and columns.
 * @param[in] x Matrix to index.
 * @param[in] name Name of variable
 * @param[in] row_idx multi index for selecting rows.
 * @param[in] col_idx uni index for selecting cols.
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename EigMat, require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline Eigen::Matrix<value_type_t<EigMat>, Eigen::Dynamic, 1> rvalue(
    EigMat&& x, const char* name, const index_multi& row_idx,
    index_uni col_idx) {
  math::check_range("matrix[multi, uni] column indexing", name, x.cols(),
                    col_idx.n_);

  return stan::math::make_holder(
      [name, &row_idx, col_idx](auto& x_ref) {
        return Eigen::Matrix<value_type_t<EigMat>, Eigen::Dynamic, 1>::
            NullaryExpr(row_idx.ns_.size(),
                        [name, &row_idx, col_i = col_idx.n_ - 1,
                         &x_ref](Eigen::Index i) {
                          math::check_range("matrix[multi, uni] row indexing",
                                            name, x_ref.rows(), row_idx.ns_[i]);
                          return x_ref.coeff(row_idx.ns_[i] - 1, col_i);
                        });
      },
      stan::math::to_ref(x));
}

/**
 * Return an Eigen matrix that is a possibly non-contiguous subset of the input
 *  Eigen matrix.
 *
 * Types:  matrix[multi, multi] = matrix
 *
 * @tparam EigMat An eigen matrix
 * @param[in] x Matrix to index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] row_idx multi index for selecting rows.
 * @param[in] col_idx multi index for selecting cols.
 * @return Result of indexing matrix.
 */
template <typename EigMat, require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline plain_type_t<EigMat> rvalue(EigMat&& x, const char* name,
                                   const index_multi& row_idx,
                                   const index_multi& col_idx) {
  const auto& x_ref = stan::math::to_ref(x);
  const int rows = row_idx.ns_.size();
  const int cols = col_idx.ns_.size();
  plain_type_t<EigMat> x_ret(rows, cols);
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      const int m = row_idx.ns_[i];
      const int n = col_idx.ns_[j];
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
 * @param[in] name String form of expression being evaluated.
 * @param[in] row_idx index for selecting rows.
 * @param[in] col_idx uni index for selecting cols.
 */
template <typename Mat, typename Idx, require_dense_dynamic_t<Mat>* = nullptr>
inline auto rvalue(Mat&& x, const char* name, const Idx& row_idx,
                   index_uni col_idx) {
  math::check_range("matrix[..., uni] column indexing", name, x.cols(),
                    col_idx.n_);
  return rvalue(x.col(col_idx.n_ - 1), name, row_idx);
}

/**
 * Return an Eigen matrix of possibly unordered columns with each column
 *  range specified by another index.
 *
 * Types:  matrix[Idx, multi] = matrix
 *
 * @tparam EigMat An eigen matrix
 * @param[in] x Eigen matrix.
 * @param[in] name String form of expression being evaluated.
 * @param[in] row_idx index for selecting rows.
 * @param[in] col_idx multi index for selecting cols.
 * @return Result of indexing matrix.
 */
template <typename EigMat, typename Idx,
          require_eigen_dense_dynamic_t<EigMat>* = nullptr,
          require_not_same_t<std::decay_t<Idx>, index_uni>* = nullptr>
inline plain_type_t<EigMat> rvalue(EigMat&& x, const char* name,
                                   const Idx& row_idx,
                                   const index_multi& col_idx) {
  const auto& x_ref = stan::math::to_ref(x);
  const int rows = rvalue_index_size(row_idx, x_ref.rows());
  const int cols = rvalue_index_size(col_idx, x_ref.cols());
  plain_type_t<EigMat> x_ret(rows, col_idx.ns_.size());
  for (int j = 0; j < col_idx.ns_.size(); ++j) {
    const int n = col_idx.ns_[j];
    math::check_range("matrix[..., multi] column indexing", name, x_ref.cols(),
                      n);
    x_ret.col(j) = rvalue(x_ref.col(n - 1), name, row_idx);
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
 * @param[in] name Name of variable
 * @param[in] row_idx index for selecting rows.
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename Mat, typename Idx, require_dense_dynamic_t<Mat>* = nullptr>
inline auto rvalue(Mat&& x, const char* name, const Idx& row_idx,
                   index_omni /*col_idx*/) {
  return rvalue(std::forward<Mat>(x), name, row_idx);
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
 * @param[in] name Name of variable
 * @param[in] row_idx index for selecting rows.
 * @param[in] col_idx min index for selecting cols.
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename Mat, typename Idx, require_dense_dynamic_t<Mat>* = nullptr>
inline auto rvalue(Mat&& x, const char* name, const Idx& row_idx,
                   index_min col_idx) {
  const auto col_size = x.cols() - (col_idx.min_ - 1);
  math::check_range("matrix[..., min] column indexing", name, x.cols(),
                    col_idx.min_);
  return rvalue(x.rightCols(col_size), name, row_idx);
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
 * @param[in] name Name of variable
 * @param[in] row_idx index for selecting rows.
 * @param[in] col_idx max index for selecting cols.
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename Mat, typename Idx, require_dense_dynamic_t<Mat>* = nullptr>
inline auto rvalue(Mat&& x, const char* name, const Idx& row_idx,
                   index_max col_idx) {
  math::check_range("matrix[..., max] column indexing", name, x.cols(),
                    col_idx.max_);
  return rvalue(x.leftCols(col_idx.max_), name, row_idx);
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
 * @param[in] name String form of expression being evaluated.
 * @param[in] row_idx index for selecting rows.
 * @param[in] col_idx min max index for selecting cols.
 * @return Result of indexing matrix.
 */
template <typename Mat, typename Idx, require_dense_dynamic_t<Mat>* = nullptr>
inline auto rvalue(Mat&& x, const char* name, const Idx& row_idx,
                   index_min_max col_idx) {
  math::check_range("matrix[..., min_max] min column indexing", name, x.cols(),
                    col_idx.min_);
  math::check_range("matrix[..., min_max] max column indexing", name, x.cols(),
                    col_idx.max_);
  if (col_idx.is_ascending()) {
    const auto col_start = col_idx.min_ - 1;
    return rvalue(x.middleCols(col_start, col_idx.max_ - col_start), name,
                  row_idx)
        .eval();
  } else {
    const auto col_start = col_idx.max_ - 1;
    return rvalue(internal::rowwise_reverse(
                      x.middleCols(col_start, col_idx.min_ - col_start)),
                  name, row_idx)
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
 * @param[in] name String form of expression being evaluated.
 * @param[in] idx1 first index.
 * @param[in] idxs remaining indices.
 * @return Result of indexing array.
 */
template <typename StdVec, typename... Idxs,
          require_std_vector_t<StdVec>* = nullptr,
          require_not_t<std::is_lvalue_reference<StdVec&&>>* = nullptr>
inline auto rvalue(StdVec&& v, const char* name, index_uni idx1,
                   const Idxs&... idxs) {
  math::check_range("array[uni, ...] index", name, v.size(), idx1.n_);
  return rvalue(std::move(v[idx1.n_ - 1]), name, idxs...);
}
template <typename StdVec, typename... Idxs,
          require_std_vector_t<StdVec>* = nullptr>
inline auto rvalue(StdVec& v, const char* name, index_uni idx1,
                   const Idxs&... idxs) {
  math::check_range("array[uni, ...] index", name, v.size(), idx1.n_);
  return rvalue(v[idx1.n_ - 1], name, idxs...);
}

/**
 * Return the result of indexing the specified array with
 * a single index.
 *
 * Types:  std::vector<T>[uni] : T
 *
 * @tparam StdVec a standard vector
 * @param[in] v Container of list elements.
 * @param[in] name String form of expression being evaluated.
 * @param[in] idx single index
 * @return Result of indexing array.
 */
template <typename StdVec, require_std_vector_t<StdVec>* = nullptr,
          require_not_t<std::is_lvalue_reference<StdVec&&>>* = nullptr>
inline auto rvalue(StdVec&& v, const char* name, index_uni idx) {
  math::check_range("array[uni, ...] index", name, v.size(), idx.n_);
  return v[idx.n_ - 1];
}
template <typename StdVec, require_std_vector_t<StdVec>* = nullptr>
inline auto& rvalue(StdVec& v, const char* name, index_uni idx) {
  math::check_range("array[uni, ...] index", name, v.size(), idx.n_);
  return v[idx.n_ - 1];
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
 * @param[in] name String form of expression being evaluated.
 * @param[in] idx1 first index
 * @param[in] idxs remaining indices
 * @return Result of indexing array.
 */
template <typename StdVec, typename Idx1, typename... Idxs,
          require_std_vector_t<StdVec>* = nullptr,
          require_not_same_t<Idx1, index_uni>* = nullptr>
inline auto rvalue(StdVec&& v, const char* name, Idx1 idx1,
                   const Idxs&... idxs) {
  using inner_type = plain_type_t<decltype(
      rvalue(v[rvalue_at(0, idx1) - 1], name, idxs...))>;
  std::vector<inner_type> result;
  const int index_size = rvalue_index_size(idx1, v.size());
  if (index_size > 0) {
    result.reserve(index_size);
  }
  for (int i = 0; i < index_size; ++i) {
    const int n = rvalue_at(i, idx1);
    math::check_range("array[..., ...] index", name, v.size(), n);
    if (std::is_rvalue_reference<StdVec>::value) {
      result.emplace_back(rvalue(std::move(v[n - 1]), name, idxs...));
    } else {
      result.emplace_back(rvalue(v[n - 1], name, idxs...));
    }
  }
  return result;
}

}  // namespace model
}  // namespace stan
#endif

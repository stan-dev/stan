#ifndef STAN_MODEL_INDEXING_ASSIGN_HPP
#define STAN_MODEL_INDEXING_ASSIGN_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/functor.hpp>
#include <stan/math/prim/fun/to_ref.hpp>
#include <stan/model/indexing/access_helpers.hpp>
#include <stan/model/indexing/index.hpp>
#include <stan/model/indexing/rvalue_at.hpp>
#include <stan/model/indexing/rvalue_index_size.hpp>
#include <type_traits>
#include <vector>
#include <tuple>

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
 *    - Assigns a subset of rows.
 *  - column/row overloads
 *    - overload on both the row and column indices.
 *  - column overloads
 *    - These take a subset of columns and then call the row slice assignment
 *       over the column subset.
 * Std vector:
 *  - single element and elementwise overloads
 *  - General overload for nested std vectors.
 */

/**
 * Assign one object to another.
 *
 * @tparam T lvalue variable type
 * @tparam U rvalue variable type, which must be assignable to `T`
 * @param[in,out] x lvalue
 * @param[in] y rvalue
 * @param[in] name Name of lvalue variable
 */
template <
    typename T, typename U,
    require_t<std::is_assignable<std::decay_t<T>&, std::decay_t<U>>>* = nullptr,
    require_all_not_t<internal::is_tuple<T>, internal::is_tuple<U>>* = nullptr>
inline void assign(T&& x, U&& y, const char* name) {
  internal::assign_impl(x, std::forward<U>(y), name);
}

/**
 * Assign to a single element of an Eigen Vector.
 *
 * Types: vector[uni] <- scalar
 *
 * @tparam Vec Eigen type with either dynamic rows or columns, but not both.
 * @tparam U Type of value (must be assignable to T).
 * @param[in] x Vector variable to be assigned.
 * @param[in] y Value to assign.
 * @param[in] name Name of variable
 * @param[in] idx index to assign to.
 * @throw std::out_of_range If the index is out of bounds.
 */
template <typename Vec, typename U, require_eigen_vector_t<Vec>* = nullptr,
          require_stan_scalar_t<U>* = nullptr>
inline void assign(Vec&& x, const U& y, const char* name, index_uni idx) {
  stan::math::check_range("vector[uni] assign", name, x.size(), idx.n_);
  x.coeffRef(idx.n_ - 1) = y;
}

/**
 * Assign to a non-contiguous subset of elements in a vector.
 *
 * Types:  vector[multi] <- vector
 *
 * @tparam Vec1 Eigen type with either dynamic rows or columns, but not both.
 * @tparam Vec2 Eigen type with either dynamic rows or columns, but not both.
 * @param[in] x Vector to be assigned.
 * @param[in] y Value vector.
 * @param[in] name Name of variable
 * @param[in] idx Index holding an `std::vector` of cells to assign to.
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the value size isn't the same as
 * the indexed size.
 */
template <typename Vec1, typename Vec2,
          require_all_eigen_vector_t<Vec1, Vec2>* = nullptr>
inline void assign(Vec1&& x, const Vec2& y, const char* name,
                   const index_multi& idx) {
  const auto& y_ref = stan::math::to_ref(y);
  stan::math::check_size_match("vector[multi] assign", name, idx.ns_.size(),
                               "right hand side", y_ref.size());
  const auto x_size = x.size();
  for (int n = 0; n < y_ref.size(); ++n) {
    stan::math::check_range("vector[multi] assign", name, x_size, idx.ns_[n]);
    x.coeffRef(idx.ns_[n] - 1) = y_ref.coeff(n);
  }
}

/**
 * Assign to a range of an Eigen vector
 *
 * Types:  vector[min_max] <- vector
 *
 * @tparam Vec1 A type with either dynamic rows or columns, but not both.
 * @tparam Vec2 A type with either dynamic rows or columns, but not both.
 * @param[in] x vector variable to be assigned.
 * @param[in] y Value vector.
 * @param[in] name Name of variable
 * @param[in] idx `index_min_max`.
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the value size isn't the same as
 * the indexed size.
 */
template <typename Vec1, typename Vec2,
          require_all_vector_t<Vec1, Vec2>* = nullptr,
          require_all_not_std_vector_t<Vec1, Vec2>* = nullptr>
inline void assign(Vec1&& x, const Vec2& y, const char* name,
                   index_min_max idx) {
  if (likely(idx.max_ >= idx.min_)) {
    stan::math::check_range("vector[min_max] min assign", name, x.size(),
                            idx.min_);
    stan::math::check_range("vector[min_max] max assign", name, x.size(),
                            idx.max_);
    const auto slice_start = idx.min_ - 1;
    const auto slice_size = idx.max_ - slice_start;
    stan::math::check_size_match("vector[min_max] assign", name, slice_size,
                                 "right hand side", y.size());
    internal::assign_impl(x.segment(slice_start, slice_size), y, name);
  } else {
    stan::math::check_size_match("vector[negative_min_max] assign", name, 0,
                                 "right hand side", y.size());
  }
}

/**
 * Assign to a tail slice of a vector
 *
 * Types:  vector[min:N] <- vector
 *
 * @tparam Vec1 A type with either dynamic rows or columns, but not both.
 * @tparam Vec2 A type with either dynamic rows or columns, but not both.
 * @param[in] x vector to be assigned to.
 * @param[in] y Value vector.
 * @param[in] name Name of variable
 * @param[in] idx An index.
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the value size isn't the same as
 * the indexed size.
 */
template <typename Vec1, typename Vec2,
          require_all_vector_t<Vec1, Vec2>* = nullptr,
          require_all_not_std_vector_t<Vec1, Vec2>* = nullptr>
inline void assign(Vec1&& x, const Vec2& y, const char* name, index_min idx) {
  stan::math::check_range("vector[min] assign", name, x.size(), idx.min_);
  stan::math::check_size_match("vector[min] assign", name,
                               x.size() - idx.min_ + 1, "right hand side",
                               y.size());
  internal::assign_impl(x.tail(x.size() - idx.min_ + 1), y, name);
}

/**
 * Assign to a head slice of the assignee vector
 *
 * Types:  vector[1:max] <- vector
 *
 * @tparam Vec1 A type with either dynamic rows or columns, but not both.
 * @tparam Vec2 A type with either dynamic rows or columns, but not both.
 * @param[in] x vector to be assigned to.
 * @param[in] y Value vector.
 * @param[in] name Name of variable
 * @param[in] idx An index.
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the value size isn't the same as
 * the indexed size.
 */
template <typename Vec1, typename Vec2,
          require_all_vector_t<Vec1, Vec2>* = nullptr,
          require_all_not_std_vector_t<Vec1, Vec2>* = nullptr>
inline void assign(Vec1&& x, const Vec2& y, const char* name, index_max idx) {
  if (likely(idx.max_ > 0)) {
    stan::math::check_range("vector[max] assign", name, x.size(), idx.max_);
    stan::math::check_size_match("vector[max] assign", name, idx.max_,
                                 "right hand side", y.size());
    internal::assign_impl(x.head(idx.max_), y, name);
  } else {
    stan::math::check_size_match("vector[max < 1] assign", name, 0,
                                 "right hand side", y.size());
  }
}

/**
 * Assign a vector to another vector.
 *
 * Types:  vector[omni] <- vector
 *
 * @tparam Vec1 A type with either dynamic rows or columns, but not both.
 * @tparam Vec2 A type with either dynamic rows or columns, but not both.
 * @param[in] x vector to be assigned to.
 * @param[in] y Value vector.
 * @param[in] name Name of variable
 * @throw std::invalid_argument If the value size isn't the same as
 * the indexed size.
 */
template <typename Vec1, typename Vec2,
          require_all_vector_t<Vec1, Vec2>* = nullptr,
          require_all_not_std_vector_t<Vec1, Vec2>* = nullptr>
inline void assign(Vec1&& x, Vec2&& y, const char* name, index_omni /* idx */) {
  stan::math::check_size_match("vector[omni] assign", name, x.size(),
                               "right hand side", y.size());
  internal::assign_impl(x, std::forward<Vec2>(y), name);
}

/**
 * Assign a row vector to a row of an eigen matrix.
 *
 * Types:  mat[uni] = row_vector
 *
 * @tparam Mat A type with dynamic rows and columns.
 * @tparam RowVec A type with dynamic columns and a compile time rows equal
 * to 1.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] y Value row vector.
 * @param[in] name Name of variable
 * @param[in] idx An index holding the row to be assigned to.
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the number of columns in the row
 * vector and matrix do not match.
 */
template <typename Mat, typename RowVec,
          require_dense_dynamic_t<Mat>* = nullptr,
          require_row_vector_t<RowVec>* = nullptr>
inline void assign(Mat&& x, const RowVec& y, const char* name, index_uni idx) {
  stan::math::check_range("matrix[uni] assign row", name, x.rows(), idx.n_);
  stan::math::check_size_match("matrix[uni] assign columns", name, x.cols(),
                               "right hand side size", y.size());
  internal::assign_impl(x.row(idx.n_ - 1), y, name);
}

/**
 * Assign to a non-contiguous subset of a matrice's rows.
 *
 * Types:  mat[multi] = mat
 *
 * @tparam Mat An Eigen type with dynamic rows and columns.
 * @tparam Mat2 An Eigen type with dynamic rows and columns.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] y Value matrix.
 * @param[in] name Name of variable
 * @param[in] idx multi index
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and right-hand side matrix do not match.
 */
template <typename Mat1, typename Mat2,
          require_all_eigen_dense_dynamic_t<Mat1, Mat2>* = nullptr>
inline void assign(Mat1&& x, const Mat2& y, const char* name,
                   const index_multi& idx) {
  const auto& y_ref = stan::math::to_ref(y);
  stan::math::check_size_match("matrix[multi] assign rows", name,
                               idx.ns_.size(), "right hand side rows",
                               y.rows());
  stan::math::check_size_match("matrix[multi] assign columns", name, x.cols(),
                               "right hand side columns", y.cols());
  for (int i = 0; i < idx.ns_.size(); ++i) {
    const int n = idx.ns_[i];
    stan::math::check_range("matrix[multi] assign row", name, x.rows(), n);
    x.row(n - 1) = y_ref.row(i);
  }
}

/**
 * Assign a matrix to another matrix
 *
 * Types:  mat[omni] = mat
 *
 * @tparam Mat1 A type with dynamic rows and columns.
 * @tparam Mat2 A type with dynamic rows and columns.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] y Value matrix.
 * @param[in] name Name of variable
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and right-hand side matrix do not match.
 */
template <typename Mat1, typename Mat2,
          require_all_dense_dynamic_t<Mat1, Mat2>* = nullptr>
inline void assign(Mat1&& x, Mat2&& y, const char* name, index_omni /* idx */) {
  stan::math::check_size_match("matrix[omni] assign rows", name, x.rows(),
                               "right hand side rows", y.rows());
  stan::math::check_size_match("matrix[omni] assign columns", name, x.cols(),
                               "right hand side columns", y.cols());
  internal::assign_impl(x, std::forward<Mat2>(y), name);
}

/**
 * Assign a matrix to the bottom rows of the assignee matrix.
 *
 * Types:  mat[min] = mat
 *
 * @tparam Mat1 A type with dynamic rows and columns.
 * @tparam Mat2 A type with dynamic rows and columns.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] y Value matrix.
 * @param[in] name Name of variable
 * @param[in] idx min index
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and right-hand side matrix do not match.
 */
template <typename Mat1, typename Mat2,
          require_dense_dynamic_t<Mat1>* = nullptr,
          require_matrix_t<Mat2>* = nullptr>
inline void assign(Mat1&& x, const Mat2& y, const char* name, index_min idx) {
  const auto row_size = x.rows() - (idx.min_ - 1);
  stan::math::check_range("matrix[min] assign row", name, x.rows(), idx.min_);
  stan::math::check_size_match("matrix[min] assign rows", name, row_size,
                               "right hand side rows", y.rows());
  stan::math::check_size_match("matrix[min] assign columns", name, x.cols(),
                               "right hand side columns", y.cols());
  internal::assign_impl(x.bottomRows(row_size), y, name);
}

/**
 * Assign a matrix to the top rows of the assignee matrix.
 *
 * Types:  mat[max] = mat
 *
 * @tparam Mat1 A type with dynamic rows and columns.
 * @tparam Mat2 A type with dynamic rows and columns.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] y Value matrix.
 * @param[in] name Name of variable
 * @param[in] idx max index
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and right-hand side matrix do not match.
 */
template <typename Mat1, typename Mat2,
          require_dense_dynamic_t<Mat1>* = nullptr,
          require_matrix_t<Mat2>* = nullptr>
inline void assign(Mat1&& x, const Mat2& y, const char* name, index_max idx) {
  stan::math::check_size_match("matrix[max] assign columns", name, x.cols(),
                               "right hand side columns", y.cols());
  if (likely(idx.max_ > 0)) {
    stan::math::check_range("matrix[max] assign row", name, x.rows(), idx.max_);
    stan::math::check_size_match("matrix[max] assign rows", name, idx.max_,
                                 "right hand side rows", y.rows());
    internal::assign_impl(x.topRows(idx.max_), y, name);
  } else {
    stan::math::check_size_match("matrix[max < 1] assign rows", name, 0,
                                 "right hand side rows", y.rows());
  }
}

/**
 * Assign a matrix to a range of rows of the assignee matrix.
 *
 * Types:  mat[min_max] = mat
 *
 * @tparam Mat1 A type with dynamic rows and columns.
 * @tparam Mat2 A type with dynamic rows and columns.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] y Value matrix.
 * @param[in] name Name of variable
 * @param[in] idx An index for a min_max range of rows
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and right-hand side matrix do not match.
 */
template <typename Mat1, typename Mat2,
          require_dense_dynamic_t<Mat1>* = nullptr,
          require_matrix_t<Mat2>* = nullptr>
inline void assign(Mat1&& x, Mat2&& y, const char* name, index_min_max idx) {
  stan::math::check_size_match("matrix[min_max] assign columns", name, x.cols(),
                               "right hand side columns", y.cols());
  if (likely(idx.max_ >= idx.min_)) {
    stan::math::check_range("matrix[min_max] min row indexing", name, x.rows(),
                            idx.min_);
    stan::math::check_range("matrix[min_max] max row indexing", name, x.rows(),
                            idx.max_);
    const auto row_size = idx.max_ - idx.min_ + 1;
    stan::math::check_size_match("matrix[min_max] assign rows", name, row_size,
                                 "right hand side rows", y.rows());
    internal::assign_impl(x.middleRows(idx.min_ - 1, row_size), y, name);
  } else {
    stan::math::check_size_match("matrix[negative_min_max] assign rows", name,
                                 0, "right hand side", y.rows());
  }
}

/**
 * Assign to a block of an Eigen matrix.
 *
 * Types:  mat[min_max, min_max] = mat
 *
 * @tparam Mat1 A type with dynamic rows and columns.
 * @tparam Mat2 A type with dynamic rows and columns.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] y Matrix variable to assign from.
 * @param[in] name Name of variable
 * @param[in] row_idx min_max index for selecting rows
 * @param[in] col_idx min_max index for selecting columns
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and right-hand side matrix do not match.
 */
template <typename Mat1, typename Mat2,
          require_dense_dynamic_t<Mat1>* = nullptr>
inline void assign(Mat1&& x, Mat2&& y, const char* name, index_min_max row_idx,
                   index_min_max col_idx) {
  if (likely((row_idx.max_ >= row_idx.min_)
             && (col_idx.max_ >= col_idx.min_))) {
    stan::math::check_range("matrix[min_max, min_max] assign min row", name,
                            x.rows(), row_idx.min_);

    stan::math::check_range("matrix[min_max, min_max] assign min column", name,
                            x.cols(), col_idx.min_);
    stan::math::check_range("matrix[min_max, min_max] assign max row", name,
                            x.rows(), row_idx.max_);
    stan::math::check_range("matrix[min_max, min_max] assign max column", name,
                            x.cols(), col_idx.max_);
    auto row_size = row_idx.max_ - (row_idx.min_ - 1);
    auto col_size = col_idx.max_ - (col_idx.min_ - 1);
    stan::math::check_size_match("matrix[min_max, min_max] assign rows", name,
                                 row_size, "right hand side rows", y.rows());
    stan::math::check_size_match("matrix[min_max, min_max] assign columns",
                                 name, col_size, "right hand side columns",
                                 y.cols());
    internal::assign_impl(
        x.block(row_idx.min_ - 1, col_idx.min_ - 1, row_size, col_size), y,
        name);
  } else {
    // Check the indexing is valid but don't actually do assignment
    if (row_idx.max_ >= row_idx.min_) {
      stan::math::check_range(
          "matrix[min_max, negative_min_max] assign min row", name, x.rows(),
          row_idx.min_);
      stan::math::check_range(
          "matrix[min_max, negative_min_max] assign max row", name, x.rows(),
          row_idx.max_);
      stan::math::check_size_match("matrix[min_max, negative_min_max] assign",
                                   name, 0, "right hand side columns",
                                   y.cols());

    } else if (col_idx.max_ >= col_idx.min_) {
      stan::math::check_range(
          "matrix[negative_min_max, min_max] assign min column", name, x.cols(),
          col_idx.min_);
      stan::math::check_range(
          "matrix[negative_min_max, min_max] assign max column", name, x.cols(),
          col_idx.max_);
      stan::math::check_size_match("matrix[min_max, min_max] assign rows", name,
                                   0, "right hand side", y.rows());
    } else {
      stan::math::check_size_match(
          "matrix[negative_min_max, negative_min_max] assign rows", name, 0,
          "right hand side", y.rows());
      stan::math::check_size_match(
          "matrix[negative_min_max, negative_min_max] assign cols", name, 0,
          "right hand side", y.cols());
    }
  }
}

/**
 * Assign to a cell of an Eigen Matrix.
 *
 * Types:  mat[single, single] = scalar
 *
 * @tparam Mat Eigen type with dynamic rows and columns.
 * @tparam U Scalar type.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] y Value scalar.
 * @param[in] name Name of variable
 * @param[in] row_idx uni index for selecting rows
 * @param[in] col_idx uni index for selecting columns
 * @throw std::out_of_range If either of the indices are out of bounds.
 */
template <typename Mat, typename U,
          require_eigen_dense_dynamic_t<Mat>* = nullptr>
inline void assign(Mat&& x, const U& y, const char* name, index_uni row_idx,
                   index_uni col_idx) {
  stan::math::check_range("matrix[uni,uni] assign row", name, x.rows(),
                          row_idx.n_);
  stan::math::check_range("matrix[uni,uni] assign column", name, x.cols(),
                          col_idx.n_);
  x.coeffRef(row_idx.n_ - 1, col_idx.n_ - 1) = y;
}

/**
 * Assign multiple possibly unordered cells of row vector to a row of an eigen
 * matrix.
 *
 * Types:  mat[uni, multi] = row_vector
 *
 * @tparam Mat1 Eigen type with dynamic rows and columns.
 * @tparam Vec Eigen type with dynamic columns and compile time rows of 1.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] y Row vector.
 * @param[in] name Name of variable
 * @param[in] row_idx uni index for selecting rows
 * @param[in] col_idx multi index for selecting columns
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and value matrix do not match.
 */
template <typename Mat1, typename Vec,
          require_eigen_dense_dynamic_t<Mat1>* = nullptr,
          require_eigen_row_vector_t<Vec>* = nullptr>
inline void assign(Mat1&& x, const Vec& y, const char* name, index_uni row_idx,
                   const index_multi& col_idx) {
  const auto& y_ref = stan::math::to_ref(y);
  stan::math::check_range("matrix[uni, multi] assign row", name, x.rows(),
                          row_idx.n_);
  stan::math::check_size_match("matrix[uni, multi] assign", name,
                               col_idx.ns_.size(), "right hand side",
                               y_ref.size());
  for (int i = 0; i < col_idx.ns_.size(); ++i) {
    stan::math::check_range("matrix[uni, multi] assign column", name, x.cols(),
                            col_idx.ns_[i]);
    x.coeffRef(row_idx.n_ - 1, col_idx.ns_[i] - 1) = y_ref.coeff(i);
  }
}

/**
 * Assign to multiple possibly unordered cell's of a matrix from an input
 * matrix.
 *
 * Types:  mat[multi, multi] = mat
 *
 * @tparam Mat1 Eigen type with dynamic rows and columns.
 * @tparam Mat2 Eigen type with dynamic rows and columns.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] y Value matrix.
 * @param[in] name Name of variable
 * @param[in] row_idx multi index for selecting rows
 * @param[in] col_idx multi index for selecting columns
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and value matrix do not match.
 */
template <typename Mat1, typename Mat2,
          require_all_eigen_dense_dynamic_t<Mat1, Mat2>* = nullptr>
inline void assign(Mat1&& x, const Mat2& y, const char* name,
                   const index_multi& row_idx, const index_multi& col_idx) {
  const auto& y_ref = stan::math::to_ref(y);
  stan::math::check_size_match("matrix[multi,multi] assign rows", name,
                               row_idx.ns_.size(), "right hand side rows",
                               y_ref.rows());
  stan::math::check_size_match("matrix[multi,multi] assign columns", name,
                               col_idx.ns_.size(), "right hand side columns",
                               y_ref.cols());
  for (int j = 0; j < y_ref.cols(); ++j) {
    const int n = col_idx.ns_[j];
    stan::math::check_range("matrix[multi,multi] assign column", name, x.cols(),
                            n);
    for (int i = 0; i < y_ref.rows(); ++i) {
      const int m = row_idx.ns_[i];
      stan::math::check_range("matrix[multi,multi] assign row", name, x.rows(),
                              m);
      x.coeffRef(m - 1, n - 1) = y_ref.coeff(i, j);
    }
  }
}

/**
 * Assign to any rows of a single column of a matrix.
 *
 * Types:  mat[Idx, uni] = mat
 *
 * @tparam Mat1 A type with dynamic rows and columns.
 * @tparam Mat2 A type that's assignable to the indexed matrix.
 * @tparam Idx The row index type
 * @param[in] x Matrix variable to be assigned.
 * @param[in] y Matrix variable to assign from.
 * @param[in] name Name of variable
 * @param[in] row_idx index for selecting rows
 * @param[in] col_idx uni index for selecting columns
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and right-hand side matrix do not match.
 */
template <typename Mat1, typename Mat2, typename Idx,
          require_dense_dynamic_t<Mat1>* = nullptr>
inline void assign(Mat1&& x, const Mat2& y, const char* name,
                   const Idx& row_idx, index_uni col_idx) {
  stan::math::check_range("matrix[..., uni] assign column", name, x.cols(),
                          col_idx.n_);
  assign(x.col(col_idx.n_ - 1), y, name, row_idx);
}

/**
 * Assign to a non-contiguous set of columns of a matrix.
 *
 * Types:  mat[Idx, multi] = mat
 *
 * @tparam Mat1 Eigen type with dynamic rows and columns.
 * @tparam Mat2 Eigen type
 * @tparam Idx The row index type
 * @param[in] x Matrix variable to be assigned.
 * @param[in] y Value matrix.
 * @param[in] name Name of variable
 * @param[in] row_idx index for selecting rows
 * @param[in] col_idx multi index for selecting columns
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and value matrix do not match.
 */
template <typename Mat1, typename Mat2, typename Idx,
          require_eigen_dense_dynamic_t<Mat1>* = nullptr>
inline void assign(Mat1&& x, const Mat2& y, const char* name,
                   const Idx& row_idx, const index_multi& col_idx) {
  const auto& y_ref = stan::math::to_ref(y);
  stan::math::check_size_match("matrix[..., multi] assign column sizes", name,
                               col_idx.ns_.size(), "right hand side columns",
                               y_ref.cols());
  for (int j = 0; j < col_idx.ns_.size(); ++j) {
    const int n = col_idx.ns_[j];
    stan::math::check_range("matrix[..., multi] assign column", name, x.cols(),
                            n);
    assign(x.col(n - 1), y_ref.col(j), name, row_idx);
  }
}

/**
 * Assign to any rows of a matrix.
 *
 * Types:  mat[Idx, omni] = mat
 *
 * @tparam Mat1 A type with dynamic rows and columns.
 * @tparam Mat2 A type assignable to the slice of the matrix.
 * @tparam Idx The row index type
 * @param[in] x Matrix variable to be assigned.
 * @param[in] y Value matrix.
 * @param[in] name Name of variable
 * @param[in] row_idx index for selecting rows
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and value matrix do not match.
 */
template <typename Mat1, typename Mat2, typename Idx,
          require_dense_dynamic_t<Mat1>* = nullptr>
inline void assign(Mat1&& x, Mat2&& y, const char* name, const Idx& row_idx,
                   index_omni /* idx */) {
  assign(x, std::forward<Mat2>(y), name, row_idx);
}

/**
 * Assign any rows and min:N columns of a matrix.
 *
 * Types:  mat[Idx, min] = mat
 *
 * @tparam Mat1 A type with dynamic rows and columns.
 * @tparam Mat2 A type assignable to the slice of the matrix.
 * @tparam Idx The row index type
 * @param[in] x Matrix variable to be assigned.
 * index (inclusive) to the end of a container.
 * @param[in] y Value matrix.
 * @param[in] name Name of variable
 * @param[in] row_idx index for selecting rows
 * @param[in] col_idx min index for selecting columns
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and right-hand side matrix do not match.
 */
template <typename Mat1, typename Mat2, typename Idx,
          require_dense_dynamic_t<Mat1>* = nullptr>
inline void assign(Mat1&& x, const Mat2& y, const char* name,
                   const Idx& row_idx, index_min col_idx) {
  const auto start_col = col_idx.min_ - 1;
  const auto col_size = x.cols() - start_col;
  stan::math::check_range("matrix[..., min] assign column", name, x.cols(),
                          col_idx.min_);
  stan::math::check_size_match("matrix[..., min] assign columns", name,
                               col_size, "right hand side columns", y.cols());
  assign(x.rightCols(col_size), y, name, row_idx);
}

/**
 * Assign to any rows and 1:max columns of a matrix.
 *
 * Types:  mat[Idx, max] = mat
 *
 * @tparam Mat1 A type with dynamic rows and columns.
 * @tparam Mat2 A type assignable to the slice of the matrix.
 * @tparam Idx The row index type
 * @param[in] x Matrix variable to be assigned.
 * container up to the specified maximum index (inclusive).
 * @param[in] y Value matrix.
 * @param[in] name Name of variable
 * @param[in] row_idx index for selecting rows
 * @param[in] col_idx max index for selecting columns
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and right-hand side matrix do not match.
 */
template <typename Mat1, typename Mat2, typename Idx,
          require_dense_dynamic_t<Mat1>* = nullptr>
inline void assign(Mat1&& x, const Mat2& y, const char* name,
                   const Idx& row_idx, index_max col_idx) {
  if (likely(col_idx.max_ > 0)) {
    stan::math::check_range("matrix[..., max] assign", name, x.cols(),
                            col_idx.max_);
    stan::math::check_size_match("matrix[..., max] assign columns", name,
                                 col_idx.max_, "right hand side columns",
                                 y.cols());
    assign(x.leftCols(col_idx.max_), y, name, row_idx);
  } else {
    stan::math::check_size_match("matrix[..., max] assign columns", name, 0,
                                 "right hand side columns", y.cols());
  }
}

/**
 * Assign to any rows and a range of columns.
 *
 * Types:  mat[Idx, min_max] = mat
 *
 * @tparam Mat1 A type with dynamic rows and columns.
 * @tparam Mat2 A type assignable to the slice of the matrix.
 * @tparam Idx The row index type
 * @param[in] x Matrix variable to be assigned.
 * @param[in] y Matrix variable to assign from.
 * @param[in] name Name of variable
 * @param[in] row_idx index for selecting rows
 * @param[in] col_idx max index for selecting columns
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and right-hand side matrix do not match.
 */
template <typename Mat1, typename Mat2, typename Idx,
          require_dense_dynamic_t<Mat1>* = nullptr>
inline void assign(Mat1&& x, Mat2&& y, const char* name, const Idx& row_idx,
                   index_min_max col_idx) {
  if (likely(col_idx.max_ >= col_idx.min_)) {
    stan::math::check_range("matrix[..., min_max] assign min column", name,
                            x.cols(), col_idx.min_);
    stan::math::check_range("matrix[..., min_max] assign max column", name,
                            x.cols(), col_idx.max_);
    const auto col_start = col_idx.min_ - 1;
    const auto col_size = col_idx.max_ - col_start;
    stan::math::check_size_match("matrix[..., min_max] assign column size",
                                 name, col_size, "right hand side", y.cols());
    assign(x.middleCols(col_start, col_size), y, name, row_idx);
  } else {
    stan::math::check_size_match("matrix[..., negative_min_max] assign columns",
                                 name, 0, "right hand side columns", y.cols());
  }
}

/**
 * Assign the elements of one standard vector to another.
 *
 *  std_vector = std_vector
 *
 * @tparam T Type of Std vector to be assigned to.
 * @tparam U Type of Std vector to be assigned from.
 * @param[in] x lvalue variable
 * @param[in] y rvalue variable
 * @param[in] name name of lvalue variable
 */
template <typename T, typename U, require_all_std_vector_t<T, U>* = nullptr,
          require_not_t<
              std::is_assignable<std::decay_t<T>&, std::decay_t<U>>>* = nullptr>
inline void assign(T&& x, U&& y, const char* name) {
  if (unlikely(x.size() != 0)) {
    stan::math::check_size_match("assign array size", name, x.size(),
                                 "right hand side", y.size());
  }
  if (std::is_rvalue_reference<U&&>::value) {
    for (size_t i = 0; i < y.size(); ++i) {
      assign(x[i], std::move(y[i]), name);
    }
  } else {
    for (size_t i = 0; i < y.size(); ++i) {
      assign(x[i], y[i], name);
    }
  }
}

/**
 * Assign to a single element of an std vector with additional subsetting on
 *  that element.
 *
 * Types:  std_vector<T>[uni | Idx] = T
 *
 * @tparam StdVec A standard vector
 * @tparam Idx Type of tail of index list.
 * @tparam U A type assignable to the value type of `StdVec`
 * @param[in] x Array variable to be assigned.
 * @param[in] y Value.
 * @param[in] name Name of variable
 * @param[in] idx1 uni index
 * @param[in] idxs Remaining indices
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions do not match in the
 * tail assignment.
 */
template <typename StdVec, typename U, typename... Idxs,
          require_std_vector_t<StdVec>* = nullptr>
inline void assign(StdVec&& x, U&& y, const char* name, index_uni idx1,
                   const Idxs&... idxs) {
  stan::math::check_range("array[uni,...] assign", name, x.size(), idx1.n_);
  assign(x[idx1.n_ - 1], std::forward<U>(y), name, idxs...);
}

/**
 * Assign to a single element of an std vector.
 *
 * Types:  x[uni] = y
 *
 * @tparam StdVec A standard vector
 * @tparam Idx Type of tail of index list.
 * @tparam U A type assignable to the value type of `StdVec`
 * @param[in] x Array variable to be assigned.
 * @param[in] y Value.
 * @param[in] name Name of variable
 * @param[in] idx uni index
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions do not match in the
 * tail assignment.
 */
template <typename StdVec, typename U, require_std_vector_t<StdVec>* = nullptr,
          require_t<std::is_assignable<value_type_t<StdVec>&, U>>* = nullptr>
inline void assign(StdVec&& x, U&& y, const char* name, index_uni idx) {
  stan::math::check_range("array[uni,...] assign", name, x.size(), idx.n_);
  x[idx.n_ - 1] = std::forward<U>(y);
}

/**
 * Assign to the elements of an std vector with additional subsetting on each
 * element.
 *
 * Types:  x[Idx1 | Idx2] = y
 *
 * @tparam T A standard vector.
 * @tparam Idx1 Type of multiple index heading index list.
 * @tparam Idx2 Type of tail of index list.
 * @tparam U A standard vector
 * @param[in] x Array variable to be assigned.
 * @param[in] y Value.
 * @param[in] name Name of variable
 * @param[in] idx1 first index
 * @param[in] idxs Remaining indices
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the size of the multiple indexing
 * and size of first dimension of value do not match, or any of
 * the recursive tail assignment dimensions do not match.
 */
template <typename T, typename Idx1, typename... Idxs, typename U,
          require_all_std_vector_t<T, U>* = nullptr,
          require_not_same_t<Idx1, index_uni>* = nullptr>
inline void assign(T&& x, U&& y, const char* name, const Idx1& idx1,
                   const Idxs&... idxs) {
  int x_idx_size = rvalue_index_size(idx1, x.size());
  // If there is a reverse min_max index or negative max index
  if (std::is_same<std::decay_t<Idx1>, index_min_max>::value
      || std::is_same<std::decay_t<Idx1>, index_max>::value) {
    if (x_idx_size == 0) {
      if (std::is_same<std::decay_t<Idx1>, index_min_max>::value) {
        stan::math::check_size_match("array[negative_min_max, ...] assign",
                                     name, 0, "right hand side", y.size());
      } else {
        stan::math::check_size_match("array[max < 1, ...] assign", name, 0,
                                     "right hand side", y.size());
      }
      return;
    }
  }
  stan::math::check_size_match("array[multi, ...] assign", name, x_idx_size,
                               "right hand side size", y.size());
  for (size_t n = 0; n < y.size(); ++n) {
    size_t i = rvalue_at(n, idx1);
    stan::math::check_range("array[multi, ...] assign", name, x.size(), i);
    if (std::is_rvalue_reference<U&&>::value) {
      assign(x[i - 1], std::move(y[n]), name, idxs...);
    } else {
      assign(x[i - 1], y[n], name, idxs...);
    }
  }
}

namespace internal {
template <typename T, T... I>
inline constexpr auto make_tuple_seq(std::integer_sequence<T, I...>) {
  return std::make_tuple(I...);
}
}  // namespace internal

/**
 * Assign one tuple to another
 * @tparam Tuple1 Tuple with the same number of elements as `Tuple2`
 * @tparam Tuple2 Tuple with the same number of elements as `Tuple1`
 * @param x A tuple with elements to be assigned to
 * @param y A tuple with elements to be assigned from
 * @param name The name of the tuple to assign to
 */
template <typename Tuple1, typename Tuple2,
          require_all_t<internal::is_tuple<Tuple1>,
                        internal::is_tuple<Tuple2>>* = nullptr>
inline void assign(Tuple1&& x, Tuple2&& y, const char* name) {
  constexpr auto t1_size = std::tuple_size<std::decay_t<Tuple1>>::value;
  stan::math::for_each(
      [name](auto&& x_sub, auto&& y_sub, auto idx_name) mutable {
        assign(std::forward<decltype(x_sub)>(x_sub),
               std::forward<decltype(y_sub)>(y_sub),
               (std::string(name) + "." + std::to_string(idx_name)).c_str());
      },
      std::forward<Tuple1>(x), std::forward<Tuple2>(y),
      internal::make_tuple_seq(std::make_index_sequence<t1_size>()));
}

}  // namespace model
}  // namespace stan
#endif

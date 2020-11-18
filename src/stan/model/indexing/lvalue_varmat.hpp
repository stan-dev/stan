#ifndef STAN_MODEL_INDEXING_LVALUE_VARMAT_HPP
#define STAN_MODEL_INDEXING_LVALUE_VARMAT_HPP

#include <stan/math/rev.hpp>
#include <stan/model/indexing/index.hpp>
#include <stan/model/indexing/index_list.hpp>
#include <stan/model/indexing/lvalue.hpp>
#include <stan/model/indexing/rvalue_at.hpp>
#include <stan/model/indexing/rvalue_index_size.hpp>
#include <type_traits>
#include <vector>

namespace stan {

namespace model {

namespace internal {
bool check_duplicate(const arena_t<std::vector<std::array<int, 2>>>& x_idx,
                     int i, int j) {
  for (size_t k = 0; k < x_idx.size(); ++k) {
    if (x_idx[k][0] == i && x_idx[k][1] == j) {
      return true;
    }
  }
  return false;
}

bool check_duplicate(const arena_t<std::vector<int>>& x_idx, int i) {
  for (size_t k = 0; k < x_idx.size(); ++k) {
    if (x_idx[k] == i) {
      return true;
    }
  }
  return false;
}

}  // namespace internal
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
 * Assign to a single element of an Eigen Vector.
 *
 * Types: vector[uni] <- scalar
 *
 * @tparam Vec `var_value` with inner Eigen type with either dynamic rows or columns, but not both.
 * @tparam U Type of value (must be assignable to T).
 * @param[in] x Vector variable to be assigned.
 * @param[in] idxs index holding which cell to assign to.
 * @param[in] y Value to assign.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If the index is out of bounds.
 */
template <typename VarVec, typename U, require_var_vector_t<VarVec>* = nullptr,
          require_var_t<U>* = nullptr,
          require_floating_point_t<value_type_t<U>>* = nullptr>
inline void assign(VarVec&& x,
                   const cons_index_list<index_uni, nil_index_list>& idxs,
                   const U& y, const char* name = "ANON", int depth = 0) {
  stan::math::check_range("var_vector[uni] assign range", name, x.size(),
                          idxs.head_.n_);
  const auto coeff_idx = idxs.head_.n_ - 1;
  double prev_val = x.val().coeffRef(coeff_idx);
  x.vi_->val_.coeffRef(coeff_idx) = y.val();
  stan::math::reverse_pass_callback([x, y, coeff_idx, prev_val]() mutable {
    x.vi_->val_.coeffRef(coeff_idx) = prev_val;
    prev_val = x.adj().coeffRef(coeff_idx);
    x.adj().coeffRef(coeff_idx) = 0.0;
    y.adj() += prev_val;
  });
}

/**
 * Assign to a non-contiguous subset of elements in a vector.
 *
 * Types:  vector[multi] <- vector
 *
 * @tparam Vec1 `var_value` with inner Eigen type with either dynamic rows or columns, but not both.
 * @tparam Vec2 `var_value` with inner Eigen type with either dynamic rows or columns, but not both.
 * @param[in] x Vector to be assigned.
 * @param[in] idxs Index holding an `std::vector` of cells to assign to.
 * @param[in] y Value vector.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the value size isn't the same as
 * the indexed size.
 */
template <typename Vec1, typename Vec2,
          require_all_var_vector_t<Vec1, Vec2>* = nullptr>
inline void assign(Vec1&& x,
                   const cons_index_list<index_multi, nil_index_list>& idxs,
                   const Vec2& y, const char* name = "ANON", int depth = 0) {
  stan::math::check_size_match("vector[multi] assign sizes", "lhs",
                               idxs.head_.ns_.size(), name, y.size());
  const auto x_size = x.size();
  const auto assign_size = idxs.head_.ns_.size();
  arena_t<std::vector<int>> x_idx;
  arena_t<std::vector<int>> y_idx;
  // Need to remove duplicates for cases like {2, 3, 2, 2}
  for (int i = assign_size - 1; i >= 0; --i) {
    if (!internal::check_duplicate(x_idx, idxs.head_.ns_[i] - 1)) {
      y_idx.push_back(i);
      x_idx.push_back(idxs.head_.ns_[i] - 1);
    }
  }
  arena_t<Eigen::Matrix<double, -1, 1>> prev_vals(x_idx.size());
  for (Eigen::Index i = 0; i < x_idx.size(); ++i) {
    stan::math::check_range("vector[multi] assign range", name, x_size,
                            x_idx[i]);
    prev_vals.coeffRef(i) = x.vi_->val_.coeffRef(x_idx[i]);
  }
  for (Eigen::Index i = 0; i < x_idx.size(); ++i) {
    x.vi_->val_.coeffRef(x_idx[i]) = y.vi_->val_.coeff(y_idx[i]);
  }
  stan::math::reverse_pass_callback([x, y, x_idx, y_idx, prev_vals]() mutable {
    for (Eigen::Index i = 0; i < x_idx.size(); ++i) {
      x.vi_->val_.coeffRef(x_idx[i]) = prev_vals.coeffRef(i);
      prev_vals.coeffRef(i) = x.adj().coeffRef(x_idx[i]);
      x.adj().coeffRef(x_idx[i]) = 0.0;
      y.adj().coeffRef(y_idx[i]) += prev_vals.coeffRef(i);
    }
  });
}

/**
 * Assign a matrix to a range of rows of the assignee matrix.
 *
 * Types:  mat[min_max] = mat
 *
 * @tparam Mat `var_value` with inner Eigen type with dynamic rows and columns.
 * @tparam Mat2 `var_value` with inner Eigen type with dynamic rows and columns.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs An index for a min_max range of rows
 * @param[in] y Value matrix.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and right-hand side matrix do not match.
 */
template <typename EigMat1, typename EigMat2,
          require_all_var_dense_dynamic_t<EigMat1, EigMat2>* = nullptr>
inline void assign(EigMat1&& x,
                   const cons_index_list<index_min_max, nil_index_list>& idxs,
                   const EigMat2& y, const char* name = "ANON", int depth = 0) {
  stan::math::check_range("matrix[min_max] max row indexing", name, x.rows(),
                          idxs.head_.max_);
  stan::math::check_range("matrix[min_max] min row indexing", name, x.rows(),
                          idxs.head_.min_);
  if (idxs.head_.is_ascending()) {
    stan::math::check_size_match("matrix[min_max] assign row sizes", "lhs",
                                 idxs.head_.min_, name, y.rows());
    x.middleRows(idxs.head_.min_ - 1, idxs.head_.max_ - 1) = y;
    return;
  } else {
    stan::math::check_size_match("matrix[reverse_min_max] assign row sizes",
                                 "lhs", idxs.head_.max_, name, y.rows());
    x.middleRows(idxs.head_.max_ - 1, idxs.head_.min_ - 1)
        = y.colwise_reverse();
    return;
  }
}

/**
 * Assign to a block of an Eigen matrix.
 *
 * Types:  mat[min_max, min_max] = mat
 *
 * @tparam Mat1 `var_value` with inner Eigen type with dynamic rows and columns.
 * @tparam Mat2 `var_value` with inner Eigen type with dynamic rows and columns.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs An index list containing two min_max indices
 * @param[in] y Matrix variable to assign from.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and right-hand side matrix do not match.
 */
template <typename Mat1, typename Mat2,
          require_all_var_dense_dynamic_t<Mat1, Mat2>* = nullptr>
inline void assign(
    Mat1&& x,
    const cons_index_list<index_min_max,
                          cons_index_list<index_min_max, nil_index_list>>& idxs,
    const Mat2& y, const char* name = "ANON", int depth = 0) {
  if (idxs.head_.is_ascending()) {
    if (idxs.tail_.head_.is_ascending()) {
      auto row_size = idxs.head_.max_ - (idxs.head_.min_ - 1);
      auto col_size = idxs.tail_.head_.max_ - (idxs.tail_.head_.min_ - 1);
      stan::math::check_range("matrix[min_max, min_max] assign col range", name,
                              x.cols(), idxs.head_.min_);
      stan::math::check_range("matrix[min_max, min_max] assign row range", name,
                              x.rows(), idxs.tail_.head_.min_);
      stan::math::check_size_match("matrix[min_max, min_max] assign row sizes",
                                   "lhs", row_size, name, y.rows());
      stan::math::check_size_match("matrix[min_max, min_max] assign col sizes",
                                   "lhs", col_size, name, y.cols());
      x.block(idxs.head_.min_ - 1, idxs.tail_.head_.min_ - 1, row_size,
              col_size)
          = y;
      return;
    } else {
      auto row_size = idxs.head_.max_ - (idxs.head_.min_ - 1);
      auto col_size = idxs.tail_.head_.min_ - (idxs.tail_.head_.max_ - 1);
      stan::math::check_range(
          "matrix[min_max, reverse_min_max] assign col range", name, x.cols(),
          idxs.head_.min_);
      stan::math::check_range(
          "matrix[min_max, reverse_min_max] assign row range", name, x.rows(),
          idxs.tail_.head_.max_);
      stan::math::check_size_match(
          "matrix[min_max, reverse_min_max] assign row sizes", "lhs", row_size,
          name, y.rows());
      stan::math::check_size_match(
          "matrix[min_max, reverse_min_max] assign col sizes", "lhs", col_size,
          name, y.cols());
      x.block(idxs.head_.min_ - 1, idxs.tail_.head_.max_ - 1, row_size,
              col_size)
          = y.rowwise_reverse();
      return;
    }
  } else {
    if (idxs.tail_.head_.is_ascending()) {
      auto row_size = idxs.head_.min_ - (idxs.head_.max_ - 1);
      auto col_size = idxs.tail_.head_.max_ - (idxs.tail_.head_.min_ - 1);
      stan::math::check_range(
          "matrix[reverse_min_max, min_max] assign col range", name, x.cols(),
          idxs.head_.max_);
      stan::math::check_range(
          "matrix[reverse_min_max, min_max] assign row range", name, x.rows(),
          idxs.tail_.head_.min_);
      stan::math::check_size_match(
          "matrix[reverse_min_max, min_max] assign row sizes", "lhs", row_size,
          name, y.rows());
      stan::math::check_size_match(
          "matrix[reverse_min_max, min_max] assign col sizes", "lhs", col_size,
          name, y.cols());
      x.block(idxs.head_.max_ - 1, idxs.tail_.head_.min_ - 1, row_size,
              col_size)
          = y.colwise_reverse();
      return;
    } else {
      auto row_size = idxs.head_.min_ - (idxs.head_.max_ - 1);
      auto col_size = idxs.tail_.head_.min_ - (idxs.tail_.head_.max_ - 1);
      stan::math::check_range(
          "matrix[reverse_min_max, reverse_min_max] assign col range", name,
          x.cols(), idxs.head_.max_);
      stan::math::check_range(
          "matrix[reverse_min_max, reverse_min_max] assign row range", name,
          x.rows(), idxs.tail_.head_.max_);
      stan::math::check_size_match(
          "matrix[reverse_min_max, reverse_min_max] assign row sizes", "lhs",
          row_size, name, y.rows());
      stan::math::check_size_match(
          "matrix[reverse_min_max, reverse_min_max] assign col sizes", "lhs",
          col_size, name, y.cols());
      x.block(idxs.head_.max_ - 1, idxs.tail_.head_.max_ - 1, row_size,
              col_size)
          = y.reverse();
      return;
    }
  }
}

/**
 * Assign to a cell of an Eigen Matrix.
 *
 * Types:  mat[single, single] = scalar
 *
 * @tparam Mat `var_value` with inner Eigen type with dynamic rows and columns.
 * @tparam U Scalar type.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs Sequence of two single indexes (from 1).
 * @param[in] y Value scalar.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If either of the indices are out of bounds.
 */
template <typename Mat, typename U, require_var_dense_dynamic_t<Mat>* = nullptr,
          require_var_t<U>* = nullptr>
inline void assign(
    Mat&& x,
    const cons_index_list<index_uni,
                          cons_index_list<index_uni, nil_index_list>>& idxs,
    const U& y, const char* name = "ANON", int depth = 0) {
  const int row_idx = idxs.head_.n_ - 1;
  const int col_idx = idxs.tail_.head_.n_ - 1;
  stan::math::check_range("matrix[uni,uni] assign range", name, x.rows(),
                          row_idx + 1);
  stan::math::check_range("matrix[uni,uni] assign range", name, x.cols(),
                          col_idx + 1);
  double prev_val = x.val().coeffRef(row_idx, col_idx);
  x.vi_->val_.coeffRef(row_idx, col_idx) = y.val();
  stan::math::reverse_pass_callback(
      [x, y, row_idx, col_idx, prev_val]() mutable {
        x.vi_->val_.coeffRef(row_idx, col_idx) = prev_val;
        prev_val = x.adj().coeffRef(row_idx, col_idx);
        x.adj().coeffRef(row_idx, col_idx) = 0.0;
        y.adj() += prev_val;
      });
}

/**
 * Assign multiple possibly unordered cells of row vector to a row of an eigen
 * matrix.
 *
 * Types:  mat[uni, multi] = row_vector
 *
 * @tparam Mat1 `var_value` with inner Eigen type with dynamic rows and columns.
 * @tparam Vec `var_value` with inner Eigen type with dynamic columns and compile time rows of 1.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs A list with a uni index for rows and multi index for columns.
 * @param[in] y Row vector.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and value matrix do not match.
 */
template <typename Mat1, typename Vec,
          require_var_dense_dynamic_t<Mat1>* = nullptr,
          require_eigen_dense_dynamic_t<value_type_t<Mat1>>* = nullptr,
          require_var_row_vector_t<Vec>* = nullptr>
inline void assign(
    Mat1&& x,
    const cons_index_list<index_uni,
                          cons_index_list<index_multi, nil_index_list>>& idxs,
    const Vec& y, const char* name = "ANON", int depth = 0) {
  stan::math::check_range("matrix[uni, multi] assign range", name, x.cols(),
                          idxs.head_.n_);
  const auto assign_cols = idxs.tail_.head_.ns_.size();
  stan::math::check_size_match("matrix[uni, multi] assign sizes", "lhs",
                               assign_cols, name, y.size());
  const int row_idx = idxs.head_.n_ - 1;
  arena_t<std::vector<int>> x_idx;
  arena_t<std::vector<int>> y_idx;
  // Need to remove duplicates for cases like {2, 3, 2, 2}
  for (int i = assign_cols - 1; i >= 0; --i) {
    if (!internal::check_duplicate(x_idx, idxs.tail_.head_.ns_[i] - 1)) {
      y_idx.push_back(i);
      stan::math::check_range("matrix[uni, multi] assign range", name, x.cols(),
                              idxs.tail_.head_.ns_[i]);
      x_idx.push_back(idxs.tail_.head_.ns_[i] - 1);
    }
  }
  arena_t<Eigen::Matrix<double, -1, 1>> prev_val(x_idx.size());
  for (size_t i = 0; i < x_idx.size(); ++i) {
    prev_val.coeffRef(i) = x.val().coeffRef(row_idx, x_idx[i]);
    x.vi_->val_.coeffRef(row_idx, x_idx[i]) = y.val().coeff(y_idx[i]);
  }
  stan::math::reverse_pass_callback(
      [x, y, row_idx, x_idx, y_idx, prev_val]() mutable {
        for (size_t i = 0; i < x_idx.size(); ++i) {
          x.vi_->val_.coeffRef(row_idx, x_idx[i]) = prev_val.coeff(i);
          prev_val.coeffRef(i) = x.adj().coeffRef(row_idx, x_idx[i]);
          x.adj().coeffRef(row_idx, x_idx[i]) = 0.0;
          y.adj().coeffRef(y_idx[i]) += prev_val.coeffRef(i);
        }
      });
}

/**
 * Assign to multiple possibly unordered rows of a matrix from an input
 * matrix.
 *
 * Types:  mat[multi] = mat
 *
 * @tparam Mat1 `var_value` with inner Eigen type with dynamic rows and columns.
 * @tparam Mat2 `var_value` with inner Eigen type with dynamic rows and columns.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs Pair of multiple indexes (from 1).
 * @param[in] y Value matrix.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and value matrix do not match.
 */
template <typename Mat1, typename Mat2,
          require_all_var_dense_dynamic_t<Mat1, Mat2>* = nullptr>
inline void assign(
    Mat1&& x,
    const cons_index_list<index_multi, nil_index_list>& idxs,
    const Mat2& y, const char* name = "ANON", int depth = 0) {
  const auto assign_rows = idxs.head_.ns_.size();
  stan::math::check_size_match("matrix[multi,multi] assign sizes", "lhs",
                               assign_rows, name, y.rows());
  arena_t<std::vector<int>> x_idx;
  arena_t<std::vector<int>> y_idx;
  x_idx.reserve(assign_rows);
  y_idx.reserve(assign_rows);
  // Need to remove duplicates for cases like {2, 3, 2, 2}
  for (int i = assign_rows - 1; i >= 0; --i) {
    if (!internal::check_duplicate(x_idx, idxs.head_.ns_[i] - 1)) {
      y_idx.push_back(i);
      stan::math::check_range("matrix[multi, multi] assign row range", name,
                              x.rows(), idxs.head_.ns_[i]);
      x_idx.push_back(idxs.head_.ns_[i] - 1);
    }
  }
  arena_t<Eigen::Matrix<double, -1, -1>> prev_vals(x_idx.size(), x.cols());
  for (size_t i = 0; i < y_idx.size(); ++i) {
    prev_vals.row(i) = x.vi_->val_.row(x_idx[i]);
    x.vi_->val_.row(x_idx[i]) = y.vi_->val_.row(y_idx[i]);
  }
  stan::math::reverse_pass_callback([x, y, prev_vals, x_idx, y_idx]() mutable {
    for (size_t i = 0; i < y_idx.size(); ++i) {
      x.vi_->val_.row(x_idx[i]) = prev_vals.row(i);
      prev_vals.row(i) = x.adj().row(x_idx[i]);
      x.adj().row(x_idx[i]) = 0;
      y.adj().row(y_idx[i]) += prev_vals.row(i);
    }
  });
}

/**
 * Assign to multiple possibly unordered cell's of a matrix from an input
 * matrix.
 *
 * Types:  mat[multi, multi] = mat
 *
 * @tparam Mat1 `var_value` with inner Eigen type with dynamic rows and columns.
 * @tparam Mat2 `var_value` with inner Eigen type with dynamic rows and columns.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs Pair of multiple indexes (from 1).
 * @param[in] y Value matrix.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and value matrix do not match.
 */
template <typename Mat1, typename Mat2,
          require_all_var_dense_dynamic_t<Mat1, Mat2>* = nullptr>
inline void assign(
    Mat1&& x,
    const cons_index_list<index_multi,
                          cons_index_list<index_multi, nil_index_list>>& idxs,
    const Mat2& y, const char* name = "ANON", int depth = 0) {
  const auto assign_rows = idxs.head_.ns_.size();
  const auto assign_cols = idxs.tail_.head_.ns_.size();
  stan::math::check_size_match("matrix[multi,multi] assign sizes", "lhs",
                               assign_rows, name, y.rows());
  stan::math::check_size_match("matrix[multi,multi] assign sizes", "lhs",
                               assign_cols, name, y.cols());
  arena_t<std::vector<std::array<int, 2>>> x_idx;
  arena_t<std::vector<std::array<int, 2>>> y_idx;
  x_idx.reserve(assign_rows * assign_cols);
  y_idx.reserve(assign_rows * assign_cols);
  // Need to remove duplicates for cases like {2, 3, 2, 2}
  for (int j = assign_cols - 1; j >= 0; --j) {
    for (int i = assign_rows - 1; i >= 0; --i) {
      if (!internal::check_duplicate(x_idx, idxs.head_.ns_[i] - 1,
                                     idxs.tail_.head_.ns_[j] - 1)) {
        y_idx.push_back(std::array<int, 2>{i, j});
        stan::math::check_range("matrix[multi, multi] assign row range", name,
                                x.rows(), idxs.head_.ns_[i]);
        stan::math::check_range("matrix[multi, multi] assign col range", name,
                                x.cols(), idxs.tail_.head_.ns_[j]);
        x_idx.push_back(std::array<int, 2>{idxs.head_.ns_[i] - 1,
                                           idxs.tail_.head_.ns_[j] - 1});
      }
    }
  }
  arena_t<Eigen::Matrix<double, -1, 1>> prev_vals(x_idx.size());
  for (size_t i = 0; i < y_idx.size(); ++i) {
    prev_vals.coeffRef(i) = x.vi_->val_(x_idx[i][0], x_idx[i][1]);
    x.vi_->val_(x_idx[i][0], x_idx[i][1])
        = y.vi_->val_(y_idx[i][0], y_idx[i][1]);
  }
  stan::math::reverse_pass_callback([x, y, prev_vals, x_idx, y_idx]() mutable {
    for (size_t i = 0; i < y_idx.size(); ++i) {
      x.vi_->val_(x_idx[i][0], x_idx[i][1]) = prev_vals.coeffRef(i);
      prev_vals.coeffRef(i) = x.adj()(x_idx[i][0], x_idx[i][1]);
      x.adj()(x_idx[i][0], x_idx[i][1]) = 0;
      y.adj()(y_idx[i][0], y_idx[i][1]) += prev_vals.coeffRef(i);
    }
  });
}

/**
 * Assign to a non-contiguous set of columns of a matrix.
 *
 * Types:  mat[Idx, multi] = mat
 *
 * @tparam Mat1 `var_value` with inner Eigen type with dynamic rows and columns.
 * @tparam Mat2 `var_value` with inner Eigen type
 * @tparam Idx The row index type
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs Pair of multiple indexes (from 1).
 * @param[in] y Value matrix.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and value matrix do not match.
 */
template <typename Mat1, typename Mat2, typename Idx,
          require_all_var_dense_dynamic_t<Mat1, Mat2>* = nullptr>
inline void assign(
    Mat1&& x,
    const cons_index_list<Idx, cons_index_list<index_multi, nil_index_list>>&
        idxs,
    const Mat2& y, const char* name = "ANON", int depth = 0) {
  const auto assign_cols = idxs.tail_.head_.ns_.size();
  stan::math::check_size_match("matrix[..., multi] assign sizes", "lhs",
                               assign_cols, name, y.cols());
  std::vector<int> x_idx;
  std::vector<int> y_idx;
  x_idx.reserve(assign_cols);
  y_idx.reserve(assign_cols);
  // Need to remove duplicates for cases like {2, 3, 2, 2}
  for (int j = assign_cols - 1; j >= 0; --j) {
    if (!internal::check_duplicate(x_idx, idxs.tail_.head_.ns_[j] - 1)) {
      y_idx.push_back(j);
      stan::math::check_range("matrix[multi, multi] assign col range", name,
                              x.cols(), idxs.tail_.head_.ns_[j]);
      x_idx.push_back(idxs.tail_.head_.ns_[j] - 1);
    }
  }
  for (int j = 0; j < y_idx.size(); ++j) {
    assign(x.col(x_idx[j]), index_list(idxs.head_), y.col(y_idx[j]), name,
           depth + 1);
  }
}

/**
 * Assign to any rows and a range of columns.
 *
 * Types:  mat[Idx, min_max] = mat
 *
 * @tparam Mat1 `var_value` with inner Eigen type with dynamic rows and columns.
 * @tparam Mat2 `var_value` with inner Eigen type
 * @tparam Idx The row index type
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs Container holding row index and a min_max index.
 * @param[in] y Matrix variable to assign from.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and right-hand side matrix do not match.
 */
template <typename Mat1, typename Mat2, typename Idx,
          require_var_dense_dynamic_t<Mat1>* = nullptr>
inline void assign(
    Mat1&& x,
    const cons_index_list<Idx, cons_index_list<index_min_max, nil_index_list>>&
        idxs,
    const Mat2& y, const char* name = "ANON", int depth = 0) {
  if (idxs.tail_.head_.is_ascending()) {
    const auto col_start = idxs.tail_.head_.min_ - 1;
    const auto col_size = idxs.tail_.head_.max_ - col_start;
    stan::math::check_range("matrix[..., min_max] assign range", name, x.cols(),
                            idxs.tail_.head_.min_);
    stan::math::check_range("matrix[..., min_max] assign range", name,
                            idxs.tail_.head_.max_, x.cols());
    stan::math::check_size_match("matrix[..., min_max] assign col size", "lhs",
                                 idxs.tail_.head_.max_, name, x.cols());
    assign(x.middleCols(col_start, col_size), index_list(idxs.head_), y, name,
           depth + 1);
    return;
  } else {
    const auto col_start = idxs.tail_.head_.max_ - 1;
    const auto col_size = idxs.tail_.head_.min_ - col_start;
    stan::math::check_range("matrix[..., reverse_min_max] assign range", name,
                            x.cols(), idxs.tail_.head_.max_);
    stan::math::check_range("matrix[..., reverse_min_max] assign range", name,
                            idxs.tail_.head_.min_, x.cols());
    stan::math::check_size_match("matrix[..., min_max] assign col size", "lhs",
                                 idxs.tail_.head_.min_, name, x.cols());
    assign(x.middleCols(col_start, col_size), index_list(idxs.head_),
           y.rowwise_reverse(), name, depth + 1);
    return;
  }
}

}  // namespace model
}  // namespace stan
#endif

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
#include <unordered_set>

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
 * Assign to a single element of an Eigen Vector.
 *
 * Types: vector[uni] <- scalar
 *
 * @tparam Vec `var_value` with inner Eigen type with either dynamic rows or
 * columns, but not both.
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
  stan::math::check_range("var_vector[uni] assign", name, x.size(),
                          idxs.head_.n_);
  const auto coeff_idx = idxs.head_.n_ - 1;
  double prev_val = x.val().coeffRef(coeff_idx);
  x.vi_->val_.coeffRef(coeff_idx) = y.val();
  stan::math::reverse_pass_callback([x, y, coeff_idx, prev_val]() mutable {
    x.vi_->val_.coeffRef(coeff_idx) = prev_val;
    y.adj() += x.adj().coeffRef(coeff_idx);
    x.adj().coeffRef(coeff_idx) = 0.0;
  });
}

/**
 * Assign to a non-contiguous subset of elements in a vector.
 *
 * Types:  vector[multi] <- vector
 *
 * @tparam Vec1 `var_value` with inner Eigen type with either dynamic rows or
 * columns, but not both.
 * @tparam Vec2 `var_value` with inner Eigen type with either dynamic rows or
 * columns, but not both.
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
  stan::math::check_size_match("vector[multi] assign", "left hand side",
                               idxs.head_.ns_.size(), name, y.size());
  const auto x_size = x.size();
  const auto assign_size = idxs.head_.ns_.size();
  arena_t<std::vector<int>> x_idx(assign_size);
  arena_t<Eigen::Matrix<double, -1, 1>> prev_vals(assign_size);
  Eigen::Matrix<double, -1, 1> y_vals(assign_size);
  std::unordered_set<int> x_set;
  x_set.reserve(assign_size);
  // We have to use two loops to avoid aliasing issues.
  for (int i = assign_size - 1; i >= 0; --i) {
    if (likely(x_set.insert(idxs.head_.ns_[i]).second)) {
      stan::math::check_range("vector[multi] assign", name, x_size,
                              idxs.head_.ns_[i]);
      x_idx[i] = idxs.head_.ns_[i] - 1;
      prev_vals.coeffRef(i) = x.vi_->val_.coeffRef(x_idx[i]);
      y_vals.coeffRef(i) = y.vi_->val_.coeff(i);
    } else {
      x_idx[i] = -1;
    }
  }
  for (int i = assign_size - 1; i >= 0; --i) {
    if (likely(x_idx[i] != -1)) {
      x.vi_->val_.coeffRef(x_idx[i]) = y_vals.coeff(i);
    }
  }

  stan::math::reverse_pass_callback([x, y, x_idx, prev_vals]() mutable {
    for (Eigen::Index i = 0; i < x_idx.size(); ++i) {
      if (likely(x_idx[i] != -1)) {
        x.vi_->val_.coeffRef(x_idx[i]) = prev_vals.coeffRef(i);
        prev_vals.coeffRef(i) = x.adj().coeffRef(x_idx[i]);
        x.adj().coeffRef(x_idx[i]) = 0.0;
      }
    }
    for (Eigen::Index i = 0; i < x_idx.size(); ++i) {
      if (likely(x_idx[i] != -1)) {
        y.adj().coeffRef(i) += prev_vals.coeffRef(i);
      }
    }
  });
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
          require_var_t<U>* = nullptr,
          require_floating_point_t<value_type_t<U>>* = nullptr>
inline void assign(
    Mat&& x,
    const cons_index_list<index_uni,
                          cons_index_list<index_uni, nil_index_list>>& idxs,
    const U& y, const char* name = "ANON", int depth = 0) {
  stan::math::check_range("matrix[uni,uni] assign", name, x.rows(),
                          idxs.head_.n_);
  stan::math::check_range("matrix[uni,uni] assign", name, x.cols(),
                          idxs.tail_.head_.n_);
  const int row_idx = idxs.head_.n_ - 1;
  const int col_idx = idxs.tail_.head_.n_ - 1;
  double prev_val = x.val().coeffRef(row_idx, col_idx);
  x.vi_->val_.coeffRef(row_idx, col_idx) = y.val();
  stan::math::reverse_pass_callback(
      [x, y, row_idx, col_idx, prev_val]() mutable {
        x.vi_->val_.coeffRef(row_idx, col_idx) = prev_val;
        y.adj() += x.adj().coeffRef(row_idx, col_idx);
        x.adj().coeffRef(row_idx, col_idx) = 0.0;
      });
}

/**
 * Assign multiple possibly unordered cells of row vector to a row of an eigen
 * matrix.
 *
 * Types:  mat[uni, multi] = row_vector
 *
 * @tparam Mat1 `var_value` with inner Eigen type with dynamic rows and columns.
 * @tparam Vec `var_value` with inner Eigen type with dynamic columns and
 * compile time rows of 1.
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
  stan::math::check_range("matrix[uni, multi] assign", name, x.rows(),
                          idxs.head_.n_);
  const auto assign_cols = idxs.tail_.head_.ns_.size();
  stan::math::check_size_match("matrix[uni, multi] assign", "left hand side",
                               assign_cols, name, y.size());
  const int row_idx = idxs.head_.n_ - 1;
  arena_t<std::vector<int>> x_idx(assign_cols);
  arena_t<Eigen::Matrix<double, -1, 1>> prev_val(assign_cols);
  Eigen::Matrix<double, -1, 1> y_vals(assign_cols);
  std::unordered_set<int> x_set;
  x_set.reserve(assign_cols);
  // Need to remove duplicates for cases like {2, 3, 2, 2}
  for (int i = assign_cols - 1; i >= 0; --i) {
    if (likely(x_set.insert(idxs.tail_.head_.ns_[i]).second)) {
      stan::math::check_range("matrix[uni, multi] assign", name, x.cols(),
                              idxs.tail_.head_.ns_[i]);
      x_idx[i] = idxs.tail_.head_.ns_[i] - 1;
      prev_val.coeffRef(i) = x.val().coeffRef(row_idx, x_idx[i]);
      y_vals.coeffRef(i) = y.val().coeff(i);
    } else {
      x_idx[i] = -1;
    }
  }
  for (int i = assign_cols - 1; i >= 0; --i) {
    if (likely(x_idx[i] != -1)) {
      x.vi_->val_.coeffRef(row_idx, x_idx[i]) = y_vals.coeff(i);
    }
  }
  stan::math::reverse_pass_callback([x, y, row_idx, x_idx, prev_val]() mutable {
    for (size_t i = 0; i < x_idx.size(); ++i) {
      if (likely(x_idx[i] != -1)) {
        x.vi_->val_.coeffRef(row_idx, x_idx[i]) = prev_val.coeff(i);
        prev_val.coeffRef(i) = x.adj().coeffRef(row_idx, x_idx[i]);
        x.adj().coeffRef(row_idx, x_idx[i]) = 0.0;
      }
    }
    for (size_t i = 0; i < x_idx.size(); ++i) {
      if (likely(x_idx[i] != -1)) {
        y.adj().coeffRef(i) += prev_val.coeffRef(i);
      }
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
inline void assign(Mat1&& x,
                   const cons_index_list<index_multi, nil_index_list>& idxs,
                   const Mat2& y, const char* name = "ANON", int depth = 0) {
  const auto assign_rows = idxs.head_.ns_.size();
  stan::math::check_size_match("matrix[multi] assign", "left hand side rows",
                               assign_rows, name, y.rows());
  stan::math::check_size_match("matrix[multi] assign", "left hand side columns",
                               x.cols(), name, y.cols());
  arena_t<std::vector<int>> x_idx(assign_rows);
  arena_t<Eigen::Matrix<double, -1, -1>> prev_vals(assign_rows, x.cols());
  Eigen::Matrix<double, -1, -1> y_vals(assign_rows, x.cols());
  std::unordered_set<int> x_set;
  x_set.reserve(assign_rows);
  // Need to remove duplicates for cases like {2, 3, 2, 2}
  for (int i = assign_rows - 1; i >= 0; --i) {
    if (likely(x_set.insert(idxs.head_.ns_[i]).second)) {
      stan::math::check_range("matrix[multi, multi] assign row", name, x.rows(),
                              idxs.head_.ns_[i]);
      x_idx[i] = idxs.head_.ns_[i] - 1;
      prev_vals.row(i) = x.vi_->val_.row(x_idx[i]);
      y_vals.row(i) = y.vi_->val_.row(i);
    } else {
      x_idx[i] = -1;
    }
  }
  for (int i = assign_rows - 1; i >= 0; --i) {
    if (likely(x_idx[i] != -1)) {
      x.vi_->val_.row(x_idx[i]) = y_vals.row(i);
    }
  }

  stan::math::reverse_pass_callback([x, y, prev_vals, x_idx]() mutable {
    for (size_t i = 0; i < x_idx.size(); ++i) {
      if (likely(x_idx[i] != -1)) {
        x.vi_->val_.row(x_idx[i]) = prev_vals.row(i);
        prev_vals.row(i) = x.adj().row(x_idx[i]);
        x.adj().row(x_idx[i]).fill(0);
      }
    }
    for (size_t i = 0; i < x_idx.size(); ++i) {
      if (likely(x_idx[i] != -1)) {
        y.adj().row(i) += prev_vals.row(i);
      }
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
          require_all_var_matrix_t<Mat1, Mat2>* = nullptr>
inline void assign(
    Mat1&& x,
    const cons_index_list<index_multi,
                          cons_index_list<index_multi, nil_index_list>>& idxs,
    const Mat2& y, const char* name = "ANON", int depth = 0) {
  const auto assign_rows = idxs.head_.ns_.size();
  const auto assign_cols = idxs.tail_.head_.ns_.size();
  stan::math::check_size_match("matrix[multi,multi] assign", "left hand side",
                               assign_rows, name, y.rows());
  stan::math::check_size_match("matrix[multi,multi] assign", "left hand side",
                               assign_cols, name, y.cols());
  using arena_vec = std::vector<int, stan::math::arena_allocator<int>>;
  using pair_type = std::pair<int, arena_vec>;
  arena_vec x_col_idx(assign_cols);
  arena_vec x_row_idx(assign_rows);
  std::unordered_set<int> x_set;
  x_set.reserve(assign_cols);
  std::unordered_set<int> x_row_set;
  x_row_set.reserve(assign_rows);
  arena_t<Eigen::Matrix<double, -1, -1>> prev_vals(assign_rows, assign_cols);
  Eigen::Matrix<double, -1, -1> y_vals(assign_rows, assign_cols);
  // Need to remove duplicates for cases like {{2, 3, 2, 2}, {1, 2, 2}}
  for (int i = assign_rows - 1; i >= 0; --i) {
    if (likely(x_row_set.insert(idxs.head_.ns_[i]).second)) {
      stan::math::check_range("matrix[multi, multi] assign row", name, x.rows(),
                              idxs.head_.ns_[i]);
      x_row_idx[i] = idxs.head_.ns_[i] - 1;
    } else {
      x_row_idx[i] = -1;
    }
  }
  for (int j = assign_cols - 1; j >= 0; --j) {
    if (likely(x_set.insert(idxs.tail_.head_.ns_[j]).second)) {
      stan::math::check_range("matrix[multi, multi] assign col", name, x.cols(),
                              idxs.tail_.head_.ns_[j]);
      x_col_idx[j] = idxs.tail_.head_.ns_[j] - 1;
      for (int i = assign_rows - 1; i >= 0; --i) {
        if (likely(x_row_idx[i] != -1)) {
          prev_vals.coeffRef(i, j) = x.vi_->val_(x_row_idx[i], x_col_idx[j]);
          y_vals(i, j) = y.vi_->val_(i, j);
        }
      }
    } else {
      x_col_idx[j] = -1;
    }
  }
  for (int j = assign_cols - 1; j >= 0; --j) {
    if (likely(x_col_idx[j] != -1)) {
      for (int i = assign_rows - 1; i >= 0; --i) {
        if (likely(x_row_idx[i] != -1)) {
          x.vi_->val_(x_row_idx[i], x_col_idx[j]) = y_vals(i, j);
        }
      }
    }
  }
  stan::math::reverse_pass_callback([x, y, prev_vals, x_col_idx,
                                     x_row_idx]() mutable {
    for (int j = 0; j < x_col_idx.size(); ++j) {
      if (likely(x_col_idx[j] != -1)) {
        for (int i = 0; i < x_row_idx.size(); ++i) {
          if (likely(x_row_idx[i] != -1)) {
            x.vi_->val_(x_row_idx[i], x_col_idx[j]) = prev_vals.coeffRef(i, j);
            prev_vals.coeffRef(i, j) = x.adj()(x_row_idx[i], x_col_idx[j]);
            x.adj()(x_row_idx[i], x_col_idx[j]) = 0;
          }
        }
      }
    }
    for (int j = 0; j < x_col_idx.size(); ++j) {
      if (likely(x_col_idx[j] != -1)) {
        for (int i = 0; i < x_row_idx.size(); ++i) {
          if (likely(x_row_idx[i] != -1)) {
            y.adj()(i, j) += prev_vals.coeffRef(i, j);
          }
        }
      }
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
  stan::math::check_size_match("matrix[..., multi] assign", "left hand side",
                               assign_cols, name, y.cols());
  std::unordered_set<int> x_set;
  auto y_eval = y.eval();
  x_set.reserve(assign_cols);
  // Need to remove duplicates for cases like {2, 3, 2, 2}
  for (int j = assign_cols - 1; j >= 0; --j) {
    if (likely(x_set.insert(idxs.tail_.head_.ns_[j]).second)) {
      stan::math::check_range("matrix[..., multi] assign col", name, x.cols(),
                              idxs.tail_.head_.ns_[j]);
      assign(x.col(idxs.tail_.head_.ns_[j] - 1), index_list(idxs.head_),
             y_eval.col(j), name, depth + 1);
    }
  }
}

}  // namespace model
}  // namespace stan
#endif

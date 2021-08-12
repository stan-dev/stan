#ifndef STAN_MODEL_INDEXING_ASSIGN_VARMAT_HPP
#define STAN_MODEL_INDEXING_ASSIGN_VARMAT_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/fun/adjoint_of.hpp>
#include <stan/model/indexing/access_helpers.hpp>
#include <stan/model/indexing/index.hpp>
#include <type_traits>
#include <vector>
#include <unordered_set>

namespace stan {

namespace model {

namespace internal {
template <typename T>
using require_var_matrix_or_arithmetic_eigen
    = require_any_t<is_var_matrix<T>, stan::math::disjunction<
                                          std::is_arithmetic<scalar_type_t<T>>,
                                          is_eigen_matrix_dynamic<T>>>;

template <typename T>
using require_var_vector_or_arithmetic_eigen
    = require_any_t<is_var_vector<T>, stan::math::disjunction<
                                          std::is_arithmetic<scalar_type_t<T>>,
                                          is_eigen_vector<T>>>;

template <typename T>
using require_var_row_vector_or_arithmetic_eigen = require_any_t<
    is_var_row_vector<T>,
    stan::math::disjunction<std::is_arithmetic<scalar_type_t<T>>,
                            is_eigen_row_vector<T>>>;

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
 * @tparam Vec `var_value` with inner Eigen type with either dynamic rows or
 * columns, but not both.
 * @tparam U Type of value (must be assignable to T).
 * @param[in] x Vector variable to be assigned.
 * @param[in] y Value to assign.
 * @param[in] name Name of variable
 * @param[in] idx index holding which cell to assign to.
 * @throw std::out_of_range If the index is out of bounds.
 */
template <typename VarVec, typename U, require_var_vector_t<VarVec>* = nullptr,
          require_stan_scalar_t<U>* = nullptr>
inline void assign(VarVec&& x, const U& y, const char* name, index_uni idx) {
  stan::math::check_range("var_vector[uni] assign", name, x.size(), idx.n_);
  const auto coeff_idx = idx.n_ - 1;
  double prev_val = x.val().coeffRef(coeff_idx);
  x.vi_->val_.coeffRef(coeff_idx) = stan::math::value_of(y);
  stan::math::reverse_pass_callback([x, y, coeff_idx, prev_val]() mutable {
    x.vi_->val_.coeffRef(coeff_idx) = prev_val;
    if (!is_constant<U>::value) {
      math::adjoint_of(y) += x.adj().coeffRef(coeff_idx);
    }
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
 * @param[in] y Value vector.
 * @param[in] name Name of variable
 * @param[in] idx Index holding an `std::vector` of cells to assign to.
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the value size isn't the same as
 * the indexed size.
 */
template <typename Vec1, typename Vec2, require_var_vector_t<Vec1>* = nullptr,
          internal::require_var_vector_or_arithmetic_eigen<Vec2>* = nullptr>
inline void assign(Vec1&& x, const Vec2& y, const char* name,
                   const index_multi& idx) {
  stan::math::check_size_match("vector[multi] assign", "left hand side",
                               idx.ns_.size(), name, y.size());
  const auto x_size = x.size();
  const auto assign_size = idx.ns_.size();
  arena_t<std::vector<int>> x_idx(assign_size);
  arena_t<Eigen::Matrix<double, -1, 1>> prev_vals(assign_size);
  Eigen::Matrix<double, -1, 1> y_idx_vals(assign_size);
  std::unordered_set<int> x_set;
  x_set.reserve(assign_size);
  const auto& y_val = stan::math::value_of(y);
  // We have to use two loops to avoid aliasing issues.
  for (int i = assign_size - 1; i >= 0; --i) {
    if (likely(x_set.insert(idx.ns_[i]).second)) {
      stan::math::check_range("vector[multi] assign", name, x_size, idx.ns_[i]);
      x_idx[i] = idx.ns_[i] - 1;
      prev_vals.coeffRef(i) = x.vi_->val_.coeffRef(x_idx[i]);
      y_idx_vals.coeffRef(i) = y_val.coeff(i);
    } else {
      x_idx[i] = -1;
    }
  }
  for (int i = assign_size - 1; i >= 0; --i) {
    if (likely(x_idx[i] != -1)) {
      x.vi_->val_.coeffRef(x_idx[i]) = y_idx_vals.coeff(i);
    }
  }

  if (!is_constant<Vec2>::value) {
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
          math::forward_as<math::promote_scalar_t<math::var, Vec2>>(y)
              .adj()
              .coeffRef(i)
              += prev_vals.coeff(i);
        }
      }
    });
  } else {
    stan::math::reverse_pass_callback([x, x_idx, prev_vals]() mutable {
      for (Eigen::Index i = 0; i < x_idx.size(); ++i) {
        if (likely(x_idx[i] != -1)) {
          x.vi_->val_.coeffRef(x_idx[i]) = prev_vals.coeff(i);
          prev_vals.coeffRef(i) = x.adj().coeff(x_idx[i]);
          x.adj().coeffRef(x_idx[i]) = 0.0;
        }
      }
    });
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
 * @param[in] y Value scalar.
 * @param[in] name Name of variable
 * @param[in] row_idx uni index for selecting rows
 * @param[in] col_idx uni index for selecting columns
 * @throw std::out_of_range If either of the indices are out of bounds.
 */
template <typename Mat, typename U, require_var_dense_dynamic_t<Mat>* = nullptr,
          require_stan_scalar_t<U>* = nullptr>
inline void assign(Mat&& x, const U& y, const char* name, index_uni row_idx,
                   index_uni col_idx) {
  stan::math::check_range("matrix[uni,uni] assign", name, x.rows(), row_idx.n_);
  stan::math::check_range("matrix[uni,uni] assign", name, x.cols(), col_idx.n_);
  const int row_idx_val = row_idx.n_ - 1;
  const int col_idx_val = col_idx.n_ - 1;
  double prev_val = x.val().coeffRef(row_idx_val, col_idx_val);
  x.vi_->val_.coeffRef(row_idx_val, col_idx_val) = stan::math::value_of(y);
  stan::math::reverse_pass_callback(
      [x, y, row_idx_val, col_idx_val, prev_val]() mutable {
        x.vi_->val_.coeffRef(row_idx_val, col_idx_val) = prev_val;
        if (!is_constant<U>::value) {
          math::adjoint_of(y) += x.adj().coeff(row_idx_val, col_idx_val);
        }
        x.adj().coeffRef(row_idx_val, col_idx_val) = 0.0;
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
 * @param[in] y Row vector.
 * @param[in] name Name of variable
 * @param[in] row_idx uni index for selecting rows
 * @param[in] col_idx multi index for selecting columns
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and value matrix do not match.
 */
template <typename Mat1, typename Vec,
          require_var_dense_dynamic_t<Mat1>* = nullptr,
          internal::require_var_row_vector_or_arithmetic_eigen<Vec>* = nullptr>
inline void assign(Mat1&& x, const Vec& y, const char* name, index_uni row_idx,
                   const index_multi& col_idx) {
  stan::math::check_range("matrix[uni, multi] assign", name, x.rows(),
                          row_idx.n_);
  const auto assign_cols = col_idx.ns_.size();
  stan::math::check_size_match("matrix[uni, multi] assign", "left hand side",
                               assign_cols, name, y.size());
  const int row_idx_val = row_idx.n_ - 1;
  arena_t<std::vector<int>> x_idx(assign_cols);
  arena_t<Eigen::Matrix<double, -1, 1>> prev_val(assign_cols);
  Eigen::Matrix<double, -1, 1> y_val_idx(assign_cols);
  std::unordered_set<int> x_set;
  x_set.reserve(assign_cols);
  const auto& y_val = stan::math::value_of(y);
  // Need to remove duplicates for cases like {2, 3, 2, 2}
  for (int i = assign_cols - 1; i >= 0; --i) {
    if (likely(x_set.insert(col_idx.ns_[i]).second)) {
      stan::math::check_range("matrix[uni, multi] assign", name, x.cols(),
                              col_idx.ns_[i]);
      x_idx[i] = col_idx.ns_[i] - 1;
      prev_val.coeffRef(i) = x.val().coeff(row_idx_val, x_idx[i]);
      y_val_idx.coeffRef(i) = y_val.coeff(i);
    } else {
      x_idx[i] = -1;
    }
  }
  for (int i = assign_cols - 1; i >= 0; --i) {
    if (likely(x_idx[i] != -1)) {
      x.vi_->val_.coeffRef(row_idx_val, x_idx[i]) = y_val_idx.coeff(i);
    }
  }
  if (!is_constant<Vec>::value) {
    stan::math::reverse_pass_callback(
        [x, y, row_idx_val, x_idx, prev_val]() mutable {
          for (size_t i = 0; i < x_idx.size(); ++i) {
            if (likely(x_idx[i] != -1)) {
              x.vi_->val_.coeffRef(row_idx_val, x_idx[i]) = prev_val.coeff(i);
              prev_val.coeffRef(i) = x.adj().coeff(row_idx_val, x_idx[i]);
              x.adj().coeffRef(row_idx_val, x_idx[i]) = 0.0;
            }
          }
          for (size_t i = 0; i < x_idx.size(); ++i) {
            if (likely(x_idx[i] != -1)) {
              math::forward_as<math::promote_scalar_t<math::var, Vec>>(y)
                  .adj()
                  .coeffRef(i)
                  += prev_val.coeff(i);
            }
          }
        });
  } else {
    stan::math::reverse_pass_callback(
        [x, row_idx_val, x_idx, prev_val]() mutable {
          for (size_t i = 0; i < x_idx.size(); ++i) {
            if (likely(x_idx[i] != -1)) {
              x.vi_->val_.coeffRef(row_idx_val, x_idx[i]) = prev_val.coeff(i);
              prev_val.coeffRef(i) = x.adj().coeffRef(row_idx_val, x_idx[i]);
              x.adj().coeffRef(row_idx_val, x_idx[i]) = 0.0;
            }
          }
        });
  }
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
 * @param[in] y Value matrix.
 * @param[in] name Name of variable
 * @param[in] idx Multiple index
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and value matrix do not match.
 */
template <typename Mat1, typename Mat2,
          require_var_dense_dynamic_t<Mat1>* = nullptr,
          internal::require_var_matrix_or_arithmetic_eigen<Mat2>* = nullptr>
inline void assign(Mat1&& x, const Mat2& y, const char* name,
                   const index_multi& idx) {
  const auto assign_rows = idx.ns_.size();
  stan::math::check_size_match("matrix[multi] assign", "left hand side rows",
                               assign_rows, name, y.rows());
  stan::math::check_size_match("matrix[multi] assign", "left hand side columns",
                               x.cols(), name, y.cols());
  arena_t<std::vector<int>> x_idx(assign_rows);
  arena_t<Eigen::Matrix<double, -1, -1>> prev_vals(assign_rows, x.cols());
  Eigen::Matrix<double, -1, -1> y_val_idx(assign_rows, x.cols());
  std::unordered_set<int> x_set;
  x_set.reserve(assign_rows);
  const auto& y_val = stan::math::value_of(y);
  // Need to remove duplicates for cases like {2, 3, 2, 2}
  for (int i = assign_rows - 1; i >= 0; --i) {
    if (likely(x_set.insert(idx.ns_[i]).second)) {
      stan::math::check_range("matrix[multi, multi] assign row", name, x.rows(),
                              idx.ns_[i]);
      x_idx[i] = idx.ns_[i] - 1;
      prev_vals.row(i) = x.vi_->val_.row(x_idx[i]);
      y_val_idx.row(i) = y_val.row(i);
    } else {
      x_idx[i] = -1;
    }
  }
  for (int i = assign_rows - 1; i >= 0; --i) {
    if (likely(x_idx[i] != -1)) {
      x.vi_->val_.row(x_idx[i]) = y_val_idx.row(i);
    }
  }

  if (!is_constant<Mat2>::value) {
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
          math::forward_as<math::promote_scalar_t<math::var, Mat2>>(y)
              .adj()
              .row(i)
              += prev_vals.row(i);
        }
      }
    });
  } else {
    stan::math::reverse_pass_callback([x, prev_vals, x_idx]() mutable {
      for (size_t i = 0; i < x_idx.size(); ++i) {
        if (likely(x_idx[i] != -1)) {
          x.vi_->val_.row(x_idx[i]) = prev_vals.row(i);
          prev_vals.row(i) = x.adj().row(x_idx[i]);
          x.adj().row(x_idx[i]).setZero();
        }
      }
    });
  }
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
 * @param[in] y Value matrix.
 * @param[in] name Name of variable
 * @param[in] row_idx multi index for selecting rows
 * @param[in] col_idx multi index for selecting columns
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and value matrix do not match.
 */
template <typename Mat1, typename Mat2, require_var_matrix_t<Mat1>* = nullptr,
          internal::require_var_matrix_or_arithmetic_eigen<Mat2>* = nullptr>
inline void assign(Mat1&& x, const Mat2& y, const char* name,
                   const index_multi& row_idx, const index_multi& col_idx) {
  const auto assign_rows = row_idx.ns_.size();
  const auto assign_cols = col_idx.ns_.size();
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
    if (likely(x_row_set.insert(row_idx.ns_[i]).second)) {
      stan::math::check_range("matrix[multi, multi] assign row", name, x.rows(),
                              row_idx.ns_[i]);
      x_row_idx[i] = row_idx.ns_[i] - 1;
    } else {
      x_row_idx[i] = -1;
    }
  }
  const auto& y_val = stan::math::value_of(y);
  for (int j = assign_cols - 1; j >= 0; --j) {
    if (likely(x_set.insert(col_idx.ns_[j]).second)) {
      stan::math::check_range("matrix[multi, multi] assign col", name, x.cols(),
                              col_idx.ns_[j]);
      x_col_idx[j] = col_idx.ns_[j] - 1;
      for (int i = assign_rows - 1; i >= 0; --i) {
        if (likely(x_row_idx[i] != -1)) {
          prev_vals.coeffRef(i, j)
              = x.vi_->val_.coeff(x_row_idx[i], x_col_idx[j]);
          y_vals.coeffRef(i, j) = y_val.coeff(i, j);
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
          x.vi_->val_.coeffRef(x_row_idx[i], x_col_idx[j]) = y_vals.coeff(i, j);
        }
      }
    }
  }
  if (!is_constant<Mat2>::value) {
    stan::math::reverse_pass_callback(
        [x, y, prev_vals, x_col_idx, x_row_idx]() mutable {
          for (int j = 0; j < x_col_idx.size(); ++j) {
            if (likely(x_col_idx[j] != -1)) {
              for (int i = 0; i < x_row_idx.size(); ++i) {
                if (likely(x_row_idx[i] != -1)) {
                  x.vi_->val_.coeffRef(x_row_idx[i], x_col_idx[j])
                      = prev_vals.coeff(i, j);
                  prev_vals.coeffRef(i, j)
                      = x.adj().coeff(x_row_idx[i], x_col_idx[j]);
                  x.adj().coeffRef(x_row_idx[i], x_col_idx[j]) = 0;
                }
              }
            }
          }
          for (int j = 0; j < x_col_idx.size(); ++j) {
            if (likely(x_col_idx[j] != -1)) {
              for (int i = 0; i < x_row_idx.size(); ++i) {
                if (likely(x_row_idx[i] != -1)) {
                  math::forward_as<math::promote_scalar_t<math::var, Mat2>>(y)
                      .adj()
                      .coeffRef(i, j)
                      += prev_vals.coeff(i, j);
                }
              }
            }
          }
        });
  } else {
    stan::math::reverse_pass_callback(
        [x, prev_vals, x_col_idx, x_row_idx]() mutable {
          for (int j = 0; j < x_col_idx.size(); ++j) {
            if (likely(x_col_idx[j] != -1)) {
              for (int i = 0; i < x_row_idx.size(); ++i) {
                if (likely(x_row_idx[i] != -1)) {
                  x.vi_->val_.coeffRef(x_row_idx[i], x_col_idx[j])
                      = prev_vals.coeff(i, j);
                  prev_vals.coeffRef(i, j)
                      = x.adj().coeff(x_row_idx[i], x_col_idx[j]);
                  x.adj().coeffRef(x_row_idx[i], x_col_idx[j]) = 0;
                }
              }
            }
          }
        });
  }
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
 * @param[in] y Value matrix.
 * @param[in] name Name of variable
 * @param[in] row_idx index for selecting rows
 * @param[in] col_idx multi index for selecting columns
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and value matrix do not match.
 */
template <typename Mat1, typename Mat2, typename Idx,
          require_var_dense_dynamic_t<Mat1>* = nullptr,
          internal::require_var_matrix_or_arithmetic_eigen<Mat2>* = nullptr>
inline void assign(Mat1&& x, const Mat2& y, const char* name,
                   const Idx& row_idx, const index_multi& col_idx) {
  const auto assign_cols = col_idx.ns_.size();
  stan::math::check_size_match("matrix[..., multi] assign", "left hand side",
                               assign_cols, name, y.cols());
  std::unordered_set<int> x_set;
  const auto& y_eval = y.eval();
  x_set.reserve(assign_cols);
  // Need to remove duplicates for cases like {2, 3, 2, 2}
  for (int j = assign_cols - 1; j >= 0; --j) {
    if (likely(x_set.insert(col_idx.ns_[j]).second)) {
      stan::math::check_range("matrix[..., multi] assign col", name, x.cols(),
                              col_idx.ns_[j]);
      assign(x.col(col_idx.ns_[j] - 1), y_eval.col(j), name, row_idx);
    }
  }
}

}  // namespace model
}  // namespace stan
#endif

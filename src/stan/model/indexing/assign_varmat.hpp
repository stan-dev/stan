#ifndef STAN_MODEL_INDEXING_ASSIGN_VARMAT_HPP
#define STAN_MODEL_INDEXING_ASSIGN_VARMAT_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/fun/adjoint_of.hpp>
#include <stan/model/indexing/access_helpers.hpp>
#include <stan/model/indexing/index.hpp>
#include <boost/unordered/unordered_map.hpp>
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
  stan::math::check_size_match("vector[multi] assign", name, idx.ns_.size(),
                               "right hand side", y.size());
  const auto x_size = x.size();
  const auto assign_size = idx.ns_.size();
  arena_t<Eigen::Matrix<double, -1, 1>> prev_vals(assign_size);
  Eigen::Matrix<double, -1, 1> y_idx_vals(assign_size);
  // Use boost unordered flat map when boost 1.82 goes into math
  boost::unordered_map<int, int, boost::hash<int>, std::equal_to<>, stan::math::arena_allocator<std::pair<const int, int>>> x_map;
  x_map.reserve(assign_size);
  // Keep track of the last place we assigned to.
  for (int i = 0; i < assign_size; ++i) {
    x_map[idx.ns_[i] - 1] = i;
  }
  const auto& y_val = stan::math::value_of(y);
  // We have to use two loops to avoid aliasing issues.
  for (auto&& x_idx : x_map) {
      stan::math::check_range("vector[multi] assign", name, x_size, x_idx.first + 1);
      prev_vals.coeffRef(x_idx.second) = x.vi_->val_.coeffRef(x_idx.first);
      y_idx_vals.coeffRef(x_idx.second) = y_val.coeff(x_idx.second);
  }
  for (auto&& x_idx : x_map) {
      x.vi_->val_.coeffRef(x_idx.first) = y_idx_vals.coeff(x_idx.second);
  }

  if (!is_constant<Vec2>::value) {
    stan::math::reverse_pass_callback([x, y, x_map, prev_vals]() mutable {
      for (auto&& x_idx : x_map) {
          x.vi_->val_.coeffRef(x_idx.first) = prev_vals.coeffRef(x_idx.second);
          prev_vals.coeffRef(x_idx.second) = x.adj().coeffRef(x_idx.first);
          x.adj().coeffRef(x_idx.first) = 0.0;
      }
      for (auto&& x_idx : x_map) {
          math::forward_as<math::promote_scalar_t<math::var, Vec2>>(y)
              .adj()
              .coeffRef(x_idx.second)
              += prev_vals.coeff(x_idx.second);
      }
    });
  } else {
    stan::math::reverse_pass_callback([x, x_map, prev_vals]() mutable {
      for (auto&& x_idx : x_map) {
        x.vi_->val_.coeffRef(x_idx.first) = prev_vals.coeff(x_idx.second);
        prev_vals.coeffRef(x_idx.second) = x.adj().coeff(x_idx.first);
        x.adj().coeffRef(x_idx.first) = 0.0;
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
  stan::math::check_size_match("matrix[uni, multi] assign columns", name,
                               assign_cols, "right hand side", y.size());
  const int row_idx_val = row_idx.n_ - 1;
  arena_t<std::vector<int>> x_idx(assign_cols);
  arena_t<Eigen::Matrix<double, -1, 1>> prev_val(assign_cols);
  Eigen::Matrix<double, -1, 1> y_val_idx(assign_cols);
  boost::unordered_map<int, int, boost::hash<int>, std::equal_to<>, stan::math::arena_allocator<std::pair<const int, int>>> col_map;
  col_map.reserve(assign_cols);
  // Keep track of the last place we assigned to.
  for (int i = 0; i < assign_cols; ++i) {
    col_map[col_idx.ns_[i] - 1] = i;
  }
  const auto& y_val = stan::math::value_of(y);
  // Need to remove duplicates for cases like {2, 3, 2, 2}
  for (auto&& col_map_idx : col_map) {
    stan::math::check_range("matrix[uni, multi] assign", name, x.cols(),
                            col_map_idx.first + 1);
    prev_val.coeffRef(col_map_idx.second) = x.val().coeff(row_idx_val, col_map_idx.first);
    y_val_idx.coeffRef(col_map_idx.second) = y_val.coeff(col_map_idx.second);
  }
  for (auto&& col_map_idx : col_map) {
      x.vi_->val_.coeffRef(row_idx_val, col_map_idx.first) = y_val_idx.coeff(col_map_idx.second);
  }
  if (!is_constant<Vec>::value) {
    stan::math::reverse_pass_callback(
        [x, y, row_idx_val, col_map, prev_val]() mutable {
          for (auto&& col_map_idx : col_map) {
            x.vi_->val_.coeffRef(row_idx_val, col_map_idx.first) = prev_val.coeff(col_map_idx.second);
            prev_val.coeffRef(col_map_idx.second) = x.adj().coeff(row_idx_val, col_map_idx.first);
            x.adj().coeffRef(row_idx_val, col_map_idx.first) = 0.0;
          }
          for (auto&& col_map_idx : col_map) {
            math::forward_as<math::promote_scalar_t<math::var, Vec>>(y)
                .adj()
                .coeffRef(col_map_idx.second)
                += prev_val.coeff(col_map_idx.second);
          }
        });
  } else {
    stan::math::reverse_pass_callback(
        [x, row_idx_val, col_map, prev_val]() mutable {
          for (auto&& col_map_idx : col_map) {
            x.vi_->val_.coeffRef(row_idx_val, col_map_idx.first) = prev_val.coeff(col_map_idx.second);
            prev_val.coeffRef(col_map_idx.second) = x.adj().coeffRef(row_idx_val, col_map_idx.first);
            x.adj().coeffRef(row_idx_val, col_map_idx.first) = 0.0;
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
  stan::math::check_size_match("matrix[multi] assign rows", name, assign_rows,
                               "right hand side rows", y.rows());
  stan::math::check_size_match("matrix[multi] assign columns", name, x.cols(),
                               "right hand side rows", y.cols());
  arena_t<Eigen::Matrix<double, -1, -1>> prev_vals(assign_rows, x.cols());
  Eigen::Matrix<double, -1, -1> y_val_idx(assign_rows, x.cols());
  boost::unordered_map<int, int, boost::hash<int>, std::equal_to<>, stan::math::arena_allocator<std::pair<const int, int>>> row_map;
  row_map.reserve(assign_rows);
  for (int i = 0; i < assign_rows; ++i) {
    row_map[idx.ns_[i] - 1] = i;
  }

  const auto& y_val = stan::math::value_of(y);
  // Need to remove duplicates for cases like {2, 3, 2, 2}
  for (auto&& row_idx : row_map) {
      stan::math::check_range("matrix[multi, multi] assign row", name, x.rows(), row_idx.first + 1);
      prev_vals.row(row_idx.second) = x.vi_->val_.row(row_idx.first);
      y_val_idx.row(row_idx.second) = y_val.row(row_idx.second);
  }
  for (auto&& row_idx : row_map) {
    x.vi_->val_.row(row_idx.first) = y_val_idx.row(row_idx.second);
  }

  if (!is_constant<Mat2>::value) {
    stan::math::reverse_pass_callback([x, y, prev_vals, row_map]() mutable {
      for (auto&& row_idx : row_map) {
          x.vi_->val_.row(row_idx.first) = prev_vals.row(row_idx.second);
          prev_vals.row(row_idx.second) = x.adj().row(row_idx.first);
          x.adj().row(row_idx.first).fill(0);
      }
      for (auto&& row_idx : row_map) {
          math::forward_as<math::promote_scalar_t<math::var, Mat2>>(y)
              .adj()
              .row(row_idx.second)
              += prev_vals.row(row_idx.second);
      }
    });
  } else {
    stan::math::reverse_pass_callback([x, prev_vals, row_map]() mutable {
      for (auto&& row_idx : row_map) {
          x.vi_->val_.row(row_idx.first) = prev_vals.row(row_idx.second);
          prev_vals.row(row_idx.second) = x.adj().row(row_idx.first);
          x.adj().row(row_idx.first).setZero();
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
  stan::math::check_size_match("matrix[multi,multi] assign rows", name,
                               assign_rows, "right hand side rows", y.rows());
  stan::math::check_size_match("matrix[multi,multi] assign columns", name,
                               assign_cols, "right hand side columns",
                               y.cols());
  using arena_vec = std::vector<int, stan::math::arena_allocator<int>>;
  boost::unordered_map<int, int, boost::hash<int>, std::equal_to<>, stan::math::arena_allocator<std::pair<const int, int>>> row_map;
  row_map.reserve(assign_rows);
  for (int i = 0; i < assign_rows; ++i) {
    stan::math::check_range("matrix[multi, multi] assign row", name, x.rows(),
                            row_idx.ns_[i]);
    row_map[row_idx.ns_[i] - 1] = i;
  }

  boost::unordered_map<int, int, boost::hash<int>, std::equal_to<>, stan::math::arena_allocator<std::pair<const int, int>>> col_map;
  col_map.reserve(assign_cols);
  for (int i = 0; i < assign_cols; ++i) {
    stan::math::check_range("matrix[multi, multi] assign col", name, x.cols(),
                            col_idx.ns_[i]);
    col_map[col_idx.ns_[i] - 1] = i;
  }

  arena_t<Eigen::Matrix<double, -1, -1>> prev_vals(assign_rows, assign_cols);
  Eigen::Matrix<double, -1, -1> dedupe_y_vals(assign_rows, assign_cols);
  // Need to remove duplicates for cases like {{2, 3, 2, 2}, {1, 2, 2}}
  const auto& y_val = stan::math::value_of(y);
  for (auto&& col_idx : col_map) {
      for (auto&& row_idx : row_map) {
          prev_vals.coeffRef(row_idx.second, col_idx.second)
              = x.vi_->val_.coeff(row_idx.first, col_idx.first);
          dedupe_y_vals.coeffRef(row_idx.second, col_idx.second) = y_val.coeff(row_idx.second, col_idx.second);
      }
  }
  for (auto&& col_idx : col_map) {
    for (auto&& row_idx : row_map) {
      x.vi_->val_.coeffRef(row_idx.first, col_idx.first) = dedupe_y_vals.coeff(row_idx.second, col_idx.second);
    }
  }
  if (!is_constant<Mat2>::value) {
    stan::math::reverse_pass_callback(
        [x, y, prev_vals, row_map, col_map]() mutable {
          for (auto&& col_idx : col_map) {
            for (auto&& row_idx : row_map) {
                  x.vi_->val_.coeffRef(row_idx.first, col_idx.first)
                      = prev_vals.coeff(row_idx.second, col_idx.second);
                  prev_vals.coeffRef(row_idx.second, col_idx.second)
                      = x.adj().coeff(row_idx.first, col_idx.first);
                  x.adj().coeffRef(row_idx.first, col_idx.first) = 0;
              }
          }
          for (auto&& col_idx : col_map) {
            for (auto&& row_idx : row_map) {
              math::forward_as<math::promote_scalar_t<math::var, Mat2>>(y)
                  .adj()
                  .coeffRef(row_idx.second, col_idx.second)
                   += prev_vals.coeff(row_idx.second, col_idx.second);
            }
          }
        });
  } else {
    stan::math::reverse_pass_callback(
        [x, prev_vals, row_map, col_map]() mutable {
          for (auto&& col_idx : col_map) {
            for (auto&& row_idx : row_map) {
                x.vi_->val_.coeffRef(row_idx.first, col_idx.first)
                    = prev_vals.coeff(row_idx.second, col_idx.second);
                prev_vals.coeffRef(row_idx.second, col_idx.second)
                    = x.adj().coeff(row_idx.first, col_idx.first);
                x.adj().coeffRef(row_idx.first, col_idx.first) = 0;
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
  stan::math::check_size_match("matrix[..., multi] assign columns", name,
                               assign_cols, "right hand side columns",
                               y.cols());
  boost::unordered_map<int, int, boost::hash<int>, std::equal_to<>, stan::math::arena_allocator<std::pair<const int, int>>> col_map;
  col_map.reserve(assign_cols);
  for (int i = 0; i < assign_cols; ++i) {
    stan::math::check_range("matrix[multi, multi] assign col", name, x.rows(),
                            col_idx.ns_[i]);
    col_map[col_idx.ns_[i] - 1] = i;
  }
  const auto& y_eval = y.eval();
  // Need to remove duplicates for cases like {2, 3, 2, 2}
  for (auto&& col_idx : col_map) {
      assign(x.col(col_idx.first), y_eval.col(col_idx.second), name, row_idx);
  }
}

}  // namespace model
}  // namespace stan
#endif

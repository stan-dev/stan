#ifndef STAN_MODEL_INDEXING_RVALUE_VARMAT_HPP
#define STAN_MODEL_INDEXING_RVALUE_VARMAT_HPP

#include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>
#include <stan/model/indexing/index.hpp>
#include <stan/model/indexing/index_list.hpp>
#include <stan/model/indexing/rvalue.hpp>
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
 * Return a non-contiguous subset of elements in a vector.
 *
 * Types:  vector[multi] = vector
 *
 * @tparam Vec `var_value` with inner Eigen type with either dynamic rows or
 * columns, but not both.
 * @param[in] v `var_value` with inner Eigen vector type.
 * @param[in] idxs Sequence of integers.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the value size isn't the same as
 * the indexed size.
 */
template <typename Vec, require_var_vector_t<Vec>* = nullptr>
inline auto rvalue(Vec&& x,
                   const cons_index_list<index_multi, nil_index_list>& idxs,
                   const char* name = "ANON", int depth = 0) {
  using stan::math::arena_allocator;
  using stan::math::check_range;
  using stan::math::reverse_pass_callback;
  using stan::math::var_value;
  using arena_std_vec = std::vector<int, arena_allocator<int>>;
  const Eigen::Index x_size = x.size();
  const auto ret_size = idxs.head_.ns_.size();
  arena_t<value_type_t<Vec>> x_ret_vals(ret_size);
  arena_std_vec row_idx(idxs.head_.ns_.begin(), idxs.head_.ns_.end());
  for (int i = 0; i < ret_size; ++i) {
    check_range("vector[multi] assign range", name, x_size, row_idx[i]);
    --row_idx[i];
    x_ret_vals.coeffRef(i) = x.vi_->val_.coeffRef(row_idx[i]);
  }
  var_value<plain_type_t<value_type_t<Vec>>> x_ret(x_ret_vals);
  reverse_pass_callback([x, x_ret, row_idx]() mutable {
    for (Eigen::Index i = 0; i < row_idx.size(); ++i) {
      x.adj().coeffRef(row_idx[i]) += x_ret.adj().coeffRef(i);
    }
  });
  return x_ret;
}

/**
 * Return the specified matrix at the specified multi index.
 *
 * Types:  matrix[multi] = matrix
 *
 * @tparam VarMat `var_value` with inner Eigen type with dynamic rows and
 * columns.
 * @param[in] x `var_value` with inner Eigen type
 * @param[in] idxs An indexing from the start of the container up to
 * the specified maximum index (inclusive).
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename VarMat, require_var_matrix_t<VarMat>* = nullptr,
          require_eigen_dense_dynamic_t<value_type_t<VarMat>>* = nullptr>
inline auto rvalue(VarMat&& x,
                   const cons_index_list<index_multi, nil_index_list>& idxs,
                   const char* name = "ANON", int depth = 0) {
  using stan::math::arena_allocator;
  using stan::math::check_range;
  using stan::math::reverse_pass_callback;
  using stan::math::var_value;
  using arena_std_vec = std::vector<int, arena_allocator<int>>;
  const auto ret_rows = idxs.head_.ns_.size();
  arena_t<value_type_t<VarMat>> x_ret_vals(ret_rows, x.cols());
  arena_std_vec row_idx(idxs.head_.ns_.begin(), idxs.head_.ns_.end());
  for (int i = 0; i < ret_rows; ++i) {
    check_range("matrix[multi] subset range", name, x.rows(), row_idx[i]);
    --row_idx[i];
    x_ret_vals.row(i) = x.val().row(row_idx[i]);
  }
  var_value<plain_type_t<value_type_t<VarMat>>> x_ret(x_ret_vals);
  reverse_pass_callback([x, x_ret, row_idx]() mutable {
    for (Eigen::Index i = 0; i < row_idx.size(); ++i) {
      x.adj().row(row_idx[i]) += x_ret.adj().row(i);
    }
  });
  return x_ret;
}

/**
 * Return a row of a matrix with possibly unordered cells.
 *
 * Types:  matrix[uni, multi] = row vector
 *
 * @tparam VarMat `var_value` with inner Eigen type with dynamic rows and
 * columns.
 * @param[in] x `var_value` to index.
 * @param[in] idxs Pair of multiple indexes (from 1).
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename VarMat, require_var_dense_dynamic_t<VarMat>* = nullptr>
inline auto rvalue(
    VarMat&& x,
    const cons_index_list<index_uni,
                          cons_index_list<index_multi, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  using stan::math::arena_allocator;
  using stan::math::check_range;
  using stan::math::reverse_pass_callback;
  using stan::math::var_value;
  using arena_std_vec = std::vector<int, arena_allocator<int>>;
  check_range("matrix[uni, multi] index range", name, x.rows(), idxs.head_.n_);
  const auto ret_size = idxs.tail_.head_.ns_.size();
  arena_t<Eigen::Matrix<double, 1, Eigen::Dynamic>> x_ret_vals(ret_size);
  arena_std_vec col_idx(idxs.tail_.head_.ns_.begin(),
                        idxs.tail_.head_.ns_.end());
  const int row_idx = idxs.head_.n_ - 1;
  for (int i = 0; i < ret_size; ++i) {
    check_range("matrix[multi] subset range", name, x.cols(), col_idx[i]);
    --col_idx[i];
    x_ret_vals.coeffRef(i) = x.val().coeffRef(row_idx, col_idx[i]);
  }
  var_value<Eigen::Matrix<double, 1, Eigen::Dynamic>> x_ret(x_ret_vals);
  reverse_pass_callback([x, x_ret, row_idx, col_idx]() mutable {
    for (Eigen::Index i = 0; i < col_idx.size(); ++i) {
      x.adj().coeffRef(row_idx, col_idx[i]) += x_ret.adj().coeffRef(i);
    }
  });
  return x_ret;
}

/**
 * Return a column of a matrix that is a possibly non-contiguous subset
 *  of the input matrix.
 *
 * Types:  matrix[multi, uni] = vector
 *
 * @tparam VarMat `var_value` with inner Eigen type with dynamic rows and
 * columns.
 * @param[in] x `var_value` to index.
 * @param[in] idxs Pair of multiple indexes (from 1).
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename VarMat, require_var_dense_dynamic_t<VarMat>* = nullptr>
inline auto rvalue(
    VarMat&& x,
    const cons_index_list<index_multi,
                          cons_index_list<index_uni, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  using stan::math::arena_allocator;
  using stan::math::check_range;
  using stan::math::reverse_pass_callback;
  using stan::math::var_value;
  using arena_std_vec = std::vector<int, arena_allocator<int>>;
  check_range("matrix[multi, uni] rvalue range", name, x.cols(),
              idxs.tail_.head_.n_);
  const auto ret_size = idxs.head_.ns_.size();
  arena_t<Eigen::Matrix<double, Eigen::Dynamic, 1>> x_ret_val(ret_size);
  arena_std_vec row_idx(idxs.head_.ns_.begin(), idxs.head_.ns_.end());
  const int col_idx = idxs.tail_.head_.n_ - 1;
  for (int i = 0; i < ret_size; ++i) {
    check_range("matrix[multi, uni] rvalue range", name, x.rows(), row_idx[i]);
    --row_idx[i];
    x_ret_val.coeffRef(i) = x.val().coeffRef(row_idx[i], col_idx);
  }
  var_value<Eigen::Matrix<double, Eigen::Dynamic, 1>> x_ret(x_ret_val);
  reverse_pass_callback([x, x_ret, col_idx, row_idx]() mutable {
    for (Eigen::Index i = 0; i < row_idx.size(); ++i) {
      x.adj().coeffRef(row_idx[i], col_idx) += x_ret.adj().coeffRef(i);
    }
  });
  return x_ret;
}

/**
 * Return a matrix that is a possibly non-contiguous subset of the input
 *  matrix.
 *
 * Types:  matrix[multi, multi] = matrix
 *
 * @tparam VarMat `var_value` with an inner eigen matrix
 * @param[in] x `var_value` to index.
 * @param[in] idxs Pair of multi indexes.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename VarMat, require_var_dense_dynamic_t<VarMat>* = nullptr>
inline auto rvalue(
    VarMat&& x,
    const cons_index_list<index_multi,
                          cons_index_list<index_multi, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  using stan::math::arena_allocator;
  using stan::math::check_range;
  using stan::math::reverse_pass_callback;
  using stan::math::var_value;
  using arena_std_vec = std::vector<int, arena_allocator<int>>;
  const auto ret_rows = idxs.head_.ns_.size();
  const auto ret_cols = idxs.tail_.head_.ns_.size();
  const Eigen::Index x_rows = x.rows();
  const Eigen::Index x_cols = x.cols();
  arena_t<plain_type_t<value_type_t<VarMat>>> x_ret_val(ret_rows, ret_cols);
  arena_std_vec row_idx(idxs.head_.ns_.begin(), idxs.head_.ns_.end());
  arena_std_vec col_idx(idxs.tail_.head_.ns_.begin(),
                        idxs.tail_.head_.ns_.end());
  // We only want to check these once
  for (int i = 0; i < ret_rows; ++i) {
    check_range("matrix[multi,multi] row index", name, x_rows, row_idx[i]);
    --row_idx[i];
  }
  for (int j = 0; j < ret_cols; ++j) {
    check_range("matrix[multi,multi] col index", name, x.cols(), col_idx[j]);
    --col_idx[j];
    for (int i = 0; i < ret_rows; ++i) {
      x_ret_val.coeffRef(i, j) = x.val().coeff(row_idx[i], col_idx[j]);
    }
  }
  var_value<plain_type_t<value_type_t<VarMat>>> x_ret(x_ret_val);
  reverse_pass_callback([x, x_ret, col_idx, row_idx]() mutable {
    for (int j = 0; j < col_idx.size(); ++j) {
      for (int i = 0; i < row_idx.size(); ++i) {
        x.adj().coeffRef(row_idx[i], col_idx[j]) += x_ret.adj().coeff(i, j);
      }
    }
  });
  return x_ret;
}

/**
 * Return a matrix of possibly unordered columns with each column
 *  range specified by another index.
 *
 * Types:  matrix[Idx, multi] = matrix
 *
 * @tparam VarMat `var_value` with inner an eigen matrix
 * @param[in] x `var_value` object.
 * @param[in] idxs Index consisting of one uni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename VarMat, typename Idx,
          require_var_dense_dynamic_t<VarMat>* = nullptr>
inline auto rvalue(
    VarMat&& x,
    const cons_index_list<Idx, cons_index_list<index_multi, nil_index_list>>&
        idxs,
    const char* name = "ANON", int depth = 0) {
  using stan::math::arena_allocator;
  using stan::math::check_range;
  using stan::math::reverse_pass_callback;
  using stan::math::var_value;
  using arena_std_vec = std::vector<int, arena_allocator<int>>;
  const auto ret_rows = rvalue_index_size(idxs.head_, x.rows());
  const auto ret_cols = idxs.tail_.head_.ns_.size();
  arena_t<value_type_t<VarMat>> x_ret_val(ret_rows, ret_cols);
  arena_std_vec col_idx(idxs.tail_.head_.ns_.begin(),
                        idxs.tail_.head_.ns_.end());
  for (int j = 0; j < ret_cols; ++j) {
    check_range("matrix[..., multi] col index", name, x.cols(), col_idx[j]);
    --col_idx[j];
    x_ret_val.col(j) = rvalue(x.val().col(col_idx[j]), index_list(idxs.head_),
                              name, depth + 1);
  }
  var_value<Eigen::Matrix<double, -1, -1>> x_ret(x_ret_val);
  // index_multi is the only index with dynamic memory so head is safe
  reverse_pass_callback(
      [head = idxs.head_, x, x_ret, col_idx, name, depth]() mutable {
        for (size_t j = 0; j < col_idx.size(); ++j) {
          for (size_t i = 0; i < x_ret.rows(); ++i) {
            const int n = rvalue_at(i, head) - 1;
            x.adj().coeffRef(n, col_idx[j]) += x_ret.adj().coeffRef(i, j);
          }
        }
      });
  return x_ret;
}

}  // namespace model
}  // namespace stan
#endif

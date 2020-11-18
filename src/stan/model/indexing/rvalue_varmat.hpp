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
 * Return the result of indexing an eigen expression type without
 * taking a subset.
 *
 * Types:  expr[omni] : plain_type
 *
 * @tparam T A type that is an expression.
 * @param[in] x an eigen expression.
 * @param[in] idxs Index consisting of one omni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of evaluating the expression.
 */
template <typename T, require_var_matrix_t<T>* = nullptr,
          require_not_plain_type_t<value_type_t<T>>* = nullptr>
inline auto rvalue(T&& x,
                   const cons_index_list<index_omni, nil_index_list>& idxs,
                   const char* name = "ANON", int depth = 0) {
  return x.eval();
}

/**
 * Return a non-contiguous subset of elements in a vector.
 *
 * Types:  vector[multi] = vector
 *
 * @tparam Vec `var_value` with inner Eigen type with either dynamic rows or columns, but not both.
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
  const Eigen::Index x_size = x.size();
  const auto ret_size = idxs.head_.ns_.size();
  arena_t<value_type_t<Vec>> x_ret_vals(ret_size);
  using multi_map = Eigen::Map<const Eigen::Matrix<int, -1, 1>>;
  arena_t<Eigen::Matrix<int, -1, 1>> row_idx(
      multi_map(idxs.head_.ns_.data(), ret_size).eval());
  for (int i = 0; i < ret_size; ++i) {
    stan::math::check_range("vector[multi] assign range", name, x_size,
                            row_idx[i]);
    --row_idx[i];
    x_ret_vals.coeffRef(i) = x.vi_->val_.coeffRef(row_idx[i]);
  }
  stan::math::var_value<plain_type_t<value_type_t<Vec>>> x_ret(x_ret_vals);
  stan::math::reverse_pass_callback([x, x_ret, row_idx]() mutable {
    for (Eigen::Index i = 0; i < row_idx.size(); ++i) {
      x.adj().coeffRef(row_idx[i]) += x_ret.adj().coeffRef(i);
    }
  });
  return x_ret;
}

/**
 * Return the specified Eigen matrix at the specified multi index.
 *
 * Types:  matrix[multi] = matrix
 *
 * @tparam VarMat `var_value` with inner Eigen type with dynamic rows and columns.
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
  const auto ret_rows = idxs.head_.ns_.size();
  arena_t<value_type_t<VarMat>> x_ret_vals(ret_rows, x.cols());
  using multi_map = Eigen::Map<const Eigen::Matrix<int, -1, 1>>;
  arena_t<Eigen::Matrix<int, -1, 1>> row_idx(
      multi_map(idxs.head_.ns_.data(), ret_rows).eval());
  for (int i = 0; i < ret_rows; ++i) {
    math::check_range("matrix[multi] subset range", name, x.rows(), row_idx[i]);
    --row_idx[i];
    x_ret_vals.row(i) = x.val().row(row_idx[i]);
  }
  stan::math::var_value<plain_type_t<value_type_t<VarMat>>> x_ret(x_ret_vals);
  stan::math::reverse_pass_callback([x, x_ret, row_idx]() mutable {
    for (Eigen::Index i = 0; i < row_idx.size(); ++i) {
      x.adj().row(row_idx[i]) += x_ret.adj().row(i);
    }
  });
  return x_ret;
}

/**
 * Return a range of rows for an Eigen matrix.
 *
 * Types:  matrix[min_max] = matrix
 *
 * @tparam Mat `var_value` with inner Eigen type with dynamic rows and columns.
 * @param[in] x `var_value` with inner Eigen type
 * @param[in] idxs An indexing from the start of the container up to
 * the specified maximum index (inclusive).
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename Mat, require_var_matrix_t<Mat>* = nullptr,
          require_eigen_dense_dynamic_t<value_type_t<Mat>>* = nullptr>
inline auto rvalue(Mat&& x,
                   const cons_index_list<index_min_max, nil_index_list>& idxs,
                   const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[min_max] max row indexing", name, x.rows(),
                    idxs.head_.max_);
  math::check_range("matrix[min_max] min row indexing", name, x.rows(),
                    idxs.head_.min_);
  if (idxs.head_.is_ascending()) {
    return x.middleRows(idxs.head_.min_ - 1, idxs.head_.max_ - 1).eval();
  } else {
    return x.middleRows(idxs.head_.max_ - 1, idxs.head_.min_ - 1)
        .colwise_reverse()
        .eval();
  }
}

/**
 * Return the result of indexing an Eigen matrix with two min_max
 * indices, returning back a block of an Eigen matrix.
 *
 * Types:  matrix[min_max, min_max] = matrix
 *
 * @tparam Mat `var_value` with inner Eigen type with dynamic rows and columns.
 * @param[in] x `var_value` with inner Eigen type
 * @param[in] idxs Index consisting of one uni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename Mat, require_var_matrix_t<Mat>* = nullptr,
          require_eigen_dense_dynamic_t<value_type_t<Mat>>* = nullptr>
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
      return x
          .block(idxs.head_.min_ - 1, idxs.tail_.head_.max_ - 1,
                 idxs.head_.max_ - (idxs.head_.min_ - 1),
                 idxs.tail_.head_.min_ - (idxs.tail_.head_.max_ - 1))
          .rowwise_reverse()
          .eval();
    }
  } else {
    if (idxs.tail_.head_.is_ascending()) {
      return x
          .block(idxs.head_.max_ - 1, idxs.tail_.head_.min_ - 1,
                 idxs.head_.min_ - (idxs.head_.max_ - 1),
                 idxs.tail_.head_.max_ - (idxs.tail_.head_.min_ - 1))
          .colwise_reverse()
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
 * Return a row of an Eigen matrix with possibly unordered cells.
 *
 * Types:  matrix[uni, multi] = row vector
 *
 * @tparam VarMat `var_value` with inner Eigen type with dynamic rows and columns.
 * @param[in] x `var_value` to index.
 * @param[in] idxs Pair of multiple indexes (from 1).
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename VarMat, require_var_matrix_t<VarMat>* = nullptr,
          require_eigen_dense_dynamic_t<value_type_t<VarMat>>* = nullptr>
inline Eigen::Matrix<value_type_t<VarMat>, 1, Eigen::Dynamic> rvalue(
    VarMat&& x,
    const cons_index_list<index_uni,
                          cons_index_list<index_multi, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[uni, multi] index range", name, x.rows(),
                    idxs.head_.n_);
  const auto ret_size = idxs.tail_.head_.ns_.size();
  arena_t<Eigen::Matrix<double, 1, Eigen::Dynamic>> x_ret_vals(ret_size);
  using multi_map = Eigen::Map<const Eigen::Matrix<int, -1, 1>>;
  arena_t<Eigen::Matrix<int, -1, 1>> col_idx(
      multi_map(idxs.tail_.head_.ns_.data(), ret_size).eval());
  const int row_idx = idxs.head_.n_ - 1;
  for (int i = 0; i < ret_size; ++i) {
    math::check_range("matrix[multi] subset range", name, x.cols(), col_idx[i]);
    --col_idx[i];
    x_ret_vals.coeffRef(i) = x.val().coeffRef(row_idx, col_idx[i]);
  }
  stan::math::var_value<Eigen::Matrix<double, 1, Eigen::Dynamic>> x_ret(
      x_ret_vals);
  stan::math::reverse_pass_callback([x, x_ret, row_idx, col_idx]() mutable {
    for (Eigen::Index i = 0; i < col_idx.size(); ++i) {
      x.adj().coeffRef(row_idx, col_idx[i]) += x_ret.adj().coeffRef(i);
    }
  });
  return x_ret;
}

/**
 * Return a column of an Eigen matrix that is a possibly non-contiguous subset
 *  of the input Eigen matrix.
 *
 * Types:  matrix[multi, uni] = vector
 *
 * @tparam VarMat `var_value` with inner Eigen type with dynamic rows and columns.
 * @param[in] x `var_value` to index.
 * @param[in] idxs Pair of multiple indexes (from 1).
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename VarMat, require_var_matrix_t<VarMat>* = nullptr,
          require_eigen_dense_dynamic_t<value_type_t<VarMat>>* = nullptr>
inline auto rvalue(
    VarMat&& x,
    const cons_index_list<index_multi,
                          cons_index_list<index_uni, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[multi, uni] rvalue range", name, x.cols(),
                    idxs.tail_.head_.n_);
  const auto ret_size = idxs.head_.ns_.size();
  arena_t<Eigen::Matrix<double, Eigen::Dynamic, 1>> x_ret_val(ret_size);
  using multi_map = Eigen::Map<const Eigen::Matrix<int, -1, 1>>;
  arena_t<Eigen::Matrix<int, -1, 1>> row_idx(
      multi_map(idxs.head_.ns_.data(), ret_size).eval());
  const int col_idx = idxs.tail_.head_.n_ - 1;
  for (int i = 0; i < ret_size; ++i) {
    math::check_range("matrix[multi, uni] rvalue range", name, x.rows(),
                      row_idx[i]);
    --row_idx[i];
    x_ret_val.coeffRef(i) = x.val().coeffRef(row_idx[i], col_idx);
  }
  stan::math::var_value<Eigen::Matrix<double, Eigen::Dynamic, 1>> x_ret(
      x_ret_val);
  stan::math::reverse_pass_callback([x, x_ret, col_idx, row_idx]() mutable {
    for (Eigen::Index i = 0; i < row_idx.size(); ++i) {
      x.adj().coeffRef(row_idx[i], col_idx) += x_ret.adj().coeffRef(i);
    }
  });
  return x_ret;
}

/**
 * Return an Eigen matrix that is a possibly non-contiguous subset of the input
 *  Eigen matrix.
 *
 * Types:  matrix[multi, multi] = matrix
 *
 * @tparam VarMat `var_value` with inner an inner eigen matrix
 * @param[in] x `var_value` to index.
 * @param[in] idxs Pair of multiple indexes.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename VarMat, require_var_matrix_t<VarMat>* = nullptr,
          require_eigen_dense_dynamic_t<value_type_t<VarMat>>* = nullptr>
inline auto rvalue(
    VarMat&& x,
    const cons_index_list<index_multi,
                          cons_index_list<index_multi, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  const auto ret_rows = idxs.head_.ns_.size();
  const auto ret_cols = idxs.tail_.head_.ns_.size();
  const Eigen::Index x_rows = x.rows();
  const Eigen::Index x_cols = x.cols();
  arena_t<value_type_t<VarMat>> x_ret_val(ret_rows, ret_cols);
  using multi_map = Eigen::Map<const Eigen::Matrix<int, -1, 1>>;
  arena_t<Eigen::Matrix<int, -1, 1>> row_idx(
      multi_map(idxs.head_.ns_.data(), ret_rows).eval());
  arena_t<Eigen::Matrix<int, -1, 1>> col_idx(
      multi_map(idxs.tail_.head_.ns_.data(), ret_cols).eval());
  // We only want to check these once
  for (int i = 0; i < ret_rows; ++i) {
    math::check_range("matrix[multi,multi] row index", name, x_rows,
                      row_idx[i]);
    --row_idx[i];
  }
  for (int j = 0; j < ret_cols; ++j) {
    math::check_range("matrix[multi,multi] col index", name, x.cols(),
                      col_idx[j]);
    --col_idx[j];
    for (int i = 0; i < ret_rows; ++i) {
      x_ret_val.coeffRef(i, j) = x.val().coeff(row_idx[i], col_idx[j]);
    }
  }
  stan::math::var_value<plain_type_t<value_type_t<VarMat>>> x_ret(x_ret_val);
  stan::math::reverse_pass_callback([x, x_ret, col_idx, row_idx]() mutable {
    for (int j = 0; j < col_idx.size(); ++j) {
      for (int i = 0; i < row_idx.size(); ++i) {
        x.adj().coeffRef(row_idx[i], col_idx[j]) += x_ret.adj().coeffRef(i, j);
      }
    }
  });
  return x_ret;
}

/**
 * Return an Eigen matrix of possibly unordered columns with each column
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
          require_var_matrix_t<VarMat>* = nullptr,
          require_eigen_dense_dynamic_t<value_type_t<VarMat>>* = nullptr>
inline auto rvalue(
    VarMat&& x,
    const cons_index_list<Idx, cons_index_list<index_multi, nil_index_list>>&
        idxs,
    const char* name = "ANON", int depth = 0) {
  const auto ret_rows = rvalue_index_size(idxs.head_, x.rows());
  const auto ret_cols = idxs.tail_.head_.ns_.size();
  arena_t<value_type_t<VarMat>> x_ret_val(ret_rows, ret_cols);
  arena_t<Eigen::Matrix<int, -1, 1>> col_idx(
      multi_map(idxs.tail_.head_.ns_.data(), ret_cols).eval());
  for (int j = 0; j < ret_cols; ++j) {
    math::check_range("matrix[..., multi] col index", name, x.cols(),
                      col_idx[j]);
    --col_idx[j];
    x_ret_val.col(j)
        = rvalue(x.val().col(col_idx), index_list(idxs.head_), name, depth + 1);
  }
  stan::math::var_value<Eigen::Matrix<double, -1, -1>> x_ret(x_ret_val);
  // index_multi is the only index with dynamic memory so head is safe
  stan::math::reverse_pass_callback(
      [head = idxs.head_, x, x_ret, col_idx, name, depth]() mutable {
        for (size_t j = 0; j < col_idx.size(); ++j) {
          x.adj().col(col_idx[j])
              += rvalue(x_ret.adj().col(j), index_list(head), name, depth + 1);
        }
      });
  return x_ret;
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * min_max_index returning a block from min to max.
 *
 * Types:  matrix[Idx, min_max] = matrix
 *
 * @tparam Mat `var_value` with inner an inner eigen matrix type.
 * @tparam Idx Type of index.
 * @param[in] x `var_value` to index.
 * @param[in] idxs Pair multiple index and single index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename Mat, typename Idx, require_var_matrix_t<Mat>* = nullptr,
          require_eigen_dense_dynamic_t<value_type_t<Mat>>* = nullptr>
inline auto rvalue(
    Mat&& x,
    const cons_index_list<Idx, cons_index_list<index_min_max, nil_index_list>>&
        idxs,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[..., min_max] indexing", name, x.rows(),
                    idxs.tail_.head_.min_);
  math::check_range("matrix[..., min_max] indexing", name, x.rows(),
                    idxs.tail_.head_.max_);
  if (idxs.tail_.head_.is_ascending()) {
    const auto col_start = idxs.tail_.head_.min_ - 1;
    return rvalue(x.middleCols(col_start, idxs.tail_.head_.max_ - col_start),
                  index_list(idxs.head_), name, depth + 1);
  } else {
    const auto col_start = idxs.tail_.head_.max_ - 1;
    return rvalue(x.middleCols(col_start, idxs.tail_.head_.min_ - col_start)
                      .rowwise_reverse(),
                  index_list(idxs.head_), name, depth + 1);
  }
}

}  // namespace model
}  // namespace stan
#endif

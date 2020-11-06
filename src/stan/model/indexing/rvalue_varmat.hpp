#ifndef STAN_MODEL_INDEXING_RVALUE_VARMAT_HPP
#define STAN_MODEL_INDEXING_RVALUE_VARMAT_HPP

#include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>
#include <stan/model/indexing/index.hpp>
#include <stan/model/indexing/index_list.hpp>
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
template <typename T, require_var_matrix_t<T>* = nullptr>
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
 * @tparam Vec Eigen type with either dynamic rows or columns, but not both.
 * @param[in] v Eigen vector type.
 * @param[in] idxs Sequence of integers.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the value size isn't the same as
 * the indexed size.
 */
template <typename Vec, require_var_vector_t<Vec>* = nullptr>
inline auto rvalue(
    Vec&& v, const cons_index_list<index_multi, nil_index_list>& idxs,
    const char* name = "ANON", int depth = 0) {
  const auto x_size = x.size();
  arena_t<value_type_t<Vec>> x_ret_vals(idxs.head_.ns_.size());
  arena_t<std::vector<int>> multi_idx(idxs.head_.ns_.size());
  for (int n = 0; n < y.size(); ++n) {
    const auto coeff_idx = idxs.head_.ns_[n] - 1;
    stan::math::check_range("vector[multi] assign range", name, x_size,
                            coeff_idx + 1);
    multi_idx[n] = coeff_idx;
    x_ret_vals.coeffRef(n) = x.vi_->val_.coeffRef(coeff_idx);
  }
  stan::math::var_value<plain_type_t<value_type_t<Vec>>> x_ret(x_ret_vals);
  stan::math::reverse_pass_callback([x_vi = x.vi_, ret_vi = x_ret.vi_, multi_idx]() mutable {
    for (Eigen::Index i = 0; i < multi_idx.size(); ++i) {
      x_vi->adj_.coeffRef(multi_idx[i]) += ret_vi->adj_.coeffRef(i);
    }
  });
  return x_ret;
}

/**
 * Return the specified Eigen matrix at the specified multi index.
 *
 * Types:  matrix[multi] = matrix
 *
 * @tparam VarMat Eigen type with dynamic rows and columns.
 * @param[in] x Eigen type
 * @param[in] idxs An indexing from the start of the container up to
 * the specified maximum index (inclusive).
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename VarMat,
          require_var_matrix_t<VarMat>* = nullptr,
          require_eigen_dense_dynamic_t<value_type_t<VarMat>>* = nullptr>
inline auto rvalue(VarMat&& x, const cons_index_list<index_multi, nil_index_list>& idxs,
    const char* name = "ANON", int depth = 0) {
  arena_t<value_type_t<VarMat>> x_ret_vals(idxs.head_.ns_.size(), x.cols());
  arena_t<std::vector<int>> multi_idx(idxs.head_.ns_.size());
  for (int i = 0; i < idxs.head_.ns_.size(); ++i) {
    const int n = idxs.head_.ns_[i];
    multi_idx[n] = n;
    math::check_range("matrix[multi] subset range", name, x_ref.rows(), n);
    x_ret_vals.row(i) = x.val().row(n - 1);
  }
  stan::math::var_value<plain_type_t<value_type_t<Var>>> x_ret(x_ret_vals);
  stan::math::reverse_pass_callback([x_vi = x.vi_, ret_vi = x_ret.vi_, multi_idx]() mutable {
    for (Eigen::Index i = 0; i < multi_idx.size(); ++i) {
      x_vi->adj_.coeffRef(multi_idx[i]) += ret_vi->adj_.coeffRef(i);
    }
  });
  return x_ret;
}

/**
 * Return a row of an Eigen matrix with possibly unordered cells.
 *
 * Types:  matrix[uni, multi] = row vector
 *
 * @tparam VarMat Eigen type with dynamic rows and columns.
 * @param[in] x Matrix to index.
 * @param[in] idxs Pair of multiple indexes (from 1).
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename VarMat,
          require_var_matrix_t<VarMat>* = nullptr>
inline Eigen::Matrix<value_type_t<VarMat>, 1, Eigen::Dynamic> rvalue(
    VarMat&& x,
    const cons_index_list<index_uni,
                          cons_index_list<index_multi, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[uni, multi] index range", name, x.rows(),
                    idxs.head_.n_);
  arena_t<Eigen::Matrix<double, 1, Eigen::Dynamic>> x_ret_vals(1, idxs.head_.ns_.size());
  arena_t<std::vector<int>> multi_idx(idxs.tail_.head_.ns_.size());
  const int n = idxs.head_.n_ - 1
  for (int i = 0; i < idxs.head_.ns_.size(); ++i) {
    const int m = idxs.tail_.head_.ns_[i] - 1;
    math::check_range("matrix[multi] subset range", name, x_ref.cols(), m + 1);
    multi_idx[n] = m;
    x_ret_vals.coeffRef(i) = x.val().coeffRef(n, m);
  }
  stan::math::var_value<Eigen::Matrix<double, 1, Eigen::Dynamic>> x_ret(x_ret_vals);
  stan::math::reverse_pass_callback([x_vi = x.vi_, ret_vi = x_ret.vi_, n, multi_idx]() mutable {
    for (Eigen::Index i = 0; i < multi_idx.size(); ++i) {
      x_vi->adj_.coeffRef(n, multi_idx[i]) += ret_vi->adj_.coeffRef(i);
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
 * @tparam VarMat Eigen type with dynamic rows and columns.
 * @param[in] x Matrix to index.
 * @param[in] idxs Pair of multiple indexes (from 1).
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename VarMat,
          require_var_matrix_t<VarMat>* = nullptr>
inline auto rvalue(VarMat&& x,
    const cons_index_list<index_multi,
                          cons_index_list<index_uni, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[multi, uni] rvalue range", name, x.cols(),
                    idxs.tail_.head_.n_);
  arena_t<Eigen::Matrix<double, Eigen::Dynamic, 1>> x_ret_val(idxs.head_.ns_.size());
  arena_t<std::vector<int>> multi_idx(idxs.head_.ns_.size());
  const int m = idxs.tail_.head_.n_ - 1;
  for (int i = 0; i < idxs.head_.ns_.size(); ++i) {
    const int n = idxs.head_.ns_[i] - 1;
    math::check_range("matrix[multi, uni] rvalue range", name, x_ref.rows(), n + 1);
    multi_idx[i] = n;
    x_ret_val.coeffRef(i) = x.val().coeffRef(n, m);
  }
  stan::math::var_value<Eigen::Matrix<double, Eigen::Dynamic, 1>> x_ret(x_ret_val);
  stan::math::reverse_pass_callback([x_vi = x.vi_, ret_vi = x_ret.vi_, m, multi_idx](){
    for (Eigen::Index i = 0; i < multi_idx.size(); ++i) {
      x_vi->adj_.coeffRef(multi_idx[i], m) += ret_vi->adj_.coeffRef(i);
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
 * @tparam VarMat An eigen matrix
 * @param[in] x Matrix to index.
 * @param[in] idxs Pair of multiple indexes.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename VarMat,
          require_var_matrix_t<VarMat>* = nullptr>
inline auto rvalue(VarMat&& x,
    const cons_index_list<index_multi,
                          cons_index_list<index_multi, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  const int rows = idxs.head_.ns_.size();
  const int cols = idxs.tail_.head_.ns_.size();
  arena_t<value_type_t<VarMat>> x_ret_val(rows, cols);
  arena_t<std::vector<std::array<int, 2>>> multi_idx(idxs.head_.ns_.size());
  for (int j = 0; j < cols; ++j) {
    const int n = idxs.tail_.head_.ns_[j];
    for (int i = 0; i < rows; ++i) {
      const int m = idxs.head_.ns_[i];
      multi_idx.push_back(std::array<int, 2>{m, n});
      math::check_range("matrix[multi,multi] row index", name, x.rows(), m);
      x_ret_val.coeffRef(i, j) = x.val().coeff(m - 1, n - 1);
    }
  }
  stan::math::var_value<plain_type_t<value_type_t<VarMat>>> x_ret(x_ret_val);
  stan::math::reverse_pass_callback([x_vi = x.vi_, ret_vi = x_ret.vi_, multi_idx](){
    for (int j = 0; j < multi_idx.size(); ++j) {
      const auto idx = multi_idx[j];
      const int n = idx[0];
      const int m = idx[1];
      x_vi->adj_.coeffRef(m - 1, n - 1) = ret_vi->adj_.coeffRef(j);
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
 * @tparam VarMat An eigen matrix
 * @param[in] x Eigen matrix.
 * @param[in] idxs Index consisting of one uni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename VarMat, typename Idx,
          require_dense_dynamic_t<VarMat>* = nullptr,
          require_not_same_t<std::decay_t<Idx>, index_uni>* = nullptr>
inline auto rvalue(VarMat&& x,
    const cons_index_list<Idx, cons_index_list<index_multi, nil_index_list>>&
        idxs,
    const char* name = "ANON", int depth = 0) {
  const int rows = rvalue_index_size(idxs.head_, x.rows());
  const int cols = rvalue_index_size(idxs.tail_.head_, x.cols());
  arena_t<value_type_t<VarMat>> x_ret_val(rows, cols);
  arena_t<std::vector<int>> multi_idx(idxs.head_.ns_.size());
  for (int j = 0; j < idxs.tail_.head_.ns_.size(); ++j) {
    const int n = idxs.tail_.head_.ns_[j] - 1;
    multi_idx.push_back(n);
    math::check_range("matrix[..., multi] col index", name, x.cols(), n + 1);
    x_ret_val.col(j) = rvalue(x.val().col(n - 1), index_list(idxs.head_), name, depth + 1);
  }
  var_value<Eigen::MatrixXd<double, -1, -1>> x_ret(rows, idxs.tail_.head_.ns_.size());
  stan::math::reverse_pass_callback([x_vi = x.vi_, ret_vi = x_ret.vi_, multi_idx, head = idxs.head_](){
    for (size_t i = 0; i < multi_idx[i]; ++i) {
      x_vi->adj_.col(multi_idx[i]) += rvalue(ret_vi->adj_.col(i), index_list(idxs.head_), name, depth + 1);      
    }
  });
  return x_ret;
}


}  // namespace model
}  // namespace stan
#endif

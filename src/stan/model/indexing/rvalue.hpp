#ifndef STAN_MODEL_INDEXING_RVALUE_HPP
#define STAN_MODEL_INDEXING_RVALUE_HPP

#include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>
#include <stan/model/indexing/deep_copy.hpp>
#include <stan/model/indexing/index.hpp>
#include <stan/model/indexing/index_list.hpp>
#include <stan/model/indexing/rvalue_at.hpp>
#include <stan/model/indexing/rvalue_index_size.hpp>
#include <type_traits>
#include <vector>

namespace stan {

namespace model {

// all indexing from 1

/**
 * Return the result of indexing a specified value with
 * a nil index list, which just returns the value.
 *
 * Types:  T[] : T
 *
 * @tparam T Scalar type.
 * @param[in] c Value to index.
 * @return Input value.
 */
template <typename T>
inline T rvalue(T&& c, const nil_index_list& /*idx*/, const char* /*name*/ = "",
                int /*depth*/ = 0) {
  return std::forward<T>(c);
}

/**
 * Return the result of indexing a type without taking a subset. Mostly used as
 * an intermediary rvalue function when doing multiple subsets.
 *
 * Types:  type[omni] : type
 *
 * @tparam T Any type.
 * @param[in] an object.
 * @param[in] idx Index consisting of one omni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename T>
inline T rvalue(T&& a, const cons_index_list<index_omni, nil_index_list>& idx,
                const char* name = "ANON", int depth = 0) {
  return std::forward<T>(a);
}

/**
 * Return the result of indexing a type without taking a subset
 *
 * Types:  type[omni, omni] : type
 *
 * @tparam T Any type.
 * @param[in] an object.
 * @param[in] idx Index consisting of two omni-indices.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename T>
inline T rvalue(
    T&& a,
    const cons_index_list<index_omni,
                          cons_index_list<index_omni, nil_index_list>>& idx,
    const char* name = "ANON", int depth = 0) {
  return std::forward<T>(a);
}

/**
 * Return the result of indexing the specified Eigen vector with a
 * sequence containing one single index, returning a scalar.
 *
 * Types:  vec[single] : scal
 *
 * @tparam EigVec An eigen vector
 * @param[in] v Vector being indexed.
 * @param[in] idx One single index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing vector.
 */
template <typename EigVec, require_eigen_vector_t<EigVec>* = nullptr>
inline auto rvalue(EigVec&& v,
                   const cons_index_list<index_uni, nil_index_list>& idx,
                   const char* name = "ANON", int depth = 0) {
  using stan::math::to_ref;
  math::check_range("vector[uni] indexing", name, v.size(), idx.head_.n_);
  return to_ref(v).coeffRef(idx.head_.n_ - 1);
}

/**
 * Return the result of indexing the specified Eigen vector min_max index,
 *  returning a vector
 *
 * Types:  vec[min_max] : vector
 *
 * @tparam EigVec An eigen vector
 * @param[in] v Vector being indexed.
 * @param[in] idx One single index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing vector.
 */
template <typename EigVec, require_eigen_vector_t<EigVec>* = nullptr>
inline auto rvalue(EigVec&& v,
                   const cons_index_list<index_min_max, nil_index_list>& idx,
                   const char* name = "ANON", int depth = 0) {
  math::check_range("vector[min_max] min indexing", name, v.size(),
                    idx.head_.min_);
  math::check_range("vector[min_max] max indexing", name, v.size(),
                    idx.head_.max_);
  if (idx.head_.positive_idx_) {
    return v.segment(idx.head_.min_ - 1, idx.head_.max_ - (idx.head_.min_ - 1))
        .eval();
  } else {
    return v.segment(idx.head_.max_ - 1, idx.head_.min_ - (idx.head_.max_ - 1))
        .reverse()
        .eval();
  }
}

/**
 * Return the result of indexing the specified Eigen vector with a random
 * access index.
 *
 * Types:  vec[multi] : vec
 *
 * @tparam Vec Eigen type with either dynamic rows or columns, but not both.
 * @param[in] x Row vector variable to be assigned.
 * @param[in] idxs Sequence of integers.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the value size isn't the same as
 * the indexed size.
 */
template <typename Vec, require_eigen_vector_t<Vec>* = nullptr>
inline plain_type_t<Vec> assign(
    Vec&& v, const cons_index_list<index_multi, nil_index_list>& idxs,
    const char* name = "ANON", int depth = 0) {
  const auto v_size = v.size();
  const auto& v_ref = stan::math::to_ref(v);
  plain_type_t<Vec> ret_v(idxs.head_.ns_.size());
  for (int i = 0; i < idxs.head_.ns_.size(); ++i) {
    math::check_range("vector[multi] indexing", name, v_ref.size(),
                      idxs.head_.ns_[i]);
    ret_v.coeffRef(i) = v_ref.coeffRef(idxs.head_.ns_[i] - 1);
  }
  return ret_v;
}

/**
 * Return the result of indexing the specified Eigen vector with a
 * sequence containing one multiple index, returning a vector.
 *
 * Types: vec[general] : vec
 *
 * @tparam EigMat An eigen vector
 * @tparam I Multi-index type.
 * @param[in] v Eigen vector.
 * @param[in] idx Index consisting of one multi-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing vector.
 */
template <typename EigVec, typename I,
          require_eigen_vector_t<EigVec>* = nullptr>
inline plain_type_t<EigVec> rvalue(
    EigVec&& v, const cons_index_list<I, nil_index_list>& idx,
    const char* name = "ANON", int depth = 0) {
  const int size = rvalue_index_size(idx.head_, v.size());
  const auto& v_ref = stan::math::to_ref(v);
  plain_type_t<EigVec> ret_v(size);
  for (int i = 0; i < size; ++i) {
    int n = rvalue_at(i, idx.head_);
    math::check_range("vector[...] indexing", name, v_ref.size(), n);
    ret_v.coeffRef(i) = v_ref.coeff(n - 1);
  }
  return ret_v;
}

/**
 * Return the result of indexing the Eigen matrix with a
 * sequence consisting of one single index, returning a row vector.
 *
 * Types:  mat[single] : rowvec
 *
 * @tparam EigMat An eigen matrix
 * @param[in] a Eigen matrix.
 * @param[in] idx Index consisting of one uni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(EigMat&& a,
                   const cons_index_list<index_uni, nil_index_list>& idx,
                   const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[uni] indexing", name, a.rows(), idx.head_.n_);
  return a.row(idx.head_.n_ - 1).eval();
}

/**
 * Return the result of indexing the Eigen matrix with a min index
 * returning back a block of rows min:N and all cols
 *
 * Types:  mat[min:] : matrix
 *
 * @tparam EigMat An eigen matrix
 * @param[in] a Eigen matrix.
 * @param[in] idx Index consisting of one uni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(EigMat&& a,
                   const cons_index_list<index_min, nil_index_list>& idx,
                   const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[min] indexing", name, a.rows(), idx.head_.min_ - 1);
  return a
      .block(idx.head_.min_ - 1, 0, a.rows() - (idx.head_.min_ - 1), a.cols())
      .eval();
}

/**
 * Return the Eigen matrix at the specified max index
 *
 * Types:  mat[:max] = mat
 *
 * @tparam EigMat Eigen type with dynamic rows and columns.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs An indexing from the start of the container up to
 * the specified maximum index (inclusive).
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename EigMat,
          stan::internal::require_all_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(EigMat&& x,
                   const cons_index_list<index_max, nil_index_list>& idxs,
                   const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[max] indexing range", name, x.cols(),
                    idxs.head_.max_);
  return x.block(0, 0, idxs.head_.max_, x.cols()).eval();
}

/**
 * Return the Eigen matrix at the specified min_max index.
 *
 * Types:  mat[min_max] = mat
 *
 * @tparam EigMat Eigen type with dynamic rows and columns.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs An indexing from the start of the container up to
 * the specified maximum index (inclusive).
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename EigMat,
          stan::internal::require_all_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(EigMat&& x,
                   const cons_index_list<index_min_max, nil_index_list>& idxs,
                   const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[min_max] max row indexing", name, x.rows(),
                    idxs.head_.max_);
  math::check_range("matrix[min_max] min row indexing", name, x.rows(),
                    idxs.head_.min_);
  if (idxs.head_.positive_idx_) {
    return x.block(idxs.head_.min_ - 1, 0, idxs.head_.max_ - 1, x.cols())
        .eval();
  } else {
    return x.block(idxs.head_.max_ - 1, 0, idxs.head_.min_ - 1, x.cols())
        .colwise()
        .reverse()
        .eval();
  }
}

/**
 * Return the specified Eigen matrix at the specified multi index.
 *
 * Types:  mat[multi] = mat
 *
 * @tparam EigMat Eigen type with dynamic rows and columns.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs An indexing from the start of the container up to
 * the specified maximum index (inclusive).
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename EigMat,
          stan::internal::require_all_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline plain_type_t<EigMat> rvalue(
    EigMat&& x, const cons_index_list<index_multi, nil_index_list>& idxs,
    const char* name = "ANON", int depth = 0) {
  const auto& x_ref = stan::math::to_ref(x);
  using eig_mat = std::decay_t<EigMat>;
  plain_type_t<EigMat> x_ret(idxs.head_.ns_.size(), x.cols());
  for (int i = 0; i < idxs.head_.ns_.size(); ++i) {
    const int n = idxs.head_.ns_[i];
    math::check_range("matrix[multi] subset range", name, x_ref.rows(), n);
    x_ret.row(i) = x_ref.row(n - 1);
  }
  return x_ret;
}

/**
 * Random access assignment to an eigen matrix.
 *
 * Types:  mat[general] = mat
 *
 * @tparam EigMat Eigen type with dynamic rows and columns.
 * @tparam L Multiple index type.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs Any of the index types.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename EigMat, typename L,
          stan::internal::require_all_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline plain_type_t<EigMat> rvalue(
    EigMat&& x, const cons_index_list<L, nil_index_list>& idxs,
    const char* name = "ANON", int depth = 0) {
  const int x_idx_rows = rvalue_index_size(idxs.head_, x.rows());
  const auto& x_ref = stan::math::to_ref(x);
  plain_type_t<EigMat> x_ret(x_idx_rows, x.cols());
  for (int i = 0; i < x_idx_rows; ++i) {
    const int m = rvalue_at(i, idxs.head_);
    math::check_range("matrix[...] assign range", name, x.rows(), m);
    x_ret.row(i) = x_ref.row(m - 1);
  }
  return x_ref;
}

/**
 * Return the result of indexing the specified Eigen matrix with two min_max
 * indices, returning back a block of the Eigen matrix.
 *
 * Types:  mat[min_max, min_max] : matrix
 *
 * @tparam EigMat An eigen matrix
 * @param[in] a Eigen matrix.
 * @param[in] idx Index consisting of one uni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(
    EigMat&& mat,
    const cons_index_list<index_min_max,
                          cons_index_list<index_min_max, nil_index_list>>& idx,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[min_max, min_max] min row indexing", name,
                    mat.rows(), idx.head_.min_);
  math::check_range("matrix[min_max, min_max] max row indexing", name,
                    mat.rows(), idx.head_.min_);
  math::check_range("matrix[min_max, min_max] min column indexing", name,
                    mat.cols(), idx.tail_.head_.min_);
  math::check_range("matrix[min_max, min_max] max column indexing", name,
                    mat.cols(), idx.tail_.head_.min_);
  if (idx.head_.positive_idx_) {
    if (idx.tail_.head_.positive_idx_) {
      return mat
          .block(idx.head_.min_ - 1, idx.tail_.head_.min_ - 1,
                 idx.head_.max_ - (idx.head_.min_ - 1),
                 idx.tail_.head_.max_ - (idx.tail_.head_.min_ - 1))
          .eval();
    } else {
      return mat
          .block(idx.head_.min_ - 1, idx.tail_.head_.max_ - 1,
                 idx.head_.max_ - (idx.head_.min_ - 1),
                 idx.tail_.head_.min_ - (idx.tail_.head_.max_ - 1))
          .rowwise()
          .reverse()
          .eval();
    }
  } else {
    if (idx.tail_.head_.positive_idx_) {
      return mat
          .block(idx.head_.max_ - 1, idx.tail_.head_.min_ - 1,
                 idx.head_.min_ - (idx.head_.max_ - 1),
                 idx.tail_.head_.max_ - (idx.tail_.head_.min_ - 1))
          .colwise()
          .reverse()
          .eval();
    } else {
      return mat
          .block(idx.head_.max_ - 1, idx.tail_.head_.max_ - 1,
                 idx.head_.min_ - (idx.head_.max_ - 1),
                 idx.tail_.head_.min_ - (idx.tail_.head_.max_ - 1))
          .reverse()
          .eval();
    }
  }
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * sequence consisting of two single indexes, returning a scalar.
 *
 * Types:  mat[single,single] : scalar
 *
 * @tparam EigMat An eigen type
 * @param[in] a Matrix to index.
 * @param[in] idx Pair of single indexes.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat, require_eigen_t<EigMat>* = nullptr>
inline auto rvalue(
    EigMat&& a,
    const cons_index_list<index_uni,
                          cons_index_list<index_uni, nil_index_list>>& idx,
    const char* name = "ANON", int depth = 0) {
  using stan::math::to_ref;
  math::check_range("matrix[uni,uni] indexing, row", name, a.rows(),
                    idx.head_.n_);
  math::check_range("matrix[uni,uni] indexing, col", name, a.cols(),
                    idx.tail_.head_.n_);
  return to_ref(a).coeffRef(idx.head_.n_ - 1, idx.tail_.head_.n_ - 1);
}

/**
 * Random access to a vector's cells to a row of an eigen matrix.
 *
 * Types:  mat[uni, multi] = vector
 *
 * @tparam EigMat Eigen type with dynamic rows and columns.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs Pair of multiple indexes (from 1).
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename EigMat,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline Eigen::Matrix<value_type_t<EigMat>, 1, -1> rvalue(
    EigMat&& x,
    const cons_index_list<index_uni,
                          cons_index_list<index_multi, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[uni, multi] assign range", name, x.cols(),
                    idxs.head_.n_);
  const auto& x_ref = stan::math::to_ref(x);
  Eigen::Matrix<value_type_t<EigMat>, 1, -1> x_ret(1,
                                                   idxs.tail_.head_.ns_.size());
  for (int i = 0; i < idxs.tail_.head_.ns_.size(); ++i) {
    math::check_range("matrix[uni, multi] assign range", name, x.cols(),
                      idxs.tail_.head_.ns_[i]);
    x_ret.coeffRef(i)
        = x_ref.coeffRef(idxs.head_.n_ - 1, idxs.tail_.head_.ns_[i] - 1);
  }
  return x_ret;
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * sequence consisting of a pair of multiple indexes, returning a
 * a matrix.
 *
 * Types:  mat[multiple,multiple] : mat
 *
 * @tparam EigMat An eigen matrix
 * @tparam I Type of multiple index.
 * @param[in] a Matrix to index.
 * @param[in] idx Pair of multiple indexes.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline plain_type_t<EigMat> rvalue(
    EigMat&& a,
    const cons_index_list<index_multi,
                          cons_index_list<index_multi, nil_index_list>>& idx,
    const char* name = "ANON", int depth = 0) {
  const auto& a_ref = stan::math::to_ref(a);
  const int rows = idx.head_.ns_.size();
  const int cols = idx.tail_.head_.ns_.size();
  plain_type_t<EigMat> a_ret(rows, cols);
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      const int m = idx.head_.ns_[i];
      const int n = idx.tail_.head_.ns_[j];
      math::check_range("matrix[multi,multi] row index", name, a_ref.rows(), m);
      math::check_range("matrix[multi,multi] col index", name, a_ref.cols(), n);
      a_ret.coeffRef(i, j) = a_ref.coeffRef(m - 1, n - 1);
    }
  }
  return a_ret;
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * sequence consisting of one single index.
 *
 * Types:  mat[L, single] : vec
 *
 * @tparam EigMat An eigen matrix
 * @param[in] a Eigen matrix.
 * @param[in] idx Index consisting of one uni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 */
template <typename EigMat, typename L,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(
    EigMat&& a,
    const cons_index_list<L, cons_index_list<index_uni, nil_index_list>>& idx,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[..., uni] indexing", name, a.cols(),
                    idx.tail_.head_.n_);
  return deep_copy(rvalue(a.col(idx.tail_.head_.n_ - 1), index_list(idx.head_),
                          name, depth + 1));
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * sequence consisting of multi index.
 *
 * Types:  mat[L, multi] : vec
 *
 * @tparam EigMat An eigen matrix
 * @param[in] a Eigen matrix.
 * @param[in] idx Index consisting of one uni-index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat, typename L,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr,
          require_not_same_t<std::decay_t<L>, index_uni>* = nullptr>
inline plain_type_t<EigMat> rvalue(
    EigMat&& a,
    const cons_index_list<L, cons_index_list<index_multi, nil_index_list>>& idx,
    const char* name = "ANON", int depth = 0) {
  const auto& a_ref = stan::math::to_ref(a);
  const int rows = rvalue_index_size(idx.head_, a_ref.rows());
  const int cols = rvalue_index_size(idx.tail_.head_, a_ref.cols());
  plain_type_t<EigMat> a_ret(rows, idx.tail_.head_.ns_.size());
  for (int j = 0; j < idx.tail_.head_.ns_.size(); ++j) {
    int n = idx.tail_.head_.ns_[j];
    math::check_range("matrix[..., multi] col index", name, a_ref.cols(), n);
    a_ret.col(j)
        = rvalue(a_ref.col(n - 1), index_list(idx.head_), name, depth + 1);
  }
  return a_ret;
}

/**
 * Return the Eigen matrix with all columns and a slice of rows.
 *
 * Types:  mat[L, omni] = mat
 *
 * @tparam EigMat Eigen type with dynamic rows and columns.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs Pair of multiple indexes (from 1).
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename EigMat, typename L,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(
    EigMat&& x,
    const cons_index_list<L, cons_index_list<index_omni, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  return deep_copy(rvalue(x, index_list(idxs.head_), name, depth + 1));
}

/**
 * Return the Eigen matrix at the specified min
 * index to the specified matrix value.
 *
 * Types:  mat[L, min] = mat
 *
 * @tparam EigMat Eigen type with dynamic rows and columns.
 * @tparam L An index.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs Container holding a row index and an index from a minimum
 * index (inclusive) to the end of a container.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename EigMat, typename L,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(
    EigMat&& x,
    const cons_index_list<L, cons_index_list<index_min, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  const int start_col = idxs.tail_.head_.min_ - 1;
  math::check_range("matrix[..., min] indexing", name, x.cols(),
                    idxs.tail_.head_.min_);
  return deep_copy(rvalue(x.block(0, start_col, x.rows(), x.cols() - start_col),
                          index_list(idxs.head_), name, depth + 1));
}

/**
 * Return the Eigen matrix at the specified max index
 *
 * Types:  mat[L, max] = mat
 *
 * @tparam EigMat Eigen type with dynamic rows and columns.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs Index holding a row index and an index from the start of the
 * container up to the specified maximum index (inclusive).
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename EigMat, typename L,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(
    EigMat&& x,
    const cons_index_list<L, cons_index_list<index_max, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[..., max] indexing", name, x.cols(),
                    idxs.tail_.head_.max_);
  return deep_copy(rvalue(x.block(0, 0, x.rows(), idxs.tail_.head_.max_),
                          index_list(idxs.head_), name, depth + 1));
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * min_max_index returning a block from min to max.
 *
 * Types:  mat[L, min_max] : matrix
 *
 * @tparam EigMat An eigen matrix
 * @tparam L Type of multiple index.
 * @param[in] a Matrix to index.
 * @param[in] idx Pair multiple index and single index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat, typename L,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline auto rvalue(
    EigMat&& a,
    const cons_index_list<L, cons_index_list<index_min_max, nil_index_list>>&
        idx,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[..., min_max] indexing", name, a.rows(),
                    idx.tail_.head_.min_);
  math::check_range("matrix[..., min_max] indexing", name, a.rows(),
                    idx.tail_.head_.max_);
  if (idx.tail_.head_.positive_idx_) {
    return deep_copy(
        rvalue(a.block(0, idx.tail_.head_.min_ - 1, a.rows(),
                       idx.tail_.head_.max_ - (idx.tail_.head_.min_ - 1)),
               index_list(idx.head_), name, depth + 1));
  } else {
    return deep_copy(
        rvalue(a.block(0, idx.tail_.head_.max_ - 1, a.rows(),
                       idx.tail_.head_.min_ - (idx.tail_.head_.max_ - 1))
                   .rowwise()
                   .reverse(),
               index_list(idx.head_), name, depth + 1));
  }
}

/**
 * Random access of a matrix column with a multi index on the row.
 * Types:  mat[multi, uni] = vector
 *
 * @tparam EigMat Eigen type with dynamic rows and columns.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs Pair of multiple indexes (from 1).
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 */
template <typename EigMat,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline Eigen::Matrix<value_type_t<EigMat>, -1, 1> rvalue(
    EigMat&& x,
    const cons_index_list<index_multi,
                          cons_index_list<index_uni, nil_index_list>>& idxs,
    const char* name = "ANON", int depth = 0) {
  math::check_range("matrix[multi, uni] rvalue range", name, x.cols(),
                    idxs.tail_.head_.n_);
  const auto& x_ref = stan::math::to_ref(x);
  Eigen::Matrix<value_type_t<EigMat>, -1, 1> x_ret(idxs.head_.ns_.size(), 1);
  for (int i = 0; i < idxs.head_.ns_.size(); ++i) {
    math::check_range("matrix[multi, uni] rvalue range", name, x_ref.rows(),
                      idxs.head_.ns_[i]);
    x_ret.coeffRef(i)
        = x_ref.coeffRef(idxs.head_.ns_[i] - 1, idxs.tail_.head_.n_ - 1);
  }
  return x_ret;
}

/**
 * Return the result of indexing the specified Eigen matrix with a
 * sequence consisting of a pair of multiple indexes, returning a
 * a matrix.
 *
 * Types:  mat[general, general] : mat
 *
 * @tparam EigMat An eigen matrix
 * @tparam I Type of multiple index.
 * @param[in] a Matrix to index.
 * @param[in] idx Pair of multiple indexes.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing matrix.
 */
template <typename EigMat, typename I1, typename I2,
          stan::internal::require_eigen_dense_dynamic_t<EigMat>* = nullptr>
inline plain_type_t<EigMat> rvalue(
    EigMat&& a,
    const cons_index_list<I1, cons_index_list<I2, nil_index_list>>& idx,
    const char* name = "ANON", int depth = 0) {
  const auto& a_ref = stan::math::to_ref(a);
  const int rows = rvalue_index_size(idx.head_, a_ref.rows());
  const int cols = rvalue_index_size(idx.tail_.head_, a_ref.cols());
  plain_type_t<EigMat> x_ret(rows, cols);
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      int m = rvalue_at(i, idx.head_);
      int n = rvalue_at(j, idx.tail_.head_);
      math::check_range("matrix[..., ...] row index", name, a_ref.rows(), m);
      math::check_range("matrix[..., ...] col index", name, a_ref.cols(), n);
      x_ret.coeffRef(i, j) = a_ref.coeffRef(m - 1, n - 1);
    }
  }
  return x_ret;
}

/**
 * Return the result of indexing the specified array with
 * a list of indexes beginning with a single index;  the result is
 * determined recursively.  Note that arrays are represented as
 * standard library vectors.
 *
 * Types:  std::vector<T>[single | L] : T[L]
 *
 * @tparam T Type of list elements.
 * @tparam L Index list type for indexes after first index.
 * @param[in] v Container of list elements.
 * @param[in] idx Index list beginning with single index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing array.
 */
template <typename StdVec, typename L, require_std_vector_t<StdVec>* = nullptr>
inline auto rvalue(StdVec&& v, const cons_index_list<index_uni, L>& idx,
                   const char* name = "ANON", int depth = 0) {
  math::check_range("array[uni, ...] index", name, v.size(), idx.head_.n_);
  return rvalue(v[idx.head_.n_ - 1], idx.tail_, name, depth + 1);
}

/**
 * Return the result of indexing the specified array with
 * a single index.
 *
 * Types:  std::vector<T>[single] : T
 *
 * @tparam StdVec a standard vector
 * @param[in] c Container of list elements.
 * @param[in] idx Index list beginning with single index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing array.
 */
template <typename StdVec, typename L, require_std_vector_t<StdVec>* = nullptr>
inline auto rvalue(StdVec&& c,
                   const cons_index_list<index_uni, nil_index_list>& idx,
                   const char* name = "ANON", int depth = 0) {
  const int n = idx.head_.n_;
  math::check_range("array[uni, ...] index", name, c.size(), n);
  return c[n - 1];
}

/**
 * Return the result of indexing the specified array with
 * a list of indexes beginning with a multiple index;  the result is
 * determined recursively.  Note that arrays are represented as
 * standard library vectors.
 *
 * Types:  std::vector<T>[multiple | L] : std::vector<T[L]>
 *
 * @tparam T Type of list elements.
 * @tparam L Index list type for indexes after first index.
 * @param[in] v Container of list elements.
 * @param[in] idx Index list beginning with multiple index.
 * @param[in] name String form of expression being evaluated.
 * @param[in] depth Depth of indexing dimension.
 * @return Result of indexing array.
 */
template <typename StdVec, typename I, typename L,
          require_std_vector_t<StdVec>* = nullptr>
inline auto rvalue(StdVec&& v, const cons_index_list<I, L>& idx,
                   const char* name = "ANON", int depth = 0) {
  using inner_type = plain_type_t<decltype(
      rvalue(v[rvalue_at(0, idx.head_) - 1], idx.tail_))>;
  std::vector<inner_type> result;
  const int index_size = rvalue_index_size(idx.head_, v.size());
  if (index_size > 0) {
    result.reserve(index_size);
  }
  for (int i = 0; i < index_size; ++i) {
    int n = rvalue_at(i, idx.head_);
    math::check_range("array[..., ...] index", name, v.size(), n);
    result.emplace_back(rvalue(v[n - 1], idx.tail_, name, depth + 1));
  }
  return result;
}

}  // namespace model
}  // namespace stan
#endif

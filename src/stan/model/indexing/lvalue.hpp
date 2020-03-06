#ifndef STAN_MODEL_INDEXING_LVALUE_HPP
#define STAN_MODEL_INDEXING_LVALUE_HPP

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <stan/math/prim.hpp>
#include <stan/model/indexing/index.hpp>
#include <stan/model/indexing/index_list.hpp>
#include <stan/model/indexing/rvalue_at.hpp>
#include <stan/model/indexing/rvalue_index_size.hpp>
#include <vector>

namespace stan {

namespace model {

/**
 * Assign the specified scalar rvalue to the specified scalar lvalue.  The index
 * list's type must be `nil_index_list`, but its value will be
 * ignored.  The last two arguments are also ignored.
 *
 * @tparam T lvalue variable type
 * @tparam U rvalue variable type, which must be assignable to `T`
 * @param[in,out] x lvalue
 * @param[in] y rvalue
 * @param[in] name Name of lvalue variable (default "ANON"); ignored
 * @param[in] depth Indexing depth (default 0; ignored
 */
template <typename T, typename U, typename = require_all_stan_scalar_t<U, T>>
inline void assign(T& x, const nil_index_list& /* idxs */, U y,
                   const char* name = "ANON", int depth = 0) {
  x = y;
}

/**
 * Assign the specified non-scalar rvalue to the specified non-scalar lvalue.
 * The index list's type must be `nil_index_list`, but its value will be
 * ignored.  The last two arguments are also ignored.
 *
 * @tparam T lvalue variable type
 * @tparam U rvalue variable type, which must be assignable to `T`
 * @param[in,out] x lvalue
 * @param[in] y rvalue
 * @param[in] name Name of lvalue variable (default "ANON"); ignored
 * @param[in] depth Indexing depth (default 0; ignored
 */
template <typename T, typename U,
          typename = require_all_not_stan_scalar_t<U, T>,
          typename = require_t<std::is_assignable<std::decay_t<T>,
            std::decay_t<U>>>>
inline void assign(T& x, const nil_index_list& /* idxs */, U&& y,
                   const char* name = "ANON", int depth = 0) {
  x = std::forward<U>(y);
}

/**
 * Assign the specified standard vector rvalue to the specified
 * standard vector lvalue.
 *
 * @tparam Vec1 vector type to be assigned to
 * @tparam Vec2 vector type with scalar that must be assignable to scalar out
 *  `Vec1`.
 * @param[in] x lvalue variable
 * @param[in] y rvalue variable
 * @param[in] name name of lvalue variable (default "ANON").
 * @param[in] depth indexing depth (default 0).
 */
template <typename Vec1, typename Vec2,
          typename = require_all_std_vector_t<Vec1, Vec2>>
inline void assign(Vec1&& x, const nil_index_list& /* idxs */, Vec2&& y,
                   const char* name = "ANON", int depth = 0) {
  x.resize(y.size());
  for (size_t i = 0; i < y.size(); ++i) {
    assign(x[i], nil_index_list(), y[i], name, depth + 1);
  }
}

/**
 * Assign the specified Eigen vector at the specified single index
 * to the specified value.
 *
 * Types: vec[uni] <- scalar
 *
 * @tparam EigVec Type type of the Eigen Row or Column Vector.
 * @tparam Scalar Type of value (must be assignable to T).
 * @param[in] x Vector variable to be assigned.
 * @param[in] idxs Sequence of one single index (from 1).
 * @param[in] y Value scalar.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If the index is out of bounds.
 */
template <typename EigVec, typename Scalar,
          typename = require_eigen_vector_t<EigVec>,
          typename = require_stan_scalar_t<Scalar>>
inline void assign(EigVec& x,
                   const cons_index_list<index_uni, nil_index_list>& idxs,
                   Scalar y, const char* name = "ANON", int depth = 0) {
  int i = idxs.head_.n_;
  math::check_range("vector[uni] assign range", name, x.size(), i);
  x.coeffRef(i - 1) = y;
}

/**
 * Assign the specified Eigen vector at the specified multiple
 * index to the specified value.
 *
 * Types:  vec[multi] <- vec
 *
 * @tparam LhsEigVec Type type of the Eigen Column or Row Vector.
 * @tparam I Type of multiple index.
 * @tparam RhsEigVec Type type of the Eigen Column or Row Vector.
 * @param[in] x Row vector variable to be assigned.
 * @param[in] idxs Sequence of one single index (from 1).
 * @param[in] y Value vector.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the value size isn't the same as
 * the indexed size.
 */
template <typename LhsEigVec, typename RhsEigVec, typename I,
          typename = require_not_same_t<index_uni, I>,
          typename = require_all_eigen_vector_t<LhsEigVec, RhsEigVec>>
inline void assign(LhsEigVec& x, const cons_index_list<I, nil_index_list>& idxs,
                   const RhsEigVec& y, const char* name = "ANON",
                   int depth = 0) {
  math::check_size_match("vector[multi] assign sizes", "lhs",
                         rvalue_index_size(idxs.head_, x.size()), name,
                         y.size());
  const Eigen::Ref<const typename RhsEigVec::PlainObject>& vec = y;
  for (int n = 0; n < y.size(); ++n) {
    int i = rvalue_at(n, idxs.head_);
    math::check_range("vector[multi] assign range", name, x.size(), i);
    x.coeffRef(i - 1) = vec.coeff(n);
  }
}

/**
 * Assign the specified Eigen matrix at the specified single index
 * to the specified row vector value.
 *
 * Types:  mat[uni,] = rowvec
 *
 * @tparam EigMat Type of matrix to be assigned to.
 * @tparam RowVec Type of Eigen Row Vector to be assigned from.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs Sequence of one single index (from 1).
 * @param[in] y Value row vector.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the number of columns in the row
 * vector and matrix do not match.
 */
template <typename EigMat, typename RowVec, typename = require_eigen_t<EigMat>,
          typename = require_not_eigen_vector_t<EigMat>,
          typename = require_t<is_eigen_row_vector<RowVec>>>
void assign(EigMat& x, const cons_index_list<index_uni, nil_index_list>& idxs,
            const RowVec& y, const char* name = "ANON", int depth = 0) {
  math::check_size_match("matrix[uni] assign sizes", "lhs", x.cols(), name,
                         y.cols());
  int i = idxs.head_.n_;
  math::check_range("matrix[uni] assign range", name, x.rows(), i);
  x.row(i - 1) = y;
}

/**
 * Assign the specified Eigen matrix at the specified single column index
 * to the specified column
 *
 * Types:  mat[,uni] = vec
 *
 * @tparam EigMat Type of Eigen Matrix to be assigned to.
 * @tparam ColVec Type of Eigen Column Matrix to be assigned from.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs Sequence of one single index (from 1).
 * @param[in] y Value row vector.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the number of columns in the row
 * vector and matrix do not match.
 */
template <typename EigMat, typename ColVec, typename = require_eigen_t<EigMat>,
          typename = require_not_eigen_vector_t<EigMat>,
          typename = require_t<is_eigen_col_vector<ColVec>>>
void assign(EigMat& x,
            const cons_index_list<
                index_omni, cons_index_list<index_uni, nil_index_list>>& idxs,
            const ColVec& y, const char* name = "ANON", int depth = 0) {
  math::check_size_match("matrix[uni] assign sizes", "lhs", x.cols(), name,
                         y.cols());
  int i = idxs.tail_.head_.n_;
  math::check_range("matrix[, uni] assign range", name, x.rows(), i);
  x.col(i - 1) = y;
}

/**
 * Assign the specified Eigen matrix at the specified multiple
 * index to the specified matrix value.
 *
 * Types:  mat[multi] = mat
 *
 * @tparam LhsEigMat Type of Eigen Matrix to be assigned to.
 * @tparam I Multiple index type.
 * @tparam RhsEigMat Type of Eigen Matrix to be assigned from.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs Sequence of one multiple index (from 1).
 * @param[in] y Value matrix.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and right-hand side matrix do not match.
 */
template <typename LhsEigMat, typename I, typename RhsEigMat,
          typename = require_all_eigen_t<LhsEigMat, RhsEigMat>,
          typename = require_all_not_eigen_vector_t<LhsEigMat, RhsEigMat>,
          typename = require_not_same_t<index_uni, I>>
inline void assign(LhsEigMat& x, const cons_index_list<I, nil_index_list>& idxs,
                   const RhsEigMat& y, const char* name = "ANON",
                   int depth = 0) {
  int x_idx_rows = rvalue_index_size(idxs.head_, x.rows());
  math::check_size_match("matrix[multi] assign row sizes", "lhs", x_idx_rows,
                         name, y.rows());
  math::check_size_match("matrix[multi] assign col sizes", "lhs", x.cols(),
                         name, y.cols());
  const Eigen::Ref<const typename RhsEigMat::PlainObject>& mat = y;

  for (int i = 0; i < y.rows(); ++i) {
    int m = rvalue_at(i, idxs.head_);
    math::check_range("matrix[multi] assign range", name, x.rows(), m);
    // recurse to allow double to var assign
    for (int j = 0; j < x.cols(); ++j)
      x.coeffRef(m - 1, j) = mat.coeffRef(i, j);
  }
}

/**
 * Assign the specified Eigen matrix at the specified pair of
 * single indexes to the specified scalar value.
 *
 * Types:  mat[single, single] = scalar
 *
 * @tparam EigMat Type of Eigen Matrix to assign to.
 * @tparam U Scalar type.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs Sequence of two single indexes (from 1).
 * @param[in] y Value scalar.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If either of the indices are out of bounds.
 */
template <typename EigMat, typename U, typename = require_eigen_t<EigMat>,
          typename = require_not_eigen_vector_t<EigMat>>
void assign(EigMat& x,
            const cons_index_list<
                index_uni, cons_index_list<index_uni, nil_index_list>>& idxs,
            const U& y, const char* name = "ANON", int depth = 0) {
  int m = idxs.head_.n_;
  int n = idxs.tail_.head_.n_;
  math::check_range("matrix[uni,uni] assign range", name, x.rows(), m);
  math::check_range("matrix[uni,uni] assign range", name, x.cols(), n);
  x.coeffRef(m - 1, n - 1) = y;
}

/**
 * Assign the specified Eigen matrix at the specified single and
 * multiple index to the specified row vector.
 *
 * Types:  mat[uni, multi] = rowvec
 *
 * @tparam EigMat Type of Eigen Matrix to be assigned to.
 * @tparam I Multi-index type.
 * @tparam RowVec Type of Eigen Row Vector to be assigned from.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs Sequence of single and multiple index (from 1).
 * @param[in] y Value row vector.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and right-hand side row vector do not match.
 */
template <typename EigMat, typename I, typename RowVec,
          typename = require_eigen_t<EigMat>,
          typename = require_not_eigen_vector_t<EigMat>,
          typename = require_not_same_t<index_uni, I>,
          typename = require_t<is_eigen_row_vector<RowVec>>>
inline void assign(
    EigMat& x,
    const cons_index_list<index_uni, cons_index_list<I, nil_index_list>>& idxs,
    const RowVec& y, const char* name = "ANON", int depth = 0) {
  int x_idxs_cols = rvalue_index_size(idxs.tail_.head_, x.cols());
  math::check_size_match("matrix[uni,multi] assign sizes", "lhs", x_idxs_cols,
                         name, y.cols());
  int m = idxs.head_.n_;
  const Eigen::Ref<const typename RowVec::PlainObject>& vec = y;
  math::check_range("matrix[uni,multi] assign range", name, x.rows(), m);
  for (int i = 0; i < y.size(); ++i) {
    int n = rvalue_at(i, idxs.tail_.head_);
    math::check_range("matrix[uni,multi] assign range", name, x.cols(), n);
    x.coeffRef(m - 1, n - 1) = vec.coeff(i);
  }
}

/**
 * Assign the specified Eigen matrix at the specified multiple and
 * single index to the specified vector.
 *
 * Types:  mat[multi, uni] = vec
 *
 * @tparam EigMat Type of Eigen Matrix to be assigned to.
 * @tparam I Multi-index type.
 * @tparam ColVec Type of Eigen Column Vector to be assigned from.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs Sequence of multiple and single index (from 1).
 * @param[in] y Value vector.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and right-hand side vector do not match.
 */
template <typename EigMat, typename I, typename ColVec,
          typename = require_eigen_t<EigMat>,
          typename = require_not_eigen_vector_t<EigMat>,
          typename = require_not_same_t<index_uni, I>,
          typename = require_t<is_eigen_col_vector<ColVec>>>
inline void assign(
    EigMat& x,
    const cons_index_list<I, cons_index_list<index_uni, nil_index_list>>& idxs,
    const ColVec& y, const char* name = "ANON", int depth = 0) {
  int x_idxs_rows = rvalue_index_size(idxs.head_, x.rows());
  math::check_size_match("matrix[multi,uni] assign sizes", "lhs", x_idxs_rows,
                         name, y.rows());
  int n = idxs.tail_.head_.n_;
  const Eigen::Ref<const typename ColVec::PlainObject>& vec = y;
  math::check_range("matrix[multi,uni] assign range", name, x.cols(), n);
  for (int i = 0; i < y.size(); ++i) {
    int m = rvalue_at(i, idxs.head_);
    math::check_range("matrix[multi,uni] assign range", name, x.rows(), m);
    x.coeffRef(m - 1, n - 1) = vec.coeff(i);
  }
}

/**
 * Assign the specified Eigen matrix at the specified pair of
 * multiple indexes to the specified matrix.
 *
 * Types:  mat[multi, multi] = mat
 *
 * @tparam LhsEigMat Type of Eigen Matrix to assign to.
 * @tparam I1 First multiple index type.
 * @tparam I2 Second multiple index type.
 * @tparam RhsEigMat Type of Eigen Matrix to be assigned from.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs Pair of multiple indexes (from 1).
 * @param[in] y Value matrix.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and value matrix do not match.
 */
template <typename LhsEigMat, typename I1, typename I2, typename RhsEigMat,
          typename = require_all_eigen_t<LhsEigMat, RhsEigMat>,
          typename = require_all_not_eigen_vector_t<LhsEigMat, RhsEigMat>,
          typename = require_any_not_same_t<index_uni, I1, I2>>
inline void assign(
    LhsEigMat& x,
    const cons_index_list<I1, cons_index_list<I2, nil_index_list>>& idxs,
    const RhsEigMat& y, const char* name = "ANON", int depth = 0) {
  int x_idxs_rows = rvalue_index_size(idxs.head_, x.rows());
  int x_idxs_cols = rvalue_index_size(idxs.tail_.head_, x.cols());
  math::check_size_match("matrix[multi,multi] assign sizes", "lhs", x_idxs_rows,
                         name, y.rows());
  math::check_size_match("matrix[multi,multi] assign sizes", "lhs", x_idxs_cols,
                         name, y.cols());
  const Eigen::Ref<const typename RhsEigMat::PlainObject>& mat = y;
  for (int j = 0; j < y.cols(); ++j) {
    int n = rvalue_at(j, idxs.tail_.head_);
    math::check_range("matrix[multi,multi] assign range", name, x.cols(), n);
    for (int i = 0; i < y.rows(); ++i) {
      int m = rvalue_at(i, idxs.head_);
      math::check_range("matrix[multi,multi] assign range", name, x.rows(), m);
      x.coeffRef(m - 1, n - 1) = mat.coeffRef(i, j);
    }
  }
}

/**
 * Assign the specified array (standard vector) at the specified
 * index list beginning with a single index to the specified value.
 *
 * This function operates recursively to carry out the tail
 * indexing.
 *
 * Types:  x[uni | L] = y
 *
 * @tparam T Assigned vector member type.
 * @tparam L Type of tail of index list.
 * @tparam U Value scalar type (must be assignable to indexed
 * variable).
 * @param[in] x Array variable to be assigned.
 * @param[in] idxs List of indexes beginning with single index
 * (from 1).
 * @param[in] y Value.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions do not match in the
 * tail assignment.
 */
template <typename T, typename L, typename U>
inline void assign(std::vector<T>& x, const cons_index_list<index_uni, L>& idxs,
                   const U& y, const char* name = "ANON", int depth = 0) {
  int i = idxs.head_.n_;
  math::check_range("vector[uni,...] assign range", name, x.size(), i);
  assign(x[i - 1], idxs.tail_, y, name, depth + 1);
}

/**
 * Assign the specified array (standard vector) at the specified
 * index list beginning with a multiple index to the specified value.
 *
 * This function operates recursively to carry out the tail
 * indexing.
 *
 * Types:  x[multi | L] = y
 *
 * @tparam T Assigned vector member type.
 * @tparam I Type of multiple index heading index list.
 * @tparam L Type of tail of index list.
 * @tparam U Value scalar type (must be assignable to indexed
 * variable).
 * @param[in] x Array variable to be assigned.
 * @param[in] idxs List of indexes beginning with multiple index
 * (from 1).
 * @param[in] y Value.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the size of the multiple indexing
 * and size of first dimension of value do not match, or any of
 * the recursive tail assignment dimensions do not match.
 */
template <typename T, typename I, typename L, typename U,
          typename = require_not_same_t<index_uni, I>>
inline void assign(std::vector<T>& x, const cons_index_list<I, L>& idxs,
                   const std::vector<U>& y, const char* name = "ANON",
                   int depth = 0) {
  int x_idx_size = rvalue_index_size(idxs.head_, x.size());
  math::check_size_match("vector[multi,...] assign sizes", "lhs", x_idx_size,
                         name, y.size());
  for (size_t n = 0; n < y.size(); ++n) {
    int i = rvalue_at(n, idxs.head_);
    math::check_range("vector[multi,...] assign range", name, x.size(), i);
    assign(x[i - 1], idxs.tail_, y[n], name, depth + 1);
  }
}

}  // namespace model
}  // namespace stan
#endif

#ifndef STAN_MODEL_INDEXING_LVALUE_HPP
#define STAN_MODEL_INDEXING_LVALUE_HPP

#include <stan/math/prim.hpp>
#include <stan/model/indexing/index.hpp>
#include <stan/model/indexing/index_list.hpp>
#include <stan/model/indexing/rvalue_at.hpp>
#include <stan/model/indexing/rvalue_index_size.hpp>
#include <type_traits>
#include <vector>

namespace stan {

namespace model {

/**
 * Assign the specified rvalue to the specified lvalue.  The index
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
template <typename T, typename U>
inline void assign(T& x, const nil_index_list& /* idxs */, U&& y,
                   const char* name = "ANON", int depth = 0) {
  x = std::forward<U>(y);
}

/**
 * Assign the specified standard vector rvalue to the specified
 * standard vector lvalue.
 *
 * @tparam T lvalue container element type
 * @tparam U rvalue container element type, which must be assignable to `T`
 * @param[in] x lvalue variable
 * @param[in] y rvalue variable
 * @param[in] name name of lvalue variable (default "ANON").
 * @param[in] depth indexing depth (default 0).
 */
template <typename T, typename U, require_std_vector_t<U>* = nullptr>
inline void assign(std::vector<T>& x, const nil_index_list& /* idxs */, U&& y,
                   const char* name = "ANON", int depth = 0) {
  x.resize(y.size());
  for (size_t i = 0; i < y.size(); ++i)
    assign(x[i], nil_index_list(), std::forward<U>(y)[i], name, depth + 1);
}

/**
 * Assign the specified Eigen vector at the specified single index
 * to the specified value.
 *
 * Types: vec[uni] <- scalar
 *
 * @tparam T Type of assigned vector scalar.
 * @tparam U Type of value (must be assignable to T).
 * @param[in] x Vector variable to be assigned.
 * @param[in] idxs Sequence of one single index (from 1).
 * @param[in] y Value scalar.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If the index is out of bounds.
 */
template <typename Vec1, typename U, require_eigen_vector_t<Vec1>* = nullptr>
inline void assign(Vec1& x,
                   const cons_index_list<index_uni, nil_index_list>& idxs,
                   const U& y, const char* name = "ANON", int depth = 0) {
  math::check_range("vector[uni] assign range", name, x.size(), idxs.head_.n_);
  x.coeffRef(idxs.head_.n_ - 1) = y;
}

/**
 * Assign the specified Eigen vector at the specified multiple
 * index to the specified value.
 *
 * Types:  vec[multi] <- vec
 *
 * @tparam T Type of assigned vector scalar.
 * @tparam I Type of multiple index.
 * @tparam U Type of vector value scalar (must be assignable to T).
 * @param[in] x Row vector variable to be assigned.
 * @param[in] idxs Sequence of one single index (from 1).
 * @param[in] y Value vector.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the value size isn't the same as
 * the indexed size.
 */
template <typename Vec1, typename Vec2, typename I,
          require_all_eigen_vector_t<Vec1, Vec2>* = nullptr,
          require_not_same_t<I, index_uni>* = nullptr>
inline void assign(Vec1& x, const cons_index_list<I, nil_index_list>& idxs,
                   const Vec2& y, const char* name = "ANON", int depth = 0) {
  const auto& y_ref = stan::math::to_ref(y);
  math::check_size_match("vector[multi] assign sizes", "lhs",
                         rvalue_index_size(idxs.head_, x.size()), name,
                         y_ref.size());
  for (int n = 0; n < y.size(); ++n) {
    int i = rvalue_at(n, idxs.head_);
    math::check_range("vector[multi] assign range", name, x.size(), i);
    x.coeffRef(i - 1) = y_ref.coeffRef(n);
  }
}

template <typename Vec1, typename Vec2, typename I,
          require_all_eigen_vector_t<Vec1, Vec2>* = nullptr>
inline void assign(Vec1& x,
                   const cons_index_list<index_min_max, nil_index_list>& idxs,
                   const Vec2& y, const char* name = "ANON", int depth = 0) {
  const auto& y_ref = stan::math::to_ref(y);
  math::check_size_match("vector[multi] assign sizes", "lhs",
                         rvalue_index_size(idxs.head_, x.size()), name,
                         y_ref.size());
  math::check_range("vector[min_max] min assign", name, x.size(),
                    idxs.head_.min_);
  math::check_range("vector[min_max] max assign", name, x.size(),
                    idxs.head_.max_);
  if (idxs.head_.min_ <= idxs.head_.max_) {
    x.segment(idxs.head_.min_ - 1, idxs.head_.max_ - 1) = y;
    return;
  } else {
    x.segment(idxs.head_.max_ - 1, idxs.head_.min_ - 1).reverse() = y;
    return;
  }
}

/**
 * Assign the specified Eigen matrix at the specified single index
 * to the specified row vector value.
 *
 * Types:  mat[uni] = rowvec
 *
 * @tparam T Assigned matrix scalar type.
 * @tparam U Type of value scalar for row vector (must be
 * assignable to T).
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs Sequence of one single index (from 1).
 * @param[in] y Value row vector.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the number of columns in the row
 * vector and matrix do not match.
 */
template <typename Mat, typename RowVec,
          require_eigen_matrix_dynamic_t<Mat>* = nullptr,
          require_eigen_row_vector_t<RowVec>* = nullptr>
inline void assign(Mat& x,
                   const cons_index_list<index_uni, nil_index_list>& idxs,
                   const RowVec& y, const char* name = "ANON", int depth = 0) {
  math::check_size_match("matrix[uni] assign sizes", "lhs", x.cols(), name,
                         y.cols());
  math::check_range("matrix[uni] assign range", name, x.rows(), idxs.head_.n_);
  x.row(idxs.head_.n_ - 1) = y;
}

/**
 * Assign the specified Eigen matrix at the specified single index
 * to the specified row vector value.
 *
 * Types:  mat[uni] = rowvec
 *
 * @tparam T Assigned matrix scalar type.
 * @tparam U Type of value scalar for row vector (must be
 * assignable to T).
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs Sequence of one single index (from 1).
 * @param[in] y Value row vector.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the number of columns in the row
 * vector and matrix do not match.
 */
template <typename Mat, typename ColVec,
          require_eigen_matrix_dynamic_t<Mat>* = nullptr,
          require_eigen_col_vector_t<ColVec>* = nullptr>
inline void assign(
    Mat& x,
    const cons_index_list<index_omni,
                          cons_index_list<index_uni, nil_index_list>>& idxs,
    const ColVec& y, const char* name = "ANON", int depth = 0) {
  math::check_size_match("matrix[uni] assign sizes", "lhs", x.rows(), name,
                         y.rows());
  math::check_range("matrix[uni] assign range", name, x.cols(),
                    idxs.tail_.head_.n_);
  x.col(idxs.tail_.head_.n_ - 1) = y;
}

/**
 * Assign the specified Eigen matrix at the specified single index
 * to the specified row vector value.
 *
 * Types:  mat[uni] = rowvec
 *
 * @tparam T Assigned matrix scalar type.
 * @tparam U Type of value scalar for row vector (must be
 * assignable to T).
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs Sequence of one single index (from 1).
 * @param[in] y Value row vector.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the number of columns in the row
 * vector and matrix do not match.
 */
template <typename Mat, typename RowVec,
          require_eigen_matrix_dynamic_t<Mat>* = nullptr,
          require_eigen_row_vector_t<RowVec>* = nullptr>
inline void assign(
    Mat& x,
    const cons_index_list<index_uni,
                          cons_index_list<index_omni, nil_index_list>>& idxs,
    const RowVec& y, const char* name = "ANON", int depth = 0) {
  math::check_size_match("matrix[uni] assign sizes", "lhs", x.cols(), name,
                         y.cols());
  x.row(idxs.head_.n_ - 1) = y;
}

/**
 * Assign the specified Eigen matrix at the specified multiple
 * index to the specified matrix value.
 *
 * Types:  mat[multi] = mat
 *
 * @tparam T Assigned matrix scalar type.
 * @tparam I Multiple index type.
 * @tparam U Value matrix scalar type (must be assignable to T).
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs Sequence of one multiple index (from 1).
 * @param[in] y Value matrix.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and right-hand side matrix do not match.
 */
template <typename Mat1, typename I, typename Mat2,
          require_all_eigen_matrix_dynamic_t<Mat2, Mat2>* = nullptr,
          require_not_same_t<I, index_uni>* = nullptr>
inline void assign(Mat1& x, const cons_index_list<I, nil_index_list>& idxs,
                   const Mat2& y, const char* name = "ANON", int depth = 0) {
  const int x_idx_rows = rvalue_index_size(idxs.head_, x.rows());
  const auto& y_ref = stan::math::to_ref(y);
  math::check_size_match("matrix[multi] assign row sizes", "lhs", x_idx_rows,
                         name, y_ref.rows());
  math::check_size_match("matrix[multi] assign col sizes", "lhs", x.cols(),
                         name, y_ref.cols());
  for (int i = 0; i < y_ref.rows(); ++i) {
    const int m = rvalue_at(i, idxs.head_);
    math::check_range("matrix[multi] assign range", name, x.rows(), m);
    x.row(m - 1) = y_ref.row(i);
  }
}

template <typename Mat1, typename Mat2,
          require_all_eigen_matrix_dynamic_t<Mat2, Mat2>* = nullptr>
inline void assign(
    Mat1& x,
    const cons_index_list<index_min_max,
                          cons_index_list<index_min_max, nil_index_list>>& idxs,
    const Mat2& y, const char* name = "ANON", int depth = 0) {
  const int x_idx_rows = rvalue_index_size(idxs.head_, x.rows());
  const int x_idx_cols = rvalue_index_size(idxs.tail_.head_, x.cols());
  math::check_size_match("matrix[multi] assign row sizes", "lhs", x_idx_rows,
                         name, y.rows());
  math::check_size_match("matrix[multi] assign col sizes", "lhs", x_idx_cols,
                         name, y.cols());
  if (idxs.head_.min_ <= idxs.head_.max_) {
    if (idxs.tail_.head_.min_ <= idxs.tail_.head_.max_) {
      x.block(idxs.head_.min_ - 1, idxs.tail_.head_.min_ - 1,
              idxs.head_.max_ - 1, idxs.tail_.head_.max_ - 1)
          = y;
      return;
    } else {
      x.block(idxs.head_.min_ - 1, idxs.tail_.head_.max_ - 1,
              idxs.head_.max_ - 1, idxs.tail_.head_.min_ - 1)
          .colwise()
          .reverse()
          = y;
      return;
    }
  } else {
    if (idxs.tail_.head_.min_ <= idxs.tail_.head_.max_) {
      x.block(idxs.head_.max_ - 1, idxs.tail_.head_.min_ - 1,
              idxs.head_.min_ - 1, idxs.tail_.head_.max_ - 1)
          .rowwise()
          .reverse()
          = y;
      return;
    } else {
      x.block(idxs.head_.max_ - 1, idxs.tail_.head_.max_ - 1,
              idxs.head_.min_ - 1, idxs.tail_.head_.min_ - 1)
          .rowwise()
          .reverse()
          .colwise()
          .reverse()
          = y;
      return;
    }
  }
}

/**
 * Assign the specified Eigen matrix at the specified pair of
 * single indexes to the specified scalar value.
 *
 * Types:  mat[single, single] = scalar
 *
 * @tparam T Matrix scalar type.
 * @tparam U Scalar type.
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs Sequence of two single indexes (from 1).
 * @param[in] y Value scalar.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If either of the indices are out of bounds.
 */
template <typename Mat, typename U,
          require_eigen_matrix_dynamic_t<Mat>* = nullptr>
inline void assign(
    Mat& x,
    const cons_index_list<index_uni,
                          cons_index_list<index_uni, nil_index_list>>& idxs,
    const U& y, const char* name = "ANON", int depth = 0) {
  const int m = idxs.head_.n_;
  const int n = idxs.tail_.head_.n_;
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
 * @tparam T Assigned matrix scalar type.
 * @tparam I Multi-index type.
 * @tparam U Value row vector scalar type (must be assignable to
 * T).
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs Sequence of single and multiple index (from 1).
 * @param[in] y Value row vector.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and right-hand side row vector do not match.
 */
template <typename Mat, typename I, typename Vec,
          require_eigen_matrix_dynamic_t<Mat>* = nullptr,
          require_eigen_row_vector_t<Vec>* = nullptr,
          require_not_same_t<I, index_uni>* = nullptr>
inline void assign(
    Mat& x,
    const cons_index_list<index_uni, cons_index_list<I, nil_index_list>>& idxs,
    const Vec& y, const char* name = "ANON", int depth = 0) {
  const auto& y_ref = stan::math::to_ref(y);
  const int x_idxs_cols = rvalue_index_size(idxs.tail_.head_, x.cols());
  math::check_size_match("matrix[uni,multi] assign sizes", "lhs", x_idxs_cols,
                         name, y_ref.cols());
  const int m = idxs.head_.n_;
  math::check_range("matrix[uni,multi] assign range", name, x.rows(), m);
  for (int i = 0; i < y_ref.size(); ++i) {
    const int n = rvalue_at(i, idxs.tail_.head_);
    math::check_range("matrix[uni,multi] assign range", name, x.cols(), n);
    x.coeffRef(m - 1, n - 1) = y_ref.coeffRef(i);
  }
}

/**
 * Assign the specified Eigen matrix at the specified multiple and
 * single index to the specified vector.
 *
 * Types:  mat[multi, uni] = vec
 *
 * @tparam T Assigned matrix scalar type.
 * @tparam I Multi-index type.
 * @tparam U Value vector scalar type (must be assignable to T).
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs Sequence of multiple and single index (from 1).
 * @param[in] y Value vector.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and right-hand side vector do not match.
 */
template <typename Mat, typename I, typename ColVec,
          require_eigen_matrix_dynamic_t<Mat>* = nullptr,
          require_eigen_col_vector_t<ColVec>* = nullptr,
          require_not_same_t<I, index_uni>* = nullptr>
inline void assign(
    Mat& x,
    const cons_index_list<I, cons_index_list<index_uni, nil_index_list>>& idxs,
    const ColVec& y, const char* name = "ANON", int depth = 0) {
  const int x_idxs_rows = rvalue_index_size(idxs.head_, x.rows());
  const auto& y_ref = stan::math::to_ref(y);
  math::check_size_match("matrix[multi,uni] assign sizes", "lhs", x_idxs_rows,
                         name, y_ref.rows());
  const int n = idxs.tail_.head_.n_;
  math::check_range("matrix[multi,uni] assign range", name, x.cols(), n);
  for (int i = 0; i < y_ref.size(); ++i) {
    const int m = rvalue_at(i, idxs.head_);
    math::check_range("matrix[multi,uni] assign range", name, x.rows(), m);
    x.coeffRef(m - 1, n - 1) = y_ref.coeffRef(i);
  }
}

/**
 * Assign the specified Eigen matrix at the specified pair of
 * multiple indexes to the specified matrix.
 *
 * Types:  mat[multi, multi] = mat
 *
 * @tparam T Assigned matrix scalar type.
 * @tparam I1 First multiple index type.
 * @tparam I2 Second multiple index type.
 * @tparam U Value matrix scalar type (must be assignable to T).
 * @param[in] x Matrix variable to be assigned.
 * @param[in] idxs Pair of multiple indexes (from 1).
 * @param[in] y Value matrix.
 * @param[in] name Name of variable (default "ANON").
 * @param[in] depth Indexing depth (default 0).
 * @throw std::out_of_range If any of the indices are out of bounds.
 * @throw std::invalid_argument If the dimensions of the indexed
 * matrix and value matrix do not match.
 */
template <typename Mat1, typename I1, typename I2, typename Mat2,
          require_all_eigen_matrix_dynamic_t<Mat1, Mat2>* = nullptr,
          require_all_not_same_t<index_uni, I1, I2>* = nullptr>
inline void assign(
    Mat1& x,
    const cons_index_list<I1, cons_index_list<I2, nil_index_list>>& idxs,
    const Mat2& y, const char* name = "ANON", int depth = 0) {
  const int x_idxs_rows = rvalue_index_size(idxs.head_, x.rows());
  const int x_idxs_cols = rvalue_index_size(idxs.tail_.head_, x.cols());
  const auto& y_ref = stan::math::to_ref(y);
  math::check_size_match("matrix[multi,multi] assign sizes", "lhs", x_idxs_rows,
                         name, y_ref.rows());
  math::check_size_match("matrix[multi,multi] assign sizes", "lhs", x_idxs_cols,
                         name, y_ref.cols());
  for (int j = 0; j < y_ref.cols(); ++j) {
    const int n = rvalue_at(j, idxs.tail_.head_);
    math::check_range("matrix[multi,multi] assign range", name, x.cols(), n);
    for (int i = 0; i < y_ref.rows(); ++i) {
      const int m = rvalue_at(i, idxs.head_);
      math::check_range("matrix[multi,multi] assign range", name, x.rows(), m);
      x.coeffRef(m - 1, n - 1) = y_ref.coeffRef(i, j);
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
  math::check_range("vector[uni,...] assign range", name, x.size(),
                    idxs.head_.n_);
  assign(x[idxs.head_.n_ - 1], idxs.tail_, y, name, depth + 1);
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
          require_not_same_t<I, index_uni>* = nullptr>
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

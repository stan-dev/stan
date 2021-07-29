#ifndef STAN_TEST_UNIT_MODEL_INDEXING_UTIL_HPP
#define STAN_TEST_UNIT_MODEL_INDEXING_UTIL_HPP

#include <stan/math/prim/meta.hpp>
#include <gtest/gtest.h>

namespace stan {
namespace model {
namespace test {

/**
 * Check a standard vector of inner types adjoints
 * @tparam Check1 Functor with one integer argument that returns bool
 * @tparam Check1 Functor with one integer argument that returns bool
 * @tparam StdVec Standard vector containing inner containers.
 * @param i_check Check whether an element of the vector should be inspected.
 * @param j_check Check whether an element of the inner container should be
 * inspected.
 * @param x A vector holding underlying containers
 * @param name A helper name to print out on failure.
 */
template <typename Check1, typename Check2, typename StdVecVar,
          require_std_vector_t<StdVecVar>* = nullptr>
void check_adjs(Check1&& i_check, Check2&& j_check, const StdVecVar& x,
                const char* name) {
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    for (Eigen::Index j = 0; j < x[i].size(); ++j) {
      if (i_check(i)) {
        if (j_check(j)) {
          EXPECT_FLOAT_EQ(x[i].adj()[j], 1)
              << "Failed on " << name << " for (i, j): (" << i << ", " << j
              << ")";
        } else {
          EXPECT_FLOAT_EQ(x[i].adj()[j], 0)
              << "Failed on " << name << " for (i, j): (" << i << ", " << j
              << ")";
        }
      } else {
        EXPECT_FLOAT_EQ(x[i].adj()[j], 0)
            << "Failed on " << name << " for (i, j): (" << i << ", " << j
            << ")";
      }
    }
  }
}

/**
 * Check an Eigen matrix's adjoints
 * @tparam Check1 Functor with one integer argument that returns bool
 * @tparam Check1 Functor with one integer argument that returns bool
 * @tparam VarMat A matrix of vars or var with inner matrix type.
 * @param i_check Check whether a row of the matrix should be inspected.
 * @param j_check Check whether a column of the matrix should be inspected.
 * @param x A matrix type.
 * @param name A helper name to print out on failure.
 * @param check_val when 1, any cell satisfying `i_check` and `j_check` are
 *  assumed to be 1 and all cells that fail either are 0. When `check_val` is
 *  0, any cell satisfying `i_check` and `j_check` are assumed to be 0, and
 *  all cells that fail are equal to 1.
 */
template <typename Check1, typename Check2, typename VarMat,
          require_var_matrix_t<VarMat>* = nullptr>
void check_adjs(Check1&& i_check, Check2&& j_check, const VarMat& x,
                const char* name = "", int check_val = 1) {
  for (Eigen::Index j = 0; j < x.cols(); ++j) {
    for (Eigen::Index i = 0; i < x.rows(); ++i) {
      if (i_check(i)) {
        if (j_check(j)) {
          EXPECT_FLOAT_EQ(x.adj()(i, j), check_val)
              << "Failed on " << name << " for (i, j): (" << i << ", " << j
              << ")";
        } else {
          EXPECT_FLOAT_EQ(x.adj()(i, j), check_val == 1 ? 0 : 1)
              << "Failed on " << name << " for col_check (i, j): (" << i << ", "
              << j << ")";
        }
      } else {
        EXPECT_FLOAT_EQ(x.adj()(i, j), check_val == 1 ? 0 : 1)
            << "Failed on " << name << " for row_check (i, j): (" << i << ", "
            << j << ")";
      }
    }
  }
}

/**
 * Check an Eigen matrix's adjoints
 * @tparam Check1 Functor with one integer argument that returns bool
 * @tparam Check1 Functor with one integer argument that returns bool
 * @tparam VarMat A matrix of vars or var with inner matrix type.
 * @param i_check Check whether a row of the matrix should be inspected.
 * @param j_check Check whether a column of the matrix should be inspected.
 * @param x A matrix type.
 * @param name A helper name to print out on failure.
 * @param check_val when 1, any cell satisfying `i_check` and `j_check` are
 *  assumed to be 1 and all cells that fail either are 0. When `check_val` is
 *  0, any cell satisfying `i_check` and `j_check` are assumed to be 0, and
 *  all cells that fail are equal to 1.
 */
template <typename Check1, typename Check2, typename VarMat,
          require_eigen_vt<std::is_arithmetic, VarMat>* = nullptr>
void check_adjs(Check1&& i_check, Check2&& j_check, const VarMat& x,
                const char* name = "", int check_val = 1) {}

void check_adjs(stan::math::var x, const char* name = "", int check_val = 1) {
  EXPECT_FLOAT_EQ(x.adj(), check_val) << "Failed on " << name;
}

void check_adjs(double x, const char* name = "", int check_val = 1) {}

/**
 * Check an Eigen vector's adjoints
 * @tparam Check1 Functor with one integer argument that returns bool
 * @tparam VarMat A vector of vars or var with inner matrix type.
 * @param i_check Check whether a cell of a vector should be inspected.
 * @param x A vector type.
 * @param name A helper name to print out on failure.
 * @param check_val when 1, any cell satisfying `i_check` is
 *  assumed to be 1 and all cells that fail are 0. When `check_val` is
 *  0, any cell satisfying `i_check` are assumed to be 0, and
 *  all cells that fail are equal to 1.
 */
template <typename Check1, typename VarMat, require_st_var<VarMat>* = nullptr>
void check_adjs(Check1&& i_check, const VarMat& x, const char* name = "",
                int check_val = 1) {
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    if (i_check(i)) {
      EXPECT_FLOAT_EQ(x.adj()(i), check_val)
          << "Failed on " << name << " for (i): (" << i << ")";
    } else {
      EXPECT_FLOAT_EQ(x.adj()(i), check_val == 1 ? 0 : 1)
          << "Failed on " << name << " for (i): (" << i << ")";
    }
  }
}
template <typename Check1, typename VarMat,
          require_st_arithmetic<VarMat>* = nullptr>
void check_adjs(Check1&& i_check, const VarMat& x, const char* name = "",
                int check_val = 1) {}

/**
 * Generate a matrix holding a linear sequence.
 * @param n Number of rows.
 * @param m Number of columns.
 * @param start Where the linear sequence should start from.
 */
auto generate_linear_matrix(Eigen::Index n, Eigen::Index m, double start = 0) {
  Eigen::Matrix<double, -1, -1> A(n, m);
  for (Eigen::Index i = 0; i < A.size(); ++i) {
    A(i) = i + start;
  }
  return A;
}

/**
 * Generate a `var_value` with inner matrix type holding a linear sequence
 *  in the values.
 * @param n Number of rows.
 * @param m Number of columns.
 * @param start Where the linear sequence should start from.
 */
template <typename RhsScalar = stan::math::var>
auto conditionally_generate_linear_var_matrix(Eigen::Index n, Eigen::Index m,
                                              double start = 0) {
  using ret_t
      = std::conditional_t<is_var<RhsScalar>::value,
                           stan::math::var_value<Eigen::Matrix<double, -1, -1>>,
                           Eigen::Matrix<double, -1, -1>>;
  return ret_t(generate_linear_matrix(n, m, start));
}

/**
 * Generate a vector holding a linear sequence
 *  in the values.
 * @tparam Vec The type of vector, either eigen row or column.
 * @param n Number of cells.
 * @param start Where the linear sequence should start from.
 */
template <typename Vec = Eigen::Matrix<double, -1, 1>>
auto generate_linear_vector(Eigen::Index n, double start = 0) {
  Vec A(n);
  for (Eigen::Index i = 0; i < A.size(); ++i) {
    A(i) = i + start;
  }
  return A;
}

/**
 * Generate a `var_value` with inner vector type holding a linear sequence
 *  in the values.
 * @param n Number of cells.
 * @param start Where the linear sequence should start from.
 */
template <typename Vec = Eigen::Matrix<double, -1, 1>,
          typename RhsScalar = stan::math::var>
auto conditionally_generate_linear_var_vector(Eigen::Index n,
                                              double start = 0) {
  using ret_t = std::conditional_t<is_var<RhsScalar>::value,
                                   stan::math::var_value<Vec>, Vec>;
  return ret_t(generate_linear_vector<Vec>(n, start));
}

template <typename T>
inline auto convert_to_multi(const index_multi& idx, const T& x,
                             bool row_or_col) {
  return idx;
}

template <typename T>
inline auto convert_to_multi(const index_omni& idx, const T& x,
                             bool row_or_col) {
  std::vector<int> v;
  if (row_or_col) {
    for (int i = 1; i <= x.cols(); ++i) {
      v.push_back(i);
    }
  } else {
    for (int i = 1; i <= x.rows(); ++i) {
      v.push_back(i);
    }
  }
  return index_multi(v);
}

template <typename T>
inline auto convert_to_multi(const index_min& idx, const T& x,
                             bool row_or_col) {
  std::vector<int> v;
  if (row_or_col) {
    for (int i = idx.min_; i <= x.cols(); ++i) {
      v.push_back(i);
    }
  } else {
    for (int i = idx.min_; i <= x.rows(); ++i) {
      v.push_back(i);
    }
  }
  return index_multi(v);
}

template <typename T>
inline auto convert_to_multi(const index_max& idx, const T& x,
                             bool row_or_col) {
  std::vector<int> v;
  for (int i = 1; i <= idx.max_; ++i) {
    v.push_back(i);
  }
  return index_multi(v);
}

template <typename T>
inline auto convert_to_multi(const index_min_max& idx, const T& x,
                             bool row_or_col) {
  std::vector<int> v;
  for (int i = idx.min_; i <= idx.max_; ++i) {
    v.push_back(i);
  }
  return index_multi(v);
}

template <typename T>
inline auto convert_to_multi(const index_uni& idx, const T& x,
                             bool row_or_col) {
  std::vector<int> v;
  v.push_back(idx.n_);
  return index_multi(v);
}

}  // namespace test
}  // namespace model
}  // namespace stan
#endif

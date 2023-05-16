#ifndef TEST_UNIT_UTIL_HPP
#define TEST_UNIT_UTIL_HPP

#include <stan/io/stan_csv_reader.hpp>

#include <boost/algorithm/string.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

#define EXPECT_THROW_MSG(expr, T_e, msg)                  \
  try {                                                   \
    expr;                                                 \
  } catch (const T_e& e) {                                \
    EXPECT_EQ(1, count_matches(msg, e.what()))            \
        << "expected message: " << msg << std::endl       \
        << "found message:    " << e.what();              \
  } catch (...) {                                         \
    FAIL() << "Wrong exception type thrown" << std::endl; \
  }

int count_matches(const std::string& target, const std::string& s) {
  if (target.size() == 0)
    return -1;  // error
  int count = 0;
  for (size_t pos = 0; (pos = s.find(target, pos)) != std::string::npos;
       pos += target.size())
    ++count;
  return count;
}

void match_csv_columns(const Eigen::MatrixXd& samples,
                       const std::string& raw_output, size_t num_rows,
                       size_t num_columns, size_t col_offset) {
  std::stringstream cell_ss;
  std::vector<std::string> cells;
  std::string line;
  std::istringstream f(raw_output);
  size_t row = 0;
  while (std::getline(f, line)) {
    if (row == 0) {
      ++row;
      continue;
    }
    if (row == num_rows + 1) {
      break;
    }
    cells.clear();
    boost::algorithm::split(cells, line, boost::is_any_of(","));
    for (size_t i = 0; i < num_columns; ++i) {
      cell_ss.str(std::string());
      cell_ss.clear();
      cell_ss << samples(row - 1, i + col_offset);
      EXPECT_EQ(cells[i], cell_ss.str());
    }
    ++row;
  }
}

/**
 * Tests for  exact elementwise equality of the input matrices
 * with the EXPECT_EQ macro from GTest.
 *
 * @param A first input matrix to compare
 * @param B second input matrix to compare
 */
#ifndef EXPECT_MATRIX_EQ
#define EXPECT_MATRIX_EQ(A, B)                                        \
  {                                                                   \
    using T_A = std::decay_t<decltype(A)>;                            \
    using T_B = std::decay_t<decltype(B)>;                            \
    const Eigen::Matrix<typename T_A::Scalar, T_A::RowsAtCompileTime, \
                        T_A::ColsAtCompileTime>                       \
        A_eval = A;                                                   \
    const Eigen::Matrix<typename T_B::Scalar, T_B::RowsAtCompileTime, \
                        T_B::ColsAtCompileTime>                       \
        B_eval = B;                                                   \
    EXPECT_EQ(A_eval.rows(), B_eval.rows());                          \
    EXPECT_EQ(A_eval.cols(), B_eval.cols());                          \
    for (int j = 0; j < A_eval.cols(); j++) {                         \
      for (int i = 0; i < A_eval.rows(); i++) {                       \
        EXPECT_EQ(A_eval(i, j), B_eval(i, j))                         \
            << "Failed at "                                           \
            << "(i, j): (" << i << ", " << j << ")";                  \
      }                                                               \
    }                                                                 \
  }
#endif

#ifndef EXPECT_MATRIX_NEAR
#define EXPECT_MATRIX_NEAR(A, B, DELTA)                               \
  {                                                                   \
    using T_A = std::decay_t<decltype(A)>;                            \
    using T_B = std::decay_t<decltype(B)>;                            \
    const Eigen::Matrix<typename T_A::Scalar, T_A::RowsAtCompileTime, \
                        T_A::ColsAtCompileTime>                       \
        A_eval = A;                                                   \
    const Eigen::Matrix<typename T_B::Scalar, T_B::RowsAtCompileTime, \
                        T_B::ColsAtCompileTime>                       \
        B_eval = B;                                                   \
    EXPECT_EQ(A_eval.rows(), B_eval.rows());                          \
    EXPECT_EQ(A_eval.cols(), B_eval.cols());                          \
    for (int i = 0; i < A_eval.size(); i++)                           \
      EXPECT_NEAR(A_eval(i), B_eval(i), DELTA);                       \
  }
#endif

/**
 * Gets the path separator for the OS.
 *
 * @return '\' for Windows, '/' otherwise.
 */
char get_slash() {
#if defined(WIN32) || defined(_WIN32) \
    || defined(__WIN32) && !defined(__CYGWIN__)
  static char path_separator = '\\';
#else
  static char path_separator = '/';
#endif
  return path_separator;
}

/**
 * Returns the path as a string with the appropriate path separator.
 *
 * @param file_path vector of strings representing path to the file
 * @return the string representation of the path
 */
std::string paths_to_fname(const std::vector<std::string>& path) {
  std::string pathstr;
  if (path.size() > 0) {
    pathstr.append(path[0]);
    for (size_t i = 1; i < path.size(); i++) {
      pathstr.append(1, get_slash());
      pathstr.append(path[i]);
    }
  }
  return pathstr;
}

namespace stan {
namespace test {
std::streambuf* cout_buf = 0;
std::streambuf* cerr_buf = 0;

std::stringstream cout_ss;
std::stringstream cerr_ss;

void capture_std_streams() {
  cout_ss.str("");
  cerr_ss.str("");

  cout_buf = std::cout.rdbuf();
  cerr_buf = std::cerr.rdbuf();

  std::cout.rdbuf(cout_ss.rdbuf());
  std::cerr.rdbuf(cerr_ss.rdbuf());
}

void reset_std_streams() {
  std::cout.rdbuf(cout_buf);
  std::cerr.rdbuf(cerr_buf);
  cout_buf = 0;
  cerr_buf = 0;
}

}  // namespace test
}  // namespace stan

#endif

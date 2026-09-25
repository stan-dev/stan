#include <stan/io/read_from_context.hpp>
#include <stan/io/json/json_data.hpp>
#include <gtest/gtest.h>
#include <complex>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

// Keep fixtures in logical JSON order. Expectations below use destination
// coordinates, independently of the parser's flat-buffer layout.
std::ifstream fixture_stream(const std::string& fixture) {
  const std::string path = "src/test/unit/io/test_json_files/read_from_context/"
                           + fixture + ".json";
  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open JSON fixture: " + path);
  }
  return in;
}

template <typename T>
void read_fixture(T& x, const std::string& fixture,
                  const std::string& name = "x") {
  auto in = fixture_stream(fixture);
  const stan::json::json_data context(in);
  stan::read_from_context(x, context, name);
}

void expect_complex(const std::complex<double>& actual, double real) {
  EXPECT_DOUBLE_EQ(real, actual.real());
  EXPECT_DOUBLE_EQ(-real - 0.5, actual.imag());
}

using complex_vector = Eigen::Matrix<std::complex<double>, -1, 1>;
using complex_matrix = Eigen::Matrix<std::complex<double>, -1, -1>;

}  // namespace

TEST(ioReadFromContext, int_scalar) {
  int x = -999;
  ASSERT_NO_THROW(read_fixture(x, "int"));
  EXPECT_EQ(-17, x);
}

TEST(ioReadFromContext, real_scalar) {
  double x = -999;
  ASSERT_NO_THROW(read_fixture(x, "double"));
  EXPECT_DOUBLE_EQ(12.25, x);
}

TEST(ioReadFromContext, complex_scalar) {
  std::complex<double> x(-999, -999);
  ASSERT_NO_THROW(read_fixture(x, "complex"));
  EXPECT_DOUBLE_EQ(-2.25, x.real());
  EXPECT_DOUBLE_EQ(7.5, x.imag());
}

TEST(ioReadFromContext, integer_backed_real) {
  double x = -999;
  ASSERT_NO_THROW(read_fixture(x, "integer_backed_real"));
  EXPECT_DOUBLE_EQ(17, x);
}

TEST(ioReadFromContext, integer_backed_complex) {
  std::complex<double> x(-999, -999);
  ASSERT_NO_THROW(read_fixture(x, "integer_backed_complex"));
  EXPECT_DOUBLE_EQ(7, x.real());
  EXPECT_DOUBLE_EQ(-19, x.imag());
}

TEST(ioReadFromContext, vector_double) {
  std::vector<double> x(4, -999);
  ASSERT_NO_THROW(read_fixture(x, "vector_double"));
  EXPECT_EQ((std::vector<double>{1.25, -2.5, 3.75, 40.125}), x);
}

TEST(ioReadFromContext, vector_int) {
  std::vector<int> x(3, -999);
  ASSERT_NO_THROW(read_fixture(x, "vector_int"));
  EXPECT_EQ((std::vector<int>{-4, 7, 19}), x);
}

TEST(ioReadFromContext, vector_complex) {
  std::vector<std::complex<double>> x(4, {-999, -999});
  ASSERT_NO_THROW(read_fixture(x, "vector_complex"));
  ASSERT_EQ(4, x.size());
  for (int i = 0; i < 4; ++i) {
    expect_complex(x[i], i + 0.25);
  }
}

TEST(ioReadFromContext, eigen_vector) {
  Eigen::VectorXd x = Eigen::VectorXd::Constant(4, -999);
  ASSERT_NO_THROW(read_fixture(x, "eigen_vector"));
  ASSERT_EQ(4, x.size());
  for (int i = 0; i < 4; ++i) {
    EXPECT_DOUBLE_EQ(i + 0.25, x(i));
  }
}

TEST(ioReadFromContext, eigen_row_vector) {
  Eigen::RowVectorXd x = Eigen::RowVectorXd::Constant(4, -999);
  ASSERT_NO_THROW(read_fixture(x, "eigen_row_vector"));
  ASSERT_EQ(4, x.size());
  for (int i = 0; i < 4; ++i) {
    EXPECT_DOUBLE_EQ(i + 0.25, x(i));
  }
}

TEST(ioReadFromContext, eigen_matrix) {
  Eigen::MatrixXd x = Eigen::MatrixXd::Constant(2, 3, -999);
  ASSERT_NO_THROW(read_fixture(x, "eigen_matrix"));
  ASSERT_EQ(2, x.rows());
  ASSERT_EQ(3, x.cols());
  for (int r = 0; r < 2; ++r) {
    for (int c = 0; c < 3; ++c) {
      EXPECT_DOUBLE_EQ(10 * r + c + 0.25, x(r, c));
    }
  }
}

TEST(ioReadFromContext, eigen_complex_matrix) {
  complex_matrix x = complex_matrix::Constant(2, 3, {-999, -999});
  ASSERT_NO_THROW(read_fixture(x, "eigen_complex_matrix"));
  ASSERT_EQ(2, x.rows());
  ASSERT_EQ(3, x.cols());
  for (int r = 0; r < 2; ++r) {
    for (int c = 0; c < 3; ++c) {
      expect_complex(x(r, c), 10 * r + c + 0.25);
    }
  }
}

TEST(ioReadFromContext, array_array_real) {
  std::vector<std::vector<double>> x(2, std::vector<double>(3, -999));
  ASSERT_NO_THROW(read_fixture(x, "array_array_real"));
  ASSERT_EQ(2, x.size());
  for (int a = 0; a < 2; ++a) {
    ASSERT_EQ(3, x[a].size());
    for (int b = 0; b < 3; ++b) {
      EXPECT_DOUBLE_EQ(10 * a + b + 0.25, x[a][b]);
    }
  }
}

TEST(ioReadFromContext, array_array_complex) {
  std::vector<std::vector<std::complex<double>>> x(
      2, std::vector<std::complex<double>>(3, {-999, -999}));
  ASSERT_NO_THROW(read_fixture(x, "array_array_complex"));
  ASSERT_EQ(2, x.size());
  for (int a = 0; a < 2; ++a) {
    ASSERT_EQ(3, x[a].size());
    for (int b = 0; b < 3; ++b) {
      expect_complex(x[a][b], 10 * a + b + 0.25);
    }
  }
}

TEST(ioReadFromContext, array_array_array_real) {
  std::vector<std::vector<std::vector<double>>> x(
      2, std::vector<std::vector<double>>(3, std::vector<double>(4, -999)));
  ASSERT_NO_THROW(read_fixture(x, "array_array_array_real"));
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 4; ++c) {
        EXPECT_DOUBLE_EQ(100 * a + 10 * b + c + 0.25, x[a][b][c]);
      }
    }
  }
}

TEST(ioReadFromContext, array_eigen_matrix) {
  std::vector<Eigen::MatrixXd> x(2, Eigen::MatrixXd::Constant(3, 4, -999));
  ASSERT_NO_THROW(read_fixture(x, "array_eigen_matrix"));
  // Array entries are interleaved in the column-major context buffer;
  // a single matrix's values are not a contiguous source block.
  for (int a = 0; a < 2; ++a) {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 4; ++c) {
        EXPECT_DOUBLE_EQ(100 * a + 10 * r + c + 0.25, x[a](r, c));
      }
    }
  }
}

TEST(ioReadFromContext, array_complex_eigen_matrix) {
  std::vector<complex_matrix> x(2,
                                complex_matrix::Constant(3, 4, {-999, -999}));
  ASSERT_NO_THROW(read_fixture(x, "array_complex_eigen_matrix"));
  for (int a = 0; a < 2; ++a) {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 4; ++c) {
        expect_complex(x[a](r, c), 100 * a + 10 * r + c + 0.25);
      }
    }
  }
}

TEST(ioReadFromContext, basic) {
  std::tuple<std::vector<double>, int> x{std::vector<double>(3, -999), -999};
  ASSERT_NO_THROW(read_fixture(x, "basic", "basic"));
  EXPECT_EQ((std::vector<double>{1.25, -2.5, 3.75}), std::get<0>(x));
  EXPECT_EQ(-17, std::get<1>(x));
}

TEST(ioReadFromContext, tuple_tuple) {
  std::tuple<int, std::tuple<double, Eigen::VectorXd>> x{
      -999, {-999, Eigen::VectorXd::Constant(3, -999)}};
  ASSERT_NO_THROW(read_fixture(x, "tuple_tuple", "tuple_tuple"));
  EXPECT_EQ(-17, std::get<0>(x));
  EXPECT_DOUBLE_EQ(8.25, std::get<0>(std::get<1>(x)));
  for (int i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(10 + i + 0.5, std::get<1>(std::get<1>(x))(i));
  }
}

TEST(ioReadFromContext, arr_tuple) {
  std::vector<std::tuple<int, std::vector<double>, Eigen::VectorXd>> x(
      2,
      {-999, std::vector<double>(3, -999), Eigen::VectorXd::Constant(4, -999)});
  ASSERT_NO_THROW(read_fixture(x, "arr_tuple", "arr_tuple"));
  for (int a = 0; a < 2; ++a) {
    SCOPED_TRACE(a);
    EXPECT_EQ(a + 7, std::get<0>(x[a]));
    for (int i = 0; i < 3; ++i) {
      EXPECT_DOUBLE_EQ(100 * a + i + 0.25, std::get<1>(x[a])[i]);
    }
    for (int i = 0; i < 4; ++i) {
      EXPECT_DOUBLE_EQ(100 * a + 10 + i + 0.5, std::get<2>(x[a])(i));
    }
  }
}

TEST(ioReadFromContext, tuple_arr_tuple) {
  std::tuple<std::vector<double>, int,
             std::vector<std::tuple<double, std::vector<int>>>>
      x{std::vector<double>(3, -999), -999,
        std::vector<std::tuple<double, std::vector<int>>>(
            2, {-999, std::vector<int>(4, -999)})};
  ASSERT_NO_THROW(read_fixture(x, "tuple_arr_tuple", "tuple_arr_tuple"));
  EXPECT_EQ((std::vector<double>{1.25, 2.5, 3.75}), std::get<0>(x));
  EXPECT_EQ(-9, std::get<1>(x));
  for (int a = 0; a < 2; ++a) {
    const auto& field = std::get<2>(x)[a];
    EXPECT_DOUBLE_EQ(10 * a + 0.25, std::get<0>(field));
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(100 * a + i, std::get<1>(field)[i]);
    }
  }
}

TEST(ioReadFromContext, arr_tuple_tuple) {
  std::vector<std::tuple<double, int,
                         std::tuple<double, std::tuple<int, Eigen::VectorXd>>>>
      x(2, {-999, -999, {-999, {-999, Eigen::VectorXd::Constant(3, -999)}}});
  ASSERT_NO_THROW(read_fixture(x, "arr_tuple_tuple", "arr_tuple_tuple"));
  for (int a = 0; a < 2; ++a) {
    SCOPED_TRACE(a);
    EXPECT_DOUBLE_EQ(100 * a + 0.25, std::get<0>(x[a]));
    EXPECT_EQ(100 * a + 1, std::get<1>(x[a]));
    const auto& inner = std::get<2>(x[a]);
    EXPECT_DOUBLE_EQ(100 * a + 2.5, std::get<0>(inner));
    EXPECT_EQ(100 * a + 3, std::get<0>(std::get<1>(inner)));
    for (int i = 0; i < 3; ++i) {
      EXPECT_DOUBLE_EQ(100 * a + 10 + i + 0.75,
                       std::get<1>(std::get<1>(inner))(i));
    }
  }
}

TEST(ioReadFromContext, arr_tuple_arr_tuple) {
  using inner = std::tuple<double, Eigen::VectorXd>;
  std::vector<std::tuple<int, std::vector<inner>>> x(
      2, {-999,
          std::vector<inner>(3, {-999, Eigen::VectorXd::Constant(4, -999)})});
  ASSERT_NO_THROW(
      read_fixture(x, "arr_tuple_arr_tuple", "arr_tuple_arr_tuple"));
  for (int a = 0; a < 2; ++a) {
    EXPECT_EQ(a + 7, std::get<0>(x[a]));
    for (int b = 0; b < 3; ++b) {
      const auto& field = std::get<1>(x[a])[b];
      EXPECT_DOUBLE_EQ(100 * a + 10 * b + 0.25, std::get<0>(field));
      for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(100 * a + 10 * b + i + 0.5, std::get<1>(field)(i));
      }
    }
  }
}

TEST(ioReadFromContext, complex_vector_deep) {
  using inner = std::tuple<double, std::vector<complex_vector>>;
  std::vector<std::tuple<int, std::vector<inner>>> x(
      2, {-999, std::vector<inner>(3, {-999, std::vector<complex_vector>(
                                                 2, complex_vector::Constant(
                                                        4, {-999, -999}))})});
  ASSERT_NO_THROW(
      read_fixture(x, "complex_vector_deep", "complex_vector_deep"));
  for (int a = 0; a < 2; ++a) {
    SCOPED_TRACE(a);
    EXPECT_EQ(a + 7, std::get<0>(x[a]));
    for (int b = 0; b < 3; ++b) {
      SCOPED_TRACE(b);
      const auto& field = std::get<1>(x[a])[b];
      EXPECT_DOUBLE_EQ(100 * a + 10 * b + 0.25, std::get<0>(field));
      for (int c = 0; c < 2; ++c) {
        for (int d = 0; d < 4; ++d) {
          expect_complex(std::get<1>(field)[c](d),
                         1000 * a + 100 * b + 10 * c + d + 0.25);
        }
      }
    }
  }
}

TEST(ioReadFromContext, very_deep) {
  using leaf = std::tuple<std::complex<double>, Eigen::MatrixXd>;
  using inner = std::tuple<double, std::vector<leaf>>;
  std::vector<std::tuple<int, std::vector<inner>>> x(
      2, {-999, std::vector<inner>(
                    3, {-999, std::vector<leaf>(4, {{-999, -999},
                                                    Eigen::MatrixXd::Constant(
                                                        2, 3, -999)})})});
  ASSERT_NO_THROW(read_fixture(x, "very_deep", "very_deep"));
  for (int a = 0; a < 2; ++a) {
    SCOPED_TRACE(a);
    EXPECT_EQ(a + 7, std::get<0>(x[a]));
    for (int b = 0; b < 3; ++b) {
      SCOPED_TRACE(b);
      const auto& field = std::get<1>(x[a])[b];
      EXPECT_DOUBLE_EQ(100 * a + 10 * b + 0.25, std::get<0>(field));
      for (int c = 0; c < 4; ++c) {
        SCOPED_TRACE(c);
        const auto& nested = std::get<1>(field)[c];
        expect_complex(std::get<0>(nested), 1000 * a + 100 * b + 10 * c + 0.25);
        for (int r = 0; r < 2; ++r) {
          for (int s = 0; s < 3; ++s) {
            EXPECT_DOUBLE_EQ(10000 * a + 1000 * b + 100 * c + 10 * r + s + 0.5,
                             std::get<1>(nested)(r, s));
          }
        }
      }
    }
  }
}

TEST(ioReadFromContext, nested_array_tuple) {
  using element = std::tuple<int, double>;
  std::vector<std::vector<element>> x(2, std::vector<element>(3, {-999, -999}));
  ASSERT_NO_THROW(read_fixture(x, "nested_array_tuple"));
  // Multiple array dimensions outside a tuple use tuple-instance order,
  // unlike the identically sized ordinary array_array_real fixture.
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 3; ++b) {
      EXPECT_EQ(10 * a + b, std::get<0>(x[a][b]));
      EXPECT_DOUBLE_EQ(100 * a + 10 * b + 0.25, std::get<1>(x[a][b]));
    }
  }
}

TEST(ioReadFromContext, empty_vector) {
  std::vector<double> x;
  ASSERT_NO_THROW(read_fixture(x, "empty_vector"));
  EXPECT_TRUE(x.empty());
}

TEST(ioReadFromContext, empty_outer_tuple) {
  std::vector<std::tuple<int, std::vector<complex_vector>>> x;
  ASSERT_NO_THROW(read_fixture(x, "empty_outer_tuple"));
  EXPECT_TRUE(x.empty());
}

TEST(ioReadFromContext, empty_inner_array) {
  std::vector<std::vector<double>> x(2);
  ASSERT_NO_THROW(read_fixture(x, "empty_inner_array"));
  ASSERT_EQ(2, x.size());
  EXPECT_TRUE(x[0].empty());
  EXPECT_TRUE(x[1].empty());
}

TEST(ioReadFromContext, empty_inner_tuple) {
  std::vector<std::tuple<int, std::vector<std::complex<double>>>> x(2);
  ASSERT_NO_THROW(read_fixture(x, "empty_inner_tuple"));
  ASSERT_EQ(2, x.size());
  for (int a = 0; a < 2; ++a) {
    EXPECT_EQ(a + 7, std::get<0>(x[a]));
    EXPECT_TRUE(std::get<1>(x[a]).empty());
  }
}

TEST(ioReadFromContext, zero_rows_matrix) {
  Eigen::MatrixXd x(0, 3);
  ASSERT_NO_THROW(read_fixture(x, "zero_rows_matrix"));
  EXPECT_EQ(0, x.rows());
  EXPECT_EQ(3, x.cols());
}

TEST(ioReadFromContext, zero_cols_matrix) {
  Eigen::MatrixXd x(2, 0);
  ASSERT_NO_THROW(read_fixture(x, "zero_cols_matrix"));
  EXPECT_EQ(2, x.rows());
  EXPECT_EQ(0, x.cols());
}

TEST(ioReadFromContext, repeated_read_has_independent_positions) {
  auto in = fixture_stream("basic");
  const stan::json::json_data context(in);
  std::tuple<std::vector<double>, int> first{std::vector<double>(3, -999),
                                             -999};
  auto second = first;
  ASSERT_NO_THROW(stan::read_from_context(first, context, "basic"));
  ASSERT_NO_THROW(stan::read_from_context(second, context, "basic"));
  EXPECT_EQ((std::vector<double>{1.25, -2.5, 3.75}), std::get<0>(first));
  EXPECT_EQ(-17, std::get<1>(first));
  EXPECT_EQ(first, second);
}

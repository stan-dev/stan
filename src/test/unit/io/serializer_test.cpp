#include <stan/io/deserializer.hpp>
#include <stan/io/serializer.hpp>
// expect_near_rel comes from lib/stan_math
#include <test/unit/math/expect_near_rel.hpp>
#include <gtest/gtest.h>

TEST(serializer_scalar, write) {
  std::vector<double> theta(2, 0.0);
  double x = 4.0;
  double y = 6.0;
  stan::io::serializer<double> serializer(theta);
  serializer.write(x);
  EXPECT_FLOAT_EQ(x, theta[0]);
  serializer.write(y);
  EXPECT_FLOAT_EQ(y, theta[1]);
  EXPECT_EQ(0U, serializer.available());
  EXPECT_THROW(serializer.write(4), std::runtime_error);
}

TEST(serializer_scalar, complex_write) {
  std::vector<double> theta(4, 0.0);
  stan::io::serializer<double> serializer(theta);
  std::complex<double> x = std::complex<double>(5.0, 6.0);
  std::complex<double> y = std::complex<double>(7.0, 8.0);

  serializer.write(x);
  EXPECT_FLOAT_EQ(x.real(), theta[0]);
  EXPECT_FLOAT_EQ(x.imag(), theta[1]);
  serializer.write(y);
  EXPECT_FLOAT_EQ(y.real(), theta[2]);
  EXPECT_FLOAT_EQ(y.imag(), theta[3]);
  EXPECT_EQ(0U, serializer.available());
  EXPECT_THROW(serializer.write(4), std::runtime_error);
}

// vector

TEST(serializer_vector, write) {
  std::vector<double> theta(10, 0.0);
  Eigen::VectorXd x(10);
  for (size_t i = 0; i < 10U; ++i) {
    x.coeffRef(i) = -static_cast<double>(i);
  }
  stan::io::serializer<double> serializer(theta);
  serializer.write(x);
  for (size_t i = 0; i < 10U; ++i) {
    EXPECT_FLOAT_EQ(theta[i], x[i]);
  }
  EXPECT_THROW(serializer.write(4), std::runtime_error);
}

TEST(serializer_vector, complex_write) {
  std::vector<double> theta(20, 0.0);
  Eigen::Matrix<std::complex<double>, -1, 1> x(10);
  for (size_t i = 0; i < 10U; ++i) {
    x.coeffRef(i) = std::complex<double>(-static_cast<double>(i),
                                         -static_cast<double>(i + 1));
  }

  stan::io::serializer<double> serializer(theta);
  serializer.write(x);
  Eigen::Index sentinal = 0;
  for (size_t i = 0; i < 20U; i += 2) {
    EXPECT_FLOAT_EQ(theta[i], x[sentinal].real());
    EXPECT_FLOAT_EQ(theta[i + 1], x[sentinal].imag());
    ++sentinal;
  }
  EXPECT_THROW(serializer.write(4), std::runtime_error);
}

// row vector

TEST(serializer_rowvector, write) {
  std::vector<double> theta(10, 0.0);
  Eigen::RowVectorXd x(10);
  for (size_t i = 0; i < 10U; ++i) {
    x.coeffRef(i) = -static_cast<double>(i);
  }
  stan::io::serializer<double> serializer(theta);
  serializer.write(x);
  for (size_t i = 0; i < 10U; ++i) {
    EXPECT_FLOAT_EQ(theta[i], x[i]);
  }
  EXPECT_THROW(serializer.write(4), std::runtime_error);
}

TEST(serializer_rowvector, complex_write) {
  std::vector<double> theta(20, 0.0);
  Eigen::Matrix<std::complex<double>, 1, -1> x(10);
  for (size_t i = 0; i < 10U; ++i) {
    x.coeffRef(i) = std::complex<double>(-static_cast<double>(i),
                                         -static_cast<double>(i + 1));
  }

  stan::io::serializer<double> serializer(theta);
  serializer.write(x);
  Eigen::Index sentinal = 0;
  for (size_t i = 0; i < 20U; i += 2) {
    EXPECT_FLOAT_EQ(theta[i], x[sentinal].real());
    EXPECT_FLOAT_EQ(theta[i + 1], x[sentinal].imag());
    ++sentinal;
  }
  EXPECT_THROW(serializer.write(4), std::runtime_error);
}

// matrix

TEST(serializer_matrix, write) {
  std::vector<double> theta(16, 0.0);
  Eigen::MatrixXd x(4, 4);
  for (size_t i = 0; i < 16U; ++i) {
    x.coeffRef(i) = -static_cast<double>(i);
  }
  stan::io::serializer<double> serializer(theta);
  serializer.write(x);
  for (size_t i = 0; i < 16U; ++i) {
    EXPECT_FLOAT_EQ(theta[i], x(i));
  }
  EXPECT_THROW(serializer.write(4), std::runtime_error);
}

TEST(serializer_matrix, complex_write) {
  std::vector<double> theta(32, 0.0);
  Eigen::Matrix<std::complex<double>, -1, -1> x(4, 4);
  for (size_t i = 0; i < 16U; ++i) {
    x.coeffRef(i) = std::complex<double>(-static_cast<double>(i),
                                         -static_cast<double>(i + 1));
  }

  stan::io::serializer<double> serializer(theta);
  serializer.write(x);
  Eigen::Index sentinal = 0;
  for (size_t i = 0; i < 32U; i += 2) {
    EXPECT_FLOAT_EQ(theta[i], x(sentinal).real());
    EXPECT_FLOAT_EQ(theta[i + 1], x(sentinal).imag());
    ++sentinal;
  }
  EXPECT_THROW(serializer.write(4), std::runtime_error);
}

// array

TEST(serializer_stdvector, write) {
  std::vector<double> theta(10, 0.0);
  std::vector<double> x;
  for (size_t i = 0; i < 10U; ++i) {
    x.push_back(-static_cast<double>(i));
  }
  stan::io::serializer<double> serializer(theta);
  serializer.write(x);
  for (size_t i = 0; i < 10U; ++i) {
    EXPECT_FLOAT_EQ(theta[i], x[i]);
  }
  EXPECT_THROW(serializer.write(4), std::runtime_error);
}

TEST(serializer_stdvector, complex_write) {
  std::vector<double> theta(20, 0.0);
  std::vector<std::complex<double>> x;
  for (size_t i = 0; i < 10U; ++i) {
    x.push_back(std::complex<double>(-static_cast<double>(i),
                                     -static_cast<double>(i + 1)));
  }

  stan::io::serializer<double> serializer(theta);
  serializer.write(x);
  Eigen::Index sentinal = 0;
  for (size_t i = 0; i < 20U; i += 2) {
    EXPECT_FLOAT_EQ(theta[i], x[sentinal].real());
    EXPECT_FLOAT_EQ(theta[i + 1], x[sentinal].imag());
    ++sentinal;
  }
  EXPECT_THROW(serializer.write(4), std::runtime_error);
}

// size zero

TEST(serializer, zeroSizeVecs) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(1.0);
  stan::io::serializer<double> serializer(theta);

  EXPECT_NO_THROW(serializer.write(1.5));  // finish available

  EXPECT_NO_THROW(serializer.write(std::vector<double>(0)));
  EXPECT_NO_THROW(serializer.write(Eigen::VectorXd(0)));
  EXPECT_NO_THROW(serializer.write(Eigen::RowVectorXd(0)));
  EXPECT_NO_THROW(serializer.write(Eigen::MatrixXd(0, 3)));
  EXPECT_NO_THROW(serializer.write(Eigen::MatrixXd(3, 0)));
  EXPECT_NO_THROW(
      serializer.write(std::vector<std::vector<Eigen::MatrixXd>>(0)));
}

// out of memory

TEST(serializer, eos_exception) {
  std::vector<double> theta;
  theta.push_back(1.0);
  theta.push_back(2.0);
  stan::io::serializer<double> serializer(theta);

  EXPECT_EQ(2U, serializer.available());

  EXPECT_NO_THROW(serializer.write(double{1}));
  EXPECT_NO_THROW(serializer.write(double{1}));
  EXPECT_THROW(serializer.write(double{1}), std::runtime_error);

  // should keep throwing
  EXPECT_THROW(serializer.write(double{1}), std::runtime_error);

  // The strategy for all the following checks is to allocate 1 less than
  // the required memory and make sure an error happens
  {
    std::vector<double> theta(1);
    stan::io::serializer<double> serializer(theta);
    EXPECT_THROW(serializer.write(std::complex<double>{-1, 1}),
                 std::runtime_error);
  }

  {
    std::vector<double> theta(2);
    stan::io::serializer<double> serializer(theta);
    EXPECT_THROW(serializer.write(std::vector<double>(3)), std::runtime_error);
  }

  {
    std::vector<double> theta(1);
    stan::io::serializer<double> serializer(theta);
    EXPECT_THROW(serializer.write(Eigen::VectorXd(2)), std::runtime_error);
  }

  {
    std::vector<double> theta(3);
    stan::io::serializer<double> serializer(theta);
    using complex_colvec
        = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;
    EXPECT_THROW((serializer.write(complex_colvec(4))), std::runtime_error);
  }

  {
    std::vector<double> theta(5);
    stan::io::serializer<double> serializer(theta);
    EXPECT_THROW(
        serializer.write(std::vector<Eigen::VectorXd>(3, Eigen::VectorXd(2))),
        std::runtime_error);
  }

  {
    std::vector<double> theta(1);
    stan::io::serializer<double> serializer(theta);
    EXPECT_THROW(serializer.write(Eigen::RowVectorXd(2)), std::runtime_error);
  }

  {
    std::vector<double> theta(3);
    stan::io::serializer<double> serializer(theta);
    using complex_rowvec
        = Eigen::Matrix<std::complex<double>, 1, Eigen::Dynamic>;
    EXPECT_THROW((serializer.write(complex_rowvec(4))), std::runtime_error);
  }

  {
    std::vector<double> theta(5);
    stan::io::serializer<double> serializer(theta);
    using std_vec_rowvec = std::vector<Eigen::RowVectorXd>;
    EXPECT_THROW((serializer.write(std_vec_rowvec(3, Eigen::RowVectorXd(2)))),
                 std::runtime_error);
  }

  {
    std::vector<double> theta(3);
    stan::io::serializer<double> serializer(theta);
    EXPECT_THROW(serializer.write(Eigen::MatrixXd(2, 2)), std::runtime_error);
  }

  {
    std::vector<double> theta(7);
    stan::io::serializer<double> serializer(theta);
    using eig_complex_mat
        = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>;
    EXPECT_THROW((serializer.write(eig_complex_mat(2, 2))), std::runtime_error);
  }

  {
    std::vector<double> theta(11);
    stan::io::serializer<double> serializer(theta);
    EXPECT_THROW(serializer.write(
                     std::vector<Eigen::MatrixXd>(2, Eigen::MatrixXd(3, 2))),
                 std::runtime_error);
  }
}

template <typename Ret, typename... Sizes>
void write_free_lb_test(Sizes... sizes) {
  double lb = 0.5;
  constexpr size_t theta_size = 100;
  Eigen::VectorXd theta1 = Eigen::VectorXd::Random(theta_size);
  std::vector<int> theta_i;

  // Read an constrained variable
  stan::io::deserializer<double> deserializer(theta1, theta_i);
  double lp = 0.0;
  Ret vec_ref = deserializer.read_constrain_lb<Ret, false>(lb, lp, sizes...);

  // Serialize a constrained variable
  Eigen::VectorXd theta2 = Eigen::VectorXd::Random(theta_size);
  stan::io::serializer<double> serializer(theta2);
  serializer.write_free_lb(lb, vec_ref);

  size_t used1 = theta1.size() - deserializer.available();
  size_t used2 = theta2.size() - serializer.available();

  // Number of variables read should equal number of variables written
  EXPECT_EQ(used1, used2);

  // Make sure the variables written back are the same
  stan::test::expect_near_rel("deserializer read free",
                              theta1.segment(0, used1),
                              theta2.segment(0, used1));
}

TEST(serializer_vectorized, write_free_lb) {
  write_free_lb_test<double>();
  write_free_lb_test<Eigen::VectorXd>(4);
  write_free_lb_test<std::vector<Eigen::VectorXd>>(2, 4);
  write_free_lb_test<std::vector<std::vector<Eigen::VectorXd>>>(3, 2, 4);
}

template <typename Ret, typename... Sizes>
void write_free_ub_test(Sizes... sizes) {
  double ub = 0.5;
  constexpr size_t theta_size = 100;
  Eigen::VectorXd theta1 = Eigen::VectorXd::Random(theta_size);
  std::vector<int> theta_i;

  // Read an constrained variable
  stan::io::deserializer<double> deserializer(theta1, theta_i);
  double lp = 0.0;
  Ret vec_ref = deserializer.read_constrain_ub<Ret, false>(ub, lp, sizes...);

  // Serialize a constrained variable
  Eigen::VectorXd theta2 = Eigen::VectorXd::Random(theta_size);
  stan::io::serializer<double> serializer(theta2);
  serializer.write_free_ub(ub, vec_ref);

  size_t used1 = theta1.size() - deserializer.available();
  size_t used2 = theta2.size() - serializer.available();

  // Number of variables read should equal number of variables written
  EXPECT_EQ(used1, used2);

  // Make sure the variables written back are the same
  stan::test::expect_near_rel("deserializer read free",
                              theta1.segment(0, used1),
                              theta2.segment(0, used1));
}

TEST(serializer_vectorized, write_free_ub) {
  write_free_ub_test<double>();
  write_free_ub_test<Eigen::VectorXd>(4);
  write_free_ub_test<std::vector<Eigen::VectorXd>>(2, 4);
  write_free_ub_test<std::vector<std::vector<Eigen::VectorXd>>>(3, 2, 4);
}

template <typename Ret, typename... Sizes>
void write_free_lub_test(Sizes... sizes) {
  double ub = 0.5;
  double lb = 0.1;
  constexpr size_t theta_size = 100;
  Eigen::VectorXd theta1 = Eigen::VectorXd::Random(theta_size);
  std::vector<int> theta_i;

  // Read an constrained variable
  stan::io::deserializer<double> deserializer(theta1, theta_i);
  double lp = 0.0;
  Ret vec_ref
      = deserializer.read_constrain_lub<Ret, false>(lb, ub, lp, sizes...);

  // Serialize a constrained variable
  Eigen::VectorXd theta2 = Eigen::VectorXd::Random(theta_size);
  stan::io::serializer<double> serializer(theta2);
  serializer.write_free_lub(lb, ub, vec_ref);

  size_t used1 = theta1.size() - deserializer.available();
  size_t used2 = theta2.size() - serializer.available();

  // Number of variables read should equal number of variables written
  EXPECT_EQ(used1, used2);

  // Make sure the variables written back are the same
  stan::test::expect_near_rel("deserializer read free",
                              theta1.segment(0, used1),
                              theta2.segment(0, used1));
}

TEST(serializer_vectorized, write_free_lub) {
  write_free_lub_test<double>();
  write_free_lub_test<Eigen::VectorXd>(4);
  write_free_lub_test<std::vector<Eigen::VectorXd>>(2, 4);
  write_free_lub_test<std::vector<std::vector<Eigen::VectorXd>>>(3, 2, 4);
}

template <typename Ret, typename... Sizes>
void write_free_offset_multiplier_test(Sizes... sizes) {
  double offset = 0.5;
  double multiplier = 0.35;
  constexpr size_t theta_size = 100;
  Eigen::VectorXd theta1 = Eigen::VectorXd::Random(theta_size);
  std::vector<int> theta_i;

  // Read an constrained variable
  stan::io::deserializer<double> deserializer(theta1, theta_i);
  double lp = 0.0;
  Ret vec_ref = deserializer.read_constrain_offset_multiplier<Ret, false>(
      offset, multiplier, lp, sizes...);

  // Serialize a constrained variable
  Eigen::VectorXd theta2 = Eigen::VectorXd::Random(theta_size);
  stan::io::serializer<double> serializer(theta2);
  serializer.write_free_offset_multiplier(offset, multiplier, vec_ref);

  size_t used1 = theta1.size() - deserializer.available();
  size_t used2 = theta2.size() - serializer.available();

  // Number of variables read should equal number of variables written
  EXPECT_EQ(used1, used2);

  // Make sure the variables written back are the same
  stan::test::expect_near_rel("deserializer read free",
                              theta1.segment(0, used1),
                              theta2.segment(0, used1));
}

TEST(serializer_vectorized, write_free_offset_multiplier) {
  write_free_offset_multiplier_test<double>();
  write_free_offset_multiplier_test<Eigen::VectorXd>(4);
  write_free_offset_multiplier_test<std::vector<Eigen::VectorXd>>(2, 4);
  write_free_offset_multiplier_test<std::vector<std::vector<Eigen::VectorXd>>>(
      3, 2, 4);
}
template <typename Ret, typename... Sizes>
void write_free_unit_vector_test(Sizes... sizes) {
  Eigen::VectorXd theta1 = Eigen::VectorXd::Random(100);
  std::vector<int> theta_i;

  // Read an constrained variable
  stan::io::deserializer<double> deserializer(theta1, theta_i);
  double lp = 0.0;
  Ret vec_ref
      = deserializer.read_constrain_unit_vector<Ret, false>(lp, sizes...);

  // Serialize a constrained variable
  Eigen::VectorXd theta2 = Eigen::VectorXd::Random(theta1.size());
  stan::io::serializer<double> serializer(theta2);
  serializer.write_free_unit_vector(vec_ref);

  // For unit vector, it's not actually doing a change of variables so we check
  // theta2 equals theta2 (freeing doesn't actually get the unconstrained
  // variable back).
  size_t used1 = theta1.size() - deserializer.available();
  size_t used2 = theta1.size() - serializer.available();

  // Number of variables read should equal number of variables written
  EXPECT_EQ(used1, used2);

  // Make sure the variables written back are the same
  stan::test::expect_near_rel("deserializer read free",
                              theta2.segment(0, used1),
                              theta2.segment(0, used2));
}

TEST(serializer_vectorized, write_free_unit_vector) {
  write_free_unit_vector_test<Eigen::VectorXd>(4);
  write_free_unit_vector_test<std::vector<Eigen::VectorXd>>(2, 4);
  write_free_unit_vector_test<std::vector<std::vector<Eigen::VectorXd>>>(3, 2,
                                                                         4);
}

template <typename Ret, typename... Sizes>
void write_free_simplex_test(Sizes... sizes) {
  constexpr size_t theta_size = 100;
  Eigen::VectorXd theta1 = Eigen::VectorXd::Random(theta_size);
  std::vector<int> theta_i;

  // Read an constrained variable
  stan::io::deserializer<double> deserializer(theta1, theta_i);
  double lp = 0.0;
  Ret vec_ref = deserializer.read_constrain_simplex<Ret, false>(lp, sizes...);

  // Serialize a constrained variable
  Eigen::VectorXd theta2 = Eigen::VectorXd::Random(theta_size);
  stan::io::serializer<double> serializer(theta2);
  serializer.write_free_simplex(vec_ref);

  size_t used1 = theta1.size() - deserializer.available();
  size_t used2 = theta2.size() - serializer.available();

  // Number of variables read should equal number of variables written
  EXPECT_EQ(used1, used2);

  // Make sure the variables written back are the same
  stan::test::expect_near_rel("deserializer read free",
                              theta1.segment(0, used1),
                              theta2.segment(0, used1));
}

TEST(serializer_vectorized, write_free_simplex) {
  write_free_simplex_test<Eigen::VectorXd>(4);
  write_free_simplex_test<std::vector<Eigen::VectorXd>>(2, 4);
  write_free_simplex_test<std::vector<std::vector<Eigen::VectorXd>>>(3, 2, 4);
}

// ordered

template <typename Ret, typename... Sizes>
void write_free_ordered_test(Sizes... sizes) {
  // Read an constrained variable
  Eigen::VectorXd theta1 = Eigen::VectorXd::Random(100);
  std::vector<int> theta_i;
  stan::io::deserializer<double> deserializer(theta1, theta_i);
  double lp = 0.0;
  Ret vec_ref = deserializer.read_constrain_ordered<Ret, false>(lp, sizes...);

  // Serialize a constrained variable
  Eigen::VectorXd theta2 = Eigen::VectorXd::Random(theta1.size());
  stan::io::serializer<double> serializer(theta2);
  serializer.write_free_ordered(vec_ref);

  size_t used1 = theta1.size() - deserializer.available();
  size_t used2 = theta2.size() - serializer.available();

  // Number of variables read should equal number of variables written
  EXPECT_EQ(used1, used2);

  // Make sure the variables written back are the same
  stan::test::expect_near_rel("deserializer read free",
                              theta1.segment(0, used1),
                              theta2.segment(0, used1));
}

TEST(serializer_vectorized, write_free_ordered) {
  write_free_ordered_test<Eigen::VectorXd>(4);
  write_free_ordered_test<std::vector<Eigen::VectorXd>>(2, 4);
  write_free_ordered_test<std::vector<std::vector<Eigen::VectorXd>>>(3, 2, 4);
}

// positive_ordered

template <typename Ret, typename... Sizes>
void write_free_positive_ordered_test(Sizes... sizes) {
  Eigen::VectorXd theta1 = Eigen::VectorXd::Random(100);
  Eigen::VectorXd theta2 = Eigen::VectorXd::Random(theta1.size());
  std::vector<int> theta_i;

  // Read an constrained variable
  stan::io::deserializer<double> deserializer(theta1, theta_i);
  double lp = 0.0;
  Ret vec_ref
      = deserializer.read_constrain_positive_ordered<Ret, false>(lp, sizes...);

  // Serialize a constrained variable
  stan::io::serializer<double> serializer(theta2);
  serializer.write_free_positive_ordered(vec_ref);

  size_t used1 = theta1.size() - deserializer.available();
  size_t used2 = theta2.size() - serializer.available();

  // Number of variables read should equal number of variables written
  EXPECT_EQ(used1, used2);

  // Make sure the variables written back are the same
  stan::test::expect_near_rel("deserializer read free",
                              theta1.segment(0, used1),
                              theta2.segment(0, used1));
}

TEST(serializer_vectorized, write_free_positive_ordered) {
  write_free_positive_ordered_test<Eigen::VectorXd>(4);
  write_free_positive_ordered_test<std::vector<Eigen::VectorXd>>(2, 4);
  write_free_positive_ordered_test<std::vector<std::vector<Eigen::VectorXd>>>(
      3, 2, 4);
}

// cholesky_factor_cov

template <typename Ret, typename... Sizes>
void write_free_cholesky_factor_cov_test(Sizes... sizes) {
  Eigen::VectorXd theta1 = Eigen::VectorXd::Random(100);
  Eigen::VectorXd theta2 = Eigen::VectorXd::Random(theta1.size());
  std::vector<int> theta_i;

  // Read an constrained variable
  stan::io::deserializer<double> deserializer(theta1, theta_i);
  double lp = 0.0;
  Ret vec_ref = deserializer.read_constrain_cholesky_factor_cov<Ret, false>(
      lp, sizes...);

  // Serialize a constrained variable
  stan::io::serializer<double> serializer(theta2);
  serializer.write_free_cholesky_factor_cov(vec_ref);

  size_t used1 = theta1.size() - deserializer.available();
  size_t used2 = theta2.size() - serializer.available();

  // Number of variables read should equal number of variables written
  EXPECT_EQ(used1, used2);

  // Make sure the variables written back are the same
  stan::test::expect_near_rel("deserializer read free",
                              theta1.segment(0, used1),
                              theta2.segment(0, used1));
}

TEST(serializer_vectorized, write_free_cholesky_factor_cov) {
  write_free_cholesky_factor_cov_test<Eigen::MatrixXd>(4, 3);
  write_free_cholesky_factor_cov_test<std::vector<Eigen::MatrixXd>>(2, 4, 3);
  write_free_cholesky_factor_cov_test<
      std::vector<std::vector<Eigen::MatrixXd>>>(3, 2, 4, 3);

  write_free_cholesky_factor_cov_test<Eigen::MatrixXd>(2, 2);
  write_free_cholesky_factor_cov_test<std::vector<Eigen::MatrixXd>>(2, 2, 2);
  write_free_cholesky_factor_cov_test<
      std::vector<std::vector<Eigen::MatrixXd>>>(3, 2, 2, 2);
}

// cholesky_factor_corr

template <typename Ret, typename... Sizes>
void write_free_cholesky_factor_corr_test(Sizes... sizes) {
  // Read an constrained variable
  Eigen::VectorXd theta1 = Eigen::VectorXd::Random(100);
  std::vector<int> theta_i;
  stan::io::deserializer<double> deserializer(theta1, theta_i);
  double lp = 0.0;
  Ret vec_ref = deserializer.read_constrain_cholesky_factor_corr<Ret, false>(
      lp, sizes...);

  // Serialize a constrained variable
  Eigen::VectorXd theta2 = Eigen::VectorXd::Random(theta1.size());
  stan::io::serializer<double> serializer(theta2);
  serializer.write_free_cholesky_factor_corr(vec_ref);

  size_t used1 = theta1.size() - deserializer.available();
  size_t used2 = theta2.size() - serializer.available();

  // Number of variables read should equal number of variables written
  EXPECT_EQ(used1, used2);

  // Make sure the variables written back are the same
  stan::test::expect_near_rel("deserializer read free",
                              theta1.segment(0, used1),
                              theta2.segment(0, used1));
}

TEST(serializer_vectorized, write_free_cholesky_factor_corr) {
  write_free_cholesky_factor_corr_test<Eigen::MatrixXd>(2);
  write_free_cholesky_factor_corr_test<std::vector<Eigen::MatrixXd>>(2, 2);
  write_free_cholesky_factor_corr_test<
      std::vector<std::vector<Eigen::MatrixXd>>>(3, 2, 2);
}

// cov_matrix

template <typename Ret, typename... Sizes>
void write_free_cov_matrix_test(Sizes... sizes) {
  Eigen::VectorXd theta1 = Eigen::VectorXd::Random(100);
  Eigen::VectorXd theta2 = Eigen::VectorXd::Random(theta1.size());
  std::vector<int> theta_i;

  // Read an constrained variable
  stan::io::deserializer<double> deserializer(theta1, theta_i);
  double lp = 0.0;
  Ret vec_ref
      = deserializer.read_constrain_cov_matrix<Ret, false>(lp, sizes...);

  // Serialize a constrained variable
  stan::io::serializer<double> serializer(theta2);
  serializer.write_free_cov_matrix(vec_ref);

  size_t used1 = theta1.size() - deserializer.available();
  size_t used2 = theta2.size() - serializer.available();

  // Number of variables read should equal number of variables written
  EXPECT_EQ(used1, used2);

  // Make sure the variables written back are the same
  stan::test::expect_near_rel("deserializer read free",
                              theta1.segment(0, used1),
                              theta2.segment(0, used1));
}

TEST(serializer_vectorized, write_free_cov_matrix) {
  write_free_cov_matrix_test<Eigen::MatrixXd>(2);
  write_free_cov_matrix_test<std::vector<Eigen::MatrixXd>>(2, 2);
  write_free_cov_matrix_test<std::vector<std::vector<Eigen::MatrixXd>>>(3, 2,
                                                                        2);
}

// corr_matrix

template <typename Ret, typename... Sizes>
void write_free_corr_matrix_test(Sizes... sizes) {
  Eigen::VectorXd theta1 = Eigen::VectorXd::Random(100);
  Eigen::VectorXd theta2 = Eigen::VectorXd::Random(theta1.size());
  std::vector<int> theta_i;

  // Read an constrained variable
  stan::io::deserializer<double> deserializer(theta1, theta_i);
  double lp = 0.0;
  Ret vec_ref
      = deserializer.read_constrain_corr_matrix<Ret, false>(lp, sizes...);

  // Serialize a constrained variable
  stan::io::serializer<double> serializer(theta2);
  serializer.write_free_corr_matrix(vec_ref);

  size_t used1 = theta1.size() - deserializer.available();
  size_t used2 = theta2.size() - serializer.available();

  // Number of variables read should equal number of variables written
  EXPECT_EQ(used1, used2);

  // Make sure the variables written back are the same
  stan::test::expect_near_rel("deserializer read free",
                              theta1.segment(0, used1),
                              theta2.segment(0, used1));
}

TEST(serializer_vectorized, write_free_corr_matrix) {
  write_free_corr_matrix_test<Eigen::MatrixXd>(2);
  write_free_corr_matrix_test<std::vector<Eigen::MatrixXd>>(2, 2);
  write_free_corr_matrix_test<std::vector<std::vector<Eigen::MatrixXd>>>(3, 2,
                                                                         2);
}

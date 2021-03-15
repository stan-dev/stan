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

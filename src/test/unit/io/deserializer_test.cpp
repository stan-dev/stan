#include <stan/io/deserializer.hpp>
// expect_near_rel comes from lib/stan_math
#include <test/unit/math/expect_near_rel.hpp>
#include <gtest/gtest.h>

TEST(deserializer_scalar, read) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(1.0);
  theta.push_back(2.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double x = deserializer.read<double>();
  EXPECT_FLOAT_EQ(1.0, x);
  double y = deserializer.read<double>();
  EXPECT_FLOAT_EQ(2.0, y);
  EXPECT_EQ(0U, deserializer.available());
}

TEST(deserializer_scalar, complex_read) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(1.0);
  theta.push_back(2.0);
  theta.push_back(3.0);
  theta.push_back(4.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  std::complex<double> x = deserializer.read<std::complex<double>>();
  EXPECT_FLOAT_EQ(1.0, x.real());
  EXPECT_FLOAT_EQ(2.0, x.imag());
  std::complex<double> y = deserializer.read<std::complex<double>>();
  EXPECT_FLOAT_EQ(3.0, y.real());
  EXPECT_FLOAT_EQ(4.0, y.imag());
  EXPECT_EQ(0U, deserializer.available());
}

TEST(deserializer, read_int) {
  Eigen::Matrix<int, -1, 1> theta_i(1);
  Eigen::VectorXd theta(2);
  theta[0] = 1.0;
  theta[1] = 2.0;
  theta_i[0] = 1;
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double x = deserializer.read<double>();
  EXPECT_FLOAT_EQ(1.0, x);
  double y = deserializer.read<double>();
  EXPECT_FLOAT_EQ(2.0, y);
  int z = deserializer.read<int>();
  EXPECT_EQ(1, z);
  EXPECT_EQ(0U, deserializer.available());
}

// vector

TEST(deserializer_vector, read) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  for (size_t i = 0; i < 7U; ++i) {
    double x = deserializer.read<double>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x);
  }
  Eigen::Matrix<double, Eigen::Dynamic, 1> y
      = deserializer.read<Eigen::VectorXd>(4);
  EXPECT_EQ(4, y.rows());
  EXPECT_EQ(1, y.cols());
  EXPECT_EQ(4, y.size());
  EXPECT_FLOAT_EQ(7.0, y[0]);
  EXPECT_FLOAT_EQ(8.0, y[1]);
  EXPECT_FLOAT_EQ(9.0, y[2]);
  EXPECT_FLOAT_EQ(10.0, y[3]);

  double z = deserializer.read<double>();
  EXPECT_FLOAT_EQ(11.0, z);
}

TEST(deserializer_vector, complex_read) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  for (size_t i = 0; i < 7U; ++i) {
    double x = deserializer.read<double>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x);
  }
  using complex_vec = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;
  complex_vec y = deserializer.read<complex_vec>(4);
  EXPECT_EQ(4, y.rows());
  EXPECT_EQ(1, y.cols());
  EXPECT_EQ(4, y.size());
  double sentinal = 7;
  for (int i = 0; i < y.size(); ++i) {
    EXPECT_FLOAT_EQ(sentinal, y[i].real());
    ++sentinal;
    EXPECT_FLOAT_EQ(sentinal, y[i].imag());
    ++sentinal;
  }

  double z = deserializer.read<double>();
  EXPECT_FLOAT_EQ(15.0, z);
}

// row vector

TEST(deserializer_row_vector, read) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  for (size_t i = 0; i < 7U; ++i) {
    double x = deserializer.read<double>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x);
  }
  Eigen::Matrix<double, 1, Eigen::Dynamic> y
      = deserializer.read<Eigen::RowVectorXd>(4);
  EXPECT_EQ(4, y.cols());
  EXPECT_EQ(1, y.rows());
  EXPECT_EQ(4, y.size());
  EXPECT_FLOAT_EQ(7.0, y[0]);
  EXPECT_FLOAT_EQ(8.0, y[1]);
  EXPECT_FLOAT_EQ(9.0, y[2]);
  EXPECT_FLOAT_EQ(10.0, y[3]);

  double z = deserializer.read<double>();
  EXPECT_FLOAT_EQ(11.0, z);
}

TEST(deserializer_row_vector, complex_read) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  for (size_t i = 0; i < 7U; ++i) {
    double x = deserializer.read<double>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x);
  }
  using complex_row_vec
      = Eigen::Matrix<std::complex<double>, 1, Eigen::Dynamic>;
  complex_row_vec y = deserializer.read<complex_row_vec>(4);
  EXPECT_EQ(1, y.rows());
  EXPECT_EQ(4, y.cols());
  EXPECT_EQ(4, y.size());
  double sentinal = 7;
  for (int i = 0; i < y.size(); ++i) {
    EXPECT_FLOAT_EQ(sentinal, y[i].real());
    ++sentinal;
    EXPECT_FLOAT_EQ(sentinal, y[i].imag());
    ++sentinal;
  }

  double z = deserializer.read<double>();
  EXPECT_FLOAT_EQ(15.0, z);
}

// matrix

TEST(deserializer_matrix, read) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = deserializer.read<double>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x);
  }
  using eig_mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  eig_mat y = deserializer.read<eig_mat>(3, 2);
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  EXPECT_FLOAT_EQ(7.0, y(0, 0));
  EXPECT_FLOAT_EQ(8.0, y(1, 0));
  EXPECT_FLOAT_EQ(9.0, y(2, 0));
  EXPECT_FLOAT_EQ(10.0, y(0, 1));
  EXPECT_FLOAT_EQ(11.0, y(1, 1));
  EXPECT_FLOAT_EQ(12.0, y(2, 1));

  double a = deserializer.read<double>();
  EXPECT_FLOAT_EQ(13.0, a);
}

TEST(deserializer_matrix, complex_read) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = deserializer.read<double>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x);
  }
  using eig_mat
      = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>;
  eig_mat y = deserializer.read<eig_mat>(3, 2);
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  double sentinal = 7;
  for (int i = 0; i < y.size(); ++i) {
    EXPECT_FLOAT_EQ(sentinal, y(i).real());
    sentinal++;
    EXPECT_FLOAT_EQ(sentinal, y(i).imag());
    sentinal++;
  }
}

// array

TEST(deserializer_array, read) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  for (size_t i = 0; i < 7U; ++i) {
    double x = deserializer.read<double>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x);
  }
  std::vector<double> y = deserializer.read<std::vector<double>>(4);
  EXPECT_EQ(4, y.size());
  EXPECT_FLOAT_EQ(7.0, y[0]);
  EXPECT_FLOAT_EQ(8.0, y[1]);
  EXPECT_FLOAT_EQ(9.0, y[2]);
  EXPECT_FLOAT_EQ(10.0, y[3]);

  double z = deserializer.read<double>();
  EXPECT_FLOAT_EQ(11.0, z);
}

TEST(deserializer_array, complex_read) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  for (size_t i = 0; i < 7U; ++i) {
    double x = deserializer.read<double>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x);
  }
  using complex_array = std::vector<std::complex<double>>;
  complex_array y = deserializer.read<complex_array>(4);
  EXPECT_EQ(4, y.size());
  double sentinal = 7;
  for (int i = 0; i < y.size(); ++i) {
    EXPECT_FLOAT_EQ(sentinal, y[i].real());
    ++sentinal;
    EXPECT_FLOAT_EQ(sentinal, y[i].imag());
    ++sentinal;
  }

  double z = deserializer.read<double>();
  EXPECT_FLOAT_EQ(15.0, z);
}

TEST(deserializer_array, read_vector) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  std::vector<Eigen::VectorXd> y
      = deserializer.read<std::vector<Eigen::VectorXd>>(4, 2);
  EXPECT_EQ(4, y.size());
  EXPECT_FLOAT_EQ(0.0, y[0](0));
  EXPECT_FLOAT_EQ(1.0, y[0](1));
  EXPECT_FLOAT_EQ(2.0, y[1](0));
  EXPECT_FLOAT_EQ(3.0, y[1](1));
  EXPECT_FLOAT_EQ(4.0, y[2](0));
  EXPECT_FLOAT_EQ(5.0, y[2](1));
  EXPECT_FLOAT_EQ(6.0, y[3](0));
  EXPECT_FLOAT_EQ(7.0, y[3](1));

  double z = deserializer.read<double>();
  EXPECT_FLOAT_EQ(8.0, z);
}

// size zero

TEST(deserializer, zeroSizeVecs) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(1.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);

  EXPECT_FLOAT_EQ(1.0, deserializer.read<double>());  // finish available

  EXPECT_EQ(0, deserializer.read<std::vector<double>>(0).size());
  EXPECT_EQ(0, deserializer.read<Eigen::VectorXd>(0).size());
  EXPECT_EQ(0, deserializer.read<Eigen::RowVectorXd>(0).size());
  EXPECT_EQ(0, deserializer.read<Eigen::MatrixXd>(0, 3).size());
  EXPECT_EQ(0, deserializer.read<Eigen::MatrixXd>(3, 0).size());
  EXPECT_EQ(0, deserializer
                   .read<std::vector<std::vector<Eigen::MatrixXd>>>(0, 0, 0, 0)
                   .size());
}

// out of memory

TEST(deserializer, eos_exception) {
  std::vector<double> theta;
  theta.push_back(1.0);
  theta.push_back(2.0);
  std::vector<int> theta_i;
  theta_i.push_back(1);
  stan::io::deserializer<double> deserializer(theta, theta_i);

  EXPECT_EQ(2U, deserializer.available());
  EXPECT_EQ(1U, deserializer.available_i());

  EXPECT_NO_THROW(deserializer.read<double>());
  EXPECT_NO_THROW(deserializer.read<double>());
  EXPECT_THROW(deserializer.read<double>(), std::runtime_error);

  // should go back to working
  EXPECT_NO_THROW(deserializer.read<int>());
  EXPECT_THROW(deserializer.read<int>(), std::runtime_error);

  // should keep throwing
  EXPECT_THROW(deserializer.read<double>(), std::runtime_error);
  EXPECT_THROW(deserializer.read<size_t>(), std::runtime_error);

  // The strategy for all the following checks is to allocate 1 less than
  // the required memory and make sure an error happens
  {
    std::vector<double> theta(1);
    stan::io::deserializer<double> deserializer(theta, theta_i);
    EXPECT_THROW(deserializer.read<std::complex<double>>(), std::runtime_error);
  }

  {
    std::vector<double> theta(2);
    stan::io::deserializer<double> deserializer(theta, theta_i);
    EXPECT_THROW(deserializer.read<std::vector<double>>(3), std::runtime_error);
  }

  {
    std::vector<double> theta(1);
    stan::io::deserializer<double> deserializer(theta, theta_i);
    EXPECT_THROW(deserializer.read<Eigen::VectorXd>(2), std::runtime_error);
  }

  {
    std::vector<double> theta(3);
    stan::io::deserializer<double> deserializer(theta, theta_i);
    EXPECT_THROW(
        (deserializer
             .read<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>>(2)),
        std::runtime_error);
  }

  {
    std::vector<double> theta(5);
    stan::io::deserializer<double> deserializer(theta, theta_i);
    EXPECT_THROW(deserializer.read<std::vector<Eigen::VectorXd>>(3, 2),
                 std::runtime_error);
  }

  {
    std::vector<double> theta(1);
    stan::io::deserializer<double> deserializer(theta, theta_i);
    EXPECT_THROW(deserializer.read<Eigen::RowVectorXd>(2), std::runtime_error);
  }

  {
    std::vector<double> theta(3);
    stan::io::deserializer<double> deserializer(theta, theta_i);
    EXPECT_THROW(
        (deserializer
             .read<Eigen::Matrix<std::complex<double>, 1, Eigen::Dynamic>>(2)),
        std::runtime_error);
  }

  {
    std::vector<double> theta(5);
    stan::io::deserializer<double> deserializer(theta, theta_i);
    EXPECT_THROW(deserializer.read<std::vector<Eigen::RowVectorXd>>(3, 2),
                 std::runtime_error);
  }

  {
    std::vector<double> theta(3);
    stan::io::deserializer<double> deserializer(theta, theta_i);
    EXPECT_THROW(deserializer.read<Eigen::MatrixXd>(2, 2), std::runtime_error);
  }

  {
    std::vector<double> theta(7);
    stan::io::deserializer<double> deserializer(theta, theta_i);
    EXPECT_THROW(
        (deserializer.read<Eigen::Matrix<std::complex<double>, Eigen::Dynamic,
                                         Eigen::Dynamic>>(2, 2)),
        std::runtime_error);
  }

  {
    std::vector<double> theta(11);
    stan::io::deserializer<double> deserializer(theta, theta_i);
    EXPECT_THROW(deserializer.read<std::vector<Eigen::MatrixXd>>(2, 3, 2),
                 std::runtime_error);
  }
}

// lb

TEST(deserializer_scalar, read_constrain_lb_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0.0;
  EXPECT_FLOAT_EQ(1.0 + exp(-2.0),
                  (deserializer.read_constrain_lb<double, false>(1.0, lp)));
  EXPECT_FLOAT_EQ(5.0 + exp(3.0),
                  (deserializer.read_constrain_lb<double, false>(5.0, lp)));
  EXPECT_FLOAT_EQ(-2.0 + exp(-1.0),
                  (deserializer.read_constrain_lb<double, false>(-2.0, lp)));
  EXPECT_FLOAT_EQ(15.0 + exp(0.0),
                  (deserializer.read_constrain_lb<double, false>(15.0, lp)));
}

TEST(deserializer_scalar, read_constrain_lb_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = -1.5;
  EXPECT_FLOAT_EQ(1.0 + exp(-2.0),
                  (deserializer.read_constrain_lb<double, true>(1.0, lp)));
  EXPECT_FLOAT_EQ(5.0 + exp(3.0),
                  (deserializer.read_constrain_lb<double, true>(5.0, lp)));
  EXPECT_FLOAT_EQ(-2.0 + exp(-1.0),
                  (deserializer.read_constrain_lb<double, true>(-2.0, lp)));
  EXPECT_FLOAT_EQ(15.0 + exp(0.0),
                  (deserializer.read_constrain_lb<double, true>(15.0, lp)));
  EXPECT_FLOAT_EQ(-1.5 - 2.0 + 3.0 - 1.0, lp);
}

// ub

TEST(deserializer_scalar, read_constrain_ub_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  EXPECT_FLOAT_EQ(1.0 - exp(-2.0),
                  (deserializer.read_constrain_ub<double, false>(1.0, lp)));
  EXPECT_FLOAT_EQ(5.0 - exp(3.0),
                  (deserializer.read_constrain_ub<double, false>(5.0, lp)));
  EXPECT_FLOAT_EQ(-2.0 - exp(-1.0),
                  (deserializer.read_constrain_ub<double, false>(-2.0, lp)));
  EXPECT_FLOAT_EQ(15.0 - exp(0.0),
                  (deserializer.read_constrain_ub<double, false>(15.0, lp)));
}
TEST(deserializer_scalar, read_constrain_ub_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = -12.9;
  EXPECT_FLOAT_EQ(1.0 - exp(-2.0),
                  (deserializer.read_constrain_ub<double, true>(1.0, lp)));
  EXPECT_FLOAT_EQ(5.0 - exp(3.0),
                  (deserializer.read_constrain_ub<double, true>(5.0, lp)));
  EXPECT_FLOAT_EQ(-2.0 - exp(-1.0),
                  (deserializer.read_constrain_ub<double, true>(-2.0, lp)));
  EXPECT_FLOAT_EQ(15.0 - exp(0.0),
                  (deserializer.read_constrain_ub<double, true>(15.0, lp)));
  EXPECT_FLOAT_EQ(-12.9 - 2.0 + 3.0 - 1.0, lp);
}

// lub

const double inv_logit_m2 = 0.1192029;  // stan::math::inv_logit(-2.0)
const double inv_logit_m1 = 0.2689414;  // stan::math::inv_logit(-1.0)
const double inv_logit_0 = 0.5;         // stan::math::inv_logit(0)
const double inv_logit_3 = 0.9525741;   // stan::math::inv_logit(3.0)

TEST(deserializer_scalar, read_constrain_lub_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  EXPECT_FLOAT_EQ(inv_logit_m2, (deserializer.read_constrain_lub<double, false>(
                                    0.0, 1.0, lp)));
  EXPECT_FLOAT_EQ(
      3.0 + 2.0 * inv_logit_3,
      (deserializer.read_constrain_lub<double, false>(3.0, 5.0, lp)));
  EXPECT_FLOAT_EQ(
      -3.0 + 5.0 * inv_logit_m1,
      (deserializer.read_constrain_lub<double, false>(-3.0, 2.0, lp)));
  EXPECT_FLOAT_EQ(
      -15.0 + 30.0 * inv_logit_0,
      (deserializer.read_constrain_lub<double, false>(-15.0, 15.0, lp)));
}
TEST(deserializer_scalar, read_constrain_lub_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = -7.2;
  EXPECT_FLOAT_EQ(
      0.0 + 1.0 * inv_logit_m2,
      (deserializer.read_constrain_lub<double, true>(0.0, 1.0, lp)));
  EXPECT_FLOAT_EQ(
      3.0 + 2.0 * inv_logit_3,
      (deserializer.read_constrain_lub<double, true>(3.0, 5.0, lp)));
  EXPECT_FLOAT_EQ(
      -3.0 + 5.0 * inv_logit_m1,
      (deserializer.read_constrain_lub<double, true>(-3.0, 2.0, lp)));
  EXPECT_FLOAT_EQ(
      -15.0 + 30.0 * inv_logit_0,
      (deserializer.read_constrain_lub<double, true>(-15.0, 15.0, lp)));
  double expected_lp = -7.2
                       + log((1.0 - 0.0) * inv_logit_m2 * (1 - inv_logit_m2))
                       + log((5.0 - 3.0) * inv_logit_3 * (1 - inv_logit_3))
                       + log((2.0 - -3.0) * inv_logit_m1 * (1 - inv_logit_m1))
                       + log((15.0 - -15.0) * inv_logit_0 * (1 - inv_logit_0));
  EXPECT_FLOAT_EQ(expected_lp, lp);
}

// offset multiplier
TEST(deserializer_scalar, offset_multiplier_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  EXPECT_FLOAT_EQ(-2.0,
                  (deserializer.read_constrain_offset_multiplier<double, false>(
                      0.0, 1.0, lp)));
  EXPECT_FLOAT_EQ(3.0 + 5.0 * 3.0,
                  (deserializer.read_constrain_offset_multiplier<double, false>(
                      3.0, 5.0, lp)));
  EXPECT_FLOAT_EQ(-3.0 + 2.0 * -1.0,
                  (deserializer.read_constrain_offset_multiplier<double, false>(
                      -3.0, 2.0, lp)));
  EXPECT_FLOAT_EQ(-15.0,
                  (deserializer.read_constrain_offset_multiplier<double, false>(
                      -15.0, 15.0, lp)));
}

TEST(deserializer_scalar, offset_multiplier_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = -7.2;
  EXPECT_FLOAT_EQ(-2.0,
                  (deserializer.read_constrain_offset_multiplier<double, true>(
                      0.0, 1.0, lp)));
  EXPECT_FLOAT_EQ(3.0 + 5.0 * 3.0,
                  (deserializer.read_constrain_offset_multiplier<double, true>(
                      3.0, 5.0, lp)));
  EXPECT_FLOAT_EQ(-3.0 + 2.0 * -1.0,
                  (deserializer.read_constrain_offset_multiplier<double, true>(
                      -3.0, 2.0, lp)));
  EXPECT_FLOAT_EQ(-15.0,
                  (deserializer.read_constrain_offset_multiplier<double, true>(
                      -15.0, 15.0, lp)));
  double expected_lp = -7.2 + log(1.0) + log(5.0) + log(2.0) + log(15.0);
  EXPECT_FLOAT_EQ(expected_lp, lp);
}

// unit vector

TEST(deserializer_vector, unit_vector_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  Eigen::VectorXd reference
      = stan::math::unit_vector_constrain(stan::math::to_vector(theta));
  Eigen::VectorXd phi(
      deserializer.read_constrain_unit_vector<Eigen::VectorXd, false>(lp, 4));
  for (size_t i = 0; i < phi.size(); ++i) {
    EXPECT_FLOAT_EQ(reference(i), phi[i]);
  }
}

TEST(deserializer_vector, unit_vector_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0.0;
  double lp_ref = 0.0;
  Eigen::VectorXd reference
      = stan::math::unit_vector_constrain(stan::math::to_vector(theta), lp_ref);
  Eigen::VectorXd phi(
      deserializer.read_constrain_unit_vector<Eigen::VectorXd, true>(lp, 4));
  for (size_t i = 0; i < phi.size(); ++i) {
    EXPECT_FLOAT_EQ(reference(i), phi[i]);
  }
  EXPECT_FLOAT_EQ(lp_ref, lp);
}

// simplex

TEST(deserializer_vector, simplex_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  Eigen::VectorXd reference
      = stan::math::simplex_constrain(stan::math::to_vector(theta));
  Eigen::VectorXd phi(
      deserializer.read_constrain_simplex<Eigen::VectorXd, false>(
          lp, theta.size() + 1));
  for (size_t i = 0; i < phi.size(); ++i) {
    EXPECT_FLOAT_EQ(reference(i), phi[i]);
  }
}

TEST(deserializer_vector, simplex_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0.0;
  double lp_ref = 0.0;
  Eigen::VectorXd reference
      = stan::math::simplex_constrain(stan::math::to_vector(theta), lp_ref);
  Eigen::VectorXd phi(
      deserializer.read_constrain_simplex<Eigen::VectorXd, true>(
          lp, theta.size() + 1));
  for (size_t i = 0; i < phi.size(); ++i) {
    EXPECT_FLOAT_EQ(reference(i), phi[i]);
  }
  EXPECT_FLOAT_EQ(lp_ref, lp);
}

// ordered

TEST(deserializer_vector, ordered_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  double v0 = 3.0;
  double v1 = v0 + exp(-1.0);
  double v2 = v1 + exp(-2.0);
  double v3 = v2 + exp(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  Eigen::Matrix<double, Eigen::Dynamic, 1> phi(
      deserializer.read_constrain_ordered<Eigen::VectorXd, false>(lp, 4));
  EXPECT_FLOAT_EQ(v0, phi[0]);
  EXPECT_FLOAT_EQ(v1, phi[1]);
  EXPECT_FLOAT_EQ(v2, phi[2]);
  EXPECT_FLOAT_EQ(v3, phi[3]);
}

TEST(deserializer_vector, ordered_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  double v0 = 3.0;
  double v1 = v0 + exp(-1.0);
  double v2 = v1 + exp(-2.0);
  double v3 = v2 + exp(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = -101.1;
  double expected_lp = lp - 1.0 - 2.0 + 0.0;
  Eigen::Matrix<double, Eigen::Dynamic, 1> phi(
      deserializer.read_constrain_ordered<Eigen::VectorXd, true>(lp, 4));
  EXPECT_FLOAT_EQ(v0, phi[0]);
  EXPECT_FLOAT_EQ(v1, phi[1]);
  EXPECT_FLOAT_EQ(v2, phi[2]);
  EXPECT_FLOAT_EQ(v3, phi[3]);
  EXPECT_FLOAT_EQ(expected_lp, lp);
}

// positive ordered

TEST(deserializer_vector, positive_ordered_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  double v0 = exp(3.0);
  double v1 = v0 + exp(-1.0);
  double v2 = v1 + exp(-2.0);
  double v3 = v2 + exp(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  Eigen::Matrix<double, Eigen::Dynamic, 1> phi(
      deserializer.read_constrain_positive_ordered<Eigen::VectorXd, false>(lp,
                                                                           4));
  EXPECT_FLOAT_EQ(v0, phi[0]);
  EXPECT_FLOAT_EQ(v1, phi[1]);
  EXPECT_FLOAT_EQ(v2, phi[2]);
  EXPECT_FLOAT_EQ(v3, phi[3]);
}

TEST(deserializer_vector, positive_ordered_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  double v0 = exp(3.0);
  double v1 = v0 + exp(-1.0);
  double v2 = v1 + exp(-2.0);
  double v3 = v2 + exp(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = -101.1;
  double expected_lp = lp + 3.0 - 1.0 - 2.0 + 0.0;
  Eigen::Matrix<double, Eigen::Dynamic, 1> phi(
      deserializer.read_constrain_positive_ordered<Eigen::VectorXd, true>(lp,
                                                                          4));
  EXPECT_FLOAT_EQ(v0, phi[0]);
  EXPECT_FLOAT_EQ(v1, phi[1]);
  EXPECT_FLOAT_EQ(v2, phi[2]);
  EXPECT_FLOAT_EQ(v3, phi[3]);
  EXPECT_FLOAT_EQ(expected_lp, lp);
}

// chol cov

TEST(deserializer_matrix, cholesky_factor_cov_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  Eigen::MatrixXd reference = stan::math::cholesky_factor_constrain(
      stan::math::to_vector(theta).segment(0, 6), 3, 3);
  Eigen::MatrixXd L(
      deserializer.read_constrain_cholesky_factor_cov<Eigen::MatrixXd, false>(
          lp, 3U, 3U));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  stan::test::expect_near_rel("deserializer tests", reference, L);
  EXPECT_EQ(2U, deserializer.available());
}

TEST(deserializer_matrix, cholesky_factor_cov_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp_ref = 0.0;
  double lp = 0.0;
  Eigen::MatrixXd reference = stan::math::cholesky_factor_constrain(
      stan::math::to_vector(theta).segment(0, 6), 3, 3, lp_ref);
  Eigen::MatrixXd L(
      deserializer.read_constrain_cholesky_factor_cov<Eigen::MatrixXd, true>(
          lp, 3U, 3U));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  stan::test::expect_near_rel("deserializer tests", reference, L);
  EXPECT_EQ(2U, deserializer.available());
  EXPECT_FLOAT_EQ(lp_ref, lp);
}

TEST(deserializer_matrix, cholesky_factor_cov_constrain_non_square) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  Eigen::MatrixXd reference = stan::math::cholesky_factor_constrain(
      stan::math::to_vector(theta).segment(0, 5), 3, 2);
  Eigen::MatrixXd L(
      deserializer.read_constrain_cholesky_factor_cov<Eigen::MatrixXd, false>(
          lp, 3U, 2U));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(2, L.cols());
  EXPECT_EQ(6, L.size());
  stan::test::expect_near_rel("deserializer tests", reference, L);
  EXPECT_EQ(3U, deserializer.available());
}

TEST(deserializer_matrix, cholesky_factor_cov_jacobian_non_square) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp_ref = 0.0;
  double lp = 0.0;
  Eigen::MatrixXd reference = stan::math::cholesky_factor_constrain(
      stan::math::to_vector(theta).segment(0, 5), 3, 2, lp_ref);
  Eigen::MatrixXd L(
      deserializer.read_constrain_cholesky_factor_cov<Eigen::MatrixXd, true>(
          lp, 3U, 2U));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(2, L.cols());
  EXPECT_EQ(6, L.size());
  stan::test::expect_near_rel("deserializer tests", reference, L);
  EXPECT_EQ(3U, deserializer.available());
  EXPECT_FLOAT_EQ(lp_ref, lp);
}

// chol corr

TEST(deserializer_matrix, cholesky_factor_corr_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  Eigen::MatrixXd reference = stan::math::cholesky_corr_constrain(
      stan::math::to_vector(theta).segment(0, 3), 3);
  Eigen::MatrixXd L(
      deserializer.read_constrain_cholesky_factor_corr<Eigen::MatrixXd, false>(
          lp, 3U));
  EXPECT_NO_THROW(stan::math::check_cholesky_factor_corr(
      "test_cholesky_factor_corr_constrain", "L", L));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  stan::test::expect_near_rel("deserializer tests", reference, L);
  EXPECT_EQ(5U, deserializer.available());
}

TEST(deserializer_matrix, cholesky_factor_corr_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp_ref = 0.0;
  double lp = 0.0;
  Eigen::MatrixXd reference = stan::math::cholesky_corr_constrain(
      stan::math::to_vector(theta).segment(0, 3), 3, lp_ref);
  Eigen::MatrixXd L(
      deserializer.read_constrain_cholesky_factor_corr<Eigen::MatrixXd, true>(
          lp, 3U));
  EXPECT_NO_THROW(stan::math::check_cholesky_factor_corr(
      "test_cholesky_factor_corr_constrain", "L", L));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  stan::test::expect_near_rel("deserializer tests", reference, L);
  EXPECT_EQ(5U, deserializer.available());
  EXPECT_FLOAT_EQ(lp_ref, lp);
}

// cov

TEST(deserializer_matrix, cov_matrix_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  Eigen::MatrixXd reference = stan::math::cov_matrix_constrain(
      stan::math::to_vector(theta).segment(0, 6), 3);
  Eigen::MatrixXd L(
      deserializer.read_constrain_cov_matrix<Eigen::MatrixXd, false>(lp, 3U));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  stan::test::expect_near_rel("deserializer tests", reference, L);
  EXPECT_EQ(2U, deserializer.available());
}

TEST(deserializer_matrix, cov_matrix_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp_ref = 0.0;
  double lp = 0.0;
  Eigen::MatrixXd reference = stan::math::cov_matrix_constrain(
      stan::math::to_vector(theta).segment(0, 6), 3, lp_ref);
  Eigen::MatrixXd L(
      deserializer.read_constrain_cov_matrix<Eigen::MatrixXd, true>(lp, 3U));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  stan::test::expect_near_rel("deserializer tests", reference, L);
  EXPECT_EQ(2U, deserializer.available());
  EXPECT_FLOAT_EQ(lp_ref, lp);
}

// corr

TEST(deserializer_matrix, corr_matrix_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  Eigen::MatrixXd reference = stan::math::corr_matrix_constrain(
      stan::math::to_vector(theta).segment(0, 3), 3);
  Eigen::MatrixXd L(
      deserializer.read_constrain_corr_matrix<Eigen::MatrixXd, false>(lp, 3U));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  stan::test::expect_near_rel("deserializer tests", reference, L);
  EXPECT_EQ(5U, deserializer.available());
}

TEST(deserializer_matrix, corr_matrix_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp_ref = 0.0;
  double lp = 0.0;
  Eigen::MatrixXd reference = stan::math::corr_matrix_constrain(
      stan::math::to_vector(theta).segment(0, 3), 3, lp_ref);
  Eigen::MatrixXd L(
      deserializer.read_constrain_corr_matrix<Eigen::MatrixXd, true>(lp, 3U));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  stan::test::expect_near_rel("deserializer tests", reference, L);
  EXPECT_EQ(5U, deserializer.available());
  EXPECT_FLOAT_EQ(lp_ref, lp);
}

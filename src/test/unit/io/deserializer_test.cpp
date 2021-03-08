#include <stan/io/deserializer.hpp>
#include <gtest/gtest.h>

TEST(deserializer, eigen_input) {
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
}

TEST(deserializer_rowvector, read) {
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

// scalar bounds
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

TEST(deserializer_scalar, read_lb_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0.0;
  EXPECT_FLOAT_EQ(1.0 + exp(-2.0),
                  (deserializer.read_lb<double, false>(1.0, lp)));
  EXPECT_FLOAT_EQ(5.0 + exp(3.0),
                  (deserializer.read_lb<double, false>(5.0, lp)));
  EXPECT_FLOAT_EQ(-2.0 + exp(-1.0),
                  (deserializer.read_lb<double, false>(-2.0, lp)));
  EXPECT_FLOAT_EQ(15.0 + exp(0.0),
                  (deserializer.read_lb<double, false>(15.0, lp)));
}

TEST(deserializer_scalar, read_lb_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = -1.5;
  EXPECT_FLOAT_EQ(1.0 + exp(-2.0),
                  (deserializer.read_lb<double, true>(1.0, lp)));
  EXPECT_FLOAT_EQ(5.0 + exp(3.0),
                  (deserializer.read_lb<double, true>(5.0, lp)));
  EXPECT_FLOAT_EQ(-2.0 + exp(-1.0),
                  (deserializer.read_lb<double, true>(-2.0, lp)));
  EXPECT_FLOAT_EQ(15.0 + exp(0.0),
                  (deserializer.read_lb<double, true>(15.0, lp)));
  EXPECT_FLOAT_EQ(-1.5 - 2.0 + 3.0 - 1.0, lp);
}

// ub

TEST(deserializer_scalar, read_ub_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  EXPECT_FLOAT_EQ(1.0 - exp(-2.0),
                  (deserializer.read_ub<double, false>(1.0, lp)));
  EXPECT_FLOAT_EQ(5.0 - exp(3.0),
                  (deserializer.read_ub<double, false>(5.0, lp)));
  EXPECT_FLOAT_EQ(-2.0 - exp(-1.0),
                  (deserializer.read_ub<double, false>(-2.0, lp)));
  EXPECT_FLOAT_EQ(15.0 - exp(0.0),
                  (deserializer.read_ub<double, false>(15.0, lp)));
}
TEST(deserializer_scalar, read_ub_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = -12.9;
  EXPECT_FLOAT_EQ(1.0 - exp(-2.0),
                  (deserializer.read_ub<double, true>(1.0, lp)));
  EXPECT_FLOAT_EQ(5.0 - exp(3.0),
                  (deserializer.read_ub<double, true>(5.0, lp)));
  EXPECT_FLOAT_EQ(-2.0 - exp(-1.0),
                  (deserializer.read_ub<double, true>(-2.0, lp)));
  EXPECT_FLOAT_EQ(15.0 - exp(0.0),
                  (deserializer.read_ub<double, true>(15.0, lp)));
  EXPECT_FLOAT_EQ(-12.9 - 2.0 + 3.0 - 1.0, lp);
}

// lub

const double inv_logit_m2 = 0.1192029;  // stan::math::inv_logit(-2.0)
const double inv_logit_m1 = 0.2689414;  // stan::math::inv_logit(-1.0)
const double inv_logit_0 = 0.5;         // stan::math::inv_logit(0)
const double inv_logit_3 = 0.9525741;   // stan::math::inv_logit(3.0)

TEST(deserializer_scalar, read_lub_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  EXPECT_FLOAT_EQ(inv_logit_m2,
                  (deserializer.read_lub<double, false>(0.0, 1.0, lp)));
  EXPECT_FLOAT_EQ(3.0 + 2.0 * inv_logit_3,
                  (deserializer.read_lub<double, false>(3.0, 5.0, lp)));
  EXPECT_FLOAT_EQ(-3.0 + 5.0 * inv_logit_m1,
                  (deserializer.read_lub<double, false>(-3.0, 2.0, lp)));
  EXPECT_FLOAT_EQ(-15.0 + 30.0 * inv_logit_0,
                  (deserializer.read_lub<double, false>(-15.0, 15.0, lp)));
}
TEST(deserializer_scalar, read_lub_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = -7.2;
  EXPECT_FLOAT_EQ(0.0 + 1.0 * inv_logit_m2,
                  (deserializer.read_lub<double, true>(0.0, 1.0, lp)));
  EXPECT_FLOAT_EQ(3.0 + 2.0 * inv_logit_3,
                  (deserializer.read_lub<double, true>(3.0, 5.0, lp)));
  EXPECT_FLOAT_EQ(-3.0 + 5.0 * inv_logit_m1,
                  (deserializer.read_lub<double, true>(-3.0, 2.0, lp)));
  EXPECT_FLOAT_EQ(-15.0 + 30.0 * inv_logit_0,
                  (deserializer.read_lub<double, true>(-15.0, 15.0, lp)));
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
  EXPECT_FLOAT_EQ(
      -2.0, (deserializer.read_offset_multiplier<double, false>(0.0, 1.0, lp)));
  EXPECT_FLOAT_EQ(
      3.0 + 5.0 * 3.0,
      (deserializer.read_offset_multiplier<double, false>(3.0, 5.0, lp)));
  EXPECT_FLOAT_EQ(
      -3.0 + 2.0 * -1.0,
      (deserializer.read_offset_multiplier<double, false>(-3.0, 2.0, lp)));
  EXPECT_FLOAT_EQ(-15.0, (deserializer.read_offset_multiplier<double, false>(
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
  EXPECT_FLOAT_EQ(
      -2.0, (deserializer.read_offset_multiplier<double, true>(0.0, 1.0, lp)));
  EXPECT_FLOAT_EQ(
      3.0 + 5.0 * 3.0,
      (deserializer.read_offset_multiplier<double, true>(3.0, 5.0, lp)));
  EXPECT_FLOAT_EQ(
      -3.0 + 2.0 * -1.0,
      (deserializer.read_offset_multiplier<double, true>(-3.0, 2.0, lp)));
  EXPECT_FLOAT_EQ(-15.0, (deserializer.read_offset_multiplier<double, true>(
                             -15.0, 15.0, lp)));
  double expected_lp = -7.2 + log(1.0) + log(5.0) + log(2.0) + log(15.0);
  EXPECT_FLOAT_EQ(expected_lp, lp);
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

// need to add unit vector, simplex tests before merge

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
      deserializer.read_ordered<Eigen::VectorXd, false>(lp, 4));
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
      deserializer.read_ordered<Eigen::VectorXd, true>(lp, 4));
  EXPECT_FLOAT_EQ(v0, phi[0]);
  EXPECT_FLOAT_EQ(v1, phi[1]);
  EXPECT_FLOAT_EQ(v2, phi[2]);
  EXPECT_FLOAT_EQ(v3, phi[3]);
  EXPECT_FLOAT_EQ(expected_lp, lp);
}

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
      deserializer.read_positive_ordered<Eigen::VectorXd, false>(lp, 4));
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
      deserializer.read_positive_ordered<Eigen::VectorXd, true>(lp, 4));
  EXPECT_FLOAT_EQ(v0, phi[0]);
  EXPECT_FLOAT_EQ(v1, phi[1]);
  EXPECT_FLOAT_EQ(v2, phi[2]);
  EXPECT_FLOAT_EQ(v3, phi[3]);
  EXPECT_FLOAT_EQ(expected_lp, lp);
}

TEST(deserializer_vector, offset_multiplier_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  Eigen::Matrix<double, Eigen::Dynamic, 1> phi(
      deserializer.read_offset_multiplier<Eigen::VectorXd, false>(0.0, 1.0, lp,
                                                                  4));
  EXPECT_FLOAT_EQ(theta[0], phi[0]);
  EXPECT_FLOAT_EQ(theta[1], phi[1]);
  EXPECT_FLOAT_EQ(theta[2], phi[2]);
  EXPECT_FLOAT_EQ(theta[3], phi[3]);
}

TEST(deserializer_vector, offset_multiplier_constrain_lp) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  Eigen::Matrix<double, Eigen::Dynamic, 1> phi(
      deserializer.read_offset_multiplier<Eigen::VectorXd, true>(0, 2, lp, 4));
  EXPECT_FLOAT_EQ(theta[0] * 2, phi[0]);
  EXPECT_FLOAT_EQ(theta[1] * 2, phi[1]);
  EXPECT_FLOAT_EQ(theta[2] * 2, phi[2]);
  EXPECT_FLOAT_EQ(theta[3] * 2, phi[3]);
  EXPECT_FLOAT_EQ(lp, std::log(2) * 4);
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

TEST(deserializer_matrix, matrix_lb_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = deserializer.read<double>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x);
  }
  double lb = -1.5;
  double lp = 0;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> y
      = deserializer.read_lb<Eigen::MatrixXd, false>(lb, lp, 3, 2);
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(7.0, lb), y(0, 0));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(8.0, lb), y(1, 0));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(9.0, lb), y(2, 0));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(10.0, lb), y(0, 1));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(11.0, lb), y(1, 1));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(12.0, lb), y(2, 1));

  double a = deserializer.read<double>();
  EXPECT_FLOAT_EQ(13.0, a);
}

TEST(deserializer_matrix, matrix_lb_constrain_lp) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = deserializer.read<double>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x);
  }
  double lb = -1.5;
  double lp = -5.0;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> y
      = deserializer.read_lb<Eigen::MatrixXd, true>(lb, lp, 3, 2);
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(7.0, lb, lp), y(0, 0));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(8.0, lb, lp), y(1, 0));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(9.0, lb, lp), y(2, 0));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(10.0, lb, lp), y(0, 1));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(11.0, lb, lp), y(1, 1));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(12.0, lb, lp), y(2, 1));

  double a = deserializer.read<double>();
  EXPECT_FLOAT_EQ(13.0, a);
}

TEST(deserializer_matrix, matrix_ub_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = deserializer.read<double>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x);
  }
  double ub = 14.1;
  double lp = 0;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> y
      = deserializer.read_ub<Eigen::MatrixXd, false>(ub, lp, 3, 2);
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(7.0, ub), y(0, 0));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(8.0, ub), y(1, 0));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(9.0, ub), y(2, 0));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(10.0, ub), y(0, 1));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(11.0, ub), y(1, 1));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(12.0, ub), y(2, 1));

  double a = deserializer.read<double>();
  EXPECT_FLOAT_EQ(13.0, a);
}

TEST(deserializer_matrix, matrix_ub_constrain_lp) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = deserializer.read<double>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x);
  }
  double ub = 12.1;
  double lp = -5.0;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> y
      = deserializer.read_ub<Eigen::MatrixXd, true>(ub, lp, 3, 2);
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(7.0, ub, lp), y(0, 0));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(8.0, ub, lp), y(1, 0));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(9.0, ub, lp), y(2, 0));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(10.0, ub, lp), y(0, 1));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(11.0, ub, lp), y(1, 1));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(12.0, ub, lp), y(2, 1));

  double a = deserializer.read<double>();
  EXPECT_FLOAT_EQ(13.0, a);
}

TEST(deserializer_matrix, matrix_lub_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = deserializer.read<double>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x);
  }
  double lb = 3.5;
  double ub = 14.1;
  double lp = 0;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> y
      = deserializer.read_lub<Eigen::MatrixXd, false>(lb, ub, lp, 3, 2);
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(7.0, lb, ub), y(0, 0));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(8.0, lb, ub), y(1, 0));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(9.0, lb, ub), y(2, 0));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(10.0, lb, ub), y(0, 1));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(11.0, lb, ub), y(1, 1));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(12.0, lb, ub), y(2, 1));

  double a = deserializer.read<double>();
  EXPECT_FLOAT_EQ(13.0, a);
}

TEST(deserializer_matrix, matrix_lub_constrain_lp) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = deserializer.read<double>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x);
  }
  double lb = 4.1;
  double ub = 12.1;
  double lp = -5.0;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> y
      = deserializer.read_lub<Eigen::MatrixXd, true>(lb, ub, lp, 3, 2);
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(7.0, lb, ub, lp), y(0, 0));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(8.0, lb, ub, lp), y(1, 0));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(9.0, lb, ub, lp), y(2, 0));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(10.0, lb, ub, lp), y(0, 1));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(11.0, lb, ub, lp), y(1, 1));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(12.0, lb, ub, lp), y(2, 1));

  double a = deserializer.read<double>();
  EXPECT_FLOAT_EQ(13.0, a);
}

TEST(deserializer_matrix, corr_matrix_constrain) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  theta.push_back(0.5);
  theta.push_back(1.0);
  theta.push_back(2.0);
  theta.push_back(1.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  Matrix<double, Dynamic, Dynamic> R(
      deserializer.read_corr<Eigen::MatrixXd, false>(lp, 3U));
  EXPECT_EQ(3, R.rows());
  EXPECT_EQ(3, R.cols());
  EXPECT_EQ(9, R.size());
  EXPECT_EQ(4U, deserializer.available());
  for (size_t i = 0; i < 3U; ++i) {
    EXPECT_FLOAT_EQ(1.0, R(i, i));
    for (size_t j = i + 1; j < 3U; ++j)
      EXPECT_FLOAT_EQ(R(i, j), R(j, i));
  }
  Eigen::SelfAdjointEigenSolver<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
      solver(R, Eigen::EigenvaluesOnly);
  assert(solver.eigenvalues()[0] > 1E-10);
}
TEST(deserializer_matrix, corr_matrix_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  theta.push_back(0.5);
  theta.push_back(1.0);
  theta.push_back(2.0);
  theta.push_back(1.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = -9.2;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> R(
      deserializer.read_corr<Eigen::MatrixXd, true>(lp, 3U));
  EXPECT_EQ(3, R.rows());
  EXPECT_EQ(3, R.cols());
  EXPECT_EQ(9, R.size());
  EXPECT_EQ(4U, deserializer.available());
  for (size_t i = 0; i < 3U; ++i) {
    EXPECT_FLOAT_EQ(1.0, R(i, i));
    for (size_t j = i + 1; j < 3U; ++j)
      EXPECT_FLOAT_EQ(R(i, j), R(j, i));
  }
  Eigen::SelfAdjointEigenSolver<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
      solver(R, Eigen::EigenvaluesOnly);
  assert(solver.eigenvalues()[0] > 1E-10);
  // FIXME: test jacobian
}

TEST(deserializer_matrix, cov_matrix_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  theta.push_back(0.5);
  theta.push_back(1.0);
  theta.push_back(2.0);
  theta.push_back(1.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> S(
      deserializer.read_cov_matrix<Eigen::MatrixXd, false>(lp, 3U));
  EXPECT_EQ(3, S.rows());
  EXPECT_EQ(3, S.cols());
  EXPECT_EQ(9, S.size());
  EXPECT_EQ(1U, deserializer.available());
  for (size_t i = 0; i < 3U; ++i)
    for (size_t j = i + 1; j < 3U; ++j)
      EXPECT_FLOAT_EQ(S(i, j), S(j, i));
  Eigen::SelfAdjointEigenSolver<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
      solver(S, Eigen::EigenvaluesOnly);
  assert(solver.eigenvalues()[0]
         > 1E-10);  // check positive definite with smallest eigenvalue > 0
}
TEST(deserializer_matrix, cov_matrix_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  theta.push_back(0.5);
  theta.push_back(1.0);
  theta.push_back(2.0);
  theta.push_back(1.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = -3.1;

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> S(
      deserializer.read_cov_matrix<Eigen::MatrixXd, true>(lp, 3U));

  EXPECT_EQ(3, S.rows());
  EXPECT_EQ(3, S.cols());
  EXPECT_EQ(9, S.size());
  EXPECT_EQ(1U, deserializer.available());
  for (size_t i = 0; i < 3U; ++i)
    for (size_t j = i + 1; j < 3U; ++j)
      EXPECT_FLOAT_EQ(S(i, j), S(j, i));
  Eigen::SelfAdjointEigenSolver<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
      solver(S, Eigen::EigenvaluesOnly);
  assert(solver.eigenvalues()[0]
         > 1E-10);  // check positive definite with smallest eigenvalue > 0
  // FIXME: test Jacobian
}

TEST(deserializer_matrix, cholesky_factor_cov_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> L(
      deserializer.read_cholesky_factor_cov<Eigen::MatrixXd, false>(lp, 3U,
                                                                    3U));
  EXPECT_NO_THROW(stan::math::check_cholesky_factor(
      "test_cholesky_factor_constrain", "L", L));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  EXPECT_EQ(2U, deserializer.available());
}

TEST(deserializer_matrix, cholesky_factor_cov_constrain_asymmetric) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> L(
      deserializer.read_cholesky_factor_cov<Eigen::MatrixXd, false>(lp, 3U,
                                                                    2U));
  EXPECT_NO_THROW(stan::math::check_cholesky_factor(
      "test_cholesky_factor_constrain", "L", L));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(2, L.cols());
  EXPECT_EQ(6, L.size());
  EXPECT_EQ(3U, deserializer.available());
}
TEST(deserializer_matrix, cholesky_factor_cov_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 1.9;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> L(
      deserializer.read_cholesky_factor_cov<Eigen::MatrixXd, true>(lp, 3U, 3U));
  EXPECT_NO_THROW(stan::math::check_cholesky_factor(
      "test_cholesky_factor_constrain", "L", L));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  EXPECT_EQ(2U, deserializer.available());
  EXPECT_EQ(1.9 + log(L(0, 0)) + log(L(1, 1)) + log(L(2, 2)), lp);
}
TEST(deserializer_matrix, cholesky_factor_cov_constrain_jacobian_asymmetric) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 12; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 1.9;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> L(
      deserializer.read_cholesky_factor_cov<Eigen::MatrixXd, true>(lp, 4U, 3U));
  EXPECT_NO_THROW(stan::math::check_cholesky_factor(
      "test_cholesky_factor_constrain", "L", L));
  EXPECT_EQ(4, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(12, L.size());
  EXPECT_EQ(3U, deserializer.available());
  EXPECT_EQ(1.9 + log(L(0, 0)) + log(L(1, 1)) + log(L(2, 2)), lp);
}

TEST(deserializer_matrix, cholesky_factor_corr_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> L(
      deserializer.read_cholesky_factor_corr<Eigen::MatrixXd, false>(lp, 3U));
  EXPECT_NO_THROW(stan::math::check_cholesky_factor_corr(
      "test_cholesky_factor_corr_constrain", "L", L));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  EXPECT_EQ(5U, deserializer.available());
}

TEST(deserializer_matrix, offset_multiplier_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> phi(
      deserializer.read_offset_multiplier<Eigen::MatrixXd, false>(0, 1, lp, 2,
                                                                  2));
  EXPECT_FLOAT_EQ(theta[0], phi(0, 0));
  EXPECT_FLOAT_EQ(theta[1], phi(1, 0));
  EXPECT_FLOAT_EQ(theta[2], phi(0, 1));
  EXPECT_FLOAT_EQ(theta[3], phi(1, 1));
}

TEST(deserializer_matrix, offset_multiplier_constrain_lp) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> phi(
      deserializer.read_offset_multiplier<Eigen::MatrixXd, true>(0, 2, lp, 2,
                                                                 2));
  EXPECT_FLOAT_EQ(theta[0] * 2, phi(0, 0));
  EXPECT_FLOAT_EQ(theta[1] * 2, phi(1, 0));
  EXPECT_FLOAT_EQ(theta[2] * 2, phi(0, 1));
  EXPECT_FLOAT_EQ(theta[3] * 2, phi(1, 1));
  EXPECT_FLOAT_EQ(lp, std::log(2) * 4);
}

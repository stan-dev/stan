#include <stan/io/deserializer.hpp>
#include <gtest/gtest.h>

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

TEST(deserializer_stdvec, std_vector_vector) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  auto y = deserializer.read<std::vector<Eigen::VectorXd>>(5, 4);
  EXPECT_EQ(5, y.size());
  int sentinal = 0;
  for (int i = 0; i < y.size(); ++i) {
    EXPECT_EQ(4, y[i].rows());
    EXPECT_EQ(1, y[i].cols());
    for (int j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(sentinal, y[i].coeff(j));
      sentinal++;
    }
  }
}

TEST(deserializer_stdvec, std_vector_std_vector_matrix) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 120U; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  auto y = deserializer.read<std::vector<std::vector<Eigen::MatrixXd>>>(5, 4, 3,
                                                                        2);
  EXPECT_EQ(5, y.size());
  int sentinal = 0;
  for (int i = 0; i < y.size(); ++i) {
    EXPECT_EQ(4, y[i].size());
    for (int j = 0; j < 4; ++j) {
      EXPECT_EQ(3, y[i][j].rows());
      EXPECT_EQ(2, y[i][j].cols());
      for (int k = 0; k < 6; ++k) {
        EXPECT_FLOAT_EQ(sentinal, y[i][j].coeff(k));
        sentinal++;
      }
    }
  }
}

// scalar bounds
TEST(deserializer_stdvec_scalar, read) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(1.0);
  theta.push_back(2.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  auto x = deserializer.read<std::vector<double>>(2);
  EXPECT_FLOAT_EQ(1.0, x[0]);
  EXPECT_FLOAT_EQ(2.0, x[1]);
  EXPECT_EQ(0U, deserializer.available());
}

TEST(deserializer_stdvec_scalar, read_lb) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-1.0);
  theta.push_back(2.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  auto x = deserializer.read_lb<std::vector<double>>(
      std::vector<double>{-2.0, 1.0}, 2);
  EXPECT_FLOAT_EQ(-1.0, x[0]);
  EXPECT_FLOAT_EQ(2.0, x[1]);
  EXPECT_EQ(0U, deserializer.available());
}
TEST(deserializer_stdvec_scalar, read_lb_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-1.0);
  theta.push_back(2.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  EXPECT_NO_THROW(deserializer.read_lb<std::vector<double>>(-1.0, 2));
  EXPECT_THROW(deserializer.read_lb<std::vector<double>>(3.0, 2),
               std::runtime_error);
}
TEST(deserializer_stdvec_scalar, read_lb_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0.0;
  auto ret = deserializer.read_lb<std::vector<double>, false>(
      std::vector<double>{1.0, 5, -2, 15}, lp, 4);
  EXPECT_FLOAT_EQ(1.0 + exp(-2.0), ret[0]);
  EXPECT_FLOAT_EQ(5.0 + exp(3.0), ret[1]);
  EXPECT_FLOAT_EQ(-2.0 + exp(-1.0), ret[2]);
  EXPECT_FLOAT_EQ(15.0 + exp(0.0), ret[3]);
}
TEST(deserializer_stdvec_scalar, read_lb_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = -1.5;
  auto ret = deserializer.read_lb<std::vector<double>, true>(
      std::vector<double>{1.0, 5.0, -2.0, 15.0}, lp, 4);
  EXPECT_FLOAT_EQ(1.0 + exp(-2.0), ret[0]);
  EXPECT_FLOAT_EQ(5.0 + exp(3.0), ret[1]);
  EXPECT_FLOAT_EQ(-2.0 + exp(-1.0), ret[2]);
  EXPECT_FLOAT_EQ(15.0 + exp(0.0), ret[3]);
  EXPECT_FLOAT_EQ(-1.5 - 2.0 + 3.0 - 1.0, lp);
}

// ub

TEST(deserializer_stdvec_scalar, read_ub) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-1.0);
  theta.push_back(3.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  auto x = deserializer.read_ub<std::vector<double>>(3.0, 2);
  EXPECT_FLOAT_EQ(-1.0, x[0]);
  EXPECT_FLOAT_EQ(3.0, x[1]);
  EXPECT_EQ(0U, deserializer.available());
}
TEST(deserializer_stdvec_scalar, read_ub_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-1.0);
  theta.push_back(2.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  EXPECT_NO_THROW(deserializer.read_ub<std::vector<double>>(3.0, 2));
  EXPECT_THROW(deserializer.read_ub<std::vector<double>>(1.0, 2),
               std::runtime_error);
}
TEST(deserializer_stdvec_scalar, read_ub_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  auto ret = deserializer.read_ub<std::vector<double>, false>(
      std::vector<double>{1.0, 5.0, -2, 15}, lp, 4);
  EXPECT_FLOAT_EQ(1.0 - exp(-2.0), ret[0]);
  EXPECT_FLOAT_EQ(5.0 - exp(3.0), ret[1]);
  EXPECT_FLOAT_EQ(-2.0 - exp(-1.0), ret[2]);
  EXPECT_FLOAT_EQ(15.0 - exp(0.0), ret[3]);
}
TEST(deserializer_stdvec_scalar, read_ub_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = -12.9;
  auto ret = deserializer.read_ub<std::vector<double>, false>(
      std::vector<double>{1.0, 5.0, -2, 15}, lp, 4);
  EXPECT_FLOAT_EQ(1.0 - exp(-2.0), ret[0]);
  EXPECT_FLOAT_EQ(5.0 - exp(3.0), ret[1]);
  EXPECT_FLOAT_EQ(-2.0 - exp(-1.0), ret[2]);
  EXPECT_FLOAT_EQ(15.0 - exp(0.0), ret[3]);
  EXPECT_FLOAT_EQ(-12.9 - 2.0 + 3.0 - 1.0, lp);
}

// lub

TEST(deserializer_stdvec_scalar, read_lub) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-1.0);
  theta.push_back(2.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  auto x = deserializer.read_lub<std::vector<double>>(-3.0, 3.0, 2);
  EXPECT_FLOAT_EQ(-1.0, x[0]);
  EXPECT_FLOAT_EQ(2.0, x[1]);

  EXPECT_EQ(0U, deserializer.available());
}
TEST(deserializer_stdvec_scalar, read_lub_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-1.0);
  theta.push_back(2.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  EXPECT_NO_THROW(deserializer.read_lub<std::vector<double>>(-2.0, 2.0, 2));
  EXPECT_THROW(deserializer.read_lub<std::vector<double>>(-1.0, 1.0, 2),
               std::runtime_error);
}
const double inv_logit_m2 = 0.1192029;  // stan::math::inv_logit(-2.0)
const double inv_logit_m1 = 0.2689414;  // stan::math::inv_logit(-1.0)
const double inv_logit_0 = 0.5;         // stan::math::inv_logit(0)
const double inv_logit_3 = 0.9525741;   // stan::math::inv_logit(3.0)

TEST(deserializer_stdvec_scalar, read_lub_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  std::vector<double> lb{0.0, 3.0, -3.0, -15.0};
  std::vector<double> ub{1.0, 5.0, 2.0, 15.0};
  auto ret = deserializer.read_lub<std::vector<double>, false>(lb, ub, lp, 4);
  EXPECT_FLOAT_EQ(inv_logit_m2, ret[0]);
  EXPECT_FLOAT_EQ(3.0 + 2.0 * inv_logit_3, ret[1]);
  EXPECT_FLOAT_EQ(-3.0 + 5.0 * inv_logit_m1, ret[2]);
  EXPECT_FLOAT_EQ(-15.0 + 30.0 * inv_logit_0, ret[3]);
}
TEST(deserializer_stdvec_scalar, read_lub_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = -7.2;
  std::vector<double> lb{0.0, 3.0, -3.0, -15.0};
  std::vector<double> ub{1.0, 5.0, 2.0, 15.0};
  auto ret = deserializer.read_lub<std::vector<double>, true>(lb, ub, lp, 4);
  EXPECT_FLOAT_EQ(0.0 + 1.0 * inv_logit_m2, ret[0]);
  EXPECT_FLOAT_EQ(3.0 + 2.0 * inv_logit_3, ret[1]);
  EXPECT_FLOAT_EQ(-3.0 + 5.0 * inv_logit_m1, ret[2]);
  EXPECT_FLOAT_EQ(-15.0 + 30.0 * inv_logit_0, ret[3]);
  double expected_lp = -7.2
                       + log((1.0 - 0.0) * inv_logit_m2 * (1 - inv_logit_m2))
                       + log((5.0 - 3.0) * inv_logit_3 * (1 - inv_logit_3))
                       + log((2.0 - -3.0) * inv_logit_m1 * (1 - inv_logit_m1))
                       + log((15.0 - -15.0) * inv_logit_0 * (1 - inv_logit_0));
  EXPECT_FLOAT_EQ(expected_lp, lp);
}

// offset multiplier

TEST(deserializer_stdvec_scalar, read_offset_multiplier) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-1.0);
  theta.push_back(2.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  auto x
      = deserializer.read_offset_multiplier<std::vector<double>>(-3.0, 3.0, 2);
  EXPECT_FLOAT_EQ(-1.0, x[0]);
  EXPECT_FLOAT_EQ(2.0, x[1]);

  EXPECT_EQ(0U, deserializer.available());
}
TEST(deserializer_stdvec_scalar, offset_multiplier_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-1.0);
  theta.push_back(2.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  EXPECT_NO_THROW(
      deserializer.read_offset_multiplier<std::vector<double>>(-2.0, -2.0, 2));
  EXPECT_THROW((deserializer.read_offset_multiplier<std::vector<double>, false>(
                   -2.0, -2.0, lp, 2)),
               std::runtime_error);
}

TEST(deserializer_stdvec_scalar, offset_multiplier_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  std::vector<double> offset{0, 3, -3, -15};
  std::vector<double> mult{1.0, 5.0, 2.0, 15};
  auto ret = deserializer.read_offset_multiplier<std::vector<double>, false>(
      offset, mult, lp, 4);
  EXPECT_FLOAT_EQ(-2.0, ret[0]);
  EXPECT_FLOAT_EQ(3.0 + 5.0 * 3.0, ret[1]);
  EXPECT_FLOAT_EQ(-3.0 + 2.0 * -1.0, ret[2]);
  EXPECT_FLOAT_EQ(-15.0, ret[3]);
}
TEST(deserializer_stdvec_scalar, offset_multiplier_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = -7.2;
  std::vector<double> offset{0, 3, -3, -15};
  std::vector<double> mult{1.0, 5.0, 2.0, 15};
  auto ret = deserializer.read_offset_multiplier<std::vector<double>, true>(
      offset, mult, lp, 4);
  EXPECT_FLOAT_EQ(-2.0, ret[0]);
  EXPECT_FLOAT_EQ(3.0 + 5.0 * 3.0, ret[1]);
  EXPECT_FLOAT_EQ(-3.0 + 2.0 * -1.0, ret[2]);
  EXPECT_FLOAT_EQ(-15.0, ret[3]);
  double expected_lp = -7.2 + log(1.0) + log(5.0) + log(2.0) + log(15.0);
  EXPECT_FLOAT_EQ(expected_lp, lp);
}

// prob

TEST(deserializer_stdvec_scalar, prob) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(0.9);
  theta.push_back(0.1);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  auto p1 = deserializer.read_prob<std::vector<double>>(3);
  EXPECT_FLOAT_EQ(0.9, p1[0]);
  EXPECT_FLOAT_EQ(0.1, p1[1]);
  EXPECT_FLOAT_EQ(0.0, p1[2]);
  EXPECT_EQ(0U, deserializer.available());
}

TEST(deserializer_stdvec_scalar, prob_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  auto ret = deserializer.read_prob<std::vector<double>, false>(lp, 4);
  EXPECT_FLOAT_EQ(inv_logit_m2, ret[0]);
  EXPECT_FLOAT_EQ(inv_logit_3, ret[1]);
  EXPECT_FLOAT_EQ(inv_logit_m1, ret[2]);
  EXPECT_FLOAT_EQ(inv_logit_0, ret[3]);
}

TEST(deserializer_stdvec_scalar, prob_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = -0.88;
  auto ret = deserializer.read_prob<std::vector<double>, true>(lp, 4);
  EXPECT_FLOAT_EQ(inv_logit_m2, ret[0]);
  EXPECT_FLOAT_EQ(inv_logit_3, ret[1]);
  EXPECT_FLOAT_EQ(inv_logit_m1, ret[2]);
  EXPECT_FLOAT_EQ(inv_logit_0, ret[3]);
  double expected_lp = -0.88 + log(inv_logit_m2 * (1.0 - inv_logit_m2))
                       + log(inv_logit_3 * (1.0 - inv_logit_3))
                       + log(inv_logit_m1 * (1.0 - inv_logit_m1))
                       + log(inv_logit_0 * (1.0 - inv_logit_0));
  EXPECT_FLOAT_EQ(expected_lp, lp);
}

// corr

TEST(deserializer_stdvec_scalar, corr) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-0.9);
  theta.push_back(0.1);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  auto rho1 = deserializer.read_corr<std::vector<double>>(3);
  EXPECT_FLOAT_EQ(-0.9, rho1[0]);
  EXPECT_FLOAT_EQ(0.1, rho1[1]);
  EXPECT_FLOAT_EQ(0.0, rho1[2]);
  EXPECT_EQ(0U, deserializer.available());
}
TEST(deserializer_stdvec_scalar, corr_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-0.9);
  theta.push_back(-1.1);
  theta.push_back(1.1);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  EXPECT_NO_THROW((deserializer.read_corr<std::vector<double>, false>(lp, 3)));
  EXPECT_THROW(deserializer.read_corr<std::vector<double>>(3),
               std::runtime_error);
}
TEST(deserializer_stdvec_scalar, corr_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  auto ret = deserializer.read_corr<std::vector<double>, false>(lp, 4);
  EXPECT_FLOAT_EQ(tanh(-2.0), ret[0]);
  EXPECT_FLOAT_EQ(tanh(3.0), ret[1]);
  EXPECT_FLOAT_EQ(tanh(-1.0), ret[2]);
  EXPECT_FLOAT_EQ(tanh(0.0), ret[3]);
}
TEST(deserializer_stdvec_scalar, corr_constrain_jacobian) {
  using std::tanh;
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = -10.0;
  auto ret = deserializer.read_corr<std::vector<double>, true>(lp, 4);
  EXPECT_FLOAT_EQ(tanh(-2.0), ret[0]);
  EXPECT_FLOAT_EQ(tanh(3.0), ret[1]);
  EXPECT_FLOAT_EQ(tanh(-1.0), ret[2]);
  EXPECT_FLOAT_EQ(tanh(0.0), ret[3]);
  double expected_lp = -10.0 + log(1.0 - tanh(-2.0) * tanh(-2.0))
                       + log(1.0 - tanh(3.0) * tanh(3.0))
                       + log(1.0 - tanh(-1.0) * tanh(-1.0))
                       + log(1.0 - tanh(0.0) * tanh(0.0));
  EXPECT_FLOAT_EQ(expected_lp, lp);
}

// vectors

TEST(deserializer_stdvec_vector, read) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  for (size_t i = 0; i < 7U; ++i) {
    double x = deserializer.read<double>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x);
  }
  auto y = deserializer.read<std::vector<Eigen::VectorXd>>(4, 4);
  EXPECT_EQ(4, y.size());
  double sentinal = 7;
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(4, y[i].rows());
    EXPECT_EQ(1, y[i].cols());
    for (int j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(sentinal, y[i][j]);
      ++sentinal;
    }
  }
}

TEST(deserializer_stdvec_vector, unit_vector) {
  std::vector<int> theta_i(0);
  std::vector<double> theta(16, sqrt(0.25));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  auto y = deserializer.read_unit_vector<std::vector<Eigen::VectorXd>>(4, 4);
  EXPECT_EQ(4, y.size());
  const double sqrt_qtr = sqrt(0.25);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(4, y[i].rows());
    EXPECT_EQ(1, y[i].cols());
    for (int j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(sqrt_qtr, y[i][j]);
    }
  }
}

TEST(deserializer_stdvec_vector, unit_vector_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 100; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  theta[0] = 0.00;
  theta[1] = -sqrt(0.29);
  theta[2] = sqrt(0.70);
  theta[3] = -sqrt(0.01);
  theta[4] = sqrt(1.0);
  theta[5] = sqrt(1.0);
  EXPECT_NO_THROW(
      deserializer.read_unit_vector<std::vector<Eigen::VectorXd>>(1, 4));
  EXPECT_THROW(
      deserializer.read_unit_vector<std::vector<Eigen::VectorXd>>(1, 2),
      std::domain_error);
  EXPECT_THROW(
      deserializer.read_unit_vector<std::vector<Eigen::VectorXd>>(1, 0),
      std::invalid_argument);
}

TEST(deserializer_stdvec_vector, simplex) {
  std::vector<int> theta_i(0);
  std::vector<double> theta(8, 0.25);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  auto y = deserializer.read_simplex<std::vector<Eigen::VectorXd>>(2, 4);
  EXPECT_EQ(2, y.size());
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(4, y[i].rows());
    EXPECT_EQ(1, y[i].cols());
    for (int j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(0.25, y[i][j]);
    }
  }
}
TEST(deserializer_stdvec_vector, simplex_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 100; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  theta[0] = 0.00;
  theta[1] = 0.29;
  theta[2] = 0.70;
  theta[3] = 0.01;
  theta[4] = 0.00;
  theta[5] = 0.29;
  theta[6] = 0.70;
  theta[7] = 0.01;
  theta[8] = 1.0;
  theta[9] = 1.0;
  EXPECT_NO_THROW(
      deserializer.read_simplex<std::vector<Eigen::VectorXd>>(2, 4));
  EXPECT_THROW(deserializer.read_simplex<std::vector<Eigen::VectorXd>>(2, 2),
               std::domain_error);
  EXPECT_THROW(deserializer.read_simplex<std::vector<Eigen::VectorXd>>(2, 0),
               std::invalid_argument);
}

TEST(deserializer_stdvec_vector, ordered) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  EXPECT_FLOAT_EQ(0.0, deserializer.read<double>());  // throw away theta[0]
  auto y = deserializer.read_ordered<std::vector<Eigen::VectorXd>>(5, 2);
  EXPECT_EQ(5, y.size());
  double k = 1;
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(2, y[i].rows());
    EXPECT_EQ(1, y[i].cols());
    for (int j = 0; j < 2; ++j) {
      EXPECT_FLOAT_EQ(k, y[i][j]);
      ++k;
    }
  }
}

TEST(deserializer_stdvec_vector, ordered_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
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
  auto phi(
      deserializer.read_ordered<std::vector<Eigen::VectorXd>, false>(lp, 2, 4));
  EXPECT_EQ(2, phi.size());
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(4, phi[i].rows());
    EXPECT_EQ(1, phi[i].cols());
    EXPECT_FLOAT_EQ(v0, phi[i][0]);
    EXPECT_FLOAT_EQ(v1, phi[i][1]);
    EXPECT_FLOAT_EQ(v2, phi[i][2]);
    EXPECT_FLOAT_EQ(v3, phi[i][3]);
  }
}
TEST(deserializer_stdvec_vector, ordered_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
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
  double expected_lp = lp - 1.0 - 2.0 + 0.0 - 1.0 - 2.0 + 0.0;
  auto phi(
      deserializer.read_ordered<std::vector<Eigen::VectorXd>, true>(lp, 2, 4));
  EXPECT_FLOAT_EQ(v0, phi[0][0]);
  EXPECT_FLOAT_EQ(v1, phi[0][1]);
  EXPECT_FLOAT_EQ(v2, phi[0][2]);
  EXPECT_FLOAT_EQ(v3, phi[0][3]);
  EXPECT_FLOAT_EQ(v0, phi[1][0]);
  EXPECT_FLOAT_EQ(v1, phi[1][1]);
  EXPECT_FLOAT_EQ(v2, phi[1][2]);
  EXPECT_FLOAT_EQ(v3, phi[1][3]);
  EXPECT_FLOAT_EQ(expected_lp, lp);
}

TEST(deserializer_stdvec_vector, positive_ordered) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  EXPECT_FLOAT_EQ(0.0, deserializer.read<double>());  // throw away theta[0]
  auto y
      = deserializer.read_positive_ordered<std::vector<Eigen::VectorXd>>(5, 2);
  EXPECT_EQ(5, y.size());
  double sentinal = 1;
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(y[i].rows(), 2);
    EXPECT_EQ(y[i].cols(), 1);
    for (int j = 0; j < 2; ++j) {
      EXPECT_FLOAT_EQ(sentinal, y[i][j]);
      ++sentinal;
    }
  }
}

TEST(deserializer_stdvec_vector, positive_ordered_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
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
  auto phi(
      deserializer.read_positive_ordered<std::vector<Eigen::VectorXd>, false>(
          lp, 2, 4));
  EXPECT_FLOAT_EQ(v0, phi[0][0]);
  EXPECT_FLOAT_EQ(v1, phi[0][1]);
  EXPECT_FLOAT_EQ(v2, phi[0][2]);
  EXPECT_FLOAT_EQ(v3, phi[0][3]);
  EXPECT_FLOAT_EQ(v0, phi[1][0]);
  EXPECT_FLOAT_EQ(v1, phi[1][1]);
  EXPECT_FLOAT_EQ(v2, phi[1][2]);
  EXPECT_FLOAT_EQ(v3, phi[1][3]);
}

TEST(deserializer_stdvec_vector, positive_ordered_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
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
  double expected_lp = lp + 3.0 - 1.0 - 2.0 + 0.0 + 3.0 - 1.0 - 2.0 + 0.0;
  auto phi(
      deserializer.read_positive_ordered<std::vector<Eigen::VectorXd>, true>(
          lp, 2, 4));
  EXPECT_FLOAT_EQ(v0, phi[0][0]);
  EXPECT_FLOAT_EQ(v1, phi[0][1]);
  EXPECT_FLOAT_EQ(v2, phi[0][2]);
  EXPECT_FLOAT_EQ(v3, phi[0][3]);
  EXPECT_FLOAT_EQ(v0, phi[1][0]);
  EXPECT_FLOAT_EQ(v1, phi[1][1]);
  EXPECT_FLOAT_EQ(v2, phi[1][2]);
  EXPECT_FLOAT_EQ(v3, phi[1][3]);
  EXPECT_FLOAT_EQ(expected_lp, lp);
}

TEST(deserializer_stdvec_vector, offset_multiplier) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  auto phi(deserializer.read_offset_multiplier<std::vector<Eigen::VectorXd>>(
      0, 1, 2, 4));
  for (int i = 0; i < 2; ++i) {
    EXPECT_FLOAT_EQ(theta[0], phi[i][0]);
    EXPECT_FLOAT_EQ(theta[1], phi[i][1]);
    EXPECT_FLOAT_EQ(theta[2], phi[i][2]);
    EXPECT_FLOAT_EQ(theta[3], phi[i][3]);
  }
}

TEST(deserializer_stdvec_vector, offset_multiplier_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  auto phi(
      deserializer.read_offset_multiplier<std::vector<Eigen::VectorXd>, false>(
          0, 1, lp, 2, 4));
  for (int i = 0; i < 2; ++i) {
    EXPECT_FLOAT_EQ(theta[0], phi[i][0]);
    EXPECT_FLOAT_EQ(theta[1], phi[i][1]);
    EXPECT_FLOAT_EQ(theta[2], phi[i][2]);
    EXPECT_FLOAT_EQ(theta[3], phi[i][3]);
  }
}

TEST(deserializer_stdvec_vector, offset_multiplier_constrain_lp) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  auto phi(
      deserializer.read_offset_multiplier<std::vector<Eigen::VectorXd>, true>(
          0, 2, lp, 2, 4));
  for (int i = 0; i < 2; ++i) {
    EXPECT_FLOAT_EQ(theta[0] * 2, phi[i][0]);
    EXPECT_FLOAT_EQ(theta[1] * 2, phi[i][1]);
    EXPECT_FLOAT_EQ(theta[2] * 2, phi[i][2]);
    EXPECT_FLOAT_EQ(theta[3] * 2, phi[i][3]);
  }
  EXPECT_FLOAT_EQ(lp, std::log(2) * 8);
}

// matrix

TEST(deserializer_stdvec_matrix, read) {
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
  auto y = deserializer.read<std::vector<eig_mat>>(2, 3, 2);
  EXPECT_EQ(2, y.size());
  double sentinal = 7;
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(3, y[i].rows());
    EXPECT_EQ(2, y[i].cols());
    for (int j = 0; j < y[i].size(); ++j) {
      EXPECT_FLOAT_EQ(sentinal, y[i](j));
      ++sentinal;
    }
  }
}

TEST(deserializer_stdvec_matrix, matrix_lb) {
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
  auto y = deserializer.read_lb<std::vector<Eigen::MatrixXd>>(lb, 2, 3, 2);
  EXPECT_EQ(2, y.size());
  double sentinal = 7;
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(3, y[i].rows());
    EXPECT_EQ(2, y[i].cols());
    for (int j = 0; j < y[i].size(); ++j) {
      EXPECT_FLOAT_EQ(sentinal, y[i](j));
      ++sentinal;
    }
  }
}

TEST(deserializer_stdvec_matrix, matrix_lb_constrain) {
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
  auto y = deserializer.read_lb<std::vector<Eigen::MatrixXd>, false>(lb, lp, 2,
                                                                     3, 2);
  EXPECT_EQ(2, y.size());
  double sentinal = 7;
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(3, y[i].rows());
    EXPECT_EQ(2, y[i].cols());
    for (int j = 0; j < y[i].size(); ++j) {
      EXPECT_FLOAT_EQ(stan::math::lb_constrain(sentinal, lb), y[i](j));
      ++sentinal;
    }
  }
}

TEST(deserializer_stdvec_matrix, matrix_lb_constrain_lp) {
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
  auto y = deserializer.read_lb<std::vector<Eigen::MatrixXd>, true>(lb, lp, 2,
                                                                    3, 2);
  EXPECT_EQ(2, y.size());
  double sentinal = 7;
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(3, y[i].rows());
    EXPECT_EQ(2, y[i].cols());
    for (int j = 0; j < y[i].size(); ++j) {
      EXPECT_FLOAT_EQ(stan::math::lb_constrain(sentinal, lb, lp), y[i](j));
      ++sentinal;
    }
  }
}

TEST(deserializer_stdvec_matrix, matrix_ub) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = deserializer.read<double>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x);
  }
  double ub = 26.5;
  auto y = deserializer.read_ub<std::vector<Eigen::MatrixXd>>(ub, 2, 3, 2);
  EXPECT_EQ(2, y.size());
  double sentinal = 7;
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(3, y[i].rows());
    EXPECT_EQ(2, y[i].cols());
    for (int j = 0; j < y[i].size(); ++j) {
      EXPECT_FLOAT_EQ(sentinal, y[i](j));
      ++sentinal;
    }
  }
}

TEST(deserializer_stdvec_matrix, matrix_ub_constrain) {
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
  auto y = deserializer.read_ub<std::vector<Eigen::MatrixXd>, false>(ub, lp, 2,
                                                                     3, 2);
  EXPECT_EQ(2, y.size());
  double sentinal = 7;
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(3, y[i].rows());
    EXPECT_EQ(2, y[i].cols());
    for (int j = 0; j < y[i].size(); ++j) {
      EXPECT_FLOAT_EQ(stan::math::ub_constrain(sentinal, ub), y[i](j));
      ++sentinal;
    }
  }
}

TEST(deserializer_stdvec_matrix, matrix_ub_constrain_lp) {
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
  auto y = deserializer.read_ub<std::vector<Eigen::MatrixXd>, true>(ub, lp, 2,
                                                                    3, 2);
  EXPECT_EQ(2, y.size());
  double sentinal = 7;
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(3, y[i].rows());
    EXPECT_EQ(2, y[i].cols());
    for (int j = 0; j < y[i].size(); ++j) {
      EXPECT_FLOAT_EQ(stan::math::ub_constrain(sentinal, ub, lp), y[i](j));
      ++sentinal;
    }
  }
}

TEST(deserializer_stdvec_matrix, matrix_lub) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = deserializer.read<double>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x);
  }
  double lb = 6.9;
  double ub = 32.5;
  auto y = deserializer.read_lub<std::vector<Eigen::MatrixXd>>(lb, ub, 2, 3, 2);
  EXPECT_EQ(2, y.size());
  double sentinal = 7;
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(3, y[i].rows());
    EXPECT_EQ(2, y[i].cols());
    for (int j = 0; j < y[i].size(); ++j) {
      EXPECT_FLOAT_EQ(sentinal, y[i](j));
      ++sentinal;
    }
  }
}

TEST(deserializer_stdvec_matrix, matrix_lub_constrain) {
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
  auto y = deserializer.read_lub<std::vector<Eigen::MatrixXd>, false>(
      lb, ub, lp, 2, 3, 2);
  EXPECT_EQ(2, y.size());
  double sentinal = 7;
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(3, y[i].rows());
    EXPECT_EQ(2, y[i].cols());
    for (int j = 0; j < y[i].size(); ++j) {
      EXPECT_FLOAT_EQ(stan::math::lub_constrain(sentinal, lb, ub), y[i](j));
      ++sentinal;
    }
  }
}

TEST(deserializer_stdvec_matrix, matrix_lub_constrain_lp) {
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
  auto y = deserializer.read_lub<std::vector<Eigen::MatrixXd>, true>(lb, ub, lp,
                                                                     2, 3, 2);
  EXPECT_EQ(2, y.size());
  double sentinal = 7;
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(3, y[i].rows());
    EXPECT_EQ(2, y[i].cols());
    for (int j = 0; j < y[i].size(); ++j) {
      EXPECT_FLOAT_EQ(stan::math::lub_constrain(sentinal, lb, ub, lp), y[i](j));
      ++sentinal;
    }
  }
}

TEST(deserializer_stdvec_matrix, corr_matrix) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  theta[0] = 1.0000000000000000;
  theta[1] = 0.1817621852191252;
  theta[2] = 0.8620926037637362;
  theta[3] = 0.1817621852191252;
  theta[4] = 1.0000000000000000;
  theta[5] = 0.2248293054822660;
  theta[6] = 0.8620926037637362;
  theta[7] = 0.2248293054822660;
  theta[8] = 1.0000000000000001;  // allow some tolerance
  theta[9] = 1.0000000000000000;
  theta[10] = 0.1817621852191252;
  theta[11] = 0.8620926037637362;
  theta[12] = 0.1817621852191252;
  theta[13] = 1.0000000000000000;
  theta[14] = 0.2248293054822660;
  theta[15] = 0.8620926037637362;
  theta[16] = 0.2248293054822660;
  theta[17] = 1.0000000000000001;  // allow some tolerance
  stan::io::deserializer<double> deserializer(theta, theta_i);
  auto S = deserializer.read_corr<std::vector<Eigen::MatrixXd>>(2, 3);
  EXPECT_FLOAT_EQ(theta[0], S[0](0, 0));
}

TEST(deserializer_stdvec_matrix, corr_matrix_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  theta[0] = 1.5;
  theta[1] = 1.0;
  theta[2] = 2.0;
  theta[3] = 0.0;
  theta[4] = 1.0;
  stan::io::deserializer<double> deserializer(theta, theta_i);
  EXPECT_THROW(deserializer.read_corr<std::vector<Eigen::MatrixXd>>(1, 1),
               std::domain_error);
  EXPECT_THROW(deserializer.read_corr<std::vector<Eigen::MatrixXd>>(2, 3),
               std::domain_error);
}
TEST(deserializer_stdvec_matrix, corr_matrix_constrain) {
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
  theta.push_back(-2.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  theta.push_back(0.5);
  theta.push_back(1.0);
  theta.push_back(2.0);
  theta.push_back(1.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  auto R(
      deserializer.read_corr<std::vector<Eigen::MatrixXd>, false>(lp, 2, 3U));
  EXPECT_EQ(2, R.size());
  for (int k = 0; k < 2; k++) {
    EXPECT_EQ(3, R[k].rows());
    EXPECT_EQ(3, R[k].cols());
    for (size_t i = 0; i < 3U; ++i) {
      EXPECT_FLOAT_EQ(1.0, R[k](i, i));
      for (size_t j = i + 1; j < 3U; ++j)
        EXPECT_FLOAT_EQ(R[k](i, j), R[k](j, i));
    }
    Eigen::SelfAdjointEigenSolver<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
        solver(R[k], Eigen::EigenvaluesOnly);
    assert(solver.eigenvalues()[0] > 1E-10);
  }
}

TEST(deserializer_stdvec_matrix, corr_matrix_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  theta.push_back(0.5);
  theta.push_back(1.0);
  theta.push_back(2.0);
  theta.push_back(1.0);
  theta.push_back(-2.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  theta.push_back(0.5);
  theta.push_back(1.0);
  theta.push_back(2.0);
  theta.push_back(1.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = -9.2;
  auto R(deserializer.read_corr<std::vector<Eigen::MatrixXd>, true>(lp, 2, 3U));
  EXPECT_EQ(2, R.size());
  for (int k = 0; k < 2; k++) {
    EXPECT_EQ(3, R[k].rows());
    EXPECT_EQ(3, R[k].cols());
    for (size_t i = 0; i < 3U; ++i) {
      EXPECT_FLOAT_EQ(1.0, R[k](i, i));
      for (size_t j = i + 1; j < 3U; ++j)
        EXPECT_FLOAT_EQ(R[k](i, j), R[k](j, i));
    }
    Eigen::SelfAdjointEigenSolver<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
        solver(R[k], Eigen::EigenvaluesOnly);
    assert(solver.eigenvalues()[0] > 1E-10);
  }
}

TEST(deserializer_stdvec_matrix, cov_matrix) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  theta[0] = 6.3295234356180128;
  theta[1] = 0.6351775806192667;
  theta[2] = 3.8081029582054304;
  theta[3] = 0.6351775806192667;
  theta[4] = 1.9293554162496527;
  theta[5] = 0.5483126868366485;
  theta[6] = 3.8081029582054304;
  theta[7] = 0.5483126868366485;
  theta[8] = 3.0827514661973088;
  theta[9] = 6.3295234356180128;
  theta[10] = 0.6351775806192667;
  theta[11] = 3.8081029582054304;
  theta[12] = 0.6351775806192667;
  theta[13] = 1.9293554162496527;
  theta[14] = 0.5483126868366485;
  theta[15] = 3.8081029582054304;
  theta[16] = 0.5483126868366485;
  theta[17] = 3.0827514661973088;
  stan::io::deserializer<double> deserializer(theta, theta_i);
  auto S = deserializer.read_cov_matrix<std::vector<Eigen::MatrixXd>>(2, 3);
  EXPECT_FLOAT_EQ(theta[0], S[0](0, 0));
  EXPECT_FLOAT_EQ(theta[1], S[0](1, 0));
  EXPECT_FLOAT_EQ(theta[7], S[0](2, 1));
  EXPECT_FLOAT_EQ(theta[8], S[0](2, 2));
}
TEST(deserializer_stdvec_matrix, cov_matrix_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  theta[0] = 6.3;
  theta[1] = 0.7;
  theta[2] = 0.6;
  theta[3] = 1.9;
  stan::io::deserializer<double> deserializer(theta, theta_i);
  EXPECT_THROW(deserializer.read_cov_matrix<std::vector<Eigen::MatrixXd>>(2, 2),
               std::domain_error);
  EXPECT_THROW(deserializer.read_cov_matrix<std::vector<Eigen::MatrixXd>>(2, 0),
               std::invalid_argument);
}

TEST(deserializer_stdvec_matrix, cov_matrix_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  theta.push_back(0.5);
  theta.push_back(1.0);
  theta.push_back(2.0);
  theta.push_back(1.0);
  theta.push_back(-2.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  theta.push_back(0.5);
  theta.push_back(1.0);
  theta.push_back(2.0);
  theta.push_back(1.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  auto S(deserializer.read_cov_matrix<std::vector<Eigen::MatrixXd>, false>(
      lp, 2, 3U));
  EXPECT_EQ(2, S.size());
  for (int k = 0; k < 2; ++k) {
    EXPECT_EQ(3, S[k].rows());
    EXPECT_EQ(3, S[k].cols());
    for (size_t i = 0; i < 3U; ++i) {
      for (size_t j = i + 1; j < 3U; ++j) {
        EXPECT_FLOAT_EQ(S[k](i, j), S[k](j, i));
      }
    }
    Eigen::SelfAdjointEigenSolver<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
        solver(S[k], Eigen::EigenvaluesOnly);
    assert(solver.eigenvalues()[0]
           > 1E-10);  // check positive definite with smallest eigenvalue > 0
  }
}

TEST(deserializer_stdvec_matrix, cov_matrix_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  theta.push_back(0.5);
  theta.push_back(1.0);
  theta.push_back(2.0);
  theta.push_back(1.0);
  theta.push_back(-2.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  theta.push_back(0.5);
  theta.push_back(1.0);
  theta.push_back(2.0);
  theta.push_back(1.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = -3.1;

  auto S(deserializer.read_cov_matrix<std::vector<Eigen::MatrixXd>, true>(lp, 2,
                                                                          3U));

  EXPECT_EQ(2, S.size());
  for (int k = 0; k < 2; ++k) {
    EXPECT_EQ(3, S[k].rows());
    EXPECT_EQ(3, S[k].cols());
    for (size_t i = 0; i < 3U; ++i) {
      for (size_t j = i + 1; j < 3U; ++j) {
        EXPECT_FLOAT_EQ(S[k](i, j), S[k](j, i));
      }
    }
    Eigen::SelfAdjointEigenSolver<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
        solver(S[k], Eigen::EigenvaluesOnly);
    assert(solver.eigenvalues()[0]
           > 1E-10);  // check positive definite with smallest eigenvalue > 0
  }
}

TEST(deserializer_stdvec_matrix, cholesky_factor_cov) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  // column major
  theta[0] = 1;
  theta[1] = 2;
  theta[2] = 3;

  theta[3] = 0;
  theta[4] = 4;
  theta[5] = 5;

  theta[6] = 0;
  theta[7] = 0;
  theta[8] = 6;

  theta[9] = 1;
  theta[10] = 2;
  theta[11] = 3;

  theta[12] = 0;
  theta[13] = 4;
  theta[14] = 5;

  theta[15] = 0;
  theta[16] = 0;
  theta[17] = 6;

  stan::io::deserializer<double> deserializer(theta, theta_i);
  auto S = deserializer.read_cholesky_factor_cov<std::vector<Eigen::MatrixXd>>(
      2, 3, 3);
  EXPECT_FLOAT_EQ(theta[0], S[0](0, 0));
  EXPECT_FLOAT_EQ(theta[1], S[0](1, 0));
  EXPECT_FLOAT_EQ(theta[7], S[0](1, 2));
  EXPECT_FLOAT_EQ(theta[8], S[0](2, 2));
}
TEST(deserializer_stdvec_matrix, cholesky_factor_cov_asymmetric) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  // column major
  theta[0] = 1;
  theta[1] = 2;
  theta[2] = 3;

  theta[3] = 0;
  theta[4] = 4;
  theta[5] = 5;

  theta[6] = 1;
  theta[7] = 2;
  theta[8] = 3;

  theta[9] = 0;
  theta[10] = 4;
  theta[11] = 5;

  stan::io::deserializer<double> deserializer(theta, theta_i);
  auto S = deserializer.read_cholesky_factor_cov<std::vector<Eigen::MatrixXd>>(
      2, 3, 2);
  EXPECT_FLOAT_EQ(theta[0], S[0](0, 0));
  EXPECT_FLOAT_EQ(theta[1], S[0](1, 0));
  EXPECT_FLOAT_EQ(theta[2], S[0](2, 0));

  EXPECT_FLOAT_EQ(theta[3], S[0](0, 1));
  EXPECT_FLOAT_EQ(theta[4], S[0](1, 1));
  EXPECT_FLOAT_EQ(theta[5], S[0](2, 1));
}

TEST(deserializer_stdvec_matrix, cholesky_factor_cov_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  theta[0] = -6.3;
  stan::io::deserializer<double> deserializer(theta, theta_i);
  EXPECT_THROW(
      deserializer.read_cholesky_factor_cov<std::vector<Eigen::MatrixXd>>(2, 2,
                                                                          2),
      std::domain_error);
  EXPECT_THROW(
      deserializer.read_cholesky_factor_cov<std::vector<Eigen::MatrixXd>>(2, 0,
                                                                          0),
      std::domain_error);

  theta[0] = 1;
  EXPECT_THROW(
      deserializer.read_cholesky_factor_cov<std::vector<Eigen::MatrixXd>>(2, 2,
                                                                          3),
      std::domain_error);
}
TEST(deserializer_stdvec_matrix, cholesky_factor_cov_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 16; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  auto L(deserializer
             .read_cholesky_factor_cov<std::vector<Eigen::MatrixXd>, false>(
                 lp, 2U, 3U, 3U));
  EXPECT_NO_THROW(stan::math::check_cholesky_factor(
      "test_cholesky_factor_constrain", "L", L[0]));
  EXPECT_EQ(2, L.size());
  EXPECT_EQ(3, L[0].rows());
  EXPECT_EQ(3, L[0].cols());
  EXPECT_EQ(9, L[0].size());
}

TEST(deserializer_stdvec_matrix, cholesky_factor_cov_constrain_asymmetric) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 16; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  auto L(deserializer
             .read_cholesky_factor_cov<std::vector<Eigen::MatrixXd>, false>(
                 lp, 2U, 3U, 2U));
  EXPECT_NO_THROW(stan::math::check_cholesky_factor(
      "test_cholesky_factor_constrain", "L", L[0]));
  EXPECT_EQ(2, L.size());
  EXPECT_EQ(3, L[0].rows());
  EXPECT_EQ(2, L[0].cols());
  EXPECT_EQ(6, L[0].size());
}
TEST(deserializer_stdvec_matrix, cholesky_factor_cov_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 16; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 1.9;
  auto L(
      deserializer.read_cholesky_factor_cov<std::vector<Eigen::MatrixXd>, true>(
          lp, 2U, 3U, 3U));
  EXPECT_NO_THROW(stan::math::check_cholesky_factor(
      "test_cholesky_factor_constrain", "L", L[0]));
  EXPECT_EQ(2, L.size());
  EXPECT_EQ(3, L[0].rows());
  EXPECT_EQ(3, L[0].cols());
  EXPECT_EQ(9, L[0].size());
  EXPECT_EQ(1.9 + log(L[0](0, 0)) + log(L[0](1, 1)) + log(L[0](2, 2))
                + +log(L[1](0, 0)) + log(L[1](1, 1)) + log(L[1](2, 2)),
            lp);
}
TEST(deserializer_stdvec_matrix,
     cholesky_factor_cov_constrain_jacobian_asymmetric) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 24; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 1.9;
  auto L(
      deserializer.read_cholesky_factor_cov<std::vector<Eigen::MatrixXd>, true>(
          lp, 2U, 4U, 3U));
  EXPECT_NO_THROW(stan::math::check_cholesky_factor(
      "test_cholesky_factor_constrain", "L", L[0]));
  EXPECT_EQ(2, L.size());
  EXPECT_EQ(4, L[0].rows());
  EXPECT_EQ(3, L[0].cols());
  EXPECT_EQ(12, L[0].size());
  EXPECT_EQ(1.9 + log(L[0](0, 0)) + log(L[0](1, 1)) + log(L[0](2, 2))
                + +log(L[1](0, 0)) + log(L[1](1, 1)) + log(L[1](2, 2)),
            lp);
}

TEST(deserializer_stdvec_matrix, cholesky_factor_corr) {
  std::vector<int> theta_i;
  std::vector<double> theta(18);
  // column major
  theta[0] = 1;
  theta[1] = 0;
  theta[2] = 0;

  theta[3] = 0;
  theta[4] = 1;
  theta[5] = 0;

  theta[6] = 0;
  theta[7] = 0;
  theta[8] = 1;
  theta[9] = 1;
  theta[10] = 0;
  theta[11] = 0;

  theta[12] = 0;
  theta[13] = 1;
  theta[14] = 0;

  theta[15] = 0;
  theta[16] = 0;
  theta[17] = 1;

  stan::io::deserializer<double> deserializer(theta, theta_i);
  auto S = deserializer.read_cholesky_factor_corr<std::vector<Eigen::MatrixXd>>(
      2, 3);
  EXPECT_FLOAT_EQ(theta[0], S[0](0, 0));
  EXPECT_FLOAT_EQ(theta[1], S[0](1, 0));
  EXPECT_FLOAT_EQ(theta[4], S[0](1, 1));
  EXPECT_FLOAT_EQ(theta[7], S[0](1, 2));
  EXPECT_FLOAT_EQ(theta[8], S[0](2, 2));
}

TEST(deserializer_stdvec_matrix, cholesky_factor_corr_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta(18);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> S;

  // non lower-triangular matrix (column major)
  // the rest of these tests are with check_cholesky_factor_corr
  theta[0] = 1;
  theta[1] = 0;
  theta[2] = 0;

  theta[3] = 0.5;
  theta[4] = 1;
  theta[5] = 0;

  theta[6] = 0;
  theta[7] = 0;
  theta[8] = 1;
  theta[9] = 1;
  theta[10] = 0;
  theta[11] = 0;

  theta[12] = 0.5;
  theta[13] = 1;
  theta[14] = 0;

  theta[15] = 0;
  theta[16] = 0;
  theta[17] = 1;
  stan::io::deserializer<double> deserializer(theta, theta_i);
  EXPECT_THROW(
      deserializer.read_cholesky_factor_corr<std::vector<Eigen::MatrixXd>>(2,
                                                                           3),
      std::domain_error);
}
TEST(deserializer_stdvec_matrix, cholesky_factor_corr_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 16; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  auto L(deserializer
             .read_cholesky_factor_corr<std::vector<Eigen::MatrixXd>, false>(
                 lp, 2U, 3U));
  EXPECT_NO_THROW(stan::math::check_cholesky_factor_corr(
      "test_cholesky_factor_corr_constrain", "L", L[0]));
  EXPECT_EQ(3, L[0].rows());
  EXPECT_EQ(3, L[0].cols());
  EXPECT_EQ(9, L[0].size());
}

TEST(deserializer_stdvec_matrix, offset_multiplier) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  auto phi(deserializer.read_offset_multiplier<std::vector<Eigen::MatrixXd>>(
      0, 1, 2, 2, 2));
  for (int i = 0; i < 2; ++i) {
    EXPECT_FLOAT_EQ(theta[0], phi[i](0, 0));
    EXPECT_FLOAT_EQ(theta[1], phi[i](1, 0));
    EXPECT_FLOAT_EQ(theta[2], phi[i](0, 1));
    EXPECT_FLOAT_EQ(theta[3], phi[i](1, 1));
  }
}

TEST(deserializer_stdvec_matrix, offset_multiplier_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  auto phi(
      deserializer.read_offset_multiplier<std::vector<Eigen::MatrixXd>, false>(
          0, 1, lp, 2, 2, 2));
  for (int i = 0; i < 2; ++i) {
    EXPECT_FLOAT_EQ(theta[0], phi[i](0, 0));
    EXPECT_FLOAT_EQ(theta[1], phi[i](1, 0));
    EXPECT_FLOAT_EQ(theta[2], phi[i](0, 1));
    EXPECT_FLOAT_EQ(theta[3], phi[i](1, 1));
  }
}

TEST(deserializer_stdvec_matrix, offset_multiplier_constrain_lp) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  stan::io::deserializer<double> deserializer(theta, theta_i);
  double lp = 0;
  auto phi(
      deserializer.read_offset_multiplier<std::vector<Eigen::MatrixXd>, true>(
          0, 2, lp, 2, 2, 2));
  for (int i = 0; i < 2; ++i) {
    EXPECT_FLOAT_EQ(theta[0] * 2, phi[i](0, 0));
    EXPECT_FLOAT_EQ(theta[1] * 2, phi[i](1, 0));
    EXPECT_FLOAT_EQ(theta[2] * 2, phi[i](0, 1));
    EXPECT_FLOAT_EQ(theta[3] * 2, phi[i](1, 1));
  }
  EXPECT_FLOAT_EQ(lp, std::log(2) * 8);
}

#include <stan/io/deserializer.hpp>
// expect_near_rel comes from lib/stan_math
#include <test/unit/math/expect_near_rel.hpp>
#include <gtest/gtest.h>

TEST(deserializer_scalar, read) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(1.0);
  theta.push_back(2.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var x = deserializer.read<stan::math::var>();
  EXPECT_FLOAT_EQ(1.0, x.val());
  stan::math::var y = deserializer.read<stan::math::var>();
  EXPECT_FLOAT_EQ(2.0, y.val());
  EXPECT_EQ(0U, deserializer.available());
}

TEST(deserializer_scalar, complex_read) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(1.0);
  theta.push_back(2.0);
  theta.push_back(3.0);
  theta.push_back(4.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  std::complex<stan::math::var> x
      = deserializer.read<std::complex<stan::math::var>>();
  EXPECT_FLOAT_EQ(1.0, x.real().val());
  EXPECT_FLOAT_EQ(2.0, x.imag().val());
  std::complex<stan::math::var> y
      = deserializer.read<std::complex<stan::math::var>>();
  EXPECT_FLOAT_EQ(3.0, y.real().val());
  EXPECT_FLOAT_EQ(4.0, y.imag().val());
  EXPECT_EQ(0U, deserializer.available());
}

TEST(deserializer, read_int) {
  Eigen::Matrix<int, -1, 1> theta_i(1);
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> theta(2);
  theta[0] = 1.0;
  theta[1] = 2.0;
  theta_i[0] = 1;
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var x = deserializer.read<stan::math::var>();
  EXPECT_FLOAT_EQ(1.0, x.val());
  stan::math::var y = deserializer.read<stan::math::var>();
  EXPECT_FLOAT_EQ(2.0, y.val());
  int z = deserializer.read<int>();
  EXPECT_EQ(1, z);
  EXPECT_EQ(0U, deserializer.available());
}

// vector

TEST(deserializer_vector, read) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  for (size_t i = 0; i < 7U; ++i) {
    stan::math::var x = deserializer.read<stan::math::var>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x.val());
  }
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> y
      = deserializer.read<Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>>(4);
  EXPECT_EQ(4, y.rows());
  EXPECT_EQ(1, y.cols());
  EXPECT_EQ(4, y.size());
  EXPECT_FLOAT_EQ(7.0, y[0].val());
  EXPECT_FLOAT_EQ(8.0, y[1].val());
  EXPECT_FLOAT_EQ(9.0, y[2].val());
  EXPECT_FLOAT_EQ(10.0, y[3].val());

  stan::math::var z = deserializer.read<stan::math::var>();
  EXPECT_FLOAT_EQ(11.0, z.val());
}

TEST(deserializer_vector, complex_read) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  for (size_t i = 0; i < 7U; ++i) {
    stan::math::var x = deserializer.read<stan::math::var>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x.val());
  }
  using complex_vec
      = Eigen::Matrix<std::complex<stan::math::var>, Eigen::Dynamic, 1>;
  complex_vec y = deserializer.read<complex_vec>(4);
  EXPECT_EQ(4, y.rows());
  EXPECT_EQ(1, y.cols());
  EXPECT_EQ(4, y.size());
  double sentinal = 7;
  for (int i = 0; i < y.size(); ++i) {
    EXPECT_FLOAT_EQ(sentinal, y[i].real().val());
    ++sentinal;
    EXPECT_FLOAT_EQ(sentinal, y[i].imag().val());
    ++sentinal;
  }

  stan::math::var z = deserializer.read<stan::math::var>();
  EXPECT_FLOAT_EQ(15.0, z.val());
}

// row vector

TEST(deserializer_row_vector, read) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  for (size_t i = 0; i < 7U; ++i) {
    stan::math::var x = deserializer.read<stan::math::var>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x.val());
  }
  Eigen::Matrix<stan::math::var, 1, Eigen::Dynamic> y
      = deserializer.read<Eigen::Matrix<stan::math::var, 1, Eigen::Dynamic>>(4);
  EXPECT_EQ(4, y.cols());
  EXPECT_EQ(1, y.rows());
  EXPECT_EQ(4, y.size());
  EXPECT_FLOAT_EQ(7.0, y[0].val());
  EXPECT_FLOAT_EQ(8.0, y[1].val());
  EXPECT_FLOAT_EQ(9.0, y[2].val());
  EXPECT_FLOAT_EQ(10.0, y[3].val());

  stan::math::var z = deserializer.read<stan::math::var>();
  EXPECT_FLOAT_EQ(11.0, z.val());
}

TEST(deserializer_row_vector, complex_read) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  for (size_t i = 0; i < 7U; ++i) {
    stan::math::var x = deserializer.read<stan::math::var>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x.val());
  }
  using complex_row_vec
      = Eigen::Matrix<std::complex<stan::math::var>, 1, Eigen::Dynamic>;
  complex_row_vec y = deserializer.read<complex_row_vec>(4);
  EXPECT_EQ(1, y.rows());
  EXPECT_EQ(4, y.cols());
  EXPECT_EQ(4, y.size());
  double sentinal = 7;
  for (int i = 0; i < y.size(); ++i) {
    EXPECT_FLOAT_EQ(sentinal, y[i].real().val());
    ++sentinal;
    EXPECT_FLOAT_EQ(sentinal, y[i].imag().val());
    ++sentinal;
  }

  stan::math::var z = deserializer.read<stan::math::var>();
  EXPECT_FLOAT_EQ(15.0, z.val());
}

// matrix

TEST(deserializer_matrix, read) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    stan::math::var x = deserializer.read<stan::math::var>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x.val());
  }
  using eig_mat
      = Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic>;
  eig_mat y = deserializer.read<eig_mat>(3, 2);
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  EXPECT_FLOAT_EQ(7.0, y(0, 0).val());
  EXPECT_FLOAT_EQ(8.0, y(1, 0).val());
  EXPECT_FLOAT_EQ(9.0, y(2, 0).val());
  EXPECT_FLOAT_EQ(10.0, y(0, 1).val());
  EXPECT_FLOAT_EQ(11.0, y(1, 1).val());
  EXPECT_FLOAT_EQ(12.0, y(2, 1).val());

  stan::math::var a = deserializer.read<stan::math::var>();
  EXPECT_FLOAT_EQ(13.0, a.val());
}

TEST(deserializer_matrix, complex_read) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    stan::math::var x = deserializer.read<stan::math::var>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x.val());
  }
  using eig_mat = Eigen::Matrix<std::complex<stan::math::var>, Eigen::Dynamic,
                                Eigen::Dynamic>;
  eig_mat y = deserializer.read<eig_mat>(3, 2);
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  double sentinal = 7;
  for (int i = 0; i < y.size(); ++i) {
    EXPECT_FLOAT_EQ(sentinal, y(i).real().val());
    sentinal++;
    EXPECT_FLOAT_EQ(sentinal, y(i).imag().val());
    sentinal++;
  }
}

// array

TEST(deserializer_array, read) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  for (size_t i = 0; i < 7U; ++i) {
    stan::math::var x = deserializer.read<stan::math::var>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x.val());
  }
  std::vector<stan::math::var> y
      = deserializer.read<std::vector<stan::math::var>>(4);
  EXPECT_EQ(4, y.size());
  EXPECT_FLOAT_EQ(7.0, y[0].val());
  EXPECT_FLOAT_EQ(8.0, y[1].val());
  EXPECT_FLOAT_EQ(9.0, y[2].val());
  EXPECT_FLOAT_EQ(10.0, y[3].val());

  stan::math::var z = deserializer.read<stan::math::var>();
  EXPECT_FLOAT_EQ(11.0, z.val());
}

TEST(deserializer_array, complex_read) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  for (size_t i = 0; i < 7U; ++i) {
    stan::math::var x = deserializer.read<stan::math::var>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x.val());
  }
  using complex_array = std::vector<std::complex<stan::math::var>>;
  complex_array y = deserializer.read<complex_array>(4);
  EXPECT_EQ(4, y.size());
  double sentinal = 7;
  for (int i = 0; i < y.size(); ++i) {
    EXPECT_FLOAT_EQ(sentinal, y[i].real().val());
    ++sentinal;
    EXPECT_FLOAT_EQ(sentinal, y[i].imag().val());
    ++sentinal;
  }

  stan::math::var z = deserializer.read<stan::math::var>();
  EXPECT_FLOAT_EQ(15.0, z.val());
}

TEST(deserializer_array, read_vector) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  std::vector<Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>> y
      = deserializer.read<
          std::vector<Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>>>(4, 2);
  EXPECT_EQ(4, y.size());
  EXPECT_FLOAT_EQ(0.0, y[0](0).val());
  EXPECT_FLOAT_EQ(1.0, y[0](1).val());
  EXPECT_FLOAT_EQ(2.0, y[1](0).val());
  EXPECT_FLOAT_EQ(3.0, y[1](1).val());
  EXPECT_FLOAT_EQ(4.0, y[2](0).val());
  EXPECT_FLOAT_EQ(5.0, y[2](1).val());
  EXPECT_FLOAT_EQ(6.0, y[3](0).val());
  EXPECT_FLOAT_EQ(7.0, y[3](1).val());

  stan::math::var z = deserializer.read<stan::math::var>();
  EXPECT_FLOAT_EQ(8.0, z.val());
}

// lb

TEST(deserializer_scalar, read_constrain_lb_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = 0.0;
  EXPECT_FLOAT_EQ(
      1.0 + exp(-2.0),
      (deserializer.read_constrain_lb<stan::math::var, false>(1.0, lp)).val());
  EXPECT_FLOAT_EQ(
      5.0 + exp(3.0),
      (deserializer.read_constrain_lb<stan::math::var, false>(5.0, lp)).val());
  EXPECT_FLOAT_EQ(
      -2.0 + exp(-1.0),
      (deserializer.read_constrain_lb<stan::math::var, false>(-2.0, lp)).val());
  EXPECT_FLOAT_EQ(
      15.0 + exp(0.0),
      (deserializer.read_constrain_lb<stan::math::var, false>(15.0, lp)).val());
}

TEST(deserializer_scalar, read_constrain_lb_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = -1.5;
  EXPECT_FLOAT_EQ(
      1.0 + exp(-2.0),
      (deserializer.read_constrain_lb<stan::math::var, true>(1.0, lp)).val());
  EXPECT_FLOAT_EQ(
      5.0 + exp(3.0),
      (deserializer.read_constrain_lb<stan::math::var, true>(5.0, lp)).val());
  EXPECT_FLOAT_EQ(
      -2.0 + exp(-1.0),
      (deserializer.read_constrain_lb<stan::math::var, true>(-2.0, lp)).val());
  EXPECT_FLOAT_EQ(
      15.0 + exp(0.0),
      (deserializer.read_constrain_lb<stan::math::var, true>(15.0, lp)).val());
  EXPECT_FLOAT_EQ(-1.5 - 2.0 + 3.0 - 1.0, lp.val());
}

// ub

TEST(deserializer_scalar, read_constrain_ub_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = 0;
  EXPECT_FLOAT_EQ(
      1.0 - exp(-2.0),
      (deserializer.read_constrain_ub<stan::math::var, false>(1.0, lp)).val());
  EXPECT_FLOAT_EQ(
      5.0 - exp(3.0),
      (deserializer.read_constrain_ub<stan::math::var, false>(5.0, lp)).val());
  EXPECT_FLOAT_EQ(
      -2.0 - exp(-1.0),
      (deserializer.read_constrain_ub<stan::math::var, false>(-2.0, lp)).val());
  EXPECT_FLOAT_EQ(
      15.0 - exp(0.0),
      (deserializer.read_constrain_ub<stan::math::var, false>(15.0, lp)).val());
}
TEST(deserializer_scalar, read_constrain_ub_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = -12.9;
  EXPECT_FLOAT_EQ(
      1.0 - exp(-2.0),
      (deserializer.read_constrain_ub<stan::math::var, true>(1.0, lp)).val());
  EXPECT_FLOAT_EQ(
      5.0 - exp(3.0),
      (deserializer.read_constrain_ub<stan::math::var, true>(5.0, lp)).val());
  EXPECT_FLOAT_EQ(
      -2.0 - exp(-1.0),
      (deserializer.read_constrain_ub<stan::math::var, true>(-2.0, lp)).val());
  EXPECT_FLOAT_EQ(
      15.0 - exp(0.0),
      (deserializer.read_constrain_ub<stan::math::var, true>(15.0, lp)).val());
  EXPECT_FLOAT_EQ(-12.9 - 2.0 + 3.0 - 1.0, lp.val());
}

// lub

const double inv_logit_m2 = 0.1192029;  // stan::math::inv_logit(-2.0)
const double inv_logit_m1 = 0.2689414;  // stan::math::inv_logit(-1.0)
const double inv_logit_0 = 0.5;         // stan::math::inv_logit(0)
const double inv_logit_3 = 0.9525741;   // stan::math::inv_logit(3.0)

TEST(deserializer_scalar, read_constrain_lub_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = 0;
  EXPECT_FLOAT_EQ(
      inv_logit_m2,
      (deserializer.read_constrain_lub<stan::math::var, false>(0.0, 1.0, lp))
          .val());
  EXPECT_FLOAT_EQ(
      3.0 + 2.0 * inv_logit_3,
      (deserializer.read_constrain_lub<stan::math::var, false>(3.0, 5.0, lp))
          .val());
  EXPECT_FLOAT_EQ(
      -3.0 + 5.0 * inv_logit_m1,
      (deserializer.read_constrain_lub<stan::math::var, false>(-3.0, 2.0, lp))
          .val());
  EXPECT_FLOAT_EQ(
      -15.0 + 30.0 * inv_logit_0,
      (deserializer.read_constrain_lub<stan::math::var, false>(-15.0, 15.0, lp))
          .val());
}
TEST(deserializer_scalar, read_constrain_lub_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = -7.2;
  EXPECT_FLOAT_EQ(
      0.0 + 1.0 * inv_logit_m2,
      (deserializer.read_constrain_lub<stan::math::var, true>(0.0, 1.0, lp))
          .val());
  EXPECT_FLOAT_EQ(
      3.0 + 2.0 * inv_logit_3,
      (deserializer.read_constrain_lub<stan::math::var, true>(3.0, 5.0, lp))
          .val());
  EXPECT_FLOAT_EQ(
      -3.0 + 5.0 * inv_logit_m1,
      (deserializer.read_constrain_lub<stan::math::var, true>(-3.0, 2.0, lp))
          .val());
  EXPECT_FLOAT_EQ(
      -15.0 + 30.0 * inv_logit_0,
      (deserializer.read_constrain_lub<stan::math::var, true>(-15.0, 15.0, lp))
          .val());
  double expected_lp = -7.2
                       + log((1.0 - 0.0) * inv_logit_m2 * (1 - inv_logit_m2))
                       + log((5.0 - 3.0) * inv_logit_3 * (1 - inv_logit_3))
                       + log((2.0 - -3.0) * inv_logit_m1 * (1 - inv_logit_m1))
                       + log((15.0 - -15.0) * inv_logit_0 * (1 - inv_logit_0));
  EXPECT_FLOAT_EQ(expected_lp, lp.val());
}

// offset multiplier
TEST(deserializer_scalar, offset_multiplier_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = 0;
  EXPECT_FLOAT_EQ(
      -2.0,
      (deserializer.read_constrain_offset_multiplier<stan::math::var, false>(
           0.0, 1.0, lp))
          .val());
  EXPECT_FLOAT_EQ(
      3.0 + 5.0 * 3.0,
      (deserializer.read_constrain_offset_multiplier<stan::math::var, false>(
           3.0, 5.0, lp))
          .val());
  EXPECT_FLOAT_EQ(
      -3.0 + 2.0 * -1.0,
      (deserializer.read_constrain_offset_multiplier<stan::math::var, false>(
           -3.0, 2.0, lp))
          .val());
  EXPECT_FLOAT_EQ(
      -15.0,
      (deserializer.read_constrain_offset_multiplier<stan::math::var, false>(
           -15.0, 15.0, lp))
          .val());
}

TEST(deserializer_scalar, offset_multiplier_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = -7.2;
  EXPECT_FLOAT_EQ(
      -2.0,
      (deserializer.read_constrain_offset_multiplier<stan::math::var, true>(
           0.0, 1.0, lp))
          .val());
  EXPECT_FLOAT_EQ(
      3.0 + 5.0 * 3.0,
      (deserializer.read_constrain_offset_multiplier<stan::math::var, true>(
           3.0, 5.0, lp))
          .val());
  EXPECT_FLOAT_EQ(
      -3.0 + 2.0 * -1.0,
      (deserializer.read_constrain_offset_multiplier<stan::math::var, true>(
           -3.0, 2.0, lp))
          .val());
  EXPECT_FLOAT_EQ(
      -15.0,
      (deserializer.read_constrain_offset_multiplier<stan::math::var, true>(
           -15.0, 15.0, lp))
          .val());
  double expected_lp = -7.2 + log(1.0) + log(5.0) + log(2.0) + log(15.0);
  EXPECT_FLOAT_EQ(expected_lp, lp.val());
}

// unit vector

TEST(deserializer_vector, unit_vector_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = 0;
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> reference
      = stan::math::unit_vector_constrain(stan::math::to_vector(theta));
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> phi(
      deserializer.read_constrain_unit_vector<
          Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>, false>(lp, 4));
  for (size_t i = 0; i < phi.size(); ++i) {
    EXPECT_FLOAT_EQ(reference(i).val(), phi[i].val());
  }
}

TEST(deserializer_vector, unit_vector_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = 0.0;
  stan::math::var lp_ref = 0.0;
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> reference
      = stan::math::unit_vector_constrain(stan::math::to_vector(theta), lp_ref);
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> phi(
      deserializer.read_constrain_unit_vector<
          Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>, true>(lp, 4));
  for (size_t i = 0; i < phi.size(); ++i) {
    EXPECT_FLOAT_EQ(reference(i).val(), phi[i].val());
  }
  EXPECT_FLOAT_EQ(lp_ref.val(), lp.val());
}

// simplex

TEST(deserializer_vector, simplex_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = 0;
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> reference
      = stan::math::simplex_constrain(stan::math::to_vector(theta));
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> phi(
      deserializer.read_constrain_simplex<
          Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>, false>(
          lp, theta.size() + 1));
  for (size_t i = 0; i < phi.size(); ++i) {
    EXPECT_FLOAT_EQ(reference(i).val(), phi[i].val());
  }
}

TEST(deserializer_vector, simplex_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = 0.0;
  stan::math::var lp_ref = 0.0;
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> reference
      = stan::math::simplex_constrain(stan::math::to_vector(theta), lp_ref);
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> phi(
      deserializer.read_constrain_simplex<
          Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>, true>(
          lp, theta.size() + 1));
  for (size_t i = 0; i < phi.size(); ++i) {
    EXPECT_FLOAT_EQ(reference(i).val(), phi[i].val());
  }
  EXPECT_FLOAT_EQ(lp_ref.val(), lp.val());
}

// ordered

TEST(deserializer_vector, ordered_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  stan::math::var v0 = 3.0;
  stan::math::var v1 = v0 + exp(-1.0);
  stan::math::var v2 = v1 + exp(-2.0);
  stan::math::var v3 = v2 + exp(0.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = 0;
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> phi(
      deserializer.read_constrain_ordered<
          Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>, false>(lp, 4));
  EXPECT_FLOAT_EQ(v0.val(), phi[0].val());
  EXPECT_FLOAT_EQ(v1.val(), phi[1].val());
  EXPECT_FLOAT_EQ(v2.val(), phi[2].val());
  EXPECT_FLOAT_EQ(v3.val(), phi[3].val());
}

TEST(deserializer_vector, ordered_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  stan::math::var v0 = 3.0;
  stan::math::var v1 = v0 + exp(-1.0);
  stan::math::var v2 = v1 + exp(-2.0);
  stan::math::var v3 = v2 + exp(0.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = -101.1;
  double expected_lp = lp.val() - 1.0 - 2.0 + 0.0;
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> phi(
      deserializer.read_constrain_ordered<
          Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>, true>(lp, 4));
  EXPECT_FLOAT_EQ(v0.val(), phi[0].val());
  EXPECT_FLOAT_EQ(v1.val(), phi[1].val());
  EXPECT_FLOAT_EQ(v2.val(), phi[2].val());
  EXPECT_FLOAT_EQ(v3.val(), phi[3].val());
  EXPECT_FLOAT_EQ(expected_lp, lp.val());
}

// positive ordered

TEST(deserializer_vector, positive_ordered_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  stan::math::var v0 = exp(3.0);
  stan::math::var v1 = v0 + exp(-1.0);
  stan::math::var v2 = v1 + exp(-2.0);
  stan::math::var v3 = v2 + exp(0.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = 0;
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> phi(
      deserializer.read_constrain_positive_ordered<
          Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>, false>(lp, 4));
  EXPECT_FLOAT_EQ(v0.val(), phi[0].val());
  EXPECT_FLOAT_EQ(v1.val(), phi[1].val());
  EXPECT_FLOAT_EQ(v2.val(), phi[2].val());
  EXPECT_FLOAT_EQ(v3.val(), phi[3].val());
}

TEST(deserializer_vector, positive_ordered_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  stan::math::var v0 = exp(3.0);
  stan::math::var v1 = v0 + exp(-1.0);
  stan::math::var v2 = v1 + exp(-2.0);
  stan::math::var v3 = v2 + exp(0.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = -101.1;
  double expected_lp = lp.val() + 3.0 - 1.0 - 2.0 + 0.0;
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> phi(
      deserializer.read_constrain_positive_ordered<
          Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>, true>(lp, 4));
  EXPECT_FLOAT_EQ(v0.val(), phi[0].val());
  EXPECT_FLOAT_EQ(v1.val(), phi[1].val());
  EXPECT_FLOAT_EQ(v2.val(), phi[2].val());
  EXPECT_FLOAT_EQ(v3.val(), phi[3].val());
  EXPECT_FLOAT_EQ(expected_lp, lp.val());
}

// chol cov

TEST(deserializer_matrix, cholesky_factor_cov_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = 0;
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> reference
      = stan::math::cholesky_factor_constrain(
          stan::math::to_vector(theta).segment(0, 6), 3, 3);
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> L(
      deserializer.read_constrain_cholesky_factor_cov<
          Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic>,
          false>(lp, 3U, 3U));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  stan::test::expect_near_rel("deserializer tests", reference.val(), L.val());
  EXPECT_EQ(2U, deserializer.available());
}

TEST(deserializer_matrix, cholesky_factor_cov_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp_ref = 0.0;
  stan::math::var lp = 0.0;
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> reference
      = stan::math::cholesky_factor_constrain(
          stan::math::to_vector(theta).segment(0, 6), 3, 3, lp_ref);
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> L(
      deserializer.read_constrain_cholesky_factor_cov<
          Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic>, true>(
          lp, 3U, 3U));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  stan::test::expect_near_rel("deserializer tests", reference.val(), L.val());
  EXPECT_EQ(2U, deserializer.available());
  EXPECT_FLOAT_EQ(lp_ref.val(), lp.val());
}

TEST(deserializer_matrix, cholesky_factor_cov_constrain_non_square) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = 0;
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> reference
      = stan::math::cholesky_factor_constrain(
          stan::math::to_vector(theta).segment(0, 5), 3, 2);
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> L(
      deserializer.read_constrain_cholesky_factor_cov<
          Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic>,
          false>(lp, 3U, 2U));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(2, L.cols());
  EXPECT_EQ(6, L.size());
  stan::test::expect_near_rel("deserializer tests", reference.val(), L.val());
  EXPECT_EQ(3U, deserializer.available());
}

TEST(deserializer_matrix, cholesky_factor_cov_jacobian_non_square) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp_ref = 0.0;
  stan::math::var lp = 0.0;
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> reference
      = stan::math::cholesky_factor_constrain(
          stan::math::to_vector(theta).segment(0, 5), 3, 2, lp_ref);
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> L(
      deserializer.read_constrain_cholesky_factor_cov<
          Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic>, true>(
          lp, 3U, 2U));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(2, L.cols());
  EXPECT_EQ(6, L.size());
  stan::test::expect_near_rel("deserializer tests", reference.val(), L.val());
  EXPECT_EQ(3U, deserializer.available());
  EXPECT_FLOAT_EQ(lp_ref.val(), lp.val());
}

// chol corr

TEST(deserializer_matrix, cholesky_factor_corr_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = 0;
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> reference
      = stan::math::cholesky_corr_constrain(
          stan::math::to_vector(theta).segment(0, 3), 3);
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> L(
      deserializer.read_constrain_cholesky_factor_corr<
          Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic>,
          false>(lp, 3U));
  EXPECT_NO_THROW(stan::math::check_cholesky_factor_corr(
      "test_cholesky_factor_corr_constrain", "L", L));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  stan::test::expect_near_rel("deserializer tests", reference.val(), L.val());
  EXPECT_EQ(5U, deserializer.available());
}

TEST(deserializer_matrix, cholesky_factor_corr_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp_ref = 0.0;
  stan::math::var lp = 0.0;
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> reference
      = stan::math::cholesky_corr_constrain(
          stan::math::to_vector(theta).segment(0, 3), 3, lp_ref);
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> L(
      deserializer.read_constrain_cholesky_factor_corr<
          Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic>, true>(
          lp, 3U));
  EXPECT_NO_THROW(stan::math::check_cholesky_factor_corr(
      "test_cholesky_factor_corr_constrain", "L", L));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  stan::test::expect_near_rel("deserializer tests", reference.val(), L.val());
  EXPECT_EQ(5U, deserializer.available());
  EXPECT_FLOAT_EQ(lp_ref.val(), lp.val());
}

// cov

TEST(deserializer_matrix, cov_matrix_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = 0;
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> reference
      = stan::math::cov_matrix_constrain(
          stan::math::to_vector(theta).segment(0, 6), 3);
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> L(
      deserializer.read_constrain_cov_matrix<
          Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic>,
          false>(lp, 3U));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  stan::test::expect_near_rel("deserializer tests", reference.val(), L.val());
  EXPECT_EQ(2U, deserializer.available());
}

TEST(deserializer_matrix, cov_matrix_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp_ref = 0.0;
  stan::math::var lp = 0.0;
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> reference
      = stan::math::cov_matrix_constrain(
          stan::math::to_vector(theta).segment(0, 6), 3, lp_ref);
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> L(
      deserializer.read_constrain_cov_matrix<
          Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic>, true>(
          lp, 3U));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  stan::test::expect_near_rel("deserializer tests", reference.val(), L.val());
  EXPECT_EQ(2U, deserializer.available());
  EXPECT_FLOAT_EQ(lp_ref.val(), lp.val());
}

// corr

TEST(deserializer_matrix, corr_matrix_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = 0;
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> reference
      = stan::math::corr_matrix_constrain(
          stan::math::to_vector(theta).segment(0, 3), 3);
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> L(
      deserializer.read_constrain_corr_matrix<
          Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic>,
          false>(lp, 3U));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  stan::test::expect_near_rel("deserializer tests", reference.val(), L.val());
  EXPECT_EQ(5U, deserializer.available());
}

TEST(deserializer_matrix, corr_matrix_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp_ref = 0.0;
  stan::math::var lp = 0.0;
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> reference
      = stan::math::corr_matrix_constrain(
          stan::math::to_vector(theta).segment(0, 3), 3, lp_ref);
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> L(
      deserializer.read_constrain_corr_matrix<
          Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic>, true>(
          lp, 3U));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  stan::test::expect_near_rel("deserializer tests", reference.val(), L.val());
  EXPECT_EQ(5U, deserializer.available());
  EXPECT_FLOAT_EQ(lp_ref.val(), lp.val());
}

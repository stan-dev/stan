#include <stan/io/deserializer.hpp>
#include <gtest/gtest.h>

using var_matrix_t = stan::math::var_value<Eigen::MatrixXd>;
using var_vector_t = stan::math::var_value<Eigen::VectorXd>;

// vectors
TEST(deserializer_var_vector, read) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  for (size_t i = 0; i < 7U; ++i) {
    stan::math::var x = deserializer.read<stan::math::var>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x.val());
  }
  var_vector_t y = deserializer.read<var_vector_t>(4);
  EXPECT_EQ(4, y.rows());
  EXPECT_EQ(1, y.cols());
  EXPECT_EQ(4, y.size());
  EXPECT_FLOAT_EQ(7.0, y.val()[0]);
  EXPECT_FLOAT_EQ(8.0, y.val()[1]);
  EXPECT_FLOAT_EQ(9.0, y.val()[2]);
  EXPECT_FLOAT_EQ(10.0, y.val()[3]);

  stan::math::var z = deserializer.read<stan::math::var>();
  EXPECT_FLOAT_EQ(11.0, z.val());
}

// need to add unit vector and simplex constrain tests

TEST(deserializer_var_vector, ordered_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  double v0 = 3.0;
  double v1 = v0 + exp(-1.0);
  double v2 = v1 + exp(-2.0);
  double v3 = v2 + exp(0.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = 0;
  var_vector_t phi(deserializer.read_ordered<var_vector_t, false>(lp, 4));
  EXPECT_FLOAT_EQ(v0, phi.val()[0]);
  EXPECT_FLOAT_EQ(v1, phi.val()[1]);
  EXPECT_FLOAT_EQ(v2, phi.val()[2]);
  EXPECT_FLOAT_EQ(v3, phi.val()[3]);
}
TEST(deserializer_var_vector, ordered_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  double v0 = 3.0;
  double v1 = v0 + exp(-1.0);
  double v2 = v1 + exp(-2.0);
  double v3 = v2 + exp(0.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = -101.1;
  double expected_lp = lp.val() - 1.0 - 2.0 + 0.0;
  var_vector_t phi(deserializer.read_ordered<var_vector_t, true>(lp, 4));
  EXPECT_FLOAT_EQ(v0, phi.val()[0]);
  EXPECT_FLOAT_EQ(v1, phi.val()[1]);
  EXPECT_FLOAT_EQ(v2, phi.val()[2]);
  EXPECT_FLOAT_EQ(v3, phi.val()[3]);
  EXPECT_FLOAT_EQ(expected_lp, lp.val());
}

TEST(deserializer_var_vector, positive_ordered_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  double v0 = exp(3.0);
  double v1 = v0 + exp(-1.0);
  double v2 = v1 + exp(-2.0);
  double v3 = v2 + exp(0.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = 0;
  var_vector_t phi(
      deserializer.read_positive_ordered<var_vector_t, false>(lp, 4));
  EXPECT_FLOAT_EQ(v0, phi.val()[0]);
  EXPECT_FLOAT_EQ(v1, phi.val()[1]);
  EXPECT_FLOAT_EQ(v2, phi.val()[2]);
  EXPECT_FLOAT_EQ(v3, phi.val()[3]);
}

TEST(deserializer_var_vector, positive_ordered_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  double v0 = exp(3.0);
  double v1 = v0 + exp(-1.0);
  double v2 = v1 + exp(-2.0);
  double v3 = v2 + exp(0.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = -101.1;
  double expected_lp = lp.val() + 3.0 - 1.0 - 2.0 + 0.0;
  var_vector_t phi(
      deserializer.read_positive_ordered<var_vector_t, true>(lp, 4));
  EXPECT_FLOAT_EQ(v0, phi.val()[0]);
  EXPECT_FLOAT_EQ(v1, phi.val()[1]);
  EXPECT_FLOAT_EQ(v2, phi.val()[2]);
  EXPECT_FLOAT_EQ(v3, phi.val()[3]);
  EXPECT_FLOAT_EQ(expected_lp, lp.val());
}

TEST(deserializer_vector, offset_multiplier_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = 0;
  var_vector_t phi(
      deserializer.read_offset_multiplier<var_vector_t, false>(0, 1, lp, 4));
  EXPECT_FLOAT_EQ(theta[0].val(), phi.val()[0]);
  EXPECT_FLOAT_EQ(theta[1].val(), phi.val()[1]);
  EXPECT_FLOAT_EQ(theta[2].val(), phi.val()[2]);
  EXPECT_FLOAT_EQ(theta[3].val(), phi.val()[3]);
}

TEST(deserializer_vector, offset_multiplier_constrain_lp) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = 0;
  var_vector_t phi(
      deserializer.read_offset_multiplier<var_vector_t, true>(0, 2, lp, 4));
  EXPECT_FLOAT_EQ(theta[0].val() * 2, phi.val()[0]);
  EXPECT_FLOAT_EQ(theta[1].val() * 2, phi.val()[1]);
  EXPECT_FLOAT_EQ(theta[2].val() * 2, phi.val()[2]);
  EXPECT_FLOAT_EQ(theta[3].val() * 2, phi.val()[3]);
  EXPECT_FLOAT_EQ(lp.val(), std::log(2) * 4);
}

// matrix

TEST(deserializer_var_matrix, read) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    stan::math::var x = deserializer.read<stan::math::var>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x.val());
  }
  var_matrix_t y = deserializer.read<var_matrix_t>(3, 2);
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  EXPECT_FLOAT_EQ(7.0, y.val()(0, 0));
  EXPECT_FLOAT_EQ(8.0, y.val()(1, 0));
  EXPECT_FLOAT_EQ(9.0, y.val()(2, 0));
  EXPECT_FLOAT_EQ(10.0, y.val()(0, 1));
  EXPECT_FLOAT_EQ(11.0, y.val()(1, 1));
  EXPECT_FLOAT_EQ(12.0, y.val()(2, 1));

  stan::math::var a = deserializer.read<stan::math::var>();
  EXPECT_FLOAT_EQ(13.0, a.val());
}

TEST(deserializer_var_matrix, matrix_lb_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    stan::math::var x = deserializer.read<stan::math::var>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x.val());
  }
  stan::math::var lb = -1.5;
  stan::math::var lp = 0;
  var_matrix_t y = deserializer.read_lb<var_matrix_t, false>(lb, lp, 3, 2);
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(7.0, lb).val(), y.val()(0, 0));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(8.0, lb).val(), y.val()(1, 0));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(9.0, lb).val(), y.val()(2, 0));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(10.0, lb).val(), y.val()(0, 1));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(11.0, lb).val(), y.val()(1, 1));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(12.0, lb).val(), y.val()(2, 1));

  stan::math::var a = deserializer.read<stan::math::var>();
  EXPECT_FLOAT_EQ(13.0, a.val());
}

TEST(deserializer_var_matrix, matrix_lb_constrain_lp) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    stan::math::var x = deserializer.read<stan::math::var>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x.val());
  }
  stan::math::var lb = -1.5;
  stan::math::var lp = -5.0;
  var_matrix_t y = deserializer.read_lb<var_matrix_t, true>(lb, lp, 3, 2);
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(7.0, lb, lp).val(), y.val()(0, 0));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(8.0, lb, lp).val(), y.val()(1, 0));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(9.0, lb, lp).val(), y.val()(2, 0));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(10.0, lb, lp).val(), y.val()(0, 1));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(11.0, lb, lp).val(), y.val()(1, 1));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(12.0, lb, lp).val(), y.val()(2, 1));

  stan::math::var a = deserializer.read<stan::math::var>();
  EXPECT_FLOAT_EQ(13.0, a.val());
}

TEST(deserializer_var_matrix, matrix_ub_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    stan::math::var x = deserializer.read<stan::math::var>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x.val());
  }
  stan::math::var ub = 14.1;
  stan::math::var lp = 0;
  var_matrix_t y = deserializer.read_ub<var_matrix_t, false>(ub, lp, 3, 2);
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(7.0, ub).val(), y.val()(0, 0));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(8.0, ub).val(), y.val()(1, 0));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(9.0, ub).val(), y.val()(2, 0));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(10.0, ub).val(), y.val()(0, 1));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(11.0, ub).val(), y.val()(1, 1));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(12.0, ub).val(), y.val()(2, 1));

  stan::math::var a = deserializer.read<stan::math::var>();
  EXPECT_FLOAT_EQ(13.0, a.val());
}

TEST(deserializer_var_matrix, matrix_ub_constrain_lp) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    stan::math::var x = deserializer.read<stan::math::var>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x.val());
  }
  stan::math::var ub = 12.1;
  stan::math::var lp = -5.0;
  var_matrix_t y = deserializer.read_ub<var_matrix_t, true>(ub, lp, 3, 2);
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(7.0, ub, lp).val(), y.val()(0, 0));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(8.0, ub, lp).val(), y.val()(1, 0));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(9.0, ub, lp).val(), y.val()(2, 0));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(10.0, ub, lp).val(), y.val()(0, 1));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(11.0, ub, lp).val(), y.val()(1, 1));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(12.0, ub, lp).val(), y.val()(2, 1));

  stan::math::var a = deserializer.read<stan::math::var>();
  EXPECT_FLOAT_EQ(13.0, a.val());
}

TEST(deserializer_var_matrix, matrix_lub_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    stan::math::var x = deserializer.read<stan::math::var>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x.val());
  }
  stan::math::var lb = 3.5;
  stan::math::var ub = 14.1;
  stan::math::var lp = 0;
  var_matrix_t y = deserializer.read_lub<var_matrix_t, false>(lb, ub, lp, 3, 2);
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(7.0, lb, ub).val(), y.val()(0, 0));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(8.0, lb, ub).val(), y.val()(1, 0));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(9.0, lb, ub).val(), y.val()(2, 0));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(10.0, lb, ub).val(), y.val()(0, 1));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(11.0, lb, ub).val(), y.val()(1, 1));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(12.0, lb, ub).val(), y.val()(2, 1));

  stan::math::var a = deserializer.read<stan::math::var>();
  EXPECT_FLOAT_EQ(13.0, a.val());
}

TEST(deserializer_var_matrix, matrix_lub_constrain_lp) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    stan::math::var x = deserializer.read<stan::math::var>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x.val());
  }
  stan::math::var lb = 4.1;
  stan::math::var ub = 12.1;
  stan::math::var lp = -5.0;
  var_matrix_t y = deserializer.read_lub<var_matrix_t, true>(lb, ub, lp, 3, 2);
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(7.0, lb, ub, lp).val(),
                  y.val()(0, 0));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(8.0, lb, ub, lp).val(),
                  y.val()(1, 0));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(9.0, lb, ub, lp).val(),
                  y.val()(2, 0));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(10.0, lb, ub, lp).val(),
                  y.val()(0, 1));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(11.0, lb, ub, lp).val(),
                  y.val()(1, 1));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(12.0, lb, ub, lp).val(),
                  y.val()(2, 1));

  stan::math::var a = deserializer.read<stan::math::var>();
  EXPECT_FLOAT_EQ(13.0, a.val());
}

TEST(deserializer_var_matrix, corr_matrix_constrain) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  theta.push_back(0.5);
  theta.push_back(1.0);
  theta.push_back(2.0);
  theta.push_back(1.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = 0;
  var_matrix_t R(deserializer.read_corr<var_matrix_t, false>(lp, 3U));
  EXPECT_EQ(3, R.rows());
  EXPECT_EQ(3, R.cols());
  EXPECT_EQ(9, R.size());
  EXPECT_EQ(4U, deserializer.available());
  for (size_t i = 0; i < 3U; ++i) {
    EXPECT_FLOAT_EQ(1.0, R.val()(i, i));
    for (size_t j = i + 1; j < 3U; ++j)
      EXPECT_FLOAT_EQ(R.val()(i, j), R.val()(j, i));
  }
  Eigen::SelfAdjointEigenSolver<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
      solver(R.val(), Eigen::EigenvaluesOnly);
  assert(solver.eigenvalues()[0] > 1E-10);
}

TEST(deserializer_var_matrix, corr_matrix_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  theta.push_back(0.5);
  theta.push_back(1.0);
  theta.push_back(2.0);
  theta.push_back(1.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = -9.2;
  var_matrix_t R(deserializer.read_corr<var_matrix_t, true>(lp, 3U));
  EXPECT_EQ(3, R.rows());
  EXPECT_EQ(3, R.cols());
  EXPECT_EQ(9, R.size());
  EXPECT_EQ(4U, deserializer.available());
  for (size_t i = 0; i < 3U; ++i) {
    EXPECT_FLOAT_EQ(1.0, R.val()(i, i));
    for (size_t j = i + 1; j < 3U; ++j)
      EXPECT_FLOAT_EQ(R.val()(i, j), R.val()(j, i));
  }
  Eigen::SelfAdjointEigenSolver<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
      solver(R.val(), Eigen::EigenvaluesOnly);
  assert(solver.eigenvalues()[0] > 1E-10);
  // FIXME: test jacobian
}

TEST(deserializer_var_matrix, cov_matrix_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  theta.push_back(0.5);
  theta.push_back(1.0);
  theta.push_back(2.0);
  theta.push_back(1.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = 0;
  var_matrix_t S(deserializer.read_cov_matrix<var_matrix_t, false>(lp, 3U));
  EXPECT_EQ(3, S.rows());
  EXPECT_EQ(3, S.cols());
  EXPECT_EQ(9, S.size());
  EXPECT_EQ(1U, deserializer.available());
  for (size_t i = 0; i < 3U; ++i) {
    for (size_t j = i + 1; j < 3U; ++j) {
      EXPECT_FLOAT_EQ(S.val()(i, j), S.val()(j, i));
    }
  }
  using eig_m = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  Eigen::SelfAdjointEigenSolver<eig_m> solver(S.val(), Eigen::EigenvaluesOnly);
  assert(solver.eigenvalues()[0]
         > 1E-10);  // check positive definite with smallest eigenvalue > 0
}

TEST(deserializer_var_matrix, cov_matrix_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  theta.push_back(0.5);
  theta.push_back(1.0);
  theta.push_back(2.0);
  theta.push_back(1.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = -3.1;

  var_matrix_t S(deserializer.read_cov_matrix<var_matrix_t, true>(lp, 3U));

  EXPECT_EQ(3, S.rows());
  EXPECT_EQ(3, S.cols());
  EXPECT_EQ(9, S.size());
  EXPECT_EQ(1U, deserializer.available());
  for (size_t i = 0; i < 3U; ++i) {
    for (size_t j = i + 1; j < 3U; ++j) {
      EXPECT_FLOAT_EQ(S.val()(i, j), S.val()(j, i));
    }
  }
  using eig_m = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  Eigen::SelfAdjointEigenSolver<eig_m> solver(S.val(), Eigen::EigenvaluesOnly);
  assert(solver.eigenvalues()[0]
         > 1E-10);  // check positive definite with smallest eigenvalue > 0
  // FIXME: test Jacobian
}

TEST(deserializer_var_matrix, cholesky_factor_cov_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = 1.9;
  var_matrix_t L(
      deserializer.read_cholesky_factor_cov<var_matrix_t, true>(lp, 3U, 3U));
  EXPECT_NO_THROW(stan::math::check_cholesky_factor(
      "test_cholesky_factor_constrain", "L", L.val()));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  EXPECT_EQ(2U, deserializer.available());
  EXPECT_EQ(1.9 + log(L.val()(0, 0)) + log(L.val()(1, 1)) + log(L.val()(2, 2)),
            lp.val());
}
TEST(deserializer_var_matrix,
     cholesky_factor_cov_constrain_jacobian_asymmetric) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (int i = 0; i < 12; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = 1.9;
  var_matrix_t L(
      deserializer.read_cholesky_factor_cov<var_matrix_t, true>(lp, 4U, 3U));
  EXPECT_NO_THROW(stan::math::check_cholesky_factor(
      "test_cholesky_factor_constrain", "L", L.val()));
  EXPECT_EQ(4, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(12, L.size());
  EXPECT_EQ(3U, deserializer.available());
  EXPECT_EQ(1.9 + log(L.val()(0, 0)) + log(L.val()(1, 1)) + log(L.val()(2, 2)),
            lp.val());
}

TEST(deserializer_var_matrix, cholesky_factor_corr_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = 0;
  var_matrix_t L(
      deserializer.read_cholesky_factor_corr<var_matrix_t, false>(lp, 3U));
  EXPECT_NO_THROW(stan::math::check_cholesky_factor_corr(
      "test_cholesky_factor_corr_constrain", "L", L.val()));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  EXPECT_EQ(5U, deserializer.available());
}
// need lp version of ^

TEST(deserializer_var_matrix, offset_multiplier_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = 0;
  var_matrix_t phi(
      deserializer.read_offset_multiplier<var_matrix_t, false>(0, 1, lp, 2, 2));
  EXPECT_FLOAT_EQ(theta[0].val(), phi.val()(0, 0));
  EXPECT_FLOAT_EQ(theta[1].val(), phi.val()(1, 0));
  EXPECT_FLOAT_EQ(theta[2].val(), phi.val()(0, 1));
  EXPECT_FLOAT_EQ(theta[3].val(), phi.val()(1, 1));
}

TEST(deserializer_var_matrix, offset_multiplier_constrain_lp) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  stan::io::deserializer<stan::math::var> deserializer(theta, theta_i);
  stan::math::var lp = 0;
  var_matrix_t phi(
      deserializer.read_offset_multiplier<var_matrix_t, true>(0, 2, lp, 2, 2));
  EXPECT_FLOAT_EQ(theta[0].val() * 2, phi.val()(0, 0));
  EXPECT_FLOAT_EQ(theta[1].val() * 2, phi.val()(1, 0));
  EXPECT_FLOAT_EQ(theta[2].val() * 2, phi.val()(0, 1));
  EXPECT_FLOAT_EQ(theta[3].val() * 2, phi.val()(1, 1));
  EXPECT_FLOAT_EQ(lp.val(), std::log(2) * 4);
}

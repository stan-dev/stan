#include <stan/io/reader.hpp>
#include <gtest/gtest.h>

struct VarReader : public testing::Test {
  void SetUp() {
    // make sure memory's clean before starting each test
    stan::math::recover_memory();
  }
  void TearDown() {
    // make sure memory's clean before starting each test
    stan::math::recover_memory();
  }
};



TEST_F(VarReader, unit_vector) {
  using stan::math::var_value;
  using stan::math::var;
  std::vector<int> theta_i(0);
  std::vector<var> theta(4, sqrt(0.25));
  stan::io::reader<var> reader(theta, theta_i);
  var_value<Eigen::Matrix<double, Eigen::Dynamic, 1>> y = reader.var_unit_vector(4);
  EXPECT_EQ(4, y.size());
  EXPECT_FLOAT_EQ(sqrt(0.25), y.val()[0]);
  EXPECT_FLOAT_EQ(sqrt(0.25), y.val()[1]);
  EXPECT_FLOAT_EQ(sqrt(0.25), y.val()[2]);
  EXPECT_FLOAT_EQ(sqrt(0.25), y.val()[3]);
}

TEST_F(VarReader, unit_vector_exception) {
  using stan::math::var_value;
  using stan::math::var;
  std::vector<int> theta_i;
  std::vector<var> theta;
  for (int i = 0; i < 100; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<var> reader(theta, theta_i);
  theta[0] = 0.00;
  theta[1] = -sqrt(0.29);
  theta[2] = sqrt(0.70);
  theta[3] = -sqrt(0.01);
  theta[4] = sqrt(1.0);
  theta[5] = sqrt(1.0);
  EXPECT_NO_THROW(reader.var_unit_vector(4));
  EXPECT_THROW(reader.var_unit_vector(2), std::domain_error);
  EXPECT_THROW(reader.var_unit_vector(0), std::invalid_argument);
}

TEST_F(VarReader, simplex) {
  using stan::math::var_value;
  using stan::math::var;
  std::vector<int> theta_i(0);
  std::vector<var> theta(4, 0.25);
  stan::io::reader<var> reader(theta, theta_i);
  var_value<Eigen::Matrix<double, Eigen::Dynamic, 1>> y = reader.var_simplex(4);
  EXPECT_EQ(4, y.size());
  EXPECT_FLOAT_EQ(0.25, y.val()[0]);
  EXPECT_FLOAT_EQ(0.25, y.val()[1]);
  EXPECT_FLOAT_EQ(0.25, y.val()[2]);
  EXPECT_FLOAT_EQ(0.25, y.val()[3]);
}
TEST_F(VarReader, simplex_exception) {
  using stan::math::var_value;
  using stan::math::var;
  std::vector<int> theta_i;
  std::vector<var> theta;
  for (int i = 0; i < 100; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<var> reader(theta, theta_i);
  theta[0] = 0.00;
  theta[1] = 0.29;
  theta[2] = 0.70;
  theta[3] = 0.01;
  theta[4] = 1.0;
  theta[5] = 1.0;
  EXPECT_NO_THROW(reader.var_simplex(4));
  EXPECT_THROW(reader.var_simplex(2), std::domain_error);
  EXPECT_THROW(reader.var_simplex(0), std::invalid_argument);
}

TEST_F(VarReader, ordered) {
  using stan::math::var_value;
  using stan::math::var;
  std::vector<int> theta_i;
  std::vector<var> theta;
  for (int i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<var> reader(theta, theta_i);
  EXPECT_FLOAT_EQ(0.0, reader.scalar().val());  // throw away theta[0]
  var_value<Eigen::Matrix<double, Eigen::Dynamic, 1>> y = reader.var_ordered(5);
  EXPECT_EQ(5, y.size());
  EXPECT_FLOAT_EQ(1.0, y.val()[0]);
  EXPECT_FLOAT_EQ(2.0, y.val()[1]);
  EXPECT_FLOAT_EQ(5.0, y.val()[4]);
  EXPECT_FLOAT_EQ(6.0, reader.scalar().val());
}
TEST_F(VarReader, ordered_exception) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  for (int i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<var> reader(theta, theta_i);
  EXPECT_FLOAT_EQ(0.0, reader.scalar().val());  // throw away theta[0]
  var_value<Eigen::Matrix<double, Eigen::Dynamic, 1>> y = reader.var_ordered(5);
  EXPECT_EQ(5, y.size());
  EXPECT_FLOAT_EQ(1.0, y.val()[0]);
  EXPECT_FLOAT_EQ(2.0, y.val()[1]);
  EXPECT_FLOAT_EQ(5.0, y.val()[4]);
  EXPECT_FLOAT_EQ(6.0, reader.scalar().val());
}

TEST_F(VarReader, ordered_constrain) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  double v0 = 3.0;
  double v1 = v0 + exp(-1.0);
  double v2 = v1 + exp(-2.0);
  double v3 = v2 + exp(0.0);
  stan::io::reader<var> reader(theta, theta_i);
  var_value<Eigen::Matrix<double, Eigen::Dynamic, 1>> phi(reader.var_ordered_constrain(4));
  EXPECT_FLOAT_EQ(v0, phi.val()[0]);
  EXPECT_FLOAT_EQ(v1, phi.val()[1]);
  EXPECT_FLOAT_EQ(v2, phi.val()[2]);
  EXPECT_FLOAT_EQ(v3, phi.val()[3]);
}
TEST_F(VarReader, ordered_constrain_jacobian) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  double v0 = 3.0;
  double v1 = v0 + exp(-1.0);
  double v2 = v1 + exp(-2.0);
  double v3 = v2 + exp(0.0);
  stan::io::reader<var> reader(theta, theta_i);
  var lp = -101.1;
  double expected_lp = lp.val() - 1.0 - 2.0 + 0.0;
  var_value<Eigen::Matrix<double, Eigen::Dynamic, 1>> phi(reader.var_ordered_constrain(4, lp));
  EXPECT_FLOAT_EQ(v0, phi.val()[0]);
  EXPECT_FLOAT_EQ(v1, phi.val()[1]);
  EXPECT_FLOAT_EQ(v2, phi.val()[2]);
  EXPECT_FLOAT_EQ(v3, phi.val()[3]);
  EXPECT_FLOAT_EQ(expected_lp, lp.val());
}

TEST_F(VarReader, positive_ordered) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  for (int i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<var> reader(theta, theta_i);
  EXPECT_FLOAT_EQ(0.0, reader.scalar().val());  // throw away theta[0]
  var_value<Eigen::Matrix<double, Eigen::Dynamic, 1>> y = reader.var_positive_ordered(5);
  EXPECT_EQ(5, y.size());
  EXPECT_FLOAT_EQ(1.0, y.val()[0]);
  EXPECT_FLOAT_EQ(2.0, y.val()[1]);
  EXPECT_FLOAT_EQ(5.0, y.val()[4]);
  EXPECT_FLOAT_EQ(6.0, reader.scalar().val());
}

TEST_F(VarReader, positive_ordered_constrain) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  double v0 = exp(3.0);
  double v1 = v0 + exp(-1.0);
  double v2 = v1 + exp(-2.0);
  double v3 = v2 + exp(0.0);
  stan::io::reader<var> reader(theta, theta_i);
  var_value<Eigen::Matrix<double, Eigen::Dynamic, 1>> phi(
      reader.var_positive_ordered_constrain(4));
  EXPECT_FLOAT_EQ(v0, phi.val()[0]);
  EXPECT_FLOAT_EQ(v1, phi.val()[1]);
  EXPECT_FLOAT_EQ(v2, phi.val()[2]);
  EXPECT_FLOAT_EQ(v3, phi.val()[3]);
}

TEST_F(VarReader, positive_ordered_constrain_jacobian) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  double v0 = exp(3.0);
  double v1 = v0 + exp(-1.0);
  double v2 = v1 + exp(-2.0);
  double v3 = v2 + exp(0.0);
  stan::io::reader<var> reader(theta, theta_i);
  var lp = -101.1;
  double expected_lp = lp.val() + 3.0 - 1.0 - 2.0 + 0.0;
  var_value<Eigen::Matrix<double, Eigen::Dynamic, 1>> phi(
      reader.var_positive_ordered_constrain(4, lp));
  EXPECT_FLOAT_EQ(v0, phi.val()[0]);
  EXPECT_FLOAT_EQ(v1, phi.val()[1]);
  EXPECT_FLOAT_EQ(v2, phi.val()[2]);
  EXPECT_FLOAT_EQ(v3, phi.val()[3]);
  EXPECT_FLOAT_EQ(expected_lp, lp.val());
}

TEST_F(VarReader, corr_matrix) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
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
  stan::io::reader<var> reader(theta, theta_i);
  var_value<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> S
      = reader.var_corr_matrix(3);
  EXPECT_FLOAT_EQ(theta[0].val(), S.val()(0, 0));
}
TEST_F(VarReader, corr_matrix_exception) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  for (int i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  theta[0] = 1.5;
  theta[1] = 1.0;
  theta[2] = 2.0;
  theta[3] = 0.0;
  theta[4] = 1.0;
  stan::io::reader<var> reader(theta, theta_i);
  EXPECT_THROW(reader.var_corr_matrix(1), std::domain_error);
  EXPECT_THROW(reader.var_corr_matrix(2), std::domain_error);
}
TEST_F(VarReader, corr_matrix_constrain) {
  using stan::math::var_value;
  using stan::math::var;
    using Eigen::Dynamic;
  using Eigen::Matrix;
  std::vector<int> theta_i;
  std::vector<var> theta;
  theta.push_back(-2.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  theta.push_back(0.5);
  theta.push_back(1.0);
  theta.push_back(2.0);
  theta.push_back(1.0);
  stan::io::reader<var> reader(theta, theta_i);
  var_value<Matrix<double, Dynamic, Dynamic>> R(reader.var_corr_matrix_constrain(3U));
  EXPECT_EQ(3, R.rows());
  EXPECT_EQ(3, R.cols());
  EXPECT_EQ(9, R.size());
  EXPECT_EQ(4U, reader.available());
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
TEST_F(VarReader, corr_matrix_constrain_jacobian) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  theta.push_back(-2.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  theta.push_back(0.5);
  theta.push_back(1.0);
  theta.push_back(2.0);
  theta.push_back(1.0);
  stan::io::reader<var> reader(theta, theta_i);
  var lp = -9.2;
  var_value<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> R(
      reader.var_corr_matrix_constrain(3U, lp));
  EXPECT_EQ(3, R.rows());
  EXPECT_EQ(3, R.cols());
  EXPECT_EQ(9, R.size());
  EXPECT_EQ(4U, reader.available());
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

TEST_F(VarReader, cov_matrix) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
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
  stan::io::reader<var> reader(theta, theta_i);
  var_value<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> S
      = reader.var_cov_matrix(3);
  EXPECT_FLOAT_EQ(theta[0].val(), S.val()(0, 0));
  EXPECT_FLOAT_EQ(theta[1].val(), S.val()(1, 0));
  EXPECT_FLOAT_EQ(theta[7].val(), S.val()(2, 1));
  EXPECT_FLOAT_EQ(theta[8].val(), S.val()(2, 2));
}
TEST_F(VarReader, cov_matrix_exception) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  for (int i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  theta[0] = 6.3;
  theta[1] = 0.7;
  theta[2] = 0.6;
  theta[3] = 1.9;
  stan::io::reader<var> reader(theta, theta_i);
  EXPECT_THROW(reader.var_cov_matrix(2), std::domain_error);
  EXPECT_THROW(reader.var_cov_matrix(0), std::invalid_argument);
}
TEST_F(VarReader, cov_matrix_constrain) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  theta.push_back(-2.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  theta.push_back(0.5);
  theta.push_back(1.0);
  theta.push_back(2.0);
  theta.push_back(1.0);
  stan::io::reader<var> reader(theta, theta_i);
  var_value<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> S(
      reader.var_cov_matrix_constrain(3U));
  EXPECT_EQ(3, S.rows());
  EXPECT_EQ(3, S.cols());
  EXPECT_EQ(9, S.size());
  EXPECT_EQ(1U, reader.available());
  for (size_t i = 0; i < 3U; ++i)
    for (size_t j = i + 1; j < 3U; ++j)
      EXPECT_FLOAT_EQ(S.val()(i, j), S.val()(j, i));
  Eigen::SelfAdjointEigenSolver<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
      solver(S.val(), Eigen::EigenvaluesOnly);
  assert(solver.eigenvalues()[0]
         > 1E-10);  // check positive definite with smallest eigenvalue > 0
}

TEST_F(VarReader, cov_matrix_constrain_jacobian) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  theta.push_back(-2.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  theta.push_back(0.5);
  theta.push_back(1.0);
  theta.push_back(2.0);
  theta.push_back(1.0);
  stan::io::reader<var> reader(theta, theta_i);
  var lp = -3.1;

  var_value<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> S(
      reader.var_cov_matrix_constrain(3U, lp));

  EXPECT_EQ(3, S.rows());
  EXPECT_EQ(3, S.cols());
  EXPECT_EQ(9, S.size());
  EXPECT_EQ(1U, reader.available());
  for (size_t i = 0; i < 3U; ++i)
    for (size_t j = i + 1; j < 3U; ++j)
      EXPECT_FLOAT_EQ(S.val()(i, j), S.val()(j, i));
  Eigen::SelfAdjointEigenSolver<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
      solver(S.val(), Eigen::EigenvaluesOnly);
  assert(solver.eigenvalues()[0]
         > 1E-10);  // check positive definite with smallest eigenvalue > 0
  // FIXME: test Jacobian
}

TEST_F(VarReader, cholesky_factor_cov) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
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
  stan::io::reader<var> reader(theta, theta_i);
  var_value<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> S
      = reader.var_cholesky_factor_cov(3, 3);
  EXPECT_FLOAT_EQ(theta[0].val(), S.val()(0, 0));
  EXPECT_FLOAT_EQ(theta[1].val(), S.val()(1, 0));
  EXPECT_FLOAT_EQ(theta[7].val(), S.val()(1, 2));
  EXPECT_FLOAT_EQ(theta[8].val(), S.val()(2, 2));
}
TEST_F(VarReader, cholesky_factor_cov_asymmetric) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  for (int i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  // column major
  theta[0] = 1;
  theta[1] = 2;
  theta[2] = 3;

  theta[3] = 0;
  theta[4] = 4;
  theta[5] = 5;

  stan::io::reader<var> reader(theta, theta_i);
  var_value<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> S
      = reader.var_cholesky_factor_cov(3, 2);
  EXPECT_FLOAT_EQ(theta[0].val(), S.val()(0, 0));
  EXPECT_FLOAT_EQ(theta[1].val(), S.val()(1, 0));
  EXPECT_FLOAT_EQ(theta[2].val(), S.val()(2, 0));

  EXPECT_FLOAT_EQ(theta[3].val(), S.val()(0, 1));
  EXPECT_FLOAT_EQ(theta[4].val(), S.val()(1, 1));
  EXPECT_FLOAT_EQ(theta[5].val(), S.val()(2, 1));
}

TEST_F(VarReader, cholesky_factor_cov_exception) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  for (int i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  theta[0] = -6.3;
  stan::io::reader<var> reader(theta, theta_i);
  EXPECT_THROW(reader.var_cholesky_factor_cov(2, 2), std::domain_error);
  EXPECT_THROW(reader.var_cholesky_factor_cov(0, 0), std::domain_error);

  theta[0] = 1;
  EXPECT_THROW(reader.var_cholesky_factor_cov(2, 3), std::domain_error);
}
TEST_F(VarReader, cholesky_factor_cov_constrain) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::reader<var> reader(theta, theta_i);
  var_value<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> L(
      reader.var_cholesky_factor_cov_constrain(3U, 3U));
  EXPECT_NO_THROW(stan::math::check_cholesky_factor(
      "test_cholesky_factor_constrain", "L", L.val()));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  EXPECT_EQ(2U, reader.available());
}
TEST_F(VarReader, cholesky_factor_cov_constrain_asymmetric) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::reader<var> reader(theta, theta_i);
  var_value<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> L(
      reader.var_cholesky_factor_cov_constrain(3U, 2U));
  EXPECT_NO_THROW(stan::math::check_cholesky_factor(
      "test_cholesky_factor_constrain", "L", L.val()));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(2, L.cols());
  EXPECT_EQ(6, L.size());
  EXPECT_EQ(3U, reader.available());
}
TEST_F(VarReader, cholesky_factor_cov_constrain_jacobian) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::reader<var> reader(theta, theta_i);
  var lp = 1.9;
  var_value<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> L(
      reader.var_cholesky_factor_cov_constrain(3U, 3U, lp));
  EXPECT_NO_THROW(stan::math::check_cholesky_factor(
      "test_cholesky_factor_constrain", "L", L.val()));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  EXPECT_EQ(2U, reader.available());
  EXPECT_EQ(1.9 + log(L(0, 0)) + log(L(1, 1)) + log(L(2, 2)), lp);
}
TEST_F(VarReader, cholesky_factor_cov_constrain_jacobian_asymmetric) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  for (int i = 0; i < 12; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::reader<var> reader(theta, theta_i);
  var lp = 1.9;
  var_value<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> L(
      reader.var_cholesky_factor_cov_constrain(4U, 3U, lp));
  EXPECT_NO_THROW(stan::math::check_cholesky_factor(
      "test_cholesky_factor_constrain", "L", L.val()));
  EXPECT_EQ(4, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(12, L.size());
  EXPECT_EQ(3U, reader.available());
  EXPECT_EQ(1.9 + log(L.val()(0, 0)) + log(L.val()(1, 1)) + log(L.val()(2, 2)), lp);
}

TEST_F(VarReader, cholesky_factor_corr) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta(9);
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
  stan::io::reader<var> reader(theta, theta_i);
  var_value<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> S
      = reader.var_cholesky_factor_corr(3);
  EXPECT_FLOAT_EQ(theta[0].val(), S.val()(0, 0));
  EXPECT_FLOAT_EQ(theta[1].val(), S.val()(1, 0));
  EXPECT_FLOAT_EQ(theta[4].val(), S.val()(1, 1));
  EXPECT_FLOAT_EQ(theta[7].val(), S.val()(1, 2));
  EXPECT_FLOAT_EQ(theta[8].val(), S.val()(2, 2));
}

TEST_F(VarReader, cholesky_factor_corr_exception) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta(9);
  var_value<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> S;

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
  stan::io::reader<var> reader(theta, theta_i);
  EXPECT_THROW(reader.var_cholesky_factor_corr(3), std::domain_error);
}
TEST_F(VarReader, cholesky_factor_corr_constrain) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::reader<var> reader(theta, theta_i);
  var_value<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> L(
      reader.var_cholesky_factor_corr_constrain(3U));
  EXPECT_NO_THROW(stan::math::check_cholesky_factor_corr(
      "test_cholesky_factor_corr_constrain", "L", L.val()));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  EXPECT_EQ(5U, reader.available());
}

TEST_F(VarReader, matrix_lb) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<var> reader(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = reader.scalar().val();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x);
  }
  double lb = -1.5;
  var_value<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> y
      = reader.var_matrix_lb(lb, 3, 2);
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  EXPECT_FLOAT_EQ(7.0, y.val()(0, 0));
  EXPECT_FLOAT_EQ(8.0, y.val()(1, 0));
  EXPECT_FLOAT_EQ(9.0, y.val()(2, 0));
  EXPECT_FLOAT_EQ(10.0, y.val()(0, 1));
  EXPECT_FLOAT_EQ(11.0, y.val()(1, 1));
  EXPECT_FLOAT_EQ(12.0, y.val()(2, 1));

  double a = reader.scalar().val();
  EXPECT_FLOAT_EQ(13.0, a);
}

TEST_F(VarReader, matrix_lb_constrain) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<var> reader(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = reader.scalar().val();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x);
  }
  double lb = -1.5;
  var_value<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> y
      = reader.var_matrix_lb_constrain(lb, 3, 2);
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(7.0, lb), y.val()(0, 0));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(8.0, lb), y.val()(1, 0));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(9.0, lb), y.val()(2, 0));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(10.0, lb), y.val()(0, 1));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(11.0, lb), y.val()(1, 1));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(12.0, lb), y.val()(2, 1));

  double a = reader.scalar().val();
  EXPECT_FLOAT_EQ(13.0, a);
}

TEST_F(VarReader, matrix_lb_constrain_lp) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<var> reader(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = reader.scalar().val();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x);
  }
  double lb = -1.5;
  var lp = -5.0;
  var_value<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> y
      = reader.var_matrix_lb_constrain(lb, 3, 2, lp);
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(7.0, lb, lp), y.val()(0, 0));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(8.0, lb, lp), y.val()(1, 0));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(9.0, lb, lp), y.val()(2, 0));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(10.0, lb, lp), y.val()(0, 1));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(11.0, lb, lp), y.val()(1, 1));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(12.0, lb, lp), y.val()(2, 1));

  double a = reader.scalar().val();
  EXPECT_FLOAT_EQ(13.0, a);
}

TEST_F(VarReader, matrix_ub) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<var> reader(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = reader.scalar().val();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x);
  }
  double ub = 12.5;
  var_value<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> y
      = reader.var_matrix_ub(ub, 3, 2);
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  EXPECT_FLOAT_EQ(7.0, y.val()(0, 0));
  EXPECT_FLOAT_EQ(8.0, y.val()(1, 0));
  EXPECT_FLOAT_EQ(9.0, y.val()(2, 0));
  EXPECT_FLOAT_EQ(10.0, y.val()(0, 1));
  EXPECT_FLOAT_EQ(11.0, y.val()(1, 1));
  EXPECT_FLOAT_EQ(12.0, y.val()(2, 1));

  double a = reader.scalar().val();
  EXPECT_FLOAT_EQ(13.0, a);
}

TEST_F(VarReader, matrix_ub_constrain) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<var> reader(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = reader.scalar().val();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x);
  }
  double ub = 14.1;
  var_value<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> y
      = reader.var_matrix_ub_constrain(ub, 3, 2);
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(7.0, ub), y.val()(0, 0));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(8.0, ub), y.val()(1, 0));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(9.0, ub), y.val()(2, 0));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(10.0, ub), y.val()(0, 1));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(11.0, ub), y.val()(1, 1));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(12.0, ub), y.val()(2, 1));

  double a = reader.scalar().val();
  EXPECT_FLOAT_EQ(13.0, a);
}

TEST_F(VarReader, matrix_ub_constrain_lp) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<var> reader(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = reader.scalar().val();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x);
  }
  double ub = 12.1;
  var lp = -5.0;
  var_value<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> y
      = reader.var_matrix_ub_constrain(ub, 3, 2, lp);
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(7.0, ub, lp), y.val()(0, 0));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(8.0, ub, lp), y.val()(1, 0));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(9.0, ub, lp), y.val()(2, 0));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(10.0, ub, lp), y.val()(0, 1));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(11.0, ub, lp), y.val()(1, 1));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(12.0, ub, lp), y.val()(2, 1));

  double a = reader.scalar().val();
  EXPECT_FLOAT_EQ(13.0, a);
}
/*
TEST_F(VarReader, matrix_lub) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<var> reader(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = reader.scalar().val();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x);
  }
  double lb = 6.9;
  double ub = 12.5;
  var_value<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> y
      = reader.var_matrix_lub(lb, ub, 3, 2);
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  EXPECT_FLOAT_EQ(7.0, y.val()(0, 0));
  EXPECT_FLOAT_EQ(8.0, y.val()(1, 0));
  EXPECT_FLOAT_EQ(9.0, y.val()(2, 0));
  EXPECT_FLOAT_EQ(10.0, y.val()(0, 1));
  EXPECT_FLOAT_EQ(11.0, y.val()(1, 1));
  EXPECT_FLOAT_EQ(12.0, y.val()(2, 1));

  double a = reader.scalar().val();
  EXPECT_FLOAT_EQ(13.0, a);
}

TEST_F(VarReader, matrix_lub_constrain) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<var> reader(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = reader.scalar().val();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x);
  }
  double lb = 3.5;
  double ub = 14.1;
  var_value<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> y
      = reader.var_matrix_lub_constrain(lb, ub, 3, 2);
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(7.0, lb, ub), y.val()(0, 0));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(8.0, lb, ub), y.val()(1, 0));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(9.0, lb, ub), y.val()(2, 0));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(10.0, lb, ub), y.val()(0, 1));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(11.0, lb, ub), y.val()(1, 1));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(12.0, lb, ub), y.val()(2, 1));

  double a = reader.scalar().val();
  EXPECT_FLOAT_EQ(13.0, a);
}

TEST_F(VarReader, matrix_lub_constrain_lp) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<int> theta_i;
  std::vector<var> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<var> reader(theta, theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = reader.scalar().val();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x);
  }
  double lb = 4.1;
  double ub = 12.1;
  var lp = -5.0;
  var_value<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> y
      = reader.var_matrix_lub_constrain(lb, ub, 3, 2, lp);
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(7.0, lb, ub, lp), y.val()(0, 0));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(8.0, lb, ub, lp), y.val()(1, 0));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(9.0, lb, ub, lp), y.val()(2, 0));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(10.0, lb, ub, lp), y.val()(0, 1));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(11.0, lb, ub, lp), y.val()(1, 1));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(12.0, lb, ub, lp), y.val()(2, 1));

  double a = reader.scalar().val();
  EXPECT_FLOAT_EQ(13.0, a);
}

TEST_F(VarReader, SimplexThrows) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<var> theta;
  std::vector<int> theta_i;
  stan::io::reader<var> reader(theta, theta_i);

  double x = 0;
  var lp = 0;
  EXPECT_THROW(reader.var_simplex(x), std::invalid_argument);
  EXPECT_THROW(reader.var_simplex_constrain(x), std::invalid_argument);
  EXPECT_THROW(reader.var_simplex_constrain(x, lp), std::invalid_argument);
}

TEST_F(VarReader, UnitVectorThrows) {
  using stan::math::var_value;
  using stan::math::var;
    std::vector<var> theta;
  std::vector<int> theta_i;
  stan::io::reader<var> reader(theta, theta_i);

  double x = 0;
  var lp = 0;
  EXPECT_THROW(reader.var_unit_vector(x), std::invalid_argument);
  EXPECT_THROW(reader.var_unit_vector_constrain(x), std::invalid_argument);
  EXPECT_THROW(reader.var_unit_vector_constrain(x, lp), std::invalid_argument);
}
*/

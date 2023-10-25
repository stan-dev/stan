#include <stan/io/opencl_deserializer.hpp>
// expect_near_rel comes from lib/stan_math
#include <test/unit/math/expect_near_rel.hpp>
#include <test/unit/pretty_print_types.hpp>
#include <gtest/gtest.h>

using var_matrix_cl_t = stan::math::var_value<stan::math::matrix_cl<double>>;
using var_vector_cl_t = stan::math::var_value<stan::math::matrix_cl<double>>;
using var_row_vector_cl_t = stan::math::var_value<stan::math::matrix_cl<double>>;
using matrix_cl_t = stan::math::matrix_cl<double>;
using vector_cl_t = stan::math::matrix_cl<double>;
using row_vector_cl_t = stan::math::matrix_cl<double>;


TEST(deserializer_var_vector, varmat_reads_as_doubles) {
  using stan::math::matrix_cl;
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));
  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<double, true> deserializer(theta_cl, theta_i_cl);
  vector_cl_t y_device = deserializer.read<vector_cl_t>(4);
  EXPECT_EQ(4, y_device.rows());
  EXPECT_EQ(1, y_device.cols());
  EXPECT_EQ(4, y_device.size());

  auto y_host = stan::math::from_matrix_cl<Eigen::VectorXd>(y_device);
  EXPECT_FLOAT_EQ(0.0, y_host[0]);
  EXPECT_FLOAT_EQ(1.0, y_host[1]);
  EXPECT_FLOAT_EQ(2.0, y_host[2]);
  EXPECT_FLOAT_EQ(3.0, y_host[3]);
  stan::math::recover_memory();
}
// vectors

TEST(deserializer_var_vector, read) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));

  auto theta_cl = var_vector_cl_t(stan::math::to_matrix_cl(theta));
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  auto z = deserializer.read<stan::math::var>();
    auto z_host = stan::math::from_matrix_cl<Eigen::VectorXd>(z.val());
    EXPECT_FLOAT_EQ(0, z_host(0, 0));
  int iter = 0;
  for (size_t i = 0; i < 7U; ++i) {
    auto x = deserializer.read<stan::math::var>();
    auto x_host = stan::math::from_matrix_cl<Eigen::VectorXd>(x.val());
    EXPECT_FLOAT_EQ(i + 1, x_host(0, 0));
  }
  var_vector_cl_t y_device = deserializer.read<var_vector_cl_t>(4);
  EXPECT_EQ(4, y_device.rows());
  EXPECT_EQ(1, y_device.cols());
  EXPECT_EQ(4, y_device.size());

  auto y_host = stan::math::from_matrix_cl<Eigen::VectorXd>(y_device.val());
  EXPECT_FLOAT_EQ(8.0, y_host[0]);
  EXPECT_FLOAT_EQ(9.0, y_host[1]);
  EXPECT_FLOAT_EQ(10.0, y_host[2]);
  EXPECT_FLOAT_EQ(11.0, y_host[3]);
}

// row vector
/*
TEST(deserializer_var_row_vector, read) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  for (size_t i = 0; i < 7U; ++i) {
    stan::math::var x = deserializer.read<stan::math::var>();
    EXPECT_FLOAT_EQ(static_cast<double>(i), x.val());
  }
  var_row_vector_cl_t y = deserializer.read<var_row_vector_cl_t>(4);
  EXPECT_EQ(1, y.rows());
  EXPECT_EQ(4, y.cols());
  EXPECT_EQ(4, y.size());
  EXPECT_FLOAT_EQ(7.0, y.val()[0]);
  EXPECT_FLOAT_EQ(8.0, y.val()[1]);
  EXPECT_FLOAT_EQ(9.0, y.val()[2]);
  EXPECT_FLOAT_EQ(10.0, y.val()[3]);

  stan::math::var z = deserializer.read<stan::math::var>();
  EXPECT_FLOAT_EQ(11.0, z.val());
}
*/

// matrix

TEST(deserializer_var_matrix, read) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  for (int i = 0; i < 7; ++i) {
    auto x = deserializer.read<stan::math::var>();
    auto x_host = stan::math::from_matrix_cl<Eigen::VectorXd>(x.val());
    EXPECT_FLOAT_EQ(i, x_host(0));
  }
  var_matrix_cl_t y_device = deserializer.read<var_matrix_cl_t>(3, 2);
  EXPECT_EQ(3, y_device.rows());
  EXPECT_EQ(2, y_device.cols());

  auto y = stan::math::from_matrix_cl<Eigen::MatrixXd>(y_device.val());
  EXPECT_EQ(3, y.rows());
  EXPECT_EQ(2, y.cols());
  EXPECT_FLOAT_EQ(7.0, y.val()(0, 0));
  EXPECT_FLOAT_EQ(8.0, y.val()(1, 0));
  EXPECT_FLOAT_EQ(9.0, y.val()(2, 0));
  EXPECT_FLOAT_EQ(10.0, y.val()(0, 1));
  EXPECT_FLOAT_EQ(11.0, y.val()(1, 1));
  EXPECT_FLOAT_EQ(12.0, y.val()(2, 1));

  auto a = deserializer.read<stan::math::var>();
  auto a_host = stan::math::from_matrix_cl<Eigen::VectorXd>(a.val());
  EXPECT_FLOAT_EQ(13.0, a_host(0));
}

// lb

TEST(deserializer, read_constrain_lb_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(1.0);
  theta.push_back(0.0);

  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  stan::math::var lp = 0.0;
  auto reference = stan::math::lb_constrain(stan::math::to_vector(theta), 1.5);
  auto y = deserializer.read_constrain_lb<var_vector_cl_t, false>(1.5, lp,
                                                               theta.size());
  auto y_host = stan::math::from_matrix_cl<Eigen::VectorXd>(y.val());

  EXPECT_TRUE((std::is_same<var_vector_cl_t, decltype(y)>::value));
  stan::test::expect_near_rel("deserializer tests", reference.val(), y_host.val());
}

TEST(deserializer, read_constrain_lb_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(1.0);
  theta.push_back(0.0);

  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  stan::math::var lp_ref = 0.0;
  stan::math::var lp = 0.0;
  auto reference
      = stan::math::lb_constrain(stan::math::to_vector(theta), 1.5, lp_ref);
  auto y = deserializer.read_constrain_lb<var_vector_cl_t, true>(1.5, lp,
                                                              theta.size());
  EXPECT_TRUE((std::is_same<var_vector_cl_t, decltype(y)>::value));
  auto y_host = stan::math::from_matrix_cl<Eigen::VectorXd>(y.val());
  stan::test::expect_near_rel("deserializer tests", reference.val(), y_host.val());
  EXPECT_FLOAT_EQ(lp_ref.val(), lp.val());
}

// ub

TEST(deserializer, read_constrain_ub_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(1.0);
  theta.push_back(0.0);
  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  stan::math::var lp = 0.0;
  auto reference = stan::math::ub_constrain(stan::math::to_vector(theta), 1.5);
  auto y = deserializer.read_constrain_ub<var_vector_cl_t, false>(1.5, lp,
                                                               theta.size());
  auto y_host = stan::math::from_matrix_cl<Eigen::VectorXd>(y.val());

  EXPECT_TRUE((std::is_same<var_vector_cl_t, decltype(y)>::value));
  stan::test::expect_near_rel("deserializer tests", reference.val(), y_host.val());
}

TEST(deserializer, read_constrain_ub_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(1.0);
  theta.push_back(0.0);
  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  stan::math::var lp_ref = 0.0;
  stan::math::var lp = 0.0;
  auto reference
      = stan::math::ub_constrain(stan::math::to_vector(theta), 1.5, lp_ref);
  auto y = deserializer.read_constrain_ub<var_vector_cl_t, true>(1.5, lp,
                                                              theta.size());
  EXPECT_TRUE((std::is_same<var_vector_cl_t, decltype(y)>::value));
  auto y_host = stan::math::from_matrix_cl<Eigen::VectorXd>(y.val());
  stan::test::expect_near_rel("deserializer tests", reference.val(), y_host.val());
  EXPECT_FLOAT_EQ(lp_ref.val(), lp.val());
}

// lub

TEST(deserializer, read_constrain_lub_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  stan::math::var lp = 0.0;
  auto reference
      = stan::math::lub_constrain(stan::math::to_vector(theta), 1.0, 2.0);
  auto y = deserializer.read_constrain_lub<var_vector_cl_t, false>(1.0, 2.0, lp,
                                                                theta.size());
  auto y_host = stan::math::from_matrix_cl<Eigen::VectorXd>(y.val());
  EXPECT_TRUE((std::is_same<var_vector_cl_t, decltype(y)>::value));
  stan::test::expect_near_rel("deserializer tests", reference.val(), y_host.val());
}

TEST(deserializer, read_constrain_lub_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  stan::math::var lp_ref = 0.0;
  stan::math::var lp = 0.0;
  auto reference = stan::math::lub_constrain(stan::math::to_vector(theta), 1.0,
                                             2.0, lp_ref);
  auto y = deserializer.read_constrain_lub<var_vector_cl_t, true>(1.0, 2.0, lp,
                                                               theta.size());
  EXPECT_TRUE((std::is_same<var_vector_cl_t, decltype(y)>::value));
  auto y_host = stan::math::from_matrix_cl<Eigen::VectorXd>(y.val());
  stan::test::expect_near_rel("deserializer tests", reference.val(), y_host.val());
  EXPECT_FLOAT_EQ(lp_ref.val(), lp.val());
}

// offset multiplier

TEST(deserializer, read_constrain_offset_multiplier_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  stan::math::var lp = 0.0;
  auto reference = stan::math::offset_multiplier_constrain(
      stan::math::to_vector(theta), 1.0, 2.0);
  auto y = deserializer.read_constrain_offset_multiplier<var_vector_cl_t, false>(
      1.0, 2.0, lp, theta.size());
  auto y_host = stan::math::from_matrix_cl<Eigen::VectorXd>(y.val());

  EXPECT_TRUE((std::is_same<var_vector_cl_t, decltype(y)>::value));
  stan::test::expect_near_rel("deserializer tests", reference.val(), y_host.val());
}

TEST(deserializer, read_constrain_offset_multiplier_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  stan::math::var lp_ref = 0.0;
  stan::math::var lp = 0.0;
  auto reference = stan::math::offset_multiplier_constrain(
      stan::math::to_vector(theta), 1.0, 2.0, lp_ref);
  auto y = deserializer.read_constrain_offset_multiplier<var_vector_cl_t, true>(
      1.0, 2.0, lp, theta.size());
  EXPECT_TRUE((std::is_same<var_vector_cl_t, decltype(y)>::value));
  auto y_host = stan::math::from_matrix_cl<Eigen::VectorXd>(y.val());
  stan::test::expect_near_rel("deserializer tests", reference.val(), y_host.val());
  EXPECT_FLOAT_EQ(lp_ref.val(), lp.val());
}

// unit vector

TEST(deserializer, read_constrain_unit_vector_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  stan::math::var lp = 0.0;
  auto reference
      = stan::math::unit_vector_constrain(stan::math::to_vector(theta));
  auto y = deserializer.read_constrain_unit_vector<var_vector_cl_t, false>(
      lp, theta.size());
  EXPECT_TRUE((std::is_same<var_vector_cl_t, decltype(y)>::value));
  auto y_host = stan::math::from_matrix_cl<Eigen::VectorXd>(y.val());
  stan::test::expect_near_rel("deserializer tests", reference.val(), y_host.val());
}

TEST(deserializer, read_constrain_unit_vector_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  stan::math::var lp_ref = 0.0;
  stan::math::var lp = 0.0;
  auto reference
      = stan::math::unit_vector_constrain(stan::math::to_vector(theta), lp_ref);
  auto y = deserializer.read_constrain_unit_vector<var_vector_cl_t, true>(
      lp, theta.size());
  EXPECT_TRUE((std::is_same<var_vector_cl_t, decltype(y)>::value));
  auto y_host = stan::math::from_matrix_cl<Eigen::VectorXd>(y.val());
  stan::test::expect_near_rel("deserializer tests", reference.val(), y_host.val());
  EXPECT_FLOAT_EQ(lp_ref.val(), lp.val());
}

// simplex

TEST(deserializer, read_constrain_simplex_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  stan::math::var lp = 0.0;
  auto reference = stan::math::simplex_constrain(stan::math::to_vector(theta));
  auto y = deserializer.read_constrain_simplex<var_vector_cl_t, false>(
      lp, theta.size() + 1);
  EXPECT_TRUE((std::is_same<var_vector_cl_t, decltype(y)>::value));
  auto y_host = stan::math::from_matrix_cl<Eigen::VectorXd>(y.val());
  stan::test::expect_near_rel("deserializer tests", reference.val(), y_host.val());

}

TEST(deserializer, read_constrain_simplex_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  stan::math::var lp_ref = 0.0;
  stan::math::var lp = 0.0;
  auto reference
      = stan::math::simplex_constrain(stan::math::to_vector(theta), lp_ref);
  auto y = deserializer.read_constrain_simplex<var_vector_cl_t, true>(
      lp, theta.size() + 1);
  EXPECT_TRUE((std::is_same<var_vector_cl_t, decltype(y)>::value));
  auto y_host = stan::math::from_matrix_cl<Eigen::VectorXd>(y.val());
  stan::test::expect_near_rel("deserializer tests", reference.val(), y_host.val());
  EXPECT_FLOAT_EQ(lp_ref.val(), lp.val());
}

// ordered

TEST(deserializer, read_constrain_ordered_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  stan::math::var lp = 0.0;
  auto reference = stan::math::ordered_constrain(stan::math::to_vector(theta));
  auto y = deserializer.read_constrain_ordered<var_vector_cl_t, false>(
      lp, theta.size());
  EXPECT_TRUE((std::is_same<var_vector_cl_t, decltype(y)>::value));
  auto y_host = stan::math::from_matrix_cl<Eigen::VectorXd>(y.val());
  stan::test::expect_near_rel("deserializer tests", reference.val(), y_host.val());
}

TEST(deserializer, read_constrain_ordered_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  stan::math::var lp_ref = 0.0;
  stan::math::var lp = 0.0;
  auto reference
      = stan::math::ordered_constrain(stan::math::to_vector(theta), lp_ref);
  auto y = deserializer.read_constrain_ordered<var_vector_cl_t, true>(
      lp, theta.size());
  EXPECT_TRUE((std::is_same<var_vector_cl_t, decltype(y)>::value));
  auto y_host = stan::math::from_matrix_cl<Eigen::VectorXd>(y.val());
  stan::test::expect_near_rel("deserializer tests", reference.val(), y_host.val());
  EXPECT_FLOAT_EQ(lp_ref.val(), lp.val());
}

// positive ordered
/*

TEST(deserializer, read_constrain_positive_ordered_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  stan::math::var lp = 0.0;
  auto reference
      = stan::math::positive_ordered_constrain(stan::math::to_vector(theta));
  auto y = deserializer.read_constrain_positive_ordered<var_vector_cl_t, false>(
      lp, theta.size());
  EXPECT_TRUE((std::is_same<var_vector_cl_t, decltype(y)>::value));
  stan::test::expect_near_rel("deserializer tests", reference.val(), y.val());
}

TEST(deserializer, read_constrain_positive_ordered_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  stan::math::var lp_ref = 0.0;
  stan::math::var lp = 0.0;
  auto reference = stan::math::positive_ordered_constrain(
      stan::math::to_vector(theta), lp_ref);
  auto y = deserializer.read_constrain_positive_ordered<var_vector_cl_t, true>(
      lp, theta.size());
  EXPECT_TRUE((std::is_same<var_vector_cl_t, decltype(y)>::value));
  stan::test::expect_near_rel("deserializer tests", reference.val(), y.val());
  EXPECT_FLOAT_EQ(lp_ref.val(), lp.val());
}

// chol cov

TEST(deserializer_var_matrix, cholesky_factor_cov_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  stan::math::var lp = 0;
  auto reference = stan::math::cholesky_factor_constrain(
      stan::math::to_vector(theta).segment(0, 6), 3, 3);
  auto L = deserializer.read_constrain_cholesky_factor_cov<var_matrix_cl_t, false>(
      lp, 3U, 3U);
  EXPECT_TRUE((std::is_same<var_matrix_cl_t, decltype(L)>::value));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  stan::test::expect_near_rel("deserializer tests", reference.val(), L.val());
  EXPECT_EQ(2U, deserializer.available());
}

TEST(deserializer_var_matrix, cholesky_factor_cov_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  stan::math::var lp_ref = 0.0;
  stan::math::var lp = 0.0;
  auto reference = stan::math::cholesky_factor_constrain(
      stan::math::to_vector(theta).segment(0, 6), 3, 3, lp_ref);
  auto L = deserializer.read_constrain_cholesky_factor_cov<var_matrix_cl_t, true>(
      lp, 3U, 3U);
  EXPECT_TRUE((std::is_same<var_matrix_cl_t, decltype(L)>::value));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  stan::test::expect_near_rel("deserializer tests", reference.val(), L.val());
  EXPECT_EQ(2U, deserializer.available());
  EXPECT_FLOAT_EQ(lp_ref.val(), lp.val());
}

TEST(deserializer_var_matrix, cholesky_factor_cov_constrain_non_square) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  stan::math::var lp = 0;
  auto reference = stan::math::cholesky_factor_constrain(
      stan::math::to_vector(theta).segment(0, 5), 3, 2);
  auto L = deserializer.read_constrain_cholesky_factor_cov<var_matrix_cl_t, false>(
      lp, 3U, 2U);
  EXPECT_TRUE((std::is_same<var_matrix_cl_t, decltype(L)>::value));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(2, L.cols());
  EXPECT_EQ(6, L.size());
  stan::test::expect_near_rel("deserializer tests", reference.val(), L.val());
  EXPECT_EQ(3U, deserializer.available());
}

TEST(deserializer_var_matrix, cholesky_factor_cov_jacobian_non_square) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  stan::math::var lp_ref = 0.0;
  stan::math::var lp = 0.0;
  auto reference = stan::math::cholesky_factor_constrain(
      stan::math::to_vector(theta).segment(0, 5), 3, 2, lp_ref);
  auto L = deserializer.read_constrain_cholesky_factor_cov<var_matrix_cl_t, true>(
      lp, 3U, 2U);
  EXPECT_TRUE((std::is_same<var_matrix_cl_t, decltype(L)>::value));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(2, L.cols());
  EXPECT_EQ(6, L.size());
  stan::test::expect_near_rel("deserializer tests", reference.val(), L.val());
  EXPECT_EQ(3U, deserializer.available());
  EXPECT_FLOAT_EQ(lp_ref.val(), lp.val());
}

// chol corr

TEST(deserializer_var_matrix, cholesky_factor_corr_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  stan::math::var lp = 0;
  auto reference = stan::math::cholesky_corr_constrain(
      stan::math::to_vector(theta).segment(0, 3), 3);
  auto L
      = deserializer.read_constrain_cholesky_factor_corr<var_matrix_cl_t, false>(
          lp, 3U);
  EXPECT_TRUE((std::is_same<var_matrix_cl_t, decltype(L)>::value));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  stan::test::expect_near_rel("deserializer tests", reference.val(), L.val());
  EXPECT_EQ(5U, deserializer.available());
}

TEST(deserializer_var_matrix, cholesky_factor_corr_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  stan::math::var lp_ref = 0.0;
  stan::math::var lp = 0.0;
  auto reference = stan::math::cholesky_corr_constrain(
      stan::math::to_vector(theta).segment(0, 3), 3, lp_ref);
  auto L = deserializer.read_constrain_cholesky_factor_corr<var_matrix_cl_t, true>(
      lp, 3U);
  EXPECT_TRUE((std::is_same<var_matrix_cl_t, decltype(L)>::value));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  stan::test::expect_near_rel("deserializer tests", reference.val(), L.val());
  EXPECT_EQ(5U, deserializer.available());
  EXPECT_FLOAT_EQ(lp_ref.val(), lp.val());
}

// cov

TEST(deserializer_var_matrix, cov_matrix_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  stan::math::var lp = 0;
  auto reference = stan::math::cov_matrix_constrain(
      stan::math::to_vector(theta).segment(0, 6), 3);
  auto L = deserializer.read_constrain_cov_matrix<var_matrix_cl_t, false>(lp, 3U);
  EXPECT_TRUE((std::is_same<var_matrix_cl_t, decltype(L)>::value));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  stan::test::expect_near_rel("deserializer tests", reference.val(), L.val());
  EXPECT_EQ(2U, deserializer.available());
}

TEST(deserializer_var_matrix, cov_matrix_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  stan::math::var lp_ref = 0.0;
  stan::math::var lp = 0.0;
  auto reference = stan::math::cov_matrix_constrain(
      stan::math::to_vector(theta).segment(0, 6), 3, lp_ref);
  auto L = deserializer.read_constrain_cov_matrix<var_matrix_cl_t, true>(lp, 3U);
  EXPECT_TRUE((std::is_same<var_matrix_cl_t, decltype(L)>::value));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  stan::test::expect_near_rel("deserializer tests", reference.val(), L.val());
  EXPECT_EQ(2U, deserializer.available());
  EXPECT_FLOAT_EQ(lp_ref.val(), lp.val());
}

// corr

TEST(deserializer_var_matrix, corr_matrix_constrain) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  stan::math::var lp = 0;
  auto reference = stan::math::corr_matrix_constrain(
      stan::math::to_vector(theta).segment(0, 3), 3);
  auto L = deserializer.read_constrain_corr_matrix<var_matrix_cl_t, false>(lp, 3U);
  EXPECT_TRUE((std::is_same<var_matrix_cl_t, decltype(L)>::value));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  stan::test::expect_near_rel("deserializer tests", reference.val(), L.val());
  EXPECT_EQ(5U, deserializer.available());
}

TEST(deserializer_var_matrix, corr_matrix_jacobian) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  auto theta_cl = stan::math::to_matrix_cl(theta);
  auto theta_i_cl = stan::math::to_matrix_cl(theta_i);
  stan::io::deserializer<stan::math::var, true> deserializer(theta_cl, theta_i_cl);
  stan::math::var lp_ref = 0.0;
  stan::math::var lp = 0.0;
  auto reference = stan::math::corr_matrix_constrain(
      stan::math::to_vector(theta).segment(0, 3), 3, lp_ref);
  auto L = deserializer.read_constrain_corr_matrix<var_matrix_cl_t, true>(lp, 3U);
  EXPECT_TRUE((std::is_same<var_matrix_cl_t, decltype(L)>::value));
  EXPECT_EQ(3, L.rows());
  EXPECT_EQ(3, L.cols());
  EXPECT_EQ(9, L.size());
  stan::test::expect_near_rel("deserializer tests", reference.val(), L.val());
  EXPECT_EQ(5U, deserializer.available());
  EXPECT_FLOAT_EQ(lp_ref.val(), lp.val());
}
*/

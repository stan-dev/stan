#include <vector>
#include <stdexcept>
#include <gtest/gtest.h>
#include <stan/io/writer.hpp>
#include <stan/math/special_functions.hpp>

TEST(io_writer, scalar_pos_unconstrain_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  double y;

  y = 1.0;
  EXPECT_NO_THROW (writer.scalar_pos_unconstrain(y));

  y = -1.0;
  EXPECT_THROW (writer.scalar_pos_unconstrain(y), std::runtime_error);
}
TEST(io_writer, scalar_lb_unconstrain_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  double y;
  double lb = 0;
  y = 1.0;
  EXPECT_NO_THROW (writer.scalar_lb_unconstrain(lb, y));

  y = -1.0;
  EXPECT_THROW (writer.scalar_lb_unconstrain(lb, y), std::runtime_error);
}
TEST(io_writer, scalar_ub_unconstrain_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  double y;
  double ub = 0;
  y = -1.0;
  EXPECT_NO_THROW (writer.scalar_ub_unconstrain(ub, y));

  y = 1.0;
  EXPECT_THROW (writer.scalar_ub_unconstrain(ub, y), std::runtime_error);
}
TEST(io_writer, scalar_lub_unconstrain_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  double y;
  double lb = 0.0;
  double ub = 1.0;
  y = 0.5;
  EXPECT_NO_THROW (writer.scalar_lub_unconstrain(lb, ub, y));

  y = 2.0;
  EXPECT_THROW (writer.scalar_lub_unconstrain(lb, ub, y), std::runtime_error);

  y = -2.0;
  EXPECT_THROW (writer.scalar_lub_unconstrain(lb, ub, y), std::runtime_error);
}
TEST(io_writer, corr_unconstrain_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  double y;
  y = 0.5;
  EXPECT_NO_THROW (writer.corr_unconstrain(y));

  y = 2.0;
  EXPECT_THROW (writer.corr_unconstrain(y), std::runtime_error);

  y = -2.0;
  EXPECT_THROW (writer.corr_unconstrain(y), std::runtime_error);
}
TEST(io_writer, prob_unconstrain_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  double y;
  y = 0.5;
  EXPECT_NO_THROW (writer.prob_unconstrain(y));

  y = 2.0;
  EXPECT_THROW (writer.prob_unconstrain(y), std::runtime_error);

  y = -0.5;
  EXPECT_THROW (writer.prob_unconstrain(y), std::runtime_error);
}
TEST(io_writer, pos_ordered_unconstrain_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  Eigen::Matrix<double,Eigen::Dynamic,1> y;
  
  y.resize(0);
  EXPECT_NO_THROW (writer.pos_ordered_unconstrain(y));
  y.resize(2);
  y << 0.1, 1.0;
  EXPECT_NO_THROW (writer.pos_ordered_unconstrain(y));

  y << -0.5, 1.0;
  EXPECT_THROW (writer.pos_ordered_unconstrain(y), std::runtime_error);

  y << 1.0, 0.1;
  EXPECT_THROW (writer.pos_ordered_unconstrain(y), std::runtime_error);
}
TEST(io_writer, simplex_unconstrain_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  Eigen::Matrix<double,Eigen::Dynamic,1> y;
  
  y.resize(2);
  y << 0.5, 0.5;
  EXPECT_NO_THROW (writer.simplex_unconstrain(y));
  
  y << 1.1, -0.1;
  EXPECT_THROW (writer.simplex_unconstrain(y), std::runtime_error);

  y << 0.1, 0.1;
  EXPECT_THROW (writer.simplex_unconstrain(y), std::runtime_error);
  
  y.resize(0);
  EXPECT_THROW (writer.simplex_unconstrain(y), std::runtime_error);
}
TEST(io_writer, corr_matrix_unconstrain_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  
  y.resize(2,2);
  y << 1.0, 0.0, 0.0, 1.0;
  EXPECT_NO_THROW (writer.corr_matrix_unconstrain(y));
  
  y << 0.0, 1.0, 0.0, 0.0;
  EXPECT_THROW (writer.corr_matrix_unconstrain(y), std::runtime_error);

  y.resize(0,0);
  EXPECT_THROW (writer.corr_matrix_unconstrain(y), std::runtime_error);

  y.resize(1,2);
  EXPECT_THROW (writer.corr_matrix_unconstrain(y), std::runtime_error);
}
TEST(io_writer, cov_matrix_unconstrain_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  
  y.resize(2,2);
  y << 1.0, 0.0, 0.0, 1.0;
  EXPECT_NO_THROW (writer.cov_matrix_unconstrain(y));
  
  y << 0.0, 1.0, 0.0, 0.0;
  EXPECT_THROW (writer.cov_matrix_unconstrain(y), std::runtime_error);

  y.resize(0,0);
  EXPECT_THROW (writer.cov_matrix_unconstrain(y), std::runtime_error);
  
  y.resize(2,1);
  EXPECT_THROW (writer.cov_matrix_unconstrain(y), std::runtime_error);
}

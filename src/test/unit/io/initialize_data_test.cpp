#include<stan/io/initialize_data.hpp>
#include <gtest/gtest.h>

TEST(initializer, scalar) {
  stan::math::stack_alloc allocator;
  double xx = stan::io::initialize_data<double>("blah", allocator);
}

TEST(initializer, eigen) {
  stan::math::stack_alloc allocator;
  Eigen::Map<Eigen::VectorXd> x_vec = stan::io::initialize_data<Eigen::VectorXd>("blah", allocator, 3);
  Eigen::Map<Eigen::RowVectorXd> x_rowvec = stan::io::initialize_data<Eigen::RowVectorXd>("blah", allocator, 3);
  Eigen::Map<Eigen::MatrixXd> x_mat = stan::io::initialize_data<Eigen::MatrixXd>("blah", allocator, 3, 4);
}

TEST(initializer, std_vector) {
  stan::math::stack_alloc allocator;
  std::vector<double> xx = stan::io::initialize_data<std::vector<double>>("blah", allocator, 3);
  std::vector<Eigen::Map<Eigen::VectorXd>> x_vec = stan::io::initialize_data<std::vector<Eigen::VectorXd>>("blah", allocator, 4, 3);
  std::vector<Eigen::Map<Eigen::RowVectorXd>> x_rowvec = stan::io::initialize_data<std::vector<Eigen::RowVectorXd>>("blah", allocator, 4, 3);
  std::vector<Eigen::Map<Eigen::MatrixXd>> x_mat = stan::io::initialize_data<std::vector<Eigen::MatrixXd>>("blah", allocator, 4, 3, 4);
}

#include<stan/io/initialize_data.hpp>
#include <gtest/gtest.h>

TEST(initializer, scalar) {
  stan::math::stack_alloc allocator;
  double xx = stan::io::initialize_data<double>(allocator);
  EXPECT_FALSE(allocator.in_stack(static_cast<void*>(&xx)));
}

TEST(initializer, eigen) {
  stan::math::stack_alloc allocator;
  Eigen::Map<Eigen::VectorXd> x_vec = stan::io::initialize_data<Eigen::VectorXd>(allocator, 3);
  EXPECT_EQ(3, x_vec.size());
  EXPECT_TRUE(allocator.in_stack(static_cast<void*>(x_vec.data())));
  Eigen::Map<Eigen::RowVectorXd> x_rowvec = stan::io::initialize_data<Eigen::RowVectorXd>(allocator, 3);
  EXPECT_EQ(3, x_rowvec.size());
  EXPECT_TRUE(allocator.in_stack(static_cast<void*>(x_rowvec.data())));
  Eigen::Map<Eigen::MatrixXd> x_mat = stan::io::initialize_data<Eigen::MatrixXd>(allocator, 3, 4);
  EXPECT_TRUE(allocator.in_stack(static_cast<void*>(x_mat.data())));
  EXPECT_EQ(3, x_mat.rows());
  EXPECT_EQ(4, x_mat.cols());
}

TEST(initializer, std_vector) {
  stan::math::stack_alloc allocator;
  std::vector<double> xx = stan::io::initialize_data<std::vector<double>>(allocator, 3);
  std::vector<Eigen::Map<Eigen::VectorXd>> x_vec = stan::io::initialize_data<std::vector<Eigen::VectorXd>>(allocator, 4, 3);
  EXPECT_EQ(4, x_vec.size());
  for (int i = 0; i < x_vec.size(); ++i) {
    EXPECT_EQ(3, x_vec[i].size());
    EXPECT_TRUE(allocator.in_stack(static_cast<void*>(x_vec[i].data())));
  }

  std::vector<Eigen::Map<Eigen::RowVectorXd>> x_rowvec = stan::io::initialize_data<std::vector<Eigen::RowVectorXd>>(allocator, 2, 3);
  EXPECT_EQ(2, x_rowvec.size());
  for (int i = 0; i < x_rowvec.size(); ++i) {
    EXPECT_EQ(3, x_rowvec[i].size());
    EXPECT_TRUE(allocator.in_stack(static_cast<void*>(x_rowvec[i].data())));
  }
  std::vector<Eigen::Map<Eigen::MatrixXd>> x_mat = stan::io::initialize_data<std::vector<Eigen::MatrixXd>>(allocator, 4, 3, 4);
  EXPECT_EQ(4, x_mat.size());
  for (int i = 0; i < x_mat.size(); ++i) {
    EXPECT_EQ(3, x_mat[i].rows());
    EXPECT_EQ(4, x_mat[i].cols());
    EXPECT_TRUE(allocator.in_stack(static_cast<void*>(x_mat[i].data())));
  }
}

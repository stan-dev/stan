#include <cmath>
#include <gtest/gtest.h>
#include <stan/maths/matrix.hpp>

using Eigen::Matrix;
using Eigen::Dynamic;

TEST(matrix_test, resize_double) {
  double x = 5;
  std::vector<unsigned int> dims;
  stan::maths::resize(x,dims);
}
TEST(matrix_test, resize_svec_double) {
  std::vector<double> y;
  std::vector<unsigned int> dims;
  EXPECT_EQ(0U, y.size());

  dims.push_back(4U);
  stan::maths::resize(y,dims);
  EXPECT_EQ(4U, y.size());

  dims[0] = 2U;
  stan::maths::resize(y,dims);
  EXPECT_EQ(2U, y.size());
}
TEST(matrix_test, resize_vec_double) {
  Matrix<double,Dynamic,1> v(2);
  std::vector<unsigned int> dims;
  EXPECT_EQ(2U, v.size());

  dims.push_back(17U);
  stan::maths::resize(v,dims);
  EXPECT_EQ(17U, v.size());

  dims[0] = 3U;
  stan::maths::resize(v,dims);
  EXPECT_EQ(3U, v.size());
}
TEST(matrix_test, resize_rvec_double) {
  Matrix<double,1,Dynamic> rv(2);
  std::vector<unsigned int> dims;
  EXPECT_EQ(2U, rv.size());

  dims.push_back(17U);
  stan::maths::resize(rv,dims);
  EXPECT_EQ(17U, rv.size());

  dims[0] = 3U;
  stan::maths::resize(rv,dims);
  EXPECT_EQ(3U, rv.size());
}
TEST(matrix_test, resize_mat_double) {
  Matrix<double,Dynamic,Dynamic> m(2,3);
  std::vector<unsigned int> dims;
  EXPECT_EQ(2U, m.rows());
  EXPECT_EQ(3U, m.cols());

  dims.push_back(7U);
  dims.push_back(17U);
  stan::maths::resize(m,dims);
  EXPECT_EQ(7U, m.rows());
  EXPECT_EQ(17U, m.cols());
}
TEST(matrix_test, resize_svec_svec_double) {
  std::vector<std::vector<double> > xx;
  EXPECT_EQ(0U,xx.size());
  std::vector<unsigned int> dims;
  dims.push_back(4U);
  dims.push_back(5U);
  stan::maths::resize(xx,dims);
  EXPECT_EQ(4U,xx.size());
  EXPECT_EQ(5U,xx[0].size());

  dims[0] = 3U;
  dims[1] = 7U;
  stan::maths::resize(xx,dims);
  EXPECT_EQ(3U,xx.size());
  EXPECT_EQ(7U,xx[1].size());  
}
TEST(matrix_test, resize_svec_v_double) {
  std::vector<Matrix<double,Dynamic,1> > xx;
  EXPECT_EQ(0U,xx.size());
  std::vector<unsigned int> dims;
  dims.push_back(4U);
  dims.push_back(5U);
  stan::maths::resize(xx,dims);
  EXPECT_EQ(4U,xx.size());
  EXPECT_EQ(5U,xx[0].size());

  dims[0] = 3U;
  dims[1] = 7U;
  stan::maths::resize(xx,dims);
  EXPECT_EQ(3U,xx.size());
  EXPECT_EQ(7U,xx[1].size());  
}
TEST(matrix_test, resize_svec_rv_double) {
  std::vector<Matrix<double,1,Dynamic> > xx;
  EXPECT_EQ(0U,xx.size());
  std::vector<unsigned int> dims;
  dims.push_back(4U);
  dims.push_back(5U);
  stan::maths::resize(xx,dims);
  EXPECT_EQ(4U,xx.size());
  EXPECT_EQ(5U,xx[0].size());

  dims[0] = 3U;
  dims[1] = 7U;
  stan::maths::resize(xx,dims);
  EXPECT_EQ(3U,xx.size());
  EXPECT_EQ(7U,xx[1].size());  
}
TEST(matrix_test, resize_svec_svec_matrix_double) {
  std::vector<std::vector<Matrix<double,Dynamic,Dynamic> > > mm;
  std::vector<unsigned int> dims;
  dims.push_back(4U);
  dims.push_back(5U);
  dims.push_back(6U);
  dims.push_back(3U);
  stan::maths::resize(mm,dims);
  EXPECT_EQ(4U,mm.size());
  EXPECT_EQ(5U,mm[0].size());
  EXPECT_EQ(6U,mm[1][2].rows());
  EXPECT_EQ(3U,mm[3][4].cols());
}

#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>
#include <stan/maths/matrix.hpp>

using Eigen::Matrix;
using Eigen::Dynamic;

typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> matrix_d;
typedef Eigen::Matrix<double,Eigen::Dynamic,1> vector_d;
typedef Eigen::Matrix<double,1,Eigen::Dynamic> row_vector_d;

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
TEST(matrix_test,add_v_exception) {
  vector_d d1, d2;

  d1.resize(3);
  d2.resize(3);
  EXPECT_NO_THROW(stan::maths::add (d1, d2));

  d1.resize(0);
  d2.resize(0);
  EXPECT_NO_THROW(stan::maths::add (d1, d2));

  d1.resize(2);
  d2.resize(3);
  EXPECT_THROW(stan::maths::add (d1, d2), std::invalid_argument);
}
TEST(matrix_test,add_rv_exception) {
  row_vector_d d1, d2;

  d1.resize(3);
  d2.resize(3);
  EXPECT_NO_THROW(stan::maths::add (d1, d2));

  d1.resize(0);
  d2.resize(0);
  EXPECT_NO_THROW(stan::maths::add (d1, d2));

  d1.resize(2);
  d2.resize(3);
  EXPECT_THROW(stan::maths::add (d1, d2), std::invalid_argument);
}
TEST(matrix_test,add_m_exception) {
  matrix_d d1, d2;

  d1.resize(2,3);
  d2.resize(2,3);
  EXPECT_NO_THROW(stan::maths::add (d1, d2));

  d1.resize(0,0);
  d2.resize(0,0);
  EXPECT_NO_THROW(stan::maths::add (d1, d2));

  d1.resize(2,3);
  d2.resize(3,3);
  EXPECT_THROW(stan::maths::add (d1, d2), std::invalid_argument);
}

TEST(matrix_test,subtract_v_exception) {
  vector_d d1, d2;

  d1.resize(3);
  d2.resize(3);
  EXPECT_NO_THROW(stan::maths::subtract (d1, d2));

  d1.resize(0);
  d2.resize(0);
  EXPECT_NO_THROW(stan::maths::subtract (d1, d2));

  d1.resize(2);
  d2.resize(3);
  EXPECT_THROW(stan::maths::subtract (d1, d2), std::invalid_argument);
}
TEST(matrix_test,subtract_rv_exception) {
  row_vector_d d1, d2;

  d1.resize(3);
  d2.resize(3);
  EXPECT_NO_THROW(stan::maths::subtract (d1, d2));

  d1.resize(0);
  d2.resize(0);
  EXPECT_NO_THROW(stan::maths::subtract (d1, d2));

  d1.resize(2);
  d2.resize(3);
  EXPECT_THROW(stan::maths::subtract (d1, d2), std::invalid_argument);
}
TEST(matrix_test,subtract_m_exception) {
  matrix_d d1, d2;

  d1.resize(2,3);
  d2.resize(2,3);
  EXPECT_NO_THROW(stan::maths::subtract (d1, d2));

  d1.resize(0,0);
  d2.resize(0,0);
  EXPECT_NO_THROW(stan::maths::subtract (d1, d2));

  d1.resize(2,3);
  d2.resize(3,3);
  EXPECT_THROW(stan::maths::subtract (d1, d2), std::invalid_argument);
}

TEST(matrix_test,multiply_rv_v_exception) {
  row_vector_d rv;
  vector_d v;
  
  rv.resize(3);
  v.resize(3);
  EXPECT_NO_THROW(stan::maths::multiply (rv, v));

  rv.resize(0);
  v.resize(0);
  EXPECT_NO_THROW(stan::maths::multiply (rv, v));

  rv.resize(2);
  v.resize(3);
  EXPECT_THROW(stan::maths::multiply (rv, v), std::invalid_argument);
}
TEST(matrix_test,multiply_m_v_exception) {
  matrix_d m;
  vector_d v;
  
  m.resize(3, 5);
  v.resize(5);
  EXPECT_NO_THROW(stan::maths::multiply (m, v));

  m.resize(3, 0);
  v.resize(0);
  EXPECT_NO_THROW(stan::maths::multiply (m, v));

  m.resize(2, 3);
  v.resize(2);
  EXPECT_THROW(stan::maths::multiply (m, v), std::invalid_argument);  
}
TEST(matrix_test,multiply_rv_m_exception) {
  row_vector_d rv;
  matrix_d m;
    
  rv.resize(3);
  m.resize(3, 5);
  EXPECT_NO_THROW(stan::maths::multiply (rv, m));

  rv.resize(0);
  m.resize(0, 3);
  EXPECT_NO_THROW(stan::maths::multiply (rv, m));

  rv.resize(3);
  m.resize(2, 3);
  EXPECT_THROW(stan::maths::multiply (rv, m), std::invalid_argument);
}
TEST(matrix_test,multiply_m_m_exception) {
  matrix_d m1, m2;
  
  m1.resize(1, 3);
  m2.resize(3, 5);
  EXPECT_NO_THROW(stan::maths::multiply (m1, m2));

  
  m1.resize(2, 0);
  m2.resize(0, 3);
  EXPECT_NO_THROW(stan::maths::multiply (m1, m2));

  m1.resize(4, 3);
  m2.resize(2, 3);
  EXPECT_THROW(stan::maths::multiply (m1, m2), std::invalid_argument);
}
TEST(matrix_test,cholesky_decompose_exception) {
  matrix_d m;
  
  m.resize(2,2);
  EXPECT_NO_THROW(stan::maths::cholesky_decompose(m));

  m.resize(0, 0);
  EXPECT_NO_THROW(stan::maths::cholesky_decompose(m));
  
  m.resize(2, 3);
  EXPECT_THROW(stan::maths::cholesky_decompose(m), std::invalid_argument);
}
TEST(matrix_test,std_vector_sum_double) {
  std::vector<double> x(3);
  EXPECT_FLOAT_EQ(0.0,stan::maths::sum(x));
  x[0] = 1.0;
  x[1] = 2.0;
  x[2] = 3.0;
  EXPECT_FLOAT_EQ(6.0,stan::maths::sum(x));
}
TEST(matrix_test,std_vector_sum_int) {
  std::vector<int> x(3);
  EXPECT_EQ(0,stan::maths::sum(x));
  x[0] = 1;
  x[1] = 2;
  x[2] = 3;
  EXPECT_EQ(6,stan::maths::sum(x));
}



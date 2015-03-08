#include <stan/math/prim/mat/fun/min.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, min) {
  using stan::math::min;
  std::vector<int> n;
  EXPECT_THROW(min(n),std::domain_error);
  n.push_back(1);
  EXPECT_EQ(1,min(n));
  n.push_back(2);
  EXPECT_EQ(1,min(n));
  n.push_back(0);
  EXPECT_EQ(0,min(n));
  
  std::vector<double> x;
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(),min(x));
  x.push_back(1.0);
  EXPECT_FLOAT_EQ(1.0,min(x));
  x.push_back(0.0);
  EXPECT_FLOAT_EQ(0.0,min(x));
  x.push_back(2.0);
  EXPECT_FLOAT_EQ(0.0,min(x));
  x.push_back(-10.0);
  EXPECT_FLOAT_EQ(-10.0,min(x));

  stan::math::vector_d v;
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(),min(v));
  v = stan::math::vector_d(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(1.0,min(v));
  v = stan::math::vector_d(2);
  v << 1.0, 0.0;
  EXPECT_FLOAT_EQ(0.0,min(v));
  v = stan::math::vector_d(3);
  v << 1.0, 0.0, 2.0;
  EXPECT_FLOAT_EQ(0.0,min(v));
  v = stan::math::vector_d(4);
  v << 1.0, 0.0, 2.0, -10.0;
  EXPECT_FLOAT_EQ(-10.0,min(v));

  stan::math::row_vector_d rv;
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(),min(rv));
  rv = stan::math::row_vector_d(1);
  rv << 1.0;
  EXPECT_FLOAT_EQ(1.0,min(rv));
  rv = stan::math::row_vector_d(2);
  rv << 1.0, 0.0;
  EXPECT_FLOAT_EQ(0.0,min(rv));
  rv = stan::math::row_vector_d(3);
  rv << 1.0, 0.0, 2.0;
  EXPECT_FLOAT_EQ(0.0,min(rv));
  rv = stan::math::row_vector_d(4);
  rv << 1.0, 0.0, 2.0, -10.0;
  EXPECT_FLOAT_EQ(-10.0,min(rv));

  stan::math::matrix_d m;
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(),min(m));
  m = stan::math::matrix_d(1,1);
  m << 1.0;
  EXPECT_FLOAT_EQ(1.0,min(m));
  m = stan::math::matrix_d(2,2);
  m << 1.0, 0.0, 2.0, -10.0;
  EXPECT_FLOAT_EQ(-10.0,min(m));
}

TEST(MathMatrix,min_exception) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using std::numeric_limits;
  Matrix<double,Dynamic,Dynamic> m;
  Matrix<double,Dynamic,1> v;
  Matrix<double,1,Dynamic> rv;
  EXPECT_EQ(numeric_limits<double>::infinity(),
            stan::math::min(m));
  EXPECT_EQ(numeric_limits<double>::infinity(),
            stan::math::min(v));
  EXPECT_EQ(numeric_limits<double>::infinity(),
            stan::math::min(rv));

  Matrix<double,Dynamic,Dynamic> m_nz(2,3);
  Matrix<double,Dynamic,1> v_nz(2);
  Matrix<double,1,Dynamic> rv_nz(3);
  EXPECT_NO_THROW(stan::math::min(m_nz));
  EXPECT_NO_THROW(stan::math::min(v_nz));
  EXPECT_NO_THROW(stan::math::min(rv_nz));

}


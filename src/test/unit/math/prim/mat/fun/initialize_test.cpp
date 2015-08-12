#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/initialize.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix,initialize) {
  // 2nd template
  using stan::math::initialize;
  double x;
  double y = 10;
  initialize(x,y);         // template 2
  EXPECT_FLOAT_EQ(y,x);

  int z = 5;
  initialize(y,z);        // template 2
  EXPECT_FLOAT_EQ(z,y);
}

  
TEST(MathMatrix, initMatrix) {
  using stan::math::initialize;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  Matrix<double,Dynamic,Dynamic> m(3,2);
  initialize(m, 13.2);      // template 3, 2
  for (int i = 0; i < m.size(); ++i)
    EXPECT_FLOAT_EQ(m(i), 13.2);

  Matrix<double,Dynamic,1> v(3);
  initialize(v,2);           // template 3, 2
  for (int i = 0; i < v.size(); ++i)
    EXPECT_FLOAT_EQ(v(i), 2);

  Matrix<double,1,Dynamic> rv(3);
  initialize(rv,12);         // template 3, 2
  for (int i = 0; i < v.size(); ++i)
    EXPECT_FLOAT_EQ(rv(i), 12);
}

TEST(MathMatrix, initStdVector) {
  using std::vector;
  using stan::math::initialize;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  vector<double> x(3);
  initialize(x,2.2);
  for (size_t i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(2.2, x[i]);  // template 4,2

  vector<Matrix<double,Dynamic,Dynamic> > z(4, Matrix<double,Dynamic,Dynamic>(3,2));
  initialize(z, 3.7);
  for (size_t i = 0; i < 4; ++i)
    for (int m = 0; m < 3; ++m)
      for (int n = 0; n < 2; ++n)
        EXPECT_FLOAT_EQ(3.7, z[i](m,n));
}

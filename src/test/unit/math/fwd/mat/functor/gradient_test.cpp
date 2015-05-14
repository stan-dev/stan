#include <stdexcept>
#include <gtest/gtest.h>
#include <stan/math/fwd/mat/functor/gradient.hpp>
#include <stan/math/fwd/core.hpp>

using Eigen::Matrix;
using Eigen::Dynamic;

// fun1(x,y) = (x^2 * y) + (3 * y^2)
struct fun1 {
  template <typename T>
  inline
  T operator()(const Matrix<T,Dynamic,1>& x) const {
    return x(0) * x(0) * x(1)
      + 3.0 * x(1) * x(1); 
  }
};

TEST(AgradAutoDiff,gradient) {
    
  
  fun1 f;
  Matrix<double,Dynamic,1> x(2);
  x << 5, 7;

  double fx2(0);
  Matrix<double,Dynamic,1> grad_fx2;
  stan::math::gradient<double>(f,x,fx2,grad_fx2);
  EXPECT_FLOAT_EQ(5 * 5 * 7 + 3 * 7 * 7, fx2);
  EXPECT_EQ(2,grad_fx2.size());
  EXPECT_FLOAT_EQ(2 * x(0) * x(1), grad_fx2(0));
  EXPECT_FLOAT_EQ(x(0) * x(0) + 3 * 2 * x(1), grad_fx2(1));
}

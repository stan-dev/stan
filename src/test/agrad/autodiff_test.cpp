#include <gtest/gtest.h>
#include <stan/agrad/autodiff.hpp>

// fun1(x,y) = (x^2 * y) + (3 * y^2)
struct fun1 {
  template <typename T>
  inline
  T operator()(const Eigen::Matrix<T,Eigen::Dynamic,1>& x) const {
    return x[0] * x[0] * x[1] 
      + 3.0 * x[1] * x[1]; 
  }
};

TEST(AgradAutoDiff,gradient) {
  fun1 f;
  Eigen::Matrix<double,Eigen::Dynamic,1> x(2);
  x << 5, 7;
  double fx;
  Eigen::Matrix<double,Eigen::Dynamic,1> grad_fx;
  stan::agrad::gradient(f,x,fx,grad_fx);
  EXPECT_FLOAT_EQ(5 * 5 * 7 + 3 * 7 * 7, fx);
  EXPECT_EQ(2,grad_fx.size());
  EXPECT_FLOAT_EQ(2 * x[0] * x[1], grad_fx[0]);
  EXPECT_FLOAT_EQ(x[0] * x[0] + 3 * 2 * x[1], grad_fx[1]);
}

TEST(AgradAutoDiff,hessianTimesVector) {
  using stan::agrad::hessian_times_vector;
  using Eigen::Matrix;  using Eigen::Dynamic;

  fun1 f;
  
  Matrix<double,Dynamic,1> x(2);
  x << 2, -3;
  
  Matrix<double,Dynamic,1> v(2);
  v << 8, 5;

  Matrix<double,Dynamic,1> Hv;
  double fx;
  stan::agrad::hessian_times_vector(f,x,v,fx,Hv);

  EXPECT_FLOAT_EQ(2 * 2 * -3 + 3.0 * -3 * -3, fx);

  EXPECT_EQ(2,Hv.size());
  EXPECT_FLOAT_EQ(2 * x(1) * v(0) + 2 * x(0) * v(1), Hv(0));
  EXPECT_FLOAT_EQ(2 * x(0) * v(0) + 6 * v(1), Hv(1));
 
}

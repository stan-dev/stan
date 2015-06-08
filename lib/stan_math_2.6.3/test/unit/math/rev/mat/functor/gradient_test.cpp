#include <stdexcept>
#include <gtest/gtest.h>
#include <stan/math/rev/mat/functor/gradient.hpp>
#include <stan/math/rev/core.hpp>

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
  double fx;
  Matrix<double,Dynamic,1> grad_fx;
  stan::math::gradient(f,x,fx,grad_fx);
  EXPECT_FLOAT_EQ(5 * 5 * 7 + 3 * 7 * 7, fx);
  EXPECT_EQ(2,grad_fx.size());
  EXPECT_FLOAT_EQ(2 * x(0) * x(1), grad_fx(0));
  EXPECT_FLOAT_EQ(x(0) * x(0) + 3 * 2 * x(1), grad_fx(1));
}

stan::math::var 
sum_and_throw(const Matrix<stan::math::var,Dynamic,1>& x) {
  stan::math::var y = 0;
  for (int i = 0; i < x.size(); ++i)
    y += x(i);
  throw std::domain_error("fooey");
  return y;
}

TEST(AgradAutoDiff, RecoverMemory) {
  using Eigen::VectorXd;
  for (int i = 0; i < 100000; ++i) {
    try {
      VectorXd x(5);
      x << 1, 2, 3, 4, 5;
      double fx;
      VectorXd grad_fx;
      stan::math::gradient(sum_and_throw,x,fx,grad_fx);
    } catch (const std::domain_error& e) {
      // ignore me
    }
  }
  // depends on starting allocation of 65K not being exceeded
  // without recovery_memory in autodiff::apply_recover(), takes 67M 
  EXPECT_TRUE(stan::math::ChainableStack::memalloc_.bytes_allocated() < 100000);
}  

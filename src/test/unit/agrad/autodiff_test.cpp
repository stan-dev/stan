#include <stdexcept>
#include <gtest/gtest.h>
#include <stan/agrad/autodiff.hpp>

struct fun0 {
  template <typename T>
  inline
  T operator()(const T& x) const {
    return 5.0 * x * x * x;
  }
};

// fun1(x,y) = (x^2 * y) + (3 * y^2)
struct fun1 {
  template <typename T>
  inline
  T operator()(const Eigen::Matrix<T,Eigen::Dynamic,1>& x) const {
    return x(0) * x(0) * x(1)
      + 3.0 * x(1) * x(1); 
  }
};

struct fun2 {
  template <typename T>
  inline
  Eigen::Matrix<T,Eigen::Dynamic,1>
  operator()(const Eigen::Matrix<T,Eigen::Dynamic,1>& x) const {
    Eigen::Matrix<T,Eigen::Dynamic,1> z(2);
    z << x(0) + x(0), 3 * x(0) * x(1);
    return z;
  }
};

TEST(AgradAutoDiff,derivative) {
  fun0 f;
  double x = 7;
  double fx;
  double d;
  stan::agrad::derivative(f,x,fx,d);
  EXPECT_FLOAT_EQ(fx, 5 * 7 * 7 * 7);
  EXPECT_FLOAT_EQ(d, 5 * 3 * 7 * 7);
}

TEST(AgradAutoDiff,partialDerivative) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  fun1 f;
  Matrix<double,Dynamic,1> x(2);
  x << 5, 7;
  
  double fx;
  double d;
  stan::agrad::partial_derivative(f,x,0,fx,d);
  EXPECT_FLOAT_EQ(5 * 5 * 7 + 3 * 7 * 7,fx);
  EXPECT_FLOAT_EQ(2 * 5 * 7, d);

  double fx2;
  double d2;
  stan::agrad::partial_derivative(f,x,1,fx2,d2);
  EXPECT_FLOAT_EQ(5 * 5 * 7 + 3 * 7 * 7,fx);
  EXPECT_FLOAT_EQ(5 * 5 + 3 * 2 * 7,d2);
}

TEST(AgradAutoDiff,gradient) {
  using Eigen::Matrix;  
  using Eigen::Dynamic;
  fun1 f;
  Matrix<double,Dynamic,1> x(2);
  x << 5, 7;
  double fx;
  Matrix<double,Dynamic,1> grad_fx;
  stan::agrad::gradient(f,x,fx,grad_fx);
  EXPECT_FLOAT_EQ(5 * 5 * 7 + 3 * 7 * 7, fx);
  EXPECT_EQ(2,grad_fx.size());
  EXPECT_FLOAT_EQ(2 * x(0) * x(1), grad_fx(0));
  EXPECT_FLOAT_EQ(x(0) * x(0) + 3 * 2 * x(1), grad_fx(1));

  double fx2(0);
  Matrix<double,Dynamic,1> grad_fx2;
  stan::agrad::gradient<double>(f,x,fx2,grad_fx2);
  EXPECT_FLOAT_EQ(5 * 5 * 7 + 3 * 7 * 7, fx2);
  EXPECT_EQ(2,grad_fx2.size());
  EXPECT_FLOAT_EQ(2 * x(0) * x(1), grad_fx2(0));
  EXPECT_FLOAT_EQ(x(0) * x(0) + 3 * 2 * x(1), grad_fx2(1));
}


TEST(AgradAutoDiff,gradientDotVector) {
  using Eigen::Matrix;  using Eigen::Dynamic;
  using stan::agrad::var;
  fun1 f;
  Matrix<double,Dynamic,1> x(2);
  x << 5, 7;
  Matrix<double,Dynamic,1> v(2);
  v << 11, 13;
  double fx;
  double grad_fx_dot_v;
  stan::agrad::gradient_dot_vector(f,x,v,fx,grad_fx_dot_v);
  
  double fx_expected;
  Matrix<double,Dynamic,1> grad_fx;
  stan::agrad::gradient(f,x,fx_expected,grad_fx);
  double grad_fx_dot_v_expected = grad_fx.dot(v);
  
  EXPECT_FLOAT_EQ(grad_fx_dot_v_expected, grad_fx_dot_v);
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
TEST(AgradAutoDiff,jacobian) {
  using Eigen::Matrix;  using Eigen::Dynamic;
  using stan::agrad::jacobian;

  fun2 f;
  Matrix<double,Dynamic,1> x(2);
  x << 2, -3;
  
  Matrix<double,Dynamic,1> fx;
  Matrix<double,Dynamic,Dynamic> J;
  jacobian(f,x,fx,J);
  
  EXPECT_EQ(2,fx.size());
  EXPECT_FLOAT_EQ(2 * 2, fx(0));
  EXPECT_FLOAT_EQ(3 * 2 * -3, fx(1));
  
  EXPECT_FLOAT_EQ(2, J(0,0));
  EXPECT_FLOAT_EQ(-9, J(0,1));
  EXPECT_FLOAT_EQ(0, J(1,0));
  EXPECT_FLOAT_EQ(6, J(1,1));
  

  Matrix<double,Dynamic,1> fx_rev;
  Matrix<double,Dynamic,Dynamic> J_rev;
  jacobian<double>(f,x,fx_rev,J_rev);

  EXPECT_EQ(2,fx_rev.size());
  EXPECT_FLOAT_EQ(2 * 2, fx_rev(0));
  EXPECT_FLOAT_EQ(3 * 2 * -3, fx_rev(1));
  
  EXPECT_FLOAT_EQ(2, J_rev(0,0));
  EXPECT_FLOAT_EQ(-9, J_rev(0,1));
  EXPECT_FLOAT_EQ(0, J_rev(1,0));
  EXPECT_FLOAT_EQ(6, J_rev(1,1));
}

TEST(AgradAutodiff,hessian) {
  using Eigen::Matrix;  using Eigen::Dynamic;
  fun1 f;
  Matrix<double,Dynamic,1> x(2);
  x << 5, 7;
  double fx(0);
  Matrix<double,Dynamic,1> grad;
  Matrix<double,Dynamic,Dynamic> H;
  stan::agrad::hessian(f,x,fx,grad,H);

  // x^2 * y + 3 * y^2
  EXPECT_FLOAT_EQ(5 * 5 * 7 + 3 * 7  * 7, fx);

  EXPECT_FLOAT_EQ(2,grad.size());
  EXPECT_FLOAT_EQ(2 * x(0) * x(1), grad(0));
  EXPECT_FLOAT_EQ(x(0) * x(0) + 3 * 2 * x(1), grad(1));

  EXPECT_EQ(2,H.rows());
  EXPECT_EQ(2,H.cols());
  EXPECT_FLOAT_EQ(2 * 7, H(0,0));
  EXPECT_FLOAT_EQ(2 * 5, H(0,1));
  EXPECT_FLOAT_EQ(2 * 5, H(1,0));
  EXPECT_FLOAT_EQ(2 * 3, H(1,1));

  double fx2;
  Matrix<double,Dynamic,1> grad2;
  Matrix<double,Dynamic,Dynamic> H2;
  stan::agrad::hessian<double>(f,x,fx2,grad2,H2);

  EXPECT_FLOAT_EQ(5 * 5 * 7 + 3 * 7  * 7, fx2);

  EXPECT_FLOAT_EQ(2,grad2.size());
  EXPECT_FLOAT_EQ(2 * x(0) * x(1), grad2(0));
  EXPECT_FLOAT_EQ(x(0) * x(0) + 3 * 2 * x(1), grad2(1));

  EXPECT_EQ(2,H2.rows());
  EXPECT_EQ(2,H2.cols());
  EXPECT_FLOAT_EQ(2 * 7, H2(0,0));
  EXPECT_FLOAT_EQ(2 * 5, H2(0,1));
  EXPECT_FLOAT_EQ(2 * 5, H2(1,0));
  EXPECT_FLOAT_EQ(2 * 3, H2(1,1));

}
  
TEST(AgradAutodiff,GradientTraceMatrixTimesHessian) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  Matrix<double,Dynamic,Dynamic> M(2,2);
  M << 11, 13, 17, 23;
  fun1 f;
  Matrix<double,Dynamic,1> x(2);
  x << 5, 7;
  Matrix<double,Dynamic,1> grad_tr_MH;
  stan::agrad::grad_tr_mat_times_hessian(f,x,M,grad_tr_MH);

  EXPECT_EQ(2,grad_tr_MH.size());
  EXPECT_FLOAT_EQ(60,grad_tr_MH(0));
  EXPECT_FLOAT_EQ(22,grad_tr_MH(1));
}

stan::agrad::var 
sum_and_throw(const Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,1>& x) {
  stan::agrad::var y = 0;
  for (int i = 0; i < x.size(); ++i)
    y += x(i);
  throw std::domain_error("fooey");
  return y;
}

TEST(AgradAutodiff, RecoverMemory) {
  using Eigen::VectorXd;
  for (int i = 0; i < 100000; ++i) {
    try {
      VectorXd x(5);
      x << 1, 2, 3, 4, 5;
      double fx;
      VectorXd grad_fx;
      gradient(sum_and_throw,x,fx,grad_fx);
    } catch (const std::domain_error& e) {
      // ignore me
    }
  }
  // depends on starting allocation of 65K not being exceeded
  // without recovery_memory in autodiff::apply_recover(), takes 67M 
  EXPECT_TRUE(stan::agrad::ChainableStack::memalloc_.bytes_allocated() < 100000);
}  

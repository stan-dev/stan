#include <gtest/gtest.h>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/math/matrix/value_of_rec.hpp>
#include <test/unit/agrad/util.hpp>
#include <stan/prob/distributions/univariate/continuous/normal.hpp>
#include <stan/agrad/finite_diff.hpp>

struct fun0 {
  template <typename T>
  inline
  T operator()(const T& x) const {
    return 5.0 * x * x * x;
  }
};

struct norm_functor {
  template <typename T>
  inline
  T operator()(const Eigen::Matrix<T,Eigen::Dynamic,1>& inp_vec) const {
    return stan::prob::normal_log(inp_vec(0), inp_vec(1), inp_vec(2)); 
  }
};

TEST(AgradFiniteDiff,gradient) {
  using Eigen::Matrix;
  using Eigen::Dynamic;

  third_order_mixed f;
  norm_functor n;
  Matrix<double,Dynamic,1> x(3);
  Matrix<double,Dynamic,1> norm_vec(3);

  x << 0.5, 0.3, 0.1;
  norm_vec << 0.5, 0.3, 0.7;

  double fx;
  double fin_diff_fx;

  Matrix<double,Dynamic,1> grad_fx;
  Matrix<double,Dynamic,1> finite_diff_fx;

  Matrix<double,Dynamic,1> grad_norm_fx;
  Matrix<double,Dynamic,1> finite_diff_norm_fx;

  stan::agrad::gradient(f,x,fx,grad_fx);
  stan::agrad::finite_diff_gradient(f,x,fin_diff_fx,finite_diff_fx);

  stan::agrad::gradient(n,norm_vec,fx,grad_norm_fx);
  stan::agrad::finite_diff_gradient(n,norm_vec,fx,finite_diff_norm_fx);

  for (size_type i = 0; i < 3; ++i){
    EXPECT_NEAR(grad_norm_fx(i), finite_diff_norm_fx(i), 1e-12);
    EXPECT_NEAR(grad_fx(i), finite_diff_fx(i), 1e-12);
  }

}

TEST(AgradFiniteDiff,hessian) {
  using Eigen::Matrix;  
  using Eigen::Dynamic;
  third_order_mixed f;
  norm_functor n;
  Matrix<double,Dynamic,1> x(3);
  Matrix<double,Dynamic,1> norm_vec(3);

  Matrix<double,3,3> third_order_mixed_hess_result;

  x << 0.5, 0.7, 0.4;

  third_order_mixed_hess_result = third_order_mixed_hess(x);

  norm_vec << 0.5, 0.3, 0.7;

  double fx(0);
  double finite_diff_x(0);
  Matrix<double,Dynamic,1> grad;
  Matrix<double,Dynamic,Dynamic> H;
  Matrix<double,Dynamic,Dynamic> finite_diff_H;
  Matrix<double,Dynamic,Dynamic> finite_diff_H_auto;

  Matrix<double,Dynamic,1> norm_grad;
  Matrix<double,Dynamic,Dynamic> norm_H;
  Matrix<double,Dynamic,Dynamic> finite_diff_norm_H;
  Matrix<double,Dynamic,Dynamic> finite_diff_norm_H_auto;

  stan::agrad::hessian(f,x,fx,grad,H);
  stan::agrad::finite_diff_hessian(f,x,finite_diff_x,finite_diff_H);
  stan::agrad::finite_diff_hessian_auto(f,x,finite_diff_x,finite_diff_H_auto);

  stan::agrad::hessian(n,norm_vec,fx,norm_grad,norm_H);
  stan::agrad::finite_diff_hessian(n,norm_vec,finite_diff_x,finite_diff_norm_H);
  stan::agrad::finite_diff_hessian(n,norm_vec,finite_diff_x,finite_diff_norm_H_auto);

  Matrix<double,3,3> norm_hess_an = norm_hess(norm_vec);

  for (size_type i = 0; i < 3; ++i)
    for (size_type j = 0; j < 3; ++j){
      EXPECT_NEAR(H(i,j), finite_diff_H(i,j), 1e-09);
      EXPECT_NEAR(third_order_mixed_hess_result(i,j), finite_diff_H(i,j),1e-09) << "i: " << i << " j: " << j;
      EXPECT_NEAR(third_order_mixed_hess_result(i,j), finite_diff_H_auto(i,j),1e-09) << "i: " << i << " j: " << j;
    }

  for (size_type i = 0; i < 3; ++i)
    for (size_type j = 0; j < 3; ++j){
      EXPECT_NEAR(norm_H(i,j), finite_diff_norm_H(i,j), 1e-09) << "i: " << i << " j: " << j;
      EXPECT_NEAR(norm_hess_an(i,j), finite_diff_norm_H(i,j), 1e-09) << "i: " << i << " j: " << j;
      EXPECT_NEAR(norm_hess_an(i,j), finite_diff_norm_H_auto(i,j), 1e-09) << "i: " << i << " j: " << j;
    }

}

// FIXME Add normal grad_hessian tests, just as in 
// autodiff_test.cpp and also change naming in this test,
// it is terrible
TEST(AgradFiniteDiff,grad_hessian) {
  using Eigen::Matrix;  
  using Eigen::Dynamic;
  norm_functor n;
  third_order_mixed fun;
  Matrix<double,Dynamic,1> norm_vec(3);
  norm_vec << 0.5, 0.3, 0.7;

  Matrix<double,Dynamic,1> fun_vec(3);
  fun_vec << 1.6666667e-01, 0.5, 1;

  double fx(0);
  double finite_diff_x(0);

  Matrix<double,Dynamic,Dynamic> norm_H_ad;
  std::vector<Matrix<double,Dynamic,Dynamic> > ad_grad_H;

  Matrix<double,Dynamic,1> norm_grad;
  Matrix<double,Dynamic,Dynamic> norm_H;
  std::vector<Matrix<double,Dynamic,Dynamic> > finite_diff_norm_H;
  std::vector<Matrix<double,Dynamic,Dynamic> > an_norm_H = 
    third_order_mixed_grad_hess(fun_vec);

  stan::agrad::grad_hessian(fun,fun_vec,fx,norm_H_ad,ad_grad_H);
  stan::agrad::finite_diff_grad_hessian_auto(fun,fun_vec,finite_diff_x,finite_diff_norm_H);


  for (size_t i = 0; i < 3; ++i){
    for (size_type j = 0; j < 3; ++j){
      for (size_type k = 0; k < 3; ++k){
        EXPECT_NEAR(ad_grad_H[i](j,k),finite_diff_norm_H[i](j,k),1e-10) << " i: " << i << " j: " << j << " k: " << k; 
        EXPECT_NEAR(an_norm_H[i](j,k),finite_diff_norm_H[i](j,k),1e-10) << " i: " << i << " j: " << j << " k: " << k; 
        EXPECT_NEAR(ad_grad_H[i](j,k),an_norm_H[i](j,k),1e-10) << " i: " << i << " j: " << j << " k: " << k; 
      }
    }
  }


}

#include <stdexcept>
#include <gtest/gtest.h>
#include <stan/math/mix/mat/functor/derivative.hpp>
#include <stan/math/mix/mat/functor/grad_hessian.hpp>
#include <stan/math/mix/mat/functor/grad_tr_mat_times_hessian.hpp>
#include <stan/math/rev/mat/functor/gradient.hpp>
#include <stan/math/mix/mat/functor/gradient_dot_vector.hpp>
#include <stan/math/mix/mat/functor/hessian.hpp>
#include <stan/math/mix/mat/functor/hessian_times_vector.hpp>
#include <stan/math/rev/mat/functor/jacobian.hpp>
#include <stan/math/fwd/mat/functor/jacobian.hpp>
#include <stan/math/mix/mat/functor/partial_derivative.hpp>
#include <stan/math/prim/scal/prob/normal_log.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/fwd/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>

using Eigen::Matrix;
using Eigen::Dynamic;

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
  T operator()(const Matrix<T,Dynamic,1>& x) const {
    return x(0) * x(0) * x(1)
      + 3.0 * x(1) * x(1); 
  }
};

// fun2: R^2 --> R^2 | (x,y) --> [(x + x), (3 * x * y)] 
struct fun2 {
  template <typename T>
  inline
  Matrix<T,Dynamic,1>
  operator()(const Matrix<T,Dynamic,1>& x) const {
    Matrix<T,Dynamic,1> z(2);
    z << x(0) + x(0), 3 * x(0) * x(1);
    return z;
  }
};

struct norm_functor {
  template <typename T>
  inline
  T operator()(const Eigen::Matrix<T,Eigen::Dynamic,1>& inp_vec) const {
    return stan::math::normal_log(inp_vec(0), inp_vec(1), inp_vec(2)); 
  }
};

TEST(AgradAutoDiff,derivative) {
  fun0 f;
  double x = 7;
  double fx;
  double d;
  stan::math::derivative(f,x,fx,d);
  EXPECT_FLOAT_EQ(fx, 5 * 7 * 7 * 7);
  EXPECT_FLOAT_EQ(d, 5 * 3 * 7 * 7);
}

TEST(AgradAutoDiff,partialDerivative) {
  
  
  fun1 f;
  Matrix<double,Dynamic,1> x(2);
  x << 5, 7;
  
  double fx;
  double d;
  stan::math::partial_derivative(f,x,0,fx,d);
  EXPECT_FLOAT_EQ(5 * 5 * 7 + 3 * 7 * 7,fx);
  EXPECT_FLOAT_EQ(2 * 5 * 7, d);

  double fx2;
  double d2;
  stan::math::partial_derivative(f,x,1,fx2,d2);
  EXPECT_FLOAT_EQ(5 * 5 * 7 + 3 * 7 * 7,fx);
  EXPECT_FLOAT_EQ(5 * 5 + 3 * 2 * 7,d2);
}

TEST(AgradAutoDiff,gradientDotVector) {
    
  using stan::math::var;
  fun1 f;
  Matrix<double,Dynamic,1> x(2);
  x << 5, 7;
  Matrix<double,Dynamic,1> v(2);
  v << 11, 13;
  double fx;
  double grad_fx_dot_v;
  stan::math::gradient_dot_vector(f,x,v,fx,grad_fx_dot_v);
  
  double fx_expected;
  Matrix<double,Dynamic,1> grad_fx;
  stan::math::gradient(f,x,fx_expected,grad_fx);
  double grad_fx_dot_v_expected = grad_fx.dot(v);
  
  EXPECT_FLOAT_EQ(grad_fx_dot_v_expected, grad_fx_dot_v);
}
TEST(AgradAutoDiff,hessianTimesVector) {
  using stan::math::hessian_times_vector;
    

  fun1 f;
  
  Matrix<double,Dynamic,1> x(2);
  x << 2, -3;
  
  Matrix<double,Dynamic,1> v(2);
  v << 8, 5;

  Matrix<double,Dynamic,1> Hv;
  double fx;
  stan::math::hessian_times_vector(f,x,v,fx,Hv);

  EXPECT_FLOAT_EQ(2 * 2 * -3 + 3.0 * -3 * -3, fx);

  EXPECT_EQ(2,Hv.size());
  EXPECT_FLOAT_EQ(2 * x(1) * v(0) + 2 * x(0) * v(1), Hv(0));
  EXPECT_FLOAT_EQ(2 * x(0) * v(0) + 6 * v(1), Hv(1));
}
TEST(AgradAutoDiff,jacobian) {
    
  using stan::math::jacobian;

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

TEST(AgradAutoDiff,hessian) {
    
  fun1 f;
  Matrix<double,Dynamic,1> x(2);
  x << 5, 7;
  double fx(0);
  Matrix<double,Dynamic,1> grad;
  Matrix<double,Dynamic,Dynamic> H;
  stan::math::hessian(f,x,fx,grad,H);

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
  stan::math::hessian<double>(f,x,fx2,grad2,H2);

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
  
TEST(AgradAutoDiff,GradientTraceMatrixTimesHessian) {
  Matrix<double,Dynamic,Dynamic> M(2,2);
  M << 11, 13, 17, 23;
  fun1 f;
  Matrix<double,Dynamic,1> x(2);
  x << 5, 7;
  Matrix<double,Dynamic,1> grad_tr_MH;
  stan::math::grad_tr_mat_times_hessian(f,x,M,grad_tr_MH);

  EXPECT_EQ(2,grad_tr_MH.size());
  EXPECT_FLOAT_EQ(60,grad_tr_MH(0));
  EXPECT_FLOAT_EQ(22,grad_tr_MH(1));
}

TEST(AgradAutoDiff,GradientHessian){
  norm_functor log_normal_density;
  third_order_mixed mixed_third_poly;

  Matrix<double,Dynamic,1> normal_eval_vec(3); 
  Matrix<double,Dynamic,1> poly_eval_vec(3); 

  normal_eval_vec << 0.7, 0.5, 0.9; 
  poly_eval_vec << 1.5, 7.1, 3.1; 

  double normal_eval_agrad;
  double poly_eval_agrad;

  double normal_eval_agrad_hessian;
  double poly_eval_agrad_hessian;

  double normal_eval_analytic;
  double poly_eval_analytic;

  Matrix<double,Dynamic,Dynamic> norm_hess_agrad;
  Matrix<double,Dynamic,Dynamic> poly_hess_agrad;

  Matrix<double,Dynamic,Dynamic> norm_hess_agrad_hessian;
  Matrix<double,Dynamic,Dynamic> poly_hess_agrad_hessian;

  Matrix<double,Dynamic,1> norm_grad_agrad_hessian;
  Matrix<double,Dynamic,1> poly_grad_agrad_hessian;

  stan::math::hessian(log_normal_density,normal_eval_vec,
                       normal_eval_agrad_hessian,
                       norm_grad_agrad_hessian,
                       norm_hess_agrad_hessian);

  stan::math::hessian(mixed_third_poly,poly_eval_vec,
                       poly_eval_agrad_hessian,
                       poly_grad_agrad_hessian,
                       poly_hess_agrad_hessian);

  Matrix<double,Dynamic,Dynamic> norm_hess_analytic;
  Matrix<double,Dynamic,Dynamic> poly_hess_analytic;

  std::vector<Matrix<double,Dynamic,Dynamic> > norm_grad_hess_agrad;
  std::vector<Matrix<double,Dynamic,Dynamic> > poly_grad_hess_agrad;

  std::vector<Matrix<double,Dynamic,Dynamic> > norm_grad_hess_analytic;
  std::vector<Matrix<double,Dynamic,Dynamic> > poly_grad_hess_analytic;

  normal_eval_analytic = log_normal_density(normal_eval_vec);
  poly_eval_analytic = mixed_third_poly(poly_eval_vec);

  stan::math::grad_hessian(log_normal_density,normal_eval_vec,
                            normal_eval_agrad,norm_hess_agrad,norm_grad_hess_agrad);
  stan::math::grad_hessian(mixed_third_poly,poly_eval_vec,
                            poly_eval_agrad,poly_hess_agrad,poly_grad_hess_agrad);
  norm_hess_analytic = norm_hess(normal_eval_vec);
  poly_hess_analytic = third_order_mixed_hess(poly_eval_vec);

  norm_grad_hess_analytic = norm_grad_hess(normal_eval_vec);
  poly_grad_hess_analytic = third_order_mixed_grad_hess(poly_eval_vec);

  EXPECT_FLOAT_EQ(normal_eval_analytic,normal_eval_agrad);
  EXPECT_FLOAT_EQ(poly_eval_analytic,poly_eval_agrad);

  for (size_t i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) {
        if (i == 0){
          EXPECT_FLOAT_EQ(norm_hess_agrad_hessian(j,k),norm_hess_analytic(j,k));
          EXPECT_FLOAT_EQ(poly_hess_agrad_hessian(j,k),poly_hess_analytic(j,k));
          EXPECT_FLOAT_EQ(norm_hess_analytic(j,k),norm_hess_agrad(j,k));
          EXPECT_FLOAT_EQ(poly_hess_analytic(j,k),poly_hess_agrad(j,k));
        }
        EXPECT_FLOAT_EQ(norm_grad_hess_analytic[i](j,k),norm_grad_hess_agrad[i](j,k));
        EXPECT_FLOAT_EQ(poly_grad_hess_analytic[i](j,k),poly_grad_hess_agrad[i](j,k));
      }
}

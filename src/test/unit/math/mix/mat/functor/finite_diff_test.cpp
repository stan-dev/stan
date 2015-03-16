#include <gtest/gtest.h>
#include <stan/math/mix/mat/functor/derivative.hpp>
#include <stan/math/mix/mat/functor/grad_hessian.hpp>
#include <stan/math/mix/mat/functor/grad_tr_mat_times_hessian.hpp>
#include <stan/math/mix/mat/functor/gradient.hpp>
#include <stan/math/mix/mat/functor/gradient_dot_vector.hpp>
#include <stan/math/mix/mat/functor/hessian.hpp>
#include <stan/math/mix/mat/functor/hessian_times_vector.hpp>
#include <stan/math/mix/mat/functor/jacobian.hpp>
#include <stan/math/mix/mat/functor/partial_derivative.hpp>
#include <stan/math/prim/scal/prob/normal_log.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/prim/mat/fun/value_of.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/fwd/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/prim/mat/functor/finite_diff_hessian.hpp>
#include <stan/math/prim/mat/functor/finite_diff_gradient.hpp>
#include <stan/math/rev/mat/functor/finite_diff_hessian.hpp>
#include <stan/math/rev/mat/functor/finite_diff_grad_hessian.hpp>

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
  norm_functor norm;
  Matrix<double,Dynamic,1> f_vec(3);
  Matrix<double,Dynamic,1> norm_vec(3);

  f_vec << 0.5, 0.3, 0.1;
  norm_vec << 0.5, 0.3, 0.7;

  double f_eval;
  double f_fin_diff_eval;
  double norm_eval;
  double norm_fin_diff_eval;

  Matrix<double,Dynamic,1> grad_f;
  Matrix<double,Dynamic,1> finite_diff_f;

  Matrix<double,Dynamic,1> grad_norm;
  Matrix<double,Dynamic,1> finite_diff_norm;

  stan::agrad::gradient(f,f_vec,f_eval,grad_f);
  stan::math::finite_diff_gradient(f,f_vec,f_fin_diff_eval,finite_diff_f);

  stan::agrad::gradient(norm,norm_vec,norm_eval,grad_norm);
  stan::math::finite_diff_gradient(norm,norm_vec,norm_fin_diff_eval,finite_diff_norm);

  for (size_type i = 0; i < 3; ++i){
    EXPECT_NEAR(grad_norm(i), finite_diff_norm(i), 1e-12);
    EXPECT_NEAR(grad_f(i), finite_diff_f(i), 1e-12);
  }

}

TEST(AgradFiniteDiff,hessian) {
  using Eigen::Matrix;  
  using Eigen::Dynamic;
  third_order_mixed f;
  norm_functor norm;
  Matrix<double,Dynamic,1> f_vec(3);
  Matrix<double,Dynamic,1> norm_vec(3);

  f_vec << 0.5, 0.7, 0.4;

  Matrix<double,3,3> an_H_f = third_order_mixed_hess(f_vec);

  norm_vec << 0.5, 0.3, 0.7;

  double f_eval(0);
  double f_fin_diff_eval(0);
  double f_fin_diff_auto_eval(0);
  double norm_eval(0);
  double norm_fin_diff_eval(0);
  double norm_fin_diff_auto_eval(0);
  Matrix<double,Dynamic,1> grad_f;
  Matrix<double,Dynamic,Dynamic> H_f;
  Matrix<double,Dynamic,Dynamic> fin_diff_H_f;
  Matrix<double,Dynamic,Dynamic> fin_diff_auto_H_f;

  Matrix<double,Dynamic,1> grad_norm;
  Matrix<double,Dynamic,Dynamic> H_norm;
  Matrix<double,Dynamic,Dynamic> fin_diff_H_norm;
  Matrix<double,Dynamic,Dynamic> fin_diff_auto_H_norm;

  stan::agrad::hessian(f,f_vec,f_eval,grad_f,H_f);
  stan::math::finite_diff_hessian(f,f_vec,f_fin_diff_eval,fin_diff_H_f);
  stan::agrad::finite_diff_hessian(f,f_vec,f_fin_diff_auto_eval,
                                      fin_diff_auto_H_f);

  EXPECT_FLOAT_EQ(f_eval,f_fin_diff_eval);
  EXPECT_FLOAT_EQ(f_eval,f_fin_diff_auto_eval);

  stan::agrad::hessian(norm,norm_vec,norm_eval,grad_norm,H_norm);
  stan::math::finite_diff_hessian(norm,norm_vec,
                                   norm_fin_diff_eval,
                                   fin_diff_H_norm);
  stan::math::finite_diff_hessian(norm,norm_vec,
                                   norm_fin_diff_auto_eval,
                                   fin_diff_auto_H_norm);

  EXPECT_FLOAT_EQ(norm_eval,norm_fin_diff_eval);
  EXPECT_FLOAT_EQ(norm_eval,norm_fin_diff_auto_eval);
  Matrix<double,3,3> an_H_norm = norm_hess(norm_vec);

  EXPECT_FLOAT_EQ(f_eval,f_fin_diff_eval);
  EXPECT_FLOAT_EQ(f_eval,f_fin_diff_auto_eval);
  for (size_type i = 0; i < 3; ++i)
    for (size_type j = 0; j < 3; ++j){
      EXPECT_NEAR(H_f(i,j), fin_diff_H_f(i,j), 1e-09);
      EXPECT_NEAR(an_H_f(i,j), fin_diff_H_f(i,j),1e-09) << "i: " << i << " j: " << j;
      EXPECT_NEAR(an_H_f(i,j), fin_diff_auto_H_f(i,j),1e-09) << "i: " << i << " j: " << j;
    }

  for (size_type i = 0; i < 3; ++i)
    for (size_type j = 0; j < 3; ++j){
      EXPECT_NEAR(H_norm(i,j), fin_diff_H_norm(i,j), 1e-09) << "i: " << i << " j: " << j;
      EXPECT_NEAR(an_H_norm(i,j), fin_diff_H_norm(i,j), 1e-09) << "i: " << i << " j: " << j;
      EXPECT_NEAR(an_H_norm(i,j), fin_diff_auto_H_norm(i,j), 1e-09) << "i: " << i << " j: " << j;
    }

}

TEST(AgradFiniteDiff,grad_hessian) {
  using Eigen::Matrix;  
  using Eigen::Dynamic;
  norm_functor norm;
  third_order_mixed f;
  Matrix<double,Dynamic,1> norm_vec(3);
  norm_vec << 0.5, 0.3, 0.7;

  Matrix<double,Dynamic,1> f_vec(3);
  f_vec << 1.6666667e-01, 0.5, 1;


  double f_eval(0);
  Matrix<double,Dynamic,Dynamic> H_f;
  std::vector<Matrix<double,Dynamic,Dynamic> > grad_H_f;
  stan::agrad::grad_hessian(f,f_vec,f_eval,H_f,grad_H_f);

  double f_fin_diff_eval(0);
  std::vector<Matrix<double,Dynamic,Dynamic> > fin_diff_grad_H_f;
  stan::agrad::finite_diff_grad_hessian(f,f_vec,f_fin_diff_eval,
                                        fin_diff_grad_H_f);

  std::vector<Matrix<double,Dynamic,Dynamic> > an_grad_H_f = 
    third_order_mixed_grad_hess(f_vec);

  EXPECT_FLOAT_EQ(f_eval,f_fin_diff_eval);

  double norm_eval;
  Matrix<double,Dynamic,Dynamic> H_norm;
  std::vector<Matrix<double,Dynamic,Dynamic> > grad_H_norm;
  stan::agrad::grad_hessian(norm,norm_vec,norm_eval,H_norm,grad_H_norm);

  double norm_fin_diff_eval;
  std::vector<Matrix<double,Dynamic,Dynamic> > fin_diff_grad_H_norm;
  stan::agrad::finite_diff_grad_hessian(norm,norm_vec,
                                        norm_fin_diff_eval,
                                        fin_diff_grad_H_norm);

  std::vector<Matrix<double,Dynamic,Dynamic> > an_grad_H_norm = 
    norm_grad_hess(norm_vec);

  EXPECT_FLOAT_EQ(norm_eval,norm_fin_diff_eval);

  for (size_t i = 0; i < 3; ++i){
    for (size_type j = 0; j < 3; ++j){
      for (size_type k = 0; k < 3; ++k){
        EXPECT_NEAR(an_grad_H_f[i](j,k),fin_diff_grad_H_f[i](j,k),1e-10) << " i: " << i << " j: " << j << " k: " << k; 
        EXPECT_NEAR(grad_H_f[i](j,k),fin_diff_grad_H_f[i](j,k),1e-10) << " i: " << i << " j: " << j << " k: " << k; 
        EXPECT_NEAR(an_grad_H_norm[i](j,k),fin_diff_grad_H_norm[i](j,k),1e-10) << " i: " << i << " j: " << j << " k: " << k; 
        EXPECT_NEAR(grad_H_norm[i](j,k),fin_diff_grad_H_norm[i](j,k),1e-10) << " i: " << i << " j: " << j << " k: " << k; 
      }
    }
  }
}

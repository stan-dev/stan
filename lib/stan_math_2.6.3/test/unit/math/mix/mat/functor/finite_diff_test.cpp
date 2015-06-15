#include <gtest/gtest.h>
#include <stan/math/mix/mat/functor/grad_hessian.hpp>
#include <stan/math/rev/mat/functor/gradient.hpp>
#include <stan/math/mix/mat/functor/hessian.hpp>
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
#include <stan/math/mix/mat/functor/finite_diff_grad_hessian.hpp>
#include <stan/math/prim/mat/fun/sum.hpp>
#include <stan/math/rev/mat/fun/sum.hpp>
#include <stan/math/fwd/mat/fun/sum.hpp>

struct norm_functor {
  template <typename T>
  inline
  T operator()(const Eigen::Matrix<T,Eigen::Dynamic,1>& inp_vec) const {
    return stan::math::normal_log(inp_vec(0), inp_vec(1), inp_vec(2)); 
  }
};

struct sum_functor {
  template <typename T>
  inline 
  T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& inp_vec) const {
    using stan::math::sum;
    return sum(inp_vec);
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

  stan::math::gradient(f,f_vec,f_eval,grad_f);
  stan::math::finite_diff_gradient(f,f_vec,f_fin_diff_eval,finite_diff_f);

  stan::math::gradient(norm,norm_vec,norm_eval,grad_norm);
  stan::math::finite_diff_gradient(norm,norm_vec,norm_fin_diff_eval,finite_diff_norm);

  for (int i = 0; i < 3; ++i){
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

  norm_vec << 0.5, 0.3, 0.7;

  double f_eval(0);
  double f_fin_diff_eval(0);
  double norm_eval(0);
  double norm_fin_diff_eval(0);
  Matrix<double,Dynamic,1> grad_f;
  Matrix<double,Dynamic,Dynamic> H_f;
  Matrix<double,Dynamic,1> fin_diff_grad_f;
  Matrix<double,Dynamic,Dynamic> fin_diff_H_f;

  Matrix<double,Dynamic,1> grad_norm;
  Matrix<double,Dynamic,Dynamic> H_norm;
  Matrix<double,Dynamic,Dynamic> fin_diff_H_norm;
  Matrix<double,Dynamic,1> fin_diff_grad_norm;
  Matrix<double,Dynamic,Dynamic> fin_diff_auto_H_norm;

  stan::math::hessian(f, f_vec, f_eval, grad_f, H_f);
  stan::math::finite_diff_hessian(f, f_vec, f_fin_diff_eval, fin_diff_grad_f, fin_diff_H_f);

  EXPECT_FLOAT_EQ(f_eval,f_fin_diff_eval);

  stan::math::hessian(norm,norm_vec,norm_eval,grad_norm,H_norm);
  stan::math::finite_diff_hessian(norm,norm_vec,
                                   norm_fin_diff_eval, fin_diff_grad_norm,
                                   fin_diff_H_norm);

  EXPECT_FLOAT_EQ(norm_eval,norm_fin_diff_eval);
  Matrix<double,3,3> an_H_norm = norm_hess(norm_vec);

  EXPECT_FLOAT_EQ(f_eval,f_fin_diff_eval);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(H_norm(i,j), fin_diff_H_norm(i,j), 1e-09) << "i: " << i << " j: " << j;
      EXPECT_NEAR(an_H_norm(i,j), fin_diff_H_norm(i,j), 1e-09) << "i: " << i << " j: " << j;
    }
    EXPECT_NEAR(grad_norm(i), fin_diff_grad_norm(i), 1e-10);
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
  stan::math::grad_hessian(f,f_vec,f_eval,H_f,grad_H_f);

  double f_fin_diff_eval(0);
  Matrix<double, Dynamic, Dynamic> fin_diff_H_f;
  std::vector<Matrix<double, Dynamic, Dynamic> > fin_diff_grad_H_f;
  stan::math::finite_diff_grad_hessian(f,f_vec,f_fin_diff_eval, fin_diff_H_f,
                                        fin_diff_grad_H_f);

  std::vector<Matrix<double,Dynamic,Dynamic> > an_grad_H_f = 
    third_order_mixed_grad_hess(f_vec);

  EXPECT_FLOAT_EQ(f_eval,f_fin_diff_eval);

  double norm_eval;
  Matrix<double, Dynamic, Dynamic> H_norm;
  std::vector<Matrix<double, Dynamic, Dynamic> > grad_H_norm;
  stan::math::grad_hessian(norm, norm_vec, norm_eval, H_norm, grad_H_norm);

  double norm_fin_diff_eval;
  Matrix<double, Dynamic, Dynamic> fin_diff_H_norm;
  std::vector<Matrix<double,Dynamic,Dynamic> > fin_diff_grad_H_norm;
  stan::math::finite_diff_grad_hessian(norm, norm_vec,
                                        norm_fin_diff_eval, fin_diff_H_norm,
                                        fin_diff_grad_H_norm);

  std::vector<Matrix<double,Dynamic,Dynamic> > an_grad_H_norm = 
    norm_grad_hess(norm_vec);

  EXPECT_FLOAT_EQ(norm_eval,norm_fin_diff_eval);

  for (size_t i = 0; i < 3; ++i){
    for (int j = 0; j < 3; ++j){
      for (int k = 0; k < 3; ++k){
        EXPECT_NEAR(an_grad_H_f[i](j,k),fin_diff_grad_H_f[i](j,k),1e-10) << " i: " << i << " j: " << j << " k: " << k; 
        EXPECT_NEAR(grad_H_f[i](j,k),fin_diff_grad_H_f[i](j,k),1e-10) << " i: " << i << " j: " << j << " k: " << k; 
        EXPECT_NEAR(an_grad_H_norm[i](j,k),fin_diff_grad_H_norm[i](j,k),1e-10) << " i: " << i << " j: " << j << " k: " << k; 
        EXPECT_NEAR(grad_H_norm[i](j,k),fin_diff_grad_H_norm[i](j,k),1e-10) << " i: " << i << " j: " << j << " k: " << k; 
        EXPECT_FLOAT_EQ(H_norm(j, k), fin_diff_H_norm(j, k));
        EXPECT_FLOAT_EQ(H_f(j, k), fin_diff_H_f(j, k));
      }
    }
  }
}

TEST(AgradFiniteDiff,gradientZeroOneArg) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::sum;

  sum_functor sum_f;

  Matrix<double, Dynamic, 1> sum_vec(1);
  sum_vec << 2;

  double grad_eval_sum;
  double f_grad_eval_sum;

  Matrix<double,Dynamic,1> grad_sum;
  Matrix<double,Dynamic,1> f_grad_sum;

  stan::math::gradient(sum_f, sum_vec, grad_eval_sum, grad_sum);
  stan::math::finite_diff_gradient(sum_f, sum_vec, f_grad_eval_sum, f_grad_sum);

  EXPECT_NEAR(grad_sum(0), f_grad_sum(0), 1e-12); 

  Matrix<double, Dynamic, 1> z_sum_vec;
  z_sum_vec.resize(0,1);

  double z_grad_eval_sum;
  double f_z_grad_eval_sum;

  Matrix<double,Dynamic,1> z_grad_sum;
  Matrix<double,Dynamic,1> f_z_grad_sum;

  stan::math::gradient(sum_f, z_sum_vec, z_grad_eval_sum, z_grad_sum);
  stan::math::finite_diff_gradient(sum_f, z_sum_vec, f_z_grad_eval_sum, f_z_grad_sum);
}

TEST(AgradFiniteDiff,hessianZeroOneArg) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::sum;

  sum_functor sum_f;

  Matrix<double, Dynamic, 1> sum_vec(1);
  sum_vec << 2;

  double eval_sum;
  double f_eval_sum;

  Matrix<double,Dynamic,1> grad_sum;
  Matrix<double,Dynamic,1> f_grad_sum;

  Matrix<double,Dynamic,Dynamic> hess_sum;
  Matrix<double,Dynamic,Dynamic> f_hess_sum;

  stan::math::hessian(sum_f,sum_vec,eval_sum,grad_sum,hess_sum);
  stan::math::finite_diff_hessian(sum_f,sum_vec,f_eval_sum,f_grad_sum,f_hess_sum);

  EXPECT_NEAR(grad_sum(0), f_grad_sum(0), 1e-12); 
  EXPECT_NEAR(hess_sum(0,0), f_hess_sum(0,0), 1e-12); 

  Matrix<double, Dynamic, 1> z_sum_vec;
  z_sum_vec.resize(0,1);

  double z_eval_sum;
  double f_z_eval_sum;

  Matrix<double,Dynamic,1> z_grad_sum;
  Matrix<double,Dynamic,1> f_z_grad_sum;

  Matrix<double,Dynamic,Dynamic> z_hess_sum;
  Matrix<double,Dynamic,Dynamic> f_z_hess_sum;

  stan::math::hessian(sum_f, z_sum_vec, z_eval_sum, z_grad_sum, z_hess_sum);
  stan::math::finite_diff_hessian(sum_f,  z_sum_vec,  f_z_eval_sum,  f_z_grad_sum,  f_z_hess_sum);
}

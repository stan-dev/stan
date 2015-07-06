#include <stan/math.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/fwd/scal/fun/value_of_rec.hpp>
#include <stan/math/fwd/scal/fun/cos.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/fwd/scal/fun/tanh.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/lgamma.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/ceil.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/fwd/mat/fun/sum.hpp>
#include <stan/math/fwd/scal/fun/tan.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/fwd/mat/fun/multiply.hpp>
#include <test/unit/math/rev/mat/prob/lkj_corr_cholesky_test_functors.hpp>
#include <stan/math/prim/scal/prob/uniform_rng.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/fwd/scal/fun/tanh.hpp>
#include <stan/math/fwd/scal/fun/lgamma.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/tan.hpp>
#include <stan/math/fwd/mat/fun/sum.hpp>
#include <stan/math/fwd/mat/fun/multiply.hpp>
#include <stan/math/mix/mat/functor/hessian.hpp>
#include <stan/math/prim/mat/functor/finite_diff_hessian.hpp>
#include <stan/math/mix/mat/functor/grad_hessian.hpp>
#include <stan/math/mix/mat/functor/finite_diff_grad_hessian.hpp>
#include <test/unit/math/mix/mat/prob/higher_order_utils.hpp>

TEST(ProbDistributionsLkjCorr,fvar_var) {
  using stan::math::fvar;
  using stan::math::var;
  boost::random::mt19937 rng;
  int K = 4;
  Eigen::Matrix<fvar<var>,Eigen::Dynamic,Eigen::Dynamic> Sigma(K,K);
  Sigma.setZero();
  Sigma.diagonal().setOnes();
  for (int i = 0; i < K*K; i++)
    Sigma(i).d_ = 1.0;
  fvar<var> eta = stan::math::uniform_rng(0,2,rng);
  fvar<var> f = stan::math::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_.val(), stan::math::lkj_corr_log(Sigma, eta).val_.val());
  EXPECT_FLOAT_EQ(2.5177896, stan::math::lkj_corr_log(Sigma, eta).d_.val());
  eta = 1.0;
  f = stan::math::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_.val(), stan::math::lkj_corr_log(Sigma, eta).val_.val());
  EXPECT_FLOAT_EQ(f.d_.val(), stan::math::lkj_corr_log(Sigma, eta).d_.val());
}

TEST(ProbDistributionsLkjCorrCholesky,fvar_var) {
  using stan::math::fvar;
  using stan::math::var;
  boost::random::mt19937 rng;
  int K = 4;
  Eigen::Matrix<fvar<var>,Eigen::Dynamic,Eigen::Dynamic> Sigma(K,K);
  Sigma.setZero();
  Sigma.diagonal().setOnes();
  for (int i = 0; i < K*K; i++)
    Sigma(i).d_ = 1.0;
  fvar<var> eta = stan::math::uniform_rng(0,2,rng);
  fvar<var> f = stan::math::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_.val(), stan::math::lkj_corr_cholesky_log(Sigma, eta).val_.val());
  EXPECT_FLOAT_EQ(6.7766843, stan::math::lkj_corr_cholesky_log(Sigma, eta).d_.val());
  eta = 1.0;
  f = stan::math::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_.val(), stan::math::lkj_corr_cholesky_log(Sigma, eta).val_.val());
  EXPECT_FLOAT_EQ(3, stan::math::lkj_corr_cholesky_log(Sigma, eta).d_.val());
}

TEST(ProbDistributionsLkjCorr,fvar_fvar_var) {
  using stan::math::fvar;
  using stan::math::var;
  boost::random::mt19937 rng;
  int K = 4;
  Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,Eigen::Dynamic> Sigma(K,K);
  Sigma.setZero();
  Sigma.diagonal().setOnes();
  for (int i = 0; i < K*K; i++)
    Sigma(i).d_.val_ = 1.0;
  fvar<fvar<var> > eta = stan::math::uniform_rng(0,2,rng);
  fvar<fvar<var> > f = stan::math::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_.val_.val(), stan::math::lkj_corr_log(Sigma, eta).val_.val_.val());
  EXPECT_FLOAT_EQ(2.5177896, stan::math::lkj_corr_log(Sigma, eta).d_.val_.val());
  eta = 1.0;
  f = stan::math::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_.val_.val(), stan::math::lkj_corr_log(Sigma, eta).val_.val_.val());
  EXPECT_FLOAT_EQ(f.d_.val_.val(), stan::math::lkj_corr_log(Sigma, eta).d_.val_.val());
}

TEST(ProbDistributionsLkjCorrCholesky,fvar_fvar_var) {
  using stan::math::fvar;
  using stan::math::var;
  boost::random::mt19937 rng;
  int K = 4;
  Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,Eigen::Dynamic> Sigma(K,K);
  Sigma.setZero();
  Sigma.diagonal().setOnes();
  for (int i = 0; i < K*K; i++)
    Sigma(i).d_.val_ = 1.0;
  fvar<fvar<var> > eta = stan::math::uniform_rng(0,2,rng);
  fvar<fvar<var> > f = stan::math::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_.val_.val(), stan::math::lkj_corr_cholesky_log(Sigma, eta).val_.val_.val());
  EXPECT_FLOAT_EQ(6.7766843, stan::math::lkj_corr_cholesky_log(Sigma, eta).d_.val_.val());
  eta = 1.0;
  f = stan::math::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_.val_.val(), stan::math::lkj_corr_cholesky_log(Sigma, eta).val_.val_.val());
  EXPECT_FLOAT_EQ(3, stan::math::lkj_corr_cholesky_log(Sigma, eta).d_.val_.val());
}

TEST(ProbDistributionsLkjCorrCholesky,hessian) {
  int dim_mat = 3;
  Eigen::Matrix<double, Eigen::Dynamic, 1> x1(dim_mat);
  Eigen::Matrix<double, Eigen::Dynamic, 1> x2(1);
  Eigen::Matrix<double, Eigen::Dynamic, 1> x3(dim_mat + 1);

  x2(0) = 2.0;

  for(int i = 0; i < dim_mat; ++i) {
    x1(i) = i / 10.0;
    x3(i + 1) = x1(i); 
  }
  x3(0) = 0.5;

  stan::math::lkj_corr_cholesky_dc test_func_1(dim_mat);
  stan::math::lkj_corr_cholesky_cd test_func_2(dim_mat);
  stan::math::lkj_corr_cholesky_dd test_func_3(dim_mat);

  using stan::math::finite_diff_hessian;
  using stan::math::hessian;

  Eigen::Matrix<double, Eigen::Dynamic, 1> grad_hess_3;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> hess_3;
  double fx_hess_3;
  Eigen::Matrix<double, Eigen::Dynamic, 1> grad_hess_ad_3;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> hess_ad_3;
  double fx_hess_ad_3;
  
  finite_diff_hessian(test_func_3,
                      x3,
                      fx_hess_3, 
                      grad_hess_3,
                      hess_3);
  hessian(test_func_3,
          x3,
          fx_hess_ad_3, 
          grad_hess_ad_3,
          hess_ad_3);

  test_hess_eq(hess_3, hess_ad_3);
  EXPECT_FLOAT_EQ(fx_hess_3, fx_hess_ad_3);

  Eigen::Matrix<double, Eigen::Dynamic, 1> grad_hess_2;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> hess_2;
  double fx_hess_2;
  Eigen::Matrix<double, Eigen::Dynamic, 1> grad_hess_ad_2;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> hess_ad_2;
  double fx_hess_ad_2;
  
  finite_diff_hessian(test_func_2,
                      x2,
                      fx_hess_2, 
                      grad_hess_2,
                      hess_2);
  hessian(test_func_2,
          x2,
          fx_hess_ad_2, 
          grad_hess_ad_2,
          hess_ad_2);

  test_hess_eq(hess_2, hess_ad_2);
  EXPECT_FLOAT_EQ(fx_hess_2, fx_hess_ad_2);

  Eigen::Matrix<double, Eigen::Dynamic, 1> grad_hess_1;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> hess_1;
  double fx_hess_1;
  Eigen::Matrix<double, Eigen::Dynamic, 1> grad_hess_ad_1;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> hess_ad_1;
  double fx_hess_ad_1;
  
  finite_diff_hessian(test_func_1,
                      x1,
                      fx_hess_1, 
                      grad_hess_1,
                      hess_1);
  hessian(test_func_1,
          x1,
          fx_hess_ad_1, 
          grad_hess_ad_1,
          hess_ad_1);

  test_hess_eq(hess_1, hess_ad_1);
  EXPECT_FLOAT_EQ(fx_hess_1, fx_hess_ad_1);

}

TEST(ProbDistributionsLkjCorrCholesky,grad_hessian) {
  int dim_mat = 3;
  Eigen::Matrix<double, Eigen::Dynamic, 1> x1(dim_mat);
  Eigen::Matrix<double, Eigen::Dynamic, 1> x2(1);
  Eigen::Matrix<double, Eigen::Dynamic, 1> x3(dim_mat + 1);

  x2(0) = 2.0;

  for(int i = 0; i < dim_mat; ++i) {
    x1(i) = i / 10.0;
    x3(i + 1) = x1(i); 
  }
  x3(0) = 0.5;

  stan::math::lkj_corr_cholesky_dc test_func_1(dim_mat);
  stan::math::lkj_corr_cholesky_cd test_func_2(dim_mat);
  stan::math::lkj_corr_cholesky_dd test_func_3(dim_mat);

  using stan::math::finite_diff_grad_hessian;
  using stan::math::grad_hessian;

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> hess_gh_3;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > gh_3;
  double fx_gh_3;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> hess_gh_ad_3;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > gh_ad_3;
  double fx_gh_ad_3;

  finite_diff_grad_hessian(test_func_3,
                           x3,
                           fx_gh_3, 
                           hess_gh_3,
                           gh_3);
  grad_hessian(test_func_3,
               x3,
               fx_gh_ad_3, 
               hess_gh_ad_3,
               gh_ad_3);

  test_grad_hess_eq(gh_3, gh_ad_3);
  EXPECT_FLOAT_EQ(fx_gh_3, fx_gh_ad_3);

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> hess_gh_2;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > gh_2;
  double fx_gh_2;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> hess_gh_ad_2;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > gh_ad_2;
  double fx_gh_ad_2;

  finite_diff_grad_hessian(test_func_2,
                           x2,
                           fx_gh_2, 
                           hess_gh_2,
                           gh_2);
  grad_hessian(test_func_2,
               x2,
               fx_gh_ad_2, 
               hess_gh_ad_2,
               gh_ad_2);

  test_grad_hess_eq(gh_2, gh_ad_2);
  EXPECT_FLOAT_EQ(fx_gh_2, fx_gh_ad_2);

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> hess_gh_1;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > gh_1;
  double fx_gh_1;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> hess_gh_ad_1;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > gh_ad_1;
  double fx_gh_ad_1;

  finite_diff_grad_hessian(test_func_1,
                           x1,
                           fx_gh_1, 
                           hess_gh_1,
                           gh_1);
  grad_hessian(test_func_1,
               x1,
               fx_gh_ad_1, 
               hess_gh_ad_1,
               gh_ad_1);

  test_grad_hess_eq(gh_1, gh_ad_1);
  EXPECT_FLOAT_EQ(fx_gh_1, fx_gh_ad_1);
}

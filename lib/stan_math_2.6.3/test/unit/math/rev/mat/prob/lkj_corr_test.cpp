#include <stan/math.hpp>
#include <stan/math/rev/mat/functor/gradient.hpp>
#include <stan/math/prim/mat/functor/finite_diff_gradient.hpp>
#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/cos.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/lgamma.hpp>
#include <stan/math/rev/scal/fun/floor.hpp>
#include <stan/math/rev/scal/fun/ceil.hpp>
#include <stan/math/rev/scal/fun/sin.hpp>
#include <stan/math/rev/scal/fun/tgamma.hpp>
#include <stan/math/rev/scal/fun/abs.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/rev/mat/fun/sum.hpp>
#include <stan/math/prim/mat/err/check_pos_definite.hpp>
#include <stan/math/prim/mat/err/check_symmetric.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/scal/fun/tan.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/rev/mat/fun/multiply.hpp>
#include <stan/math/prim/mat/prob/lkj_corr_log.hpp>
#include <stan/math/prim/mat/prob/lkj_corr_rng.hpp>
#include <stan/math/prim/mat/prob/lkj_corr_cholesky_log.hpp>
#include <stan/math/prim/mat/prob/lkj_corr_cholesky_rng.hpp>
#include <stan/math/prim/scal/prob/uniform_rng.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <test/unit/math/rev/mat/prob/lkj_corr_cholesky_test_functors.hpp>
#include <test/unit/math/rev/mat/prob/test_gradients.hpp>
#include <stan/math/prim/mat/functor/finite_diff_gradient.hpp>

TEST(ProbDistributionsLkjCorr,var) {
  using stan::math::var;
  boost::random::mt19937 rng;
  int K = 4;
  Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic> Sigma(K,K);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Sigma_d(K,K);
  Sigma.setZero();
  Sigma.diagonal().setOnes();
  Sigma_d.setZero();
  Sigma_d.diagonal().setOnes();
  var eta = stan::math::uniform_rng(0,2,rng);
  var f = stan::math::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val(), stan::math::lkj_corr_log(Sigma, eta).val());
  EXPECT_FLOAT_EQ(f.val(), stan::math::lkj_corr_log(Sigma_d, eta).val());
  eta = 1.0;
  double eta_d = 1.0;
  f = stan::math::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val(), stan::math::lkj_corr_log(Sigma, eta).val());
  EXPECT_FLOAT_EQ(f.val(), stan::math::lkj_corr_log(Sigma, eta_d).val());
}

TEST(ProbDistributionsLkjCorrCholesky,var) {
  using stan::math::var;
  boost::random::mt19937 rng;
  int K = 4;
  Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic> Sigma(K,K);
  Sigma.setZero();
  Sigma.diagonal().setOnes();
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Sigma_d(K,K);
  Sigma_d.setZero();
  Sigma_d.diagonal().setOnes();
  var eta = stan::math::uniform_rng(0,2,rng);
  var f = stan::math::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val(), stan::math::lkj_corr_cholesky_log(Sigma, eta).val());
  EXPECT_FLOAT_EQ(f.val(), stan::math::lkj_corr_cholesky_log(Sigma_d, eta).val());
  eta = 1.0;
  double eta_d = 1.0;
  f = stan::math::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val(), stan::math::lkj_corr_cholesky_log(Sigma, eta).val());
  EXPECT_FLOAT_EQ(f.val(), stan::math::lkj_corr_cholesky_log(Sigma, eta_d).val());
}

TEST(ProbDistributionsLkjCorrCholesky,gradients) {
  using stan::math::var;
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

  using stan::math::finite_diff_gradient;
  using stan::math::gradient;

  Eigen::Matrix<double, Eigen::Dynamic, 1> grad;
  double fx;
  Eigen::Matrix<double, Eigen::Dynamic, 1> grad_ad;
  double fx_ad;

  finite_diff_gradient(test_func_3,
                       x3,
                       fx, grad);
  gradient(test_func_3,
           x3, fx_ad, grad_ad);

  test_grad_eq(grad, grad_ad);
  EXPECT_FLOAT_EQ(fx, fx_ad);

  finite_diff_gradient(test_func_2,
                       x2,
                       fx, grad);
  gradient(test_func_2,
           x2, fx_ad, grad_ad);
  test_grad_eq(grad, grad_ad);
  EXPECT_FLOAT_EQ(fx, fx_ad);

  Eigen::Matrix<double, Eigen::Dynamic, 1> grad_1;
  double fx_1;
  Eigen::Matrix<double, Eigen::Dynamic, 1> grad_ad_1;
  double fx_ad_1;

  finite_diff_gradient(test_func_1,
                       x1,
                       fx_1, grad_1);
  gradient(test_func_1,
           x1, fx_ad_1, grad_ad_1);
  test_grad_eq(grad_1, grad_ad_1);
  EXPECT_FLOAT_EQ(fx, fx_ad);
}

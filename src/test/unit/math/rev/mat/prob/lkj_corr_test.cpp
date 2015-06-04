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
//  EXPECT_FLOAT_EQ(2.5177896, stan::math::lkj_corr_log(Sigma, eta).d_.val());
  eta = 1.0;
  double eta_d = 1.0;
  f = stan::math::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val(), stan::math::lkj_corr_log(Sigma, eta).val());
  EXPECT_FLOAT_EQ(f.val(), stan::math::lkj_corr_log(Sigma, eta_d).val());
 // EXPECT_FLOAT_EQ(f.d_.val(), stan::math::lkj_corr_log(Sigma, eta).d_.val());
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
//  EXPECT_FLOAT_EQ(6.7766843, stan::math::lkj_corr_cholesky_log(Sigma, eta).d_.val());
  eta = 1.0;
  double eta_d = 1.0;
  f = stan::math::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val(), stan::math::lkj_corr_cholesky_log(Sigma, eta).val());
  EXPECT_FLOAT_EQ(f.val(), stan::math::lkj_corr_cholesky_log(Sigma, eta_d).val());
//  EXPECT_FLOAT_EQ(3, stan::math::lkj_corr_cholesky_log(Sigma, eta).d_.val());
}

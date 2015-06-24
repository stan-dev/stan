#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/fwd/scal/fun/value_of_rec.hpp>
#include <stan/math/fwd/scal/fun/cos.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/lgamma.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/ceil.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/fwd/mat/fun/sum.hpp>
#include <stan/math/prim/mat/err/check_symmetric.hpp>
#include <stan/math/fwd/scal/fun/tan.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/prim/mat/prob/lkj_corr_log.hpp>
#include <stan/math/prim/mat/prob/lkj_corr_rng.hpp>
#include <stan/math/prim/mat/prob/lkj_corr_cholesky_log.hpp>
#include <stan/math/prim/mat/prob/lkj_corr_cholesky_rng.hpp>
#include <stan/math/fwd/mat/fun/multiply.hpp>
#include <stan/math/prim/scal/prob/uniform_rng.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>

TEST(ProbDistributionsLkjCorr,fvar_double) {
  using stan::math::fvar;
  boost::random::mt19937 rng;
  int K = 4;
  Eigen::Matrix<fvar<double>,Eigen::Dynamic,Eigen::Dynamic> Sigma(K,K);
  Sigma.setZero();
  Sigma.diagonal().setOnes();
  for (int i = 0; i < K*K; i++)
    Sigma(i).d_ = 1.0;
  fvar<double> eta = stan::math::uniform_rng(0,2,rng);
  fvar<double> f = stan::math::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_, stan::math::lkj_corr_log(Sigma, eta).val_);
  EXPECT_FLOAT_EQ(2.5177896, stan::math::lkj_corr_log(Sigma, eta).d_);
  eta = 1.0;
  f = stan::math::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_, stan::math::lkj_corr_log(Sigma, eta).val_);
  EXPECT_FLOAT_EQ(f.d_, stan::math::lkj_corr_log(Sigma, eta).d_);
}

TEST(ProbDistributionsLkjCorrCholesky,fvar_double) {
  using stan::math::fvar;
  boost::random::mt19937 rng;
  int K = 4;
  Eigen::Matrix<fvar<double>,Eigen::Dynamic,Eigen::Dynamic> Sigma(K,K);
  Sigma.setZero();
  Sigma.diagonal().setOnes();
  for (int i = 0; i < K*K; i++)
    Sigma(i).d_ = 1.0;
  fvar<double> eta = stan::math::uniform_rng(0,2,rng);
  fvar<double> f = stan::math::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_, stan::math::lkj_corr_cholesky_log(Sigma, eta).val_);
  EXPECT_FLOAT_EQ(6.7766843, stan::math::lkj_corr_cholesky_log(Sigma, eta).d_);
  eta = 1.0;
  f = stan::math::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_, stan::math::lkj_corr_cholesky_log(Sigma, eta).val_);
  EXPECT_FLOAT_EQ(3, stan::math::lkj_corr_cholesky_log(Sigma, eta).d_);
}

TEST(ProbDistributionsLkjCorr,fvar_fvar_double) {
  using stan::math::fvar;
  boost::random::mt19937 rng;
  int K = 4;
  Eigen::Matrix<fvar<fvar<double> >,Eigen::Dynamic,Eigen::Dynamic> Sigma(K,K);
  Sigma.setZero();
  Sigma.diagonal().setOnes();
  for (int i = 0; i < K*K; i++)
    Sigma(i).d_.val_ = 1.0;
  fvar<fvar<double> > eta = stan::math::uniform_rng(0,2,rng);
  fvar<fvar<double> > f = stan::math::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_.val_, stan::math::lkj_corr_log(Sigma, eta).val_.val_);
  EXPECT_FLOAT_EQ(2.5177896, stan::math::lkj_corr_log(Sigma, eta).d_.val_);
  eta = 1.0;
  f = stan::math::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_.val_, stan::math::lkj_corr_log(Sigma, eta).val_.val_);
  EXPECT_FLOAT_EQ(f.d_.val_, stan::math::lkj_corr_log(Sigma, eta).d_.val_);
}

TEST(ProbDistributionsLkjCorrCholesky,fvar_fvar_double) {
  using stan::math::fvar;
  boost::random::mt19937 rng;
  int K = 4;
  Eigen::Matrix<fvar<fvar<double> >,Eigen::Dynamic,Eigen::Dynamic> Sigma(K,K);
  Sigma.setZero();
  Sigma.diagonal().setOnes();
  for (int i = 0; i < K*K; i++)
    Sigma(i).d_.val_ = 1.0;
  fvar<fvar<double> > eta = stan::math::uniform_rng(0,2,rng);
  fvar<fvar<double> > f = stan::math::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_.val_, stan::math::lkj_corr_cholesky_log(Sigma, eta).val_.val_);
  EXPECT_FLOAT_EQ(6.7766843, stan::math::lkj_corr_cholesky_log(Sigma, eta).d_.val_);
  eta = 1.0;
  f = stan::math::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_.val_, stan::math::lkj_corr_cholesky_log(Sigma, eta).val_.val_);
  EXPECT_FLOAT_EQ(3, stan::math::lkj_corr_cholesky_log(Sigma, eta).d_.val_);
}


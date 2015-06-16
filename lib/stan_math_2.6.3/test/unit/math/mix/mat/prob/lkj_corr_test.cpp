#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/scal/fun/value_of_rec.hpp>
#include <stan/math/fwd/scal/fun/cos.hpp>
#include <stan/math/rev/scal/fun/cos.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/lgamma.hpp>
#include <stan/math/rev/scal/fun/lgamma.hpp>
#include <stan/math/fwd/scal/fun/floor.hpp>
#include <stan/math/rev/scal/fun/floor.hpp>
#include <stan/math/fwd/scal/fun/ceil.hpp>
#include <stan/math/rev/scal/fun/ceil.hpp>
#include <stan/math/rev/scal/fun/sin.hpp>
#include <stan/math/rev/scal/fun/tgamma.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/rev/scal/fun/abs.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/rev/mat/fun/sum.hpp>
#include <stan/math/fwd/mat/fun/sum.hpp>
#include <stan/math/prim/mat/err/check_pos_definite.hpp>
#include <stan/math/prim/mat/err/check_symmetric.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/scal/fun/tan.hpp>
#include <stan/math/fwd/scal/fun/tan.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/prim/mat/prob/lkj_corr_log.hpp>
#include <stan/math/prim/mat/prob/lkj_corr_rng.hpp>
#include <stan/math/prim/mat/prob/lkj_corr_cholesky_log.hpp>
#include <stan/math/prim/mat/prob/lkj_corr_cholesky_rng.hpp>
#include <stan/math/prim/scal/prob/uniform_rng.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>

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

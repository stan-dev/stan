#include <gtest/gtest.h>
#include <stan/agrad/rev/matrix.hpp>
#include <stan/agrad/fwd/matrix.hpp>
#include <stan/prob/distributions/multivariate/continuous/lkj_corr.hpp>
#include <stan/prob/distributions/univariate/continuous/uniform.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>

TEST(ProbDistributionsLkjCorr,testIdentity) {
  boost::random::mt19937 rng;
  unsigned int K = 4;
  Eigen::MatrixXd Sigma(K,K);
  Sigma.setZero();
  Sigma.diagonal().setOnes();
  double eta = stan::prob::uniform_rng(0,2,rng);
  double f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f, stan::prob::lkj_corr_log(Sigma, eta));
  eta = 1.0;
  f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f, stan::prob::lkj_corr_log(Sigma, eta));
}


TEST(ProbDistributionsLkjCorr,testHalf) {
  boost::random::mt19937 rng;
  unsigned int K = 4;
  Eigen::MatrixXd Sigma(K,K);
  Sigma.setConstant(0.5);
  Sigma.diagonal().setOnes();
  double eta = stan::prob::uniform_rng(0,2,rng);
  double f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f + (eta - 1.0) * log(0.3125), stan::prob::lkj_corr_log(Sigma, eta));
  eta = 1.0;
  f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f, stan::prob::lkj_corr_log(Sigma, eta));
}

TEST(ProbDistributionsLkjCorr,Sigma) {
  boost::random::mt19937 rng;
  unsigned int K = 4;
  Eigen::MatrixXd Sigma(K,K);
  Sigma.setZero();
  Sigma.diagonal().setOnes();
  double eta = stan::prob::uniform_rng(0,2,rng);
  EXPECT_NO_THROW (stan::prob::lkj_corr_log(Sigma, eta));
  
  EXPECT_THROW (stan::prob::lkj_corr_log(Sigma, -eta), std::domain_error);

  Sigma = Sigma * -1.0;
  EXPECT_THROW (stan::prob::lkj_corr_log(Sigma, eta), std::domain_error);
  Sigma = Sigma * (0.0 / 0.0);
  EXPECT_THROW (stan::prob::lkj_corr_log(Sigma, eta), std::domain_error);

  Sigma.setConstant(0.5);
  Sigma.diagonal().setOnes();
  EXPECT_THROW (stan::prob::lkj_corr_cholesky_log(Sigma, eta), std::domain_error);
}

TEST(ProbDistributionsLKJCorr, error_check) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::prob::lkj_corr_cholesky_rng(5, 1.0,rng));
  EXPECT_NO_THROW(stan::prob::lkj_corr_rng(5, 1.0,rng));

  EXPECT_THROW(stan::prob::lkj_corr_cholesky_rng(5, -1.0,rng),std::domain_error);
  EXPECT_THROW(stan::prob::lkj_corr_rng(5, -1.0,rng),std::domain_error);
}

TEST(ProbDistributionsLKJCorr, chiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  int N = 10000;
  int K = boost::math::round(2 * std::pow(N, 0.4));
  boost::math::beta_distribution<>dist (2.5,2.5);
  boost::math::chi_squared mydist(K-1);

  double loc[K - 1];
  for(int i = 1; i < K; i++)
    loc[i - 1] = quantile(dist, i * std::pow(K, -1.0));

  int count = 0;
  int bin [K];
  double expect [K];
  for(int i = 0 ; i < K; i++)
  {
    bin[i] = 0;
    expect[i] = N / K;
  }

  while (count < N) {
    double a = 0.5 * (1.0 + stan::prob::lkj_corr_rng(5,1.0,rng)(3,4));
    int i = 0;
    while (i < K-1 && a > loc[i])
  ++i;
    ++bin[i];
    count++;
   }

  double chi = 0;

  for(int j = 0; j < K; j++)
    chi += ((bin[j] - expect[j]) * (bin[j] - expect[j]) / expect[j]);

  EXPECT_TRUE(chi < quantile(complement(mydist, 1e-6)));
}

TEST(ProbDistributionsLkjCorrCholesky,testIdentity) {
  boost::random::mt19937 rng;
  unsigned int K = 4;
  Eigen::MatrixXd Sigma(K,K);
  Sigma.setZero();
  Sigma.diagonal().setOnes();
  double eta = stan::prob::uniform_rng(0,2,rng);
  double f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f, stan::prob::lkj_corr_cholesky_log(Sigma, eta));
  eta = 1.0;
  f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f, stan::prob::lkj_corr_cholesky_log(Sigma, eta));
}

TEST(ProbDistributionsLkjCorrCholesky,testHalf) {
  boost::random::mt19937 rng;
  unsigned int K = 4;
  Eigen::MatrixXd Sigma(K,K);
  Sigma.setConstant(0.5);
  Sigma.diagonal().setOnes();
  Eigen::MatrixXd L = Sigma.llt().matrixL();
  double eta = stan::prob::uniform_rng(0,2,rng);
  double f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(stan::prob::lkj_corr_log(Sigma, eta) - 0.4904146,
                  stan::prob::lkj_corr_cholesky_log(L, eta));
  eta = 1.0;
  f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f - 0.4904146,
                  stan::prob::lkj_corr_cholesky_log(L, eta));
}

TEST(ProbDistributionsLkjCorr,fvar_double) {
  using stan::agrad::fvar;
  boost::random::mt19937 rng;
  unsigned int K = 4;
  Eigen::Matrix<fvar<double>,Eigen::Dynamic,Eigen::Dynamic> Sigma(K,K);
  Sigma.setZero();
  Sigma.diagonal().setOnes();
  for (int i = 0; i < K*K; i++)
    Sigma(i).d_ = 1.0;
  fvar<double> eta = stan::prob::uniform_rng(0,2,rng);
  fvar<double> f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_, stan::prob::lkj_corr_log(Sigma, eta).val_);
  EXPECT_FLOAT_EQ(2.5177896, stan::prob::lkj_corr_log(Sigma, eta).d_);
  eta = 1.0;
  f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_, stan::prob::lkj_corr_log(Sigma, eta).val_);
  EXPECT_FLOAT_EQ(f.d_, stan::prob::lkj_corr_log(Sigma, eta).d_);
}

TEST(ProbDistributionsLkjCorrCholesky,fvar_double) {
  using stan::agrad::fvar;
  boost::random::mt19937 rng;
  unsigned int K = 4;
  Eigen::Matrix<fvar<double>,Eigen::Dynamic,Eigen::Dynamic> Sigma(K,K);
  Sigma.setZero();
  Sigma.diagonal().setOnes();
  for (int i = 0; i < K*K; i++)
    Sigma(i).d_ = 1.0;
  fvar<double> eta = stan::prob::uniform_rng(0,2,rng);
  fvar<double> f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_, stan::prob::lkj_corr_cholesky_log(Sigma, eta).val_);
  EXPECT_FLOAT_EQ(6.7766843, stan::prob::lkj_corr_cholesky_log(Sigma, eta).d_);
  eta = 1.0;
  f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_, stan::prob::lkj_corr_cholesky_log(Sigma, eta).val_);
  EXPECT_FLOAT_EQ(3, stan::prob::lkj_corr_cholesky_log(Sigma, eta).d_);
}

TEST(ProbDistributionsLkjCorr,fvar_var) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  boost::random::mt19937 rng;
  unsigned int K = 4;
  Eigen::Matrix<fvar<var>,Eigen::Dynamic,Eigen::Dynamic> Sigma(K,K);
  Sigma.setZero();
  Sigma.diagonal().setOnes();
  for (int i = 0; i < K*K; i++)
    Sigma(i).d_ = 1.0;
  fvar<var> eta = stan::prob::uniform_rng(0,2,rng);
  fvar<var> f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_.val(), stan::prob::lkj_corr_log(Sigma, eta).val_.val());
  EXPECT_FLOAT_EQ(2.5177896, stan::prob::lkj_corr_log(Sigma, eta).d_.val());
  eta = 1.0;
  f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_.val(), stan::prob::lkj_corr_log(Sigma, eta).val_.val());
  EXPECT_FLOAT_EQ(f.d_.val(), stan::prob::lkj_corr_log(Sigma, eta).d_.val());
}

TEST(ProbDistributionsLkjCorrCholesky,fvar_var) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  boost::random::mt19937 rng;
  unsigned int K = 4;
  Eigen::Matrix<fvar<var>,Eigen::Dynamic,Eigen::Dynamic> Sigma(K,K);
  Sigma.setZero();
  Sigma.diagonal().setOnes();
  for (int i = 0; i < K*K; i++)
    Sigma(i).d_ = 1.0;
  fvar<var> eta = stan::prob::uniform_rng(0,2,rng);
  fvar<var> f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_.val(), stan::prob::lkj_corr_cholesky_log(Sigma, eta).val_.val());
  EXPECT_FLOAT_EQ(6.7766843, stan::prob::lkj_corr_cholesky_log(Sigma, eta).d_.val());
  eta = 1.0;
  f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_.val(), stan::prob::lkj_corr_cholesky_log(Sigma, eta).val_.val());
  EXPECT_FLOAT_EQ(3, stan::prob::lkj_corr_cholesky_log(Sigma, eta).d_.val());
}

TEST(ProbDistributionsLkjCorr,fvar_fvar_double) {
  using stan::agrad::fvar;
  boost::random::mt19937 rng;
  unsigned int K = 4;
  Eigen::Matrix<fvar<fvar<double> >,Eigen::Dynamic,Eigen::Dynamic> Sigma(K,K);
  Sigma.setZero();
  Sigma.diagonal().setOnes();
  for (int i = 0; i < K*K; i++)
    Sigma(i).d_.val_ = 1.0;
  fvar<fvar<double> > eta = stan::prob::uniform_rng(0,2,rng);
  fvar<fvar<double> > f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_.val_, stan::prob::lkj_corr_log(Sigma, eta).val_.val_);
  EXPECT_FLOAT_EQ(2.5177896, stan::prob::lkj_corr_log(Sigma, eta).d_.val_);
  eta = 1.0;
  f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_.val_, stan::prob::lkj_corr_log(Sigma, eta).val_.val_);
  EXPECT_FLOAT_EQ(f.d_.val_, stan::prob::lkj_corr_log(Sigma, eta).d_.val_);
}

TEST(ProbDistributionsLkjCorrCholesky,fvar_fvar_double) {
  using stan::agrad::fvar;
  boost::random::mt19937 rng;
  unsigned int K = 4;
  Eigen::Matrix<fvar<fvar<double> >,Eigen::Dynamic,Eigen::Dynamic> Sigma(K,K);
  Sigma.setZero();
  Sigma.diagonal().setOnes();
  for (int i = 0; i < K*K; i++)
    Sigma(i).d_.val_ = 1.0;
  fvar<fvar<double> > eta = stan::prob::uniform_rng(0,2,rng);
  fvar<fvar<double> > f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_.val_, stan::prob::lkj_corr_cholesky_log(Sigma, eta).val_.val_);
  EXPECT_FLOAT_EQ(6.7766843, stan::prob::lkj_corr_cholesky_log(Sigma, eta).d_.val_);
  eta = 1.0;
  f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_.val_, stan::prob::lkj_corr_cholesky_log(Sigma, eta).val_.val_);
  EXPECT_FLOAT_EQ(3, stan::prob::lkj_corr_cholesky_log(Sigma, eta).d_.val_);
}

TEST(ProbDistributionsLkjCorr,fvar_fvar_var) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  boost::random::mt19937 rng;
  unsigned int K = 4;
  Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,Eigen::Dynamic> Sigma(K,K);
  Sigma.setZero();
  Sigma.diagonal().setOnes();
  for (int i = 0; i < K*K; i++)
    Sigma(i).d_.val_ = 1.0;
  fvar<fvar<var> > eta = stan::prob::uniform_rng(0,2,rng);
  fvar<fvar<var> > f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_.val_.val(), stan::prob::lkj_corr_log(Sigma, eta).val_.val_.val());
  EXPECT_FLOAT_EQ(2.5177896, stan::prob::lkj_corr_log(Sigma, eta).d_.val_.val());
  eta = 1.0;
  f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_.val_.val(), stan::prob::lkj_corr_log(Sigma, eta).val_.val_.val());
  EXPECT_FLOAT_EQ(f.d_.val_.val(), stan::prob::lkj_corr_log(Sigma, eta).d_.val_.val());
}

TEST(ProbDistributionsLkjCorrCholesky,fvar_fvar_var) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  boost::random::mt19937 rng;
  unsigned int K = 4;
  Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,Eigen::Dynamic> Sigma(K,K);
  Sigma.setZero();
  Sigma.diagonal().setOnes();
  for (int i = 0; i < K*K; i++)
    Sigma(i).d_.val_ = 1.0;
  fvar<fvar<var> > eta = stan::prob::uniform_rng(0,2,rng);
  fvar<fvar<var> > f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_.val_.val(), stan::prob::lkj_corr_cholesky_log(Sigma, eta).val_.val_.val());
  EXPECT_FLOAT_EQ(6.7766843, stan::prob::lkj_corr_cholesky_log(Sigma, eta).d_.val_.val());
  eta = 1.0;
  f = stan::prob::do_lkj_constant(eta, K);
  EXPECT_FLOAT_EQ(f.val_.val_.val(), stan::prob::lkj_corr_cholesky_log(Sigma, eta).val_.val_.val());
  EXPECT_FLOAT_EQ(3, stan::prob::lkj_corr_cholesky_log(Sigma, eta).d_.val_.val());
}

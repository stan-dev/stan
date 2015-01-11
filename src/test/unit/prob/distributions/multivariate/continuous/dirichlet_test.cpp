#include <gtest/gtest.h>
#include <stan/prob/distributions/multivariate/continuous/dirichlet.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;

TEST(ProbDistributions,Dirichlet) {
  Matrix<double,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;
  Matrix<double,Dynamic,1> alpha(3,1);
  alpha << 1.0, 1.0, 1.0;
  EXPECT_FLOAT_EQ(0.6931472, stan::prob::dirichlet_log(theta,alpha));
  
  Matrix<double,Dynamic,1> theta2(4,1);
  theta2 << 0.01, 0.01, 0.8, 0.18;
  Matrix<double,Dynamic,1> alpha2(4,1);
  alpha2 << 10.5, 11.5, 19.3, 5.1;
  EXPECT_FLOAT_EQ(-43.40045, stan::prob::dirichlet_log(theta2,alpha2));
}

TEST(ProbDistributions,DirichletPropto) {
  Matrix<double,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;
  Matrix<double,Dynamic,1> alpha(3,1);
  alpha << 1.0, 1.0, 1.0;
  EXPECT_FLOAT_EQ(0.0, stan::prob::dirichlet_log<true>(theta,alpha));
  
  Matrix<double,Dynamic,1> theta2(4,1);
  theta2 << 0.01, 0.01, 0.8, 0.18;
  Matrix<double,Dynamic,1> alpha2(4,1);
  alpha2 << 10.5, 11.5, 19.3, 5.1;
  EXPECT_FLOAT_EQ(0.0, stan::prob::dirichlet_log<true>(theta2,alpha2));
}

TEST(ProbDistributions,DirichletBounds) {
  Matrix<double,Dynamic,1> good_alpha(2,1), bad_alpha(2,1);
  Matrix<double,Dynamic,1> good_theta(2,1), bad_theta(2,1);

  good_theta << 0.25, 0.75;
  good_alpha << 2, 3;
  EXPECT_NO_THROW(stan::prob::dirichlet_log(good_theta,good_alpha));

  good_theta << 1.0, 0.0;
  good_alpha << 2, 3;
  EXPECT_NO_THROW(stan::prob::dirichlet_log(good_theta,good_alpha))
    << "elements of theta can be 0";


  bad_theta << 0.25, 0.25;
  EXPECT_THROW(stan::prob::dirichlet_log(bad_theta,good_alpha),
               std::domain_error)
    << "sum of theta is not 1";

  bad_theta << -0.25, 1.25;
  EXPECT_THROW(stan::prob::dirichlet_log(bad_theta,good_alpha),
               std::domain_error)
    << "theta has element less than 0";

  bad_theta << -0.25, 1.25;
  EXPECT_THROW(stan::prob::dirichlet_log(bad_theta,good_alpha),
               std::domain_error)
    << "theta has element less than 0";

  bad_alpha << 0.0, 1.0;
  EXPECT_THROW(stan::prob::dirichlet_log(good_theta,bad_alpha),
               std::domain_error)
    << "alpha has element equal to 0";

  bad_alpha << -0.5, 1.0;
  EXPECT_THROW(stan::prob::dirichlet_log(good_theta,bad_alpha),
               std::domain_error)
    << "alpha has element less than 0";

  bad_alpha = Matrix<double,Dynamic,1>(4,1);
  bad_alpha << 1, 2, 3, 4;
  EXPECT_THROW(stan::prob::dirichlet_log(good_theta,bad_alpha),
               std::domain_error)
    << "size mismatch: theta is a 2-vector, alpha is a 4-vector";
}

TEST(ProbDistributionsDirichlet, random) {
  boost::random::mt19937 rng;
  Matrix<double,Dynamic,Dynamic> alpha(3,1);
  alpha << 2.0, 
    3.0,
    11.0;

  EXPECT_NO_THROW(stan::prob::dirichlet_rng(alpha,rng));
}

TEST(ProbDistributionsDirichlet, marginalOneChiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  Matrix<double,Dynamic,Dynamic> alpha(3,1);
  alpha << 2.0, 
    3.0,
    11.0;
  int N = 10000;
  int K = boost::math::round(2 * std::pow(N, 0.4));
  boost::math::beta_distribution<>dist (2.0,3.0 + 11.0);
  boost::math::chi_squared mydist(K-1);

  double loc[K - 1];
  for(int i = 1; i < K; i++)
    loc[i - 1] = quantile(dist, i * std::pow(K, -1.0));

  int count = 0;
  int bin [K];
  double expect [K];
  for(int i = 0 ; i < K; i++) {
    bin[i] = 0;
    expect[i] = N / K;
  }

  Eigen::VectorXd a(alpha.rows());

  while (count < N) {
    a = stan::prob::dirichlet_rng(alpha,rng);
    int i = 0;
    while (i < K-1 && a(0) > loc[i]) 
      ++i;
    ++bin[i];
    count++;
   }

  double chi = 0;

  for(int j = 0; j < K; j++)
    chi += ((bin[j] - expect[j]) * (bin[j] - expect[j]) / expect[j]);

  EXPECT_TRUE(chi < quantile(complement(mydist, 1e-6)));
}


TEST(ProbDistributionsDirichlet, marginalTwoChiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  Matrix<double,Dynamic,Dynamic> alpha(3,1);
  alpha << 2.0, 
    3.0,
    11.0;
  int N = 10000;
  int K = boost::math::round(2 * std::pow(N, 0.4));
  boost::math::beta_distribution<>dist (3.0,13.0);
  boost::math::chi_squared mydist(K-1);

  double loc[K - 1];
  for(int i = 1; i < K; i++)
    loc[i - 1] = quantile(dist, i * std::pow(K, -1.0));

  int count = 0;
  int bin [K];
  double expect [K];
  for(int i = 0 ; i < K; i++) {
    bin[i] = 0;
    expect[i] = N / K;
  }

  Eigen::VectorXd a(alpha.rows());

  while (count < N) {
    a = stan::prob::dirichlet_rng(alpha,rng);
    int i = 0;
    while (i < K-1 && a(1) > loc[i]) 
      ++i;
    ++bin[i];
    count++;
   }

  double chi = 0;

  for(int j = 0; j < K; j++)
    chi += ((bin[j] - expect[j]) * (bin[j] - expect[j]) / expect[j]);

  EXPECT_TRUE(chi < quantile(complement(mydist, 1e-6)));
}

TEST(ProbDistributions,fvar_double) {
  using stan::agrad::fvar;

  Matrix<fvar<double>,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;
  Matrix<fvar<double>,Dynamic,1> alpha(3,1);
  alpha << 1.0, 1.0, 1.0;
  for (int i = 0; i < 3; i++) {
    theta(i).d_ = 1.0;
    alpha(i).d_ = 1.0;
  }

  EXPECT_FLOAT_EQ(0.6931472, stan::prob::dirichlet_log(theta,alpha).val_);
  EXPECT_FLOAT_EQ(0.99344212, stan::prob::dirichlet_log(theta,alpha).d_);
  
  Matrix<fvar<double>,Dynamic,1> theta2(4,1);
  theta2 << 0.01, 0.01, 0.8, 0.18;
  Matrix<fvar<double>,Dynamic,1> alpha2(4,1);
  alpha2 << 10.5, 11.5, 19.3, 5.1;
  for (int i = 0; i < 3; i++) {
    theta2(i).d_ = 1.0;
    alpha2(i).d_ = 1.0;
  }

  EXPECT_FLOAT_EQ(-43.40045, stan::prob::dirichlet_log(theta2,alpha2).val_);
  EXPECT_FLOAT_EQ(2017.2858, stan::prob::dirichlet_log(theta2,alpha2).d_);
}

TEST(ProbDistributions,fvar_fvar_double) {
  using stan::agrad::fvar;

  Matrix<fvar<fvar<double> >,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;
  Matrix<fvar<fvar<double> >,Dynamic,1> alpha(3,1);
  alpha << 1.0, 1.0, 1.0;
  for (int i = 0; i < 3; i++) {
    theta(i).d_ = 1.0;
    alpha(i).d_ = 1.0;
  }

  EXPECT_FLOAT_EQ(0.6931472, stan::prob::dirichlet_log(theta,alpha).val_.val_);
  EXPECT_FLOAT_EQ(0.99344212, stan::prob::dirichlet_log(theta,alpha).d_.val_);
  
  Matrix<fvar<fvar<double> >,Dynamic,1> theta2(4,1);
  theta2 << 0.01, 0.01, 0.8, 0.18;
  Matrix<fvar<fvar<double> >,Dynamic,1> alpha2(4,1);
  alpha2 << 10.5, 11.5, 19.3, 5.1;
  for (int i = 0; i < 3; i++) {
    theta2(i).d_ = 1.0;
    alpha2(i).d_ = 1.0;
  }

  EXPECT_FLOAT_EQ(-43.40045, stan::prob::dirichlet_log(theta2,alpha2).val_.val_);
  EXPECT_FLOAT_EQ(2017.2858, stan::prob::dirichlet_log(theta2,alpha2).d_.val_);
}

TEST(ProbDistributions,fvar_var) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  Matrix<fvar<var>,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;
  Matrix<fvar<var>,Dynamic,1> alpha(3,1);
  alpha << 1.0, 1.0, 1.0;
  for (int i = 0; i < 3; i++) {
    theta(i).d_ = 1.0;
    alpha(i).d_ = 1.0;
  }

  EXPECT_FLOAT_EQ(0.6931472, stan::prob::dirichlet_log(theta,alpha).val_.val());
  EXPECT_FLOAT_EQ(0.99344212, stan::prob::dirichlet_log(theta,alpha).d_.val());
  
  Matrix<fvar<var>,Dynamic,1> theta2(4,1);
  theta2 << 0.01, 0.01, 0.8, 0.18;
  Matrix<fvar<var>,Dynamic,1> alpha2(4,1);
  alpha2 << 10.5, 11.5, 19.3, 5.1;
  for (int i = 0; i < 3; i++) {
    theta2(i).d_ = 1.0;
    alpha2(i).d_ = 1.0;
  }

  EXPECT_FLOAT_EQ(-43.40045, stan::prob::dirichlet_log(theta2,alpha2).val_.val());
  EXPECT_FLOAT_EQ(2017.2858, stan::prob::dirichlet_log(theta2,alpha2).d_.val());
}

TEST(ProbDistributions,fvar_fvar_var) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  Matrix<fvar<fvar<var> >,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;
  Matrix<fvar<fvar<var> >,Dynamic,1> alpha(3,1);
  alpha << 1.0, 1.0, 1.0;
  for (int i = 0; i < 3; i++) {
    theta(i).d_ = 1.0;
    alpha(i).d_ = 1.0;
  }

  EXPECT_FLOAT_EQ(0.6931472, stan::prob::dirichlet_log(theta,alpha).val_.val_.val());
  EXPECT_FLOAT_EQ(0.99344212, stan::prob::dirichlet_log(theta,alpha).d_.val_.val());
  
  Matrix<fvar<fvar<var> >,Dynamic,1> theta2(4,1);
  theta2 << 0.01, 0.01, 0.8, 0.18;
  Matrix<fvar<fvar<var> >,Dynamic,1> alpha2(4,1);
  alpha2 << 10.5, 11.5, 19.3, 5.1;
  for (int i = 0; i < 3; i++) {
    theta2(i).d_ = 1.0;
    alpha2(i).d_ = 1.0;
  }

  EXPECT_FLOAT_EQ(-43.40045, stan::prob::dirichlet_log(theta2,alpha2).val_.val_.val());
  EXPECT_FLOAT_EQ(2017.2858, stan::prob::dirichlet_log(theta2,alpha2).d_.val_.val());
}

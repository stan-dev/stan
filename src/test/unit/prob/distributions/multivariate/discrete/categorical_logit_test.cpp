#include <gtest/gtest.h>
#include <limits>
#include <stan/math/matrix/log_softmax.hpp>
#include <stan/prob/distributions/multivariate/discrete/categorical_logit.hpp>
#include <boost/math/distributions.hpp>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/fwd/matrix.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/agrad/rev/matrix.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;
using stan::math::log_softmax;
using stan::math::log_softmax;

TEST(ProbDistributionsCategoricalLogit,Categorical) {
  Matrix<double,Dynamic,1> theta(3,1);
  theta << -1, 2, -10;
  Matrix<double,Dynamic,1> theta_log_softmax = log_softmax(theta);

  EXPECT_FLOAT_EQ(theta_log_softmax[0], stan::prob::categorical_logit_log(1,theta));
  EXPECT_FLOAT_EQ(theta_log_softmax[1], stan::prob::categorical_logit_log(2,theta));
  EXPECT_FLOAT_EQ(theta_log_softmax[2], stan::prob::categorical_logit_log(3,theta));
}

TEST(ProbDistributionsCategoricalLogit,CategoricalVectorized) {
  Matrix<double,Dynamic,1> theta(3);
  theta << -1, 2, -10;

  std::vector<int> ns(0);
  EXPECT_FLOAT_EQ(0.0, stan::prob::categorical_logit_log(ns,theta));

  Matrix<double,Dynamic,1> theta_log_softmax = log_softmax(theta);

  std::vector<int> ms(3);
  ms[0] = 1;
  ms[1] = 2;
  ms[2] = 1;
  EXPECT_FLOAT_EQ(theta_log_softmax[0] + theta_log_softmax[1] + theta_log_softmax[0],
                  stan::prob::categorical_logit_log(ms,theta));
}



TEST(ProbDistributionsCategoricalLogit,Propto) {
  Matrix<double,Dynamic,1> theta(3,1);
  theta << -1, 2, 10;
  EXPECT_FLOAT_EQ(0, stan::prob::categorical_logit_log<true>(1,theta));
  EXPECT_FLOAT_EQ(0, stan::prob::categorical_logit_log<true>(3,theta));
}


TEST(ProbDistributionsCategoricalLogit,DefaultPolicy) {
  using stan::prob::categorical_logit_log;

  unsigned int n = 1;
  unsigned int N = 3;
  Matrix<double,Dynamic,1> theta(N,1);
  theta << 0.3, 0.5, 0.2;

  EXPECT_NO_THROW(categorical_logit_log(N, theta));
  EXPECT_NO_THROW(categorical_logit_log(n, theta));
  EXPECT_NO_THROW(categorical_logit_log(2, theta));
  EXPECT_THROW(categorical_logit_log(N+1, theta), std::domain_error);
  EXPECT_THROW(categorical_logit_log(0, theta), std::domain_error);

  theta(1) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(categorical_logit_log(1, theta), std::domain_error);

  theta(1) = std::numeric_limits<double>::infinity();
  EXPECT_THROW(categorical_logit_log(1, theta), std::domain_error);

  std::vector<int> ns(2);
  ns[0] = 1;
  ns[1] = 2;
  EXPECT_THROW(categorical_logit_log(ns, theta), std::domain_error);

  theta << 0.3, 0.5, 0.2;
  EXPECT_NO_THROW(categorical_logit_log(ns, theta));
  
  ns[0] = -1;
  EXPECT_THROW(categorical_logit_log(ns, theta), std::domain_error);

  ns[0] = 1;
  ns[1] = 12;
  EXPECT_THROW(categorical_logit_log(ns, theta), std::domain_error);
}

TEST(ProbDistributionsCategoricalLogit,fvar_double) {
  using stan::agrad::fvar;
  Matrix<fvar<double>,Dynamic,1> theta(3,1);
  theta << -1, 2, -10;
  for (int i = 0; i < 3; i++)
    theta(i).d_ = i;
  Matrix<fvar<double>,Dynamic,1> theta_log_softmax = log_softmax(theta);
  EXPECT_FLOAT_EQ(theta_log_softmax[0].val_,
                  stan::prob::categorical_logit_log(1,theta).val_);
  EXPECT_FLOAT_EQ(theta_log_softmax[1].val_, 
                  stan::prob::categorical_logit_log(2,theta).val_);
  EXPECT_FLOAT_EQ(theta_log_softmax[2].val_,
                  stan::prob::categorical_logit_log(3,theta).val_);
  EXPECT_FLOAT_EQ(theta_log_softmax[0].d_,
                  stan::prob::categorical_logit_log(1,theta).d_);
  EXPECT_FLOAT_EQ(theta_log_softmax[1].d_, 
                  stan::prob::categorical_logit_log(2,theta).d_);
  EXPECT_FLOAT_EQ(theta_log_softmax[2].d_, 
                  stan::prob::categorical_logit_log(3,theta).d_);
}

TEST(ProbDistributionsCategoricalLogit,fvar_double_vectorized) {
  using stan::agrad::fvar;
  Matrix<fvar<double>,Dynamic,1> theta(3);
  theta << -1, 2, -10;
  for (int i = 0; i < 3; i++)
    theta(i).d_ = i;

  std::vector<int> ns(0);
  EXPECT_FLOAT_EQ(0.0, stan::prob::categorical_logit_log(ns,theta).val_);

  Matrix<fvar<double>,Dynamic,1> theta_log_softmax = log_softmax(theta);

  std::vector<int> ms(3);
  ms[0] = 1;
  ms[1] = 2;
  ms[2] = 1;
  EXPECT_FLOAT_EQ(theta_log_softmax[0].val_ + theta_log_softmax[1].val_ 
                  + theta_log_softmax[0].val_,
                  stan::prob::categorical_logit_log(ms,theta).val_);
  EXPECT_FLOAT_EQ(theta_log_softmax[0].d_ + theta_log_softmax[1].d_ 
                  + theta_log_softmax[0].d_,
                  stan::prob::categorical_logit_log(ms,theta).d_);
}

TEST(ProbDistributionsCategoricalLogit,fvar_var) {
  using stan::agrad::var;
  using stan::agrad::fvar;
  Matrix<fvar<var>,Dynamic,1> theta(3,1);
  theta << -1, 2, -10;
  for (int i = 0; i < 3; i++)
    theta(i).d_ = i;
  Matrix<fvar<var>,Dynamic,1> theta_log_softmax = log_softmax(theta);
  EXPECT_FLOAT_EQ(theta_log_softmax[0].val_.val(), 
                  stan::prob::categorical_logit_log(1,theta).val_.val());
  EXPECT_FLOAT_EQ(theta_log_softmax[1].val_.val(), 
                  stan::prob::categorical_logit_log(2,theta).val_.val());
  EXPECT_FLOAT_EQ(theta_log_softmax[2].val_.val(), 
                  stan::prob::categorical_logit_log(3,theta).val_.val());
  EXPECT_FLOAT_EQ(theta_log_softmax[0].d_.val(), 
                  stan::prob::categorical_logit_log(1,theta).d_.val());
  EXPECT_FLOAT_EQ(theta_log_softmax[1].d_.val(), 
                  stan::prob::categorical_logit_log(2,theta).d_.val());
  EXPECT_FLOAT_EQ(theta_log_softmax[2].d_.val(), 
                  stan::prob::categorical_logit_log(3,theta).d_.val());
}

TEST(ProbDistributionsCategoricalLogit,fvar_var_vectorized) {
  using stan::agrad::var;
  using stan::agrad::fvar;
  Matrix<fvar<var>,Dynamic,1> theta(3);
  theta << -1, 2, -10;
  for (int i = 0; i < 3; i++)
    theta(i).d_ = i;

  std::vector<int> ns(0);
  EXPECT_FLOAT_EQ(0.0, stan::prob::categorical_logit_log(ns,theta).val_.val());

  Matrix<fvar<var>,Dynamic,1> theta_log_softmax = log_softmax(theta);

  std::vector<int> ms(3);
  ms[0] = 1;
  ms[1] = 2;
  ms[2] = 1;
  EXPECT_FLOAT_EQ(theta_log_softmax[0].val_.val() + theta_log_softmax[1].val_.val() 
                  + theta_log_softmax[0].val_.val(),
                  stan::prob::categorical_logit_log(ms,theta).val_.val());
  EXPECT_FLOAT_EQ(theta_log_softmax[0].d_.val() + theta_log_softmax[1].d_.val() 
                  + theta_log_softmax[0].d_.val(),
                  stan::prob::categorical_logit_log(ms,theta).d_.val());
}

TEST(ProbDistributionsCategoricalLogit,fvar_fvar_double) {
  using stan::agrad::fvar;
  Matrix<fvar<fvar<double> >,Dynamic,1> theta(3,1);
  theta << -1, 2, -10;
  for (int i = 0; i < 3; i++)
    theta(i).d_.val_ = i;
  Matrix<fvar<fvar<double> >,Dynamic,1> theta_log_softmax = log_softmax(theta);
  EXPECT_FLOAT_EQ(theta_log_softmax[0].val_.val_,
                  stan::prob::categorical_logit_log(1,theta).val_.val_);
  EXPECT_FLOAT_EQ(theta_log_softmax[1].val_.val_,
                  stan::prob::categorical_logit_log(2,theta).val_.val_);
  EXPECT_FLOAT_EQ(theta_log_softmax[2].val_.val_, 
                  stan::prob::categorical_logit_log(3,theta).val_.val_);
  EXPECT_FLOAT_EQ(theta_log_softmax[0].d_.val_, 
                  stan::prob::categorical_logit_log(1,theta).d_.val_);
  EXPECT_FLOAT_EQ(theta_log_softmax[1].d_.val_,
                  stan::prob::categorical_logit_log(2,theta).d_.val_);
  EXPECT_FLOAT_EQ(theta_log_softmax[2].d_.val_, 
                  stan::prob::categorical_logit_log(3,theta).d_.val_);
}

TEST(ProbDistributionsCategoricalLogit,fvar_fvar_double_vectorized) {
  using stan::agrad::fvar;
  Matrix<fvar<fvar<double> >,Dynamic,1> theta(3);
  theta << -1, 2, -10;
  for (int i = 0; i < 3; i++)
    theta(i).d_.val_ = i;

  std::vector<int> ns(0);
  EXPECT_FLOAT_EQ(0.0, stan::prob::categorical_logit_log(ns,theta).val_.val_);

  Matrix<fvar<fvar<double> >,Dynamic,1> theta_log_softmax = log_softmax(theta);

  std::vector<int> ms(3);
  ms[0] = 1;
  ms[1] = 2;
  ms[2] = 1;
  EXPECT_FLOAT_EQ(theta_log_softmax[0].val_.val_ + theta_log_softmax[1].val_.val_ 
                  + theta_log_softmax[0].val_.val_,
                  stan::prob::categorical_logit_log(ms,theta).val_.val_);
  EXPECT_FLOAT_EQ(theta_log_softmax[0].d_.val_ + theta_log_softmax[1].d_.val_ 
                  + theta_log_softmax[0].d_.val_,
                  stan::prob::categorical_logit_log(ms,theta).d_.val_);
}

TEST(ProbDistributionsCategoricalLogit,fvar_fvar_var) {
  using stan::agrad::var;
  using stan::agrad::fvar;
  Matrix<fvar<fvar<var> >,Dynamic,1> theta(3,1);
  theta << -1, 2, -10;
  for (int i = 0; i < 3; i++)
    theta(i).d_.val_ = i;
  Matrix<fvar<fvar<var> >,Dynamic,1> theta_log_softmax = log_softmax(theta);
  EXPECT_FLOAT_EQ(theta_log_softmax[0].val_.val_.val(), 
                  stan::prob::categorical_logit_log(1,theta).val_.val_.val());
  EXPECT_FLOAT_EQ(theta_log_softmax[1].val_.val_.val(), 
                  stan::prob::categorical_logit_log(2,theta).val_.val_.val());
  EXPECT_FLOAT_EQ(theta_log_softmax[2].val_.val_.val(), 
                  stan::prob::categorical_logit_log(3,theta).val_.val_.val());
  EXPECT_FLOAT_EQ(theta_log_softmax[0].d_.val_.val(), 
                  stan::prob::categorical_logit_log(1,theta).d_.val_.val());
  EXPECT_FLOAT_EQ(theta_log_softmax[1].d_.val_.val(),
                  stan::prob::categorical_logit_log(2,theta).d_.val_.val());
  EXPECT_FLOAT_EQ(theta_log_softmax[2].d_.val_.val(),
                  stan::prob::categorical_logit_log(3,theta).d_.val_.val());
}

TEST(ProbDistributionsCategoricalLogit,fvar_fvar_var_vectorized) {
  using stan::agrad::var;
  using stan::agrad::fvar;
  Matrix<fvar<fvar<var> >,Dynamic,1> theta(3);
  theta << -1, 2, -10;
  for (int i = 0; i < 3; i++)
    theta(i).d_.val_ = i;

  std::vector<int> ns(0);
  EXPECT_FLOAT_EQ(0.0, stan::prob::categorical_logit_log(ns,theta).val_.val_.val());

  Matrix<fvar<fvar<var> >,Dynamic,1> theta_log_softmax = log_softmax(theta);

  std::vector<int> ms(3);
  ms[0] = 1;
  ms[1] = 2;
  ms[2] = 1;
  EXPECT_FLOAT_EQ(theta_log_softmax[0].val_.val_.val() 
                  + theta_log_softmax[1].val_.val_.val() 
                  + theta_log_softmax[0].val_.val_.val(),
                  stan::prob::categorical_logit_log(ms,theta).val_.val_.val());
  EXPECT_FLOAT_EQ(theta_log_softmax[0].d_.val_.val() 
                  + theta_log_softmax[1].d_.val_.val() 
                  + theta_log_softmax[0].d_.val_.val(),
                  stan::prob::categorical_logit_log(ms,theta).d_.val_.val());
}

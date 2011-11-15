#include <vector>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include "stan/prob/transform.hpp"

using Eigen::Matrix;
using Eigen::Dynamic;

TEST(prob_transform,identity) {
  EXPECT_FLOAT_EQ(4.0, stan::prob::identity_constrain(4.0));
}
TEST(prob_transform,identity_j) {
  double lp = 1.0;
  EXPECT_FLOAT_EQ(4.0, stan::prob::identity_constrain(4.0,lp));
  EXPECT_FLOAT_EQ(1.0,lp);
}
TEST(prob_transform,identity_free) {
  EXPECT_FLOAT_EQ(4.0, stan::prob::identity_free(4.0));
}
TEST(prob_transform,identity_rt) {
  double x = 1.2;
  double xc = stan::prob::identity_constrain(x);
  double xcf = stan::prob::identity_free(xc);
  EXPECT_FLOAT_EQ(x,xcf);

  double y = -1.0;
  double yf = stan::prob::identity_free(y);
  double yfc = stan::prob::identity_constrain(yf);
  EXPECT_FLOAT_EQ(y,yfc);
}
TEST(prob_transform,identity_val) {
  double x = 1.2;
  EXPECT_EQ(true, stan::prob::identity_validate(x));
}


TEST(prob_transform, positive) {
  EXPECT_FLOAT_EQ(exp(-1.0), stan::prob::positive_constrain(-1.0));
}
TEST(prob_transform, positive_j) {
  double lp = 15.0;
  EXPECT_FLOAT_EQ(exp(-1.0), stan::prob::positive_constrain(-1.0,lp));
  EXPECT_FLOAT_EQ(15.0 - 1.0, lp);
}
TEST(prob_transform, positive_f) {
  EXPECT_FLOAT_EQ(log(0.5), stan::prob::positive_free(0.5));
}
TEST(prob_transform, positive_f_exception) {
  EXPECT_THROW (stan::prob::positive_free(-1.0), std::domain_error);
}
TEST(prob_transform, positive_rt) {
  double x = -1.0;
  double xc = stan::prob::positive_constrain(x);
  double xcf = stan::prob::positive_free(xc);
  EXPECT_FLOAT_EQ(x,xcf);
  double xcfc = stan::prob::positive_constrain(xcf);
  EXPECT_FLOAT_EQ(xc,xcfc);
}


TEST(prob_transform, lb) {
  EXPECT_FLOAT_EQ(exp(-1.0) + 2.0, stan::prob::lb_constrain(-1.0,2.0));
}
TEST(prob_transform, lb_j) {
  double lp = 15.0;
  EXPECT_FLOAT_EQ(exp(-1.0) + 2.0, stan::prob::lb_constrain(-1.0,2.0,lp));
  EXPECT_FLOAT_EQ(15.0 - 1.0, lp);
}
TEST(prob_transform, lb_f) {
  EXPECT_FLOAT_EQ(log(3.0 - 2.0), stan::prob::lb_free(3.0,2.0));
}
TEST(prob_transform, lb_f_exception) {
  double lb = 2.0;
  EXPECT_THROW (stan::prob::lb_free(lb - 0.01, lb), std::invalid_argument);
}
TEST(prob_transform, lb_rt) {
  double x = -1.0;
  double xc = stan::prob::lb_constrain(x,2.0);
  double xcf = stan::prob::lb_free(xc,2.0);
  EXPECT_FLOAT_EQ(x,xcf);
  double xcfc = stan::prob::lb_constrain(xcf,2.0);
  EXPECT_FLOAT_EQ(xc,xcfc);
}

TEST(prob_transform, ub) {
  EXPECT_FLOAT_EQ(2.0 - exp(-1.0), stan::prob::ub_constrain(-1.0,2.0));
}
TEST(prob_transform, ub_j) {
  double lp = 15.0;
  EXPECT_FLOAT_EQ(2.0 - exp(-1.0), stan::prob::ub_constrain(-1.0,2.0,lp));
  EXPECT_FLOAT_EQ(15.0 + 1.0, lp);
}
TEST(prob_transform, ub_f) {
  double y = 2.0;
  double U = 4.0;
  EXPECT_FLOAT_EQ(log(-(y - U)), stan::prob::ub_free(2.0,4.0));
}
TEST(prob_transform, ub_f_exception) {
  double ub = 4.0;
  EXPECT_THROW (stan::prob::ub_free(ub+0.01, ub), std::invalid_argument);
}
TEST(prob_transform, ub_rt) {
  double x = -1.0;
  double xc = stan::prob::ub_constrain(x,2.0);
  double xcf = stan::prob::ub_free(xc,2.0);
  EXPECT_FLOAT_EQ(x,xcf);
  double xcfc = stan::prob::ub_constrain(xcf,2.0);
  EXPECT_FLOAT_EQ(xc,xcfc);
}


TEST(prob_transform, lub) {
  EXPECT_FLOAT_EQ(2.0 + (5.0 - 2.0) * stan::maths::inv_logit(-1.0), 
		  stan::prob::lub_constrain(-1.0,2.0,5.0));
}
TEST(prob_transform, lub_j) {
  double lp = -17.0;
  double L = 2.0;
  double U = 5.0;
  double x = -1.0;
  EXPECT_FLOAT_EQ(L + (U - L) * stan::maths::inv_logit(x), 
		  stan::prob::lub_constrain(x,L,U,lp));
  EXPECT_FLOAT_EQ(-17.0 + log(U - L) + log(stan::maths::inv_logit(x)) 
		  + log(1.0 - stan::maths::inv_logit(x)),
		  lp);
}
TEST(prob_transform, lub_f) {
  double L = -10.0;
  double U = 27.0;
  double y = 3.0;
  EXPECT_FLOAT_EQ(stan::maths::logit((y - L) / (U - L)),
		  stan::prob::lub_free(y,L,U));
}
TEST(prob_transform, lub_f_exception) {
  double L = -10.0;
  double U = 27.0;
  EXPECT_THROW(stan::prob::lub_free (L-0.01,L,U), std::invalid_argument);
  EXPECT_THROW(stan::prob::lub_free (U+0.01,L,U), std::invalid_argument);

  EXPECT_THROW(stan::prob::lub_free ((L+U)/2,U,L), std::invalid_argument);
}
TEST(prob_transform, lub_rt) {
  double x = -1.0;
  double xc = stan::prob::lub_constrain(x,2.0,4.0);
  double xcf = stan::prob::lub_free(xc,2.0,4.0);
  EXPECT_FLOAT_EQ(x,xcf);
  double xcfc = stan::prob::lub_constrain(xcf,2.0,4.0);
  EXPECT_FLOAT_EQ(xc,xcfc);
}


TEST(prob_transform, prob) {
  EXPECT_FLOAT_EQ(stan::maths::inv_logit(-1.0), 
		  stan::prob::prob_constrain(-1.0));
}
TEST(prob_transform, prob_j) {
  double lp = -17.0;
  double L = 0.0;
  double U = 1.0;
  double x = -1.0;
  EXPECT_FLOAT_EQ(L + (U - L) * stan::maths::inv_logit(x), 
		  stan::prob::prob_constrain(x,lp));
  EXPECT_FLOAT_EQ(-17.0 + log(U - L) + log(stan::maths::inv_logit(x)) 
		  + log(1.0 - stan::maths::inv_logit(x)),
		  lp);
}
TEST(prob_transform, prob_f) {
  double L = 0.0;
  double U = 1.0;
  double y = 0.4;
  EXPECT_FLOAT_EQ(stan::maths::logit((y - L) / (U - L)),
		  stan::prob::prob_free(y));
}
TEST(prob_transform, prob_f_exception) {
  EXPECT_THROW (stan::prob::prob_free(1.1), std::domain_error);
  EXPECT_THROW (stan::prob::prob_free(-0.1), std::domain_error);
}
TEST(prob_transform, prob_rt) {
  double x = -1.0;
  double xc = stan::prob::prob_constrain(x);
  double xcf = stan::prob::prob_free(xc);
  EXPECT_FLOAT_EQ(x,xcf);
  double xcfc = stan::prob::prob_constrain(xcf);
  EXPECT_FLOAT_EQ(xc,xcfc);
}


TEST(prob_transform, corr) {
  EXPECT_FLOAT_EQ(std::tanh(-1.0), 
		  stan::prob::corr_constrain(-1.0));
}
TEST(prob_transform, corr_j) {
  double lp = -17.0;
  double x = -1.0;
  EXPECT_FLOAT_EQ(std::tanh(x), 
		  stan::prob::corr_constrain(x,lp));
  EXPECT_FLOAT_EQ(-17.0 + (log(1.0 - std::tanh(x) * std::tanh(x))),
		  lp);
}
TEST(prob_transform, corr_f) {
  EXPECT_FLOAT_EQ(atanh(-0.4), 0.5 * std::log((1.0 + -0.4)/(1.0 - -0.4)));
  double y = -0.4;
  EXPECT_FLOAT_EQ(atanh(y),
		  stan::prob::corr_free(y));
}
TEST(prob_transform, corr_rt) {
  double x = -1.0;
  double xc = stan::prob::corr_constrain(x);
  double xcf = stan::prob::corr_free(xc);
  EXPECT_FLOAT_EQ(x,xcf);
  double xcfc = stan::prob::corr_constrain(xcf);
  EXPECT_FLOAT_EQ(xc,xcfc);
}


TEST(prob_transform, simplex) {
  Matrix<double,Dynamic,1> x(3);
  x << 3.0, -1.0, -2.0;
  double pp0 = exp(3.0);
  double pp1 = exp(-1.0);
  double pp2 = exp(-2.0);
  double pp3 = exp(0.0);
  double sum = pp0 + pp1 + pp2 + pp3;
  double p0 = pp0 / sum;
  double p1 = pp1 / sum;
  double p2 = pp2 / sum;
  double p3 = pp3 / sum;
  Matrix<double,Dynamic,1> y = stan::prob::simplex_constrain(x);
  EXPECT_EQ(x.size()+1, y.size());
  EXPECT_FLOAT_EQ(p0, y[0]);
  EXPECT_FLOAT_EQ(p1, y[1]);
  EXPECT_FLOAT_EQ(p2, y[2]);
  EXPECT_FLOAT_EQ(p3, y[3]);
}
TEST(prob_transform, simplex_j) {
  Matrix<double,Dynamic,1> x(3);
  x << 3.0, -1.0, -2.0;
  double pp0 = exp(3.0);
  double pp1 = exp(-1.0);
  double pp2 = exp(-2.0);
  double pp3 = exp(0.0);
  double sum = pp0 + pp1 + pp2 + pp3;
  double p0 = pp0 / sum;
  double p1 = pp1 / sum;
  double p2 = pp2 / sum;
  double p3 = pp3 / sum;
  double lp = -12.0;
  double expected_lp = lp;
  Matrix<double,Dynamic,1> y = stan::prob::simplex_constrain(x,lp);
  EXPECT_EQ(x.size()+1, y.size());
  EXPECT_FLOAT_EQ(p0, y[0]);
  EXPECT_FLOAT_EQ(p1, y[1]);
  EXPECT_FLOAT_EQ(p2, y[2]);
  EXPECT_FLOAT_EQ(p3, y[3]);
  Matrix<double,Dynamic,Dynamic> J(3,3);
  J(0,0) = p0 * (1 - p0);
  J(1,1) = p1 * (1 - p1);
  J(2,2) = p2 * (1 - p2);
  J(0,1) = (J(1,0) = p0 * p1);
  J(0,2) = (J(2,0) = p0 * p2);
  J(1,2) = (J(2,1) = p1 * p2);
  expected_lp += log(fabs(J.determinant()));
  EXPECT_FLOAT_EQ(expected_lp,lp);
}
TEST(prob_transform,simplex_f) {
  Matrix<double,Dynamic,1> y(4);
  y << 0.2, 0.3, 0.4, 0.1;
  Matrix<double,Dynamic,1> x = stan::prob::simplex_free(y);
  EXPECT_EQ(y.size() - 1, x.size());
  EXPECT_EQ(log(y[0]) - log(0.1), x[0]);
  EXPECT_EQ(log(y[1]) - log(0.1), x[1]);
  EXPECT_EQ(log(y[2]) - log(0.1), x[2]);
}
TEST(prob_transform,simplex_f_exception) {
  Matrix<double,Dynamic,1> y(2);
  y << 0.5, 0.55;
  EXPECT_THROW(stan::prob::simplex_free(y), std::domain_error);
  y << 1.1, -0.1;
  EXPECT_THROW(stan::prob::simplex_free(y), std::domain_error);
  y.resize(0);
  EXPECT_THROW(stan::prob::simplex_free(y), std::domain_error);
}
TEST(prob_transform,simplex_rt) {
  Matrix<double,Dynamic,1> x(3);
  x << 1.0, -1.0, 2.0;
  Matrix<double,Dynamic,1> y = stan::prob::simplex_constrain(x);
  Matrix<double,Dynamic,1> xrt = stan::prob::simplex_free(y);
  EXPECT_EQ(x.size()+1,y.size());
  EXPECT_EQ(x.size(),xrt.size());
  for (unsigned int i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(x[i],xrt[i]);
  }
}


TEST(prob_transform,pos_ordered) {
  Matrix<double,Dynamic,1> x(3);
  x << 1.0, -2.0, -5.0;
  Matrix<double,Dynamic,1> y = stan::prob::pos_ordered_constrain(x);
  EXPECT_EQ(x.size(), y.size());
  EXPECT_EQ(exp(1.0), y[0]);
  EXPECT_EQ(exp(1.0) + exp(-2.0), y[1]);
  EXPECT_EQ(exp(1.0) + exp(-2.0) + exp(-5.0), y[2]);
}
TEST(prob_transform,pos_ordered_j) {
  Matrix<double,Dynamic,1> x(3);
  x << 1.0, -2.0, -5.0;
  double lp = -152.1;
  Matrix<double,Dynamic,1> y = stan::prob::pos_ordered_constrain(x,lp);
  EXPECT_EQ(x.size(), y.size());
  EXPECT_EQ(exp(1.0), y[0]);
  EXPECT_EQ(exp(1.0) + exp(-2.0), y[1]);
  EXPECT_EQ(exp(1.0) + exp(-2.0) + exp(-5.0), y[2]);
  EXPECT_EQ(-152.1 + 1.0 - 2.0 - 5.0,lp);
}
TEST(prob_transform,pos_ordered_f) {
  Matrix<double,Dynamic,1> y(3);
  y << 1.0, 1.1, 172.1;
  Matrix<double,Dynamic,1> x = stan::prob::pos_ordered_free(y);
  EXPECT_EQ(y.size(),x.size());
  EXPECT_FLOAT_EQ(log(1.0), x[0]);
  EXPECT_FLOAT_EQ(log(1.1 - 1.0), x[1]);
  EXPECT_FLOAT_EQ(log(172.1 - 1.1), x[2]);
}
TEST(prob_transform,pos_ordered_f_exception) {
  Matrix<double,Dynamic,1> y(3);
  y << -0.1, 0.0, 1.0;
  EXPECT_THROW(stan::prob::pos_ordered_free(y), std::domain_error);
  y << 0.0, 0.0, 0.0;
  EXPECT_THROW(stan::prob::pos_ordered_free(y), std::domain_error);
  y << 0.0, 1, 0.9;
  EXPECT_THROW(stan::prob::pos_ordered_free(y), std::domain_error);
}
TEST(prob_transform,pos_ordered_rt) {
  Matrix<double,Dynamic,1> x(3);
  x << -1.0, 8.0, -3.9;
  Matrix<double,Dynamic,1> y = stan::prob::pos_ordered_constrain(x);
  Matrix<double,Dynamic,1> xrt = stan::prob::pos_ordered_free(y);
  EXPECT_EQ(x.size(), xrt.size());
  for (unsigned int i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(x[i], xrt[i]);
  }
}

TEST(prob_transform,corr_matrix_j) {
  unsigned int K = 4;
  unsigned int K_choose_2 = 6; 
  Matrix<double,Dynamic,1> x(K_choose_2);
  x << -1.0, 2.0, 0.0, 1.0, 3.0, -1.5;
  double lp = -12.9;
  Matrix<double,Dynamic,Dynamic> y = stan::prob::corr_matrix_constrain(x,K,lp);
  // std::cout << "y=\n" << y;
  Matrix<double,Dynamic,1> xrt = stan::prob::corr_matrix_free(y);
  EXPECT_EQ(x.size(), xrt.size());
  for (unsigned int i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(x[i], xrt[i]);
  }
}
TEST(prob_transform,corr_matrix_constrain_exception) {
  unsigned int K = 4;
  unsigned int K_choose_2 = 6; 
  Matrix<double,Dynamic,1> x(K_choose_2-1);
  double lp = -12.9;

  EXPECT_THROW(stan::prob::corr_matrix_constrain(x, K), std::invalid_argument);
  EXPECT_THROW(stan::prob::corr_matrix_constrain(x, K, lp), std::invalid_argument);
  
  x.resize(K_choose_2+1);
  EXPECT_THROW(stan::prob::corr_matrix_constrain(x, K), std::invalid_argument);
  EXPECT_THROW(stan::prob::corr_matrix_constrain(x, K, lp), std::invalid_argument);
}
TEST(prob_transform,corr_matrix_rt) {
  unsigned int K = 4;
  unsigned int K_choose_2 = 6; 
  Matrix<double,Dynamic,1> x(K_choose_2);
  x << -1.0, 2.0, 0.0, 1.0, 3.0, -1.5;
  Matrix<double,Dynamic,Dynamic> y = stan::prob::corr_matrix_constrain(x,K);
  // std::cout << "y=\n" << y;
  Matrix<double,Dynamic,1> xrt = stan::prob::corr_matrix_free(y);
  EXPECT_EQ(x.size(), xrt.size());
  for (unsigned int i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(x[i], xrt[i]);
  }
}
TEST(prob_transform,corr_matrix_free_exception) {
  Matrix<double,Dynamic,Dynamic> y;
  
  EXPECT_THROW(stan::prob::corr_matrix_free(y), std::domain_error);
  y.resize(0,10);
  EXPECT_THROW(stan::prob::corr_matrix_free(y), std::domain_error);
  y.resize(10,0);
  EXPECT_THROW(stan::prob::corr_matrix_free(y), std::domain_error);
  y.resize(1,2);
  EXPECT_THROW(stan::prob::corr_matrix_free(y), std::domain_error);

  y.resize(2,2);
  y << 0, 0, 0, 0;
  EXPECT_THROW(stan::prob::corr_matrix_free(y), std::runtime_error);
}
TEST(prob_transform,cov_matrix_rt) {
  unsigned int K = 4;
  unsigned int K_choose_2 = 6; 
  Matrix<double,Dynamic,1> x(K_choose_2 + K);
  x << -1.0, 2.0, 0.0, 1.0, 3.0, -1.5,
    1.0, 2.0, -1.5, 2.5;
  Matrix<double,Dynamic,Dynamic> y = stan::prob::cov_matrix_constrain(x,K);
  // std::cout << "y=\n" << y;
  Matrix<double,Dynamic,1> xrt = stan::prob::cov_matrix_free(y);
  EXPECT_EQ(x.size(), xrt.size());
  for (unsigned int i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(x[i], xrt[i]);
  }
}
TEST(prob_transform,cov_matrix_free_exception) {
  Matrix<double,Dynamic,Dynamic> y;
  
  EXPECT_THROW(stan::prob::cov_matrix_free(y), std::domain_error);
  y.resize(0,10);
  EXPECT_THROW(stan::prob::cov_matrix_free(y), std::domain_error);
  y.resize(10,0);
  EXPECT_THROW(stan::prob::cov_matrix_free(y), std::domain_error);
  y.resize(1,2);
  EXPECT_THROW(stan::prob::cov_matrix_free(y), std::domain_error);

  y.resize(2,2);
  y << 0, 0, 0, 0;
  EXPECT_THROW(stan::prob::cov_matrix_free(y), std::runtime_error);
}





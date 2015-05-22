#include <stan/math/prim/mat/fun/factor_U.hpp>
#include <stan/math/prim/mat/fun/determinant.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/factor_cov_matrix.hpp>
#include <stan/math/prim/mat/fun/read_corr_L.hpp>
#include <stan/math/prim/mat/fun/read_corr_matrix.hpp>
#include <stan/math/prim/mat/fun/read_cov_L.hpp>
#include <stan/math/prim/mat/fun/read_cov_matrix.hpp>
#include <stan/math/prim/mat/fun/make_nu.hpp>
#include <stan/math/prim/scal/fun/identity_constrain.hpp>
#include <stan/math/prim/scal/fun/identity_free.hpp>
#include <stan/math/prim/scal/fun/positive_constrain.hpp>
#include <stan/math/prim/scal/fun/positive_free.hpp>
#include <stan/math/prim/scal/fun/lb_constrain.hpp>
#include <stan/math/prim/scal/fun/lb_free.hpp>
#include <stan/math/prim/scal/fun/ub_constrain.hpp>
#include <stan/math/prim/scal/fun/ub_free.hpp>
#include <stan/math/prim/scal/fun/lub_constrain.hpp>
#include <stan/math/prim/scal/fun/lub_free.hpp>
#include <stan/math/prim/scal/fun/prob_constrain.hpp>
#include <stan/math/prim/scal/fun/prob_free.hpp>
#include <stan/math/prim/scal/fun/corr_constrain.hpp>
#include <stan/math/prim/scal/fun/corr_free.hpp>
#include <stan/math/prim/mat/fun/unit_vector_constrain.hpp>
#include <stan/math/prim/mat/fun/unit_vector_free.hpp>
#include <stan/math/prim/mat/fun/simplex_constrain.hpp>
#include <stan/math/prim/mat/fun/simplex_free.hpp>
#include <stan/math/prim/mat/fun/ordered_constrain.hpp>
#include <stan/math/prim/mat/fun/ordered_free.hpp>
#include <stan/math/prim/mat/fun/positive_ordered_constrain.hpp>
#include <stan/math/prim/mat/fun/positive_ordered_free.hpp>
#include <stan/math/prim/mat/fun/cholesky_factor_constrain.hpp>
#include <stan/math/prim/mat/fun/cholesky_factor_free.hpp>
#include <stan/math/prim/mat/fun/cholesky_corr_constrain.hpp>
#include <stan/math/prim/mat/fun/cholesky_corr_free.hpp>
#include <stan/math/prim/mat/fun/corr_matrix_constrain.hpp>
#include <stan/math/prim/mat/fun/corr_matrix_free.hpp>
#include <stan/math/prim/mat/fun/cov_matrix_constrain.hpp>
#include <stan/math/prim/mat/fun/cov_matrix_free.hpp>
#include <stan/math/prim/mat/fun/cov_matrix_constrain_lkj.hpp>
#include <stan/math/prim/mat/fun/cov_matrix_free_lkj.hpp>
#include <test/unit/util.hpp>

#include <gtest/gtest.h>

using Eigen::Matrix;
using Eigen::Dynamic;

TEST(prob_transform,identity) {
  EXPECT_FLOAT_EQ(4.0, stan::math::identity_constrain(4.0));
}
TEST(prob_transform,identity_j) {
  double lp = 1.0;
  EXPECT_FLOAT_EQ(4.0, stan::math::identity_constrain(4.0,lp));
  EXPECT_FLOAT_EQ(1.0,lp);
}
TEST(prob_transform,identity_free) {
  EXPECT_FLOAT_EQ(4.0, stan::math::identity_free(4.0));
}
TEST(prob_transform,identity_rt) {
  double x = 1.2;
  double xc = stan::math::identity_constrain(x);
  double xcf = stan::math::identity_free(xc);
  EXPECT_FLOAT_EQ(x,xcf);

  double y = -1.0;
  double yf = stan::math::identity_free(y);
  double yfc = stan::math::identity_constrain(yf);
  EXPECT_FLOAT_EQ(y,yfc);
}


TEST(prob_transform, positive) {
  EXPECT_FLOAT_EQ(exp(-1.0), stan::math::positive_constrain(-1.0));
}
TEST(prob_transform, positive_j) {
  double lp = 15.0;
  EXPECT_FLOAT_EQ(exp(-1.0), stan::math::positive_constrain(-1.0,lp));
  EXPECT_FLOAT_EQ(15.0 - 1.0, lp);
}
TEST(prob_transform, positive_f) {
  EXPECT_FLOAT_EQ(log(0.5), stan::math::positive_free(0.5));
}
TEST(prob_transform, positive_f_exception) {
  EXPECT_THROW (stan::math::positive_free(-1.0), std::domain_error);
}
TEST(prob_transform, positive_rt) {
  double x = -1.0;
  double xc = stan::math::positive_constrain(x);
  double xcf = stan::math::positive_free(xc);
  EXPECT_FLOAT_EQ(x,xcf);
  double xcfc = stan::math::positive_constrain(xcf);
  EXPECT_FLOAT_EQ(xc,xcfc);
}


TEST(prob_transform, lb) {
  EXPECT_FLOAT_EQ(exp(-1.0) + 2.0, stan::math::lb_constrain(-1.0,2.0));
  EXPECT_FLOAT_EQ(7.9, 
                  stan::math::lb_constrain(7.9, -std::numeric_limits<double>::infinity()));
}
TEST(prob_transform, lb_j) {
  double lp = 15.0;
  EXPECT_FLOAT_EQ(exp(-1.0) + 2.0, stan::math::lb_constrain(-1.0,2.0,lp));
  EXPECT_FLOAT_EQ(15.0 - 1.0, lp);

  double lp2 = 8.6;
  EXPECT_FLOAT_EQ(7.9, 
                  stan::math::lb_constrain(7.9, -std::numeric_limits<double>::infinity(),
                                           lp2));
  EXPECT_FLOAT_EQ(8.6, lp2);
}
TEST(prob_transform, lb_f) {
  EXPECT_FLOAT_EQ(log(3.0 - 2.0), stan::math::lb_free(3.0,2.0));
  EXPECT_FLOAT_EQ(1.7, stan::math::lb_free(1.7, -std::numeric_limits<double>::infinity()));
}
TEST(prob_transform, lb_f_exception) {
  double lb = 2.0;
  EXPECT_THROW (stan::math::lb_free(lb - 0.01, lb), std::domain_error);
}
TEST(prob_transform, lb_rt) {
  double x = -1.0;
  double xc = stan::math::lb_constrain(x,2.0);
  double xcf = stan::math::lb_free(xc,2.0);
  EXPECT_FLOAT_EQ(x,xcf);
  double xcfc = stan::math::lb_constrain(xcf,2.0);
  EXPECT_FLOAT_EQ(xc,xcfc);
}

TEST(prob_transform, ub) {
  EXPECT_FLOAT_EQ(2.0 - exp(-1.0), stan::math::ub_constrain(-1.0,2.0));
  EXPECT_FLOAT_EQ(1.7, 
                  stan::math::ub_constrain(1.7, 
                                           std::numeric_limits<double>::infinity()));
}
TEST(prob_transform, ub_j) {
  double lp = 15.0;
  EXPECT_FLOAT_EQ(2.0 - exp(-1.0), stan::math::ub_constrain(-1.0,2.0,lp));
  EXPECT_FLOAT_EQ(15.0 - 1.0, lp);

  double lp2 = 1.87;
  EXPECT_FLOAT_EQ(-5.2, stan::math::ub_constrain(-5.2,
                                                 std::numeric_limits<double>::infinity(),
                                                 lp2));
  EXPECT_FLOAT_EQ(1.87,lp2);
}
TEST(prob_transform, ub_f) {
  double y = 2.0;
  double U = 4.0;
  EXPECT_FLOAT_EQ(log(-(y - U)), stan::math::ub_free(2.0,4.0));

  EXPECT_FLOAT_EQ(19.765, 
                  stan::math::ub_free(19.765,
                                      std::numeric_limits<double>::infinity()));
}
TEST(prob_transform, ub_f_exception) {
  double ub = 4.0;
  EXPECT_THROW (stan::math::ub_free(ub+0.01, ub), std::domain_error);
}
TEST(prob_transform, ub_rt) {
  double x = -1.0;
  double xc = stan::math::ub_constrain(x,2.0);
  double xcf = stan::math::ub_free(xc,2.0);
  EXPECT_FLOAT_EQ(x,xcf);
  double xcfc = stan::math::ub_constrain(xcf,2.0);
  EXPECT_FLOAT_EQ(xc,xcfc);
}


TEST(prob_transform, lub) {
  EXPECT_FLOAT_EQ(2.0 + (5.0 - 2.0) * stan::math::inv_logit(-1.0), 
                  stan::math::lub_constrain(-1.0,2.0,5.0));

  EXPECT_FLOAT_EQ(1.7, 
                  stan::math::lub_constrain(1.7,
                                            -std::numeric_limits<double>::infinity(),
                                            +std::numeric_limits<double>::infinity()));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(1.8,3.0),
                  stan::math::lub_constrain(1.8,
                                            3.0,
                                            +std::numeric_limits<double>::infinity()));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(1.9,-12.5),
                  stan::math::lub_constrain(1.9,
                                            -std::numeric_limits<double>::infinity(),
                                            -12.5));
}
TEST(prob_transform, lub_j) {
  double lp = -17.0;
  double L = 2.0;
  double U = 5.0;
  double x = -1.0;
  EXPECT_FLOAT_EQ(L + (U - L) * stan::math::inv_logit(x), 
                  stan::math::lub_constrain(x,L,U,lp));
  EXPECT_FLOAT_EQ(-17.0 + log(U - L) + log(stan::math::inv_logit(x)) 
                  + log(1.0 - stan::math::inv_logit(x)),
                  lp);

  double lp1 = -12.9;
  EXPECT_FLOAT_EQ(1.7, 
                  stan::math::lub_constrain(1.7,
                                            -std::numeric_limits<double>::infinity(),
                                            +std::numeric_limits<double>::infinity(),
                                            lp1));
  EXPECT_FLOAT_EQ(-12.9,lp1);

  double lp2 = -19.8;
  double lp2_expected = -19.8;
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(1.8,3.0,lp2_expected),
                  stan::math::lub_constrain(1.8,
                                            3.0,
                                            +std::numeric_limits<double>::infinity(),
                                            lp2));
  EXPECT_FLOAT_EQ(lp2_expected, lp2);

  double lp3 = -422;
  double lp3_expected = -422;
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(1.9,-12.5,lp3_expected),
                  stan::math::lub_constrain(1.9,
                                            -std::numeric_limits<double>::infinity(),
                                            -12.5,
                                            lp3));
  EXPECT_FLOAT_EQ(lp3_expected,lp3);
  
}
TEST(ProbTransform, lubException) {
  using stan::math::lub_constrain;
  EXPECT_THROW(lub_constrain(5.0,1.0,1.0), std::domain_error);
  EXPECT_NO_THROW(lub_constrain(5.0,1.0,1.01));
  double lp = 12;
  EXPECT_THROW(lub_constrain(5.0,1.0,1.0,lp), std::domain_error);
  EXPECT_NO_THROW(lub_constrain(5.0,1.0,1.01,lp));
}
TEST(prob_transform, lub_f) {
  double L = -10.0;
  double U = 27.0;
  double y = 3.0;
  EXPECT_FLOAT_EQ(stan::math::logit((y - L) / (U - L)),
                  stan::math::lub_free(y,L,U));
  
  EXPECT_FLOAT_EQ(14.2,
                  stan::math::lub_free(14.2,
                                       -std::numeric_limits<double>::infinity(),
                                       std::numeric_limits<double>::infinity()));
  EXPECT_FLOAT_EQ(stan::math::ub_free(-18.3,7.6),
                  stan::math::lub_free(-18.3,
                                       -std::numeric_limits<double>::infinity(),
                                       7.6));
  EXPECT_FLOAT_EQ(stan::math::lb_free(763.9, -3122.2),
                  stan::math::lub_free(763.9,
                                       -3122.2,
                                       std::numeric_limits<double>::infinity()));
}
TEST(prob_transform, lub_f_exception) {
  double L = -10.0;
  double U = 27.0;
  EXPECT_THROW(stan::math::lub_free (L-0.01,L,U), std::domain_error);
  EXPECT_THROW(stan::math::lub_free (U+0.01,L,U), std::domain_error);

  EXPECT_THROW(stan::math::lub_free ((L+U)/2,U,L), std::domain_error);
}
TEST(prob_transform, lub_rt) {
  double x = -1.0;
  double xc = stan::math::lub_constrain(x,2.0,4.0);
  double xcf = stan::math::lub_free(xc,2.0,4.0);
  EXPECT_FLOAT_EQ(x,xcf);
  double xcfc = stan::math::lub_constrain(xcf,2.0,4.0);
  EXPECT_FLOAT_EQ(xc,xcfc);
}


TEST(prob_transform, prob) {
  EXPECT_FLOAT_EQ(stan::math::inv_logit(-1.0), 
                  stan::math::prob_constrain(-1.0));
}
TEST(prob_transform, prob_j) {
  double lp = -17.0;
  double L = 0.0;
  double U = 1.0;
  double x = -1.0;
  EXPECT_FLOAT_EQ(L + (U - L) * stan::math::inv_logit(x), 
                  stan::math::prob_constrain(x,lp));
  EXPECT_FLOAT_EQ(-17.0 + log(U - L) + log(stan::math::inv_logit(x)) 
                  + log(1.0 - stan::math::inv_logit(x)),
                  lp);
}
TEST(prob_transform, prob_f) {
  double L = 0.0;
  double U = 1.0;
  double y = 0.4;
  EXPECT_FLOAT_EQ(stan::math::logit((y - L) / (U - L)),
                  stan::math::prob_free(y));
}
TEST(prob_transform, prob_f_exception) {
  EXPECT_THROW (stan::math::prob_free(1.1), std::domain_error);
  EXPECT_THROW (stan::math::prob_free(-0.1), std::domain_error);
}
TEST(prob_transform, prob_rt) {
  double x = -1.0;
  double xc = stan::math::prob_constrain(x);
  double xcf = stan::math::prob_free(xc);
  EXPECT_FLOAT_EQ(x,xcf);
  double xcfc = stan::math::prob_constrain(xcf);
  EXPECT_FLOAT_EQ(xc,xcfc);
}


TEST(prob_transform, corr) {
  EXPECT_FLOAT_EQ(std::tanh(-1.0), 
                  stan::math::corr_constrain(-1.0));
}
TEST(prob_transform, corr_j) {
  double lp = -17.0;
  double x = -1.0;
  EXPECT_FLOAT_EQ(std::tanh(x), 
                  stan::math::corr_constrain(x,lp));
  EXPECT_FLOAT_EQ(-17.0 + (log(1.0 - std::tanh(x) * std::tanh(x))),
                  lp);
}
TEST(prob_transform, corr_f) {
  EXPECT_FLOAT_EQ(atanh(-0.4), 0.5 * std::log((1.0 + -0.4)/(1.0 - -0.4)));
  double y = -0.4;
  EXPECT_FLOAT_EQ(atanh(y),
                  stan::math::corr_free(y));
}
TEST(prob_transform, corr_rt) {
  double x = -1.0;
  double xc = stan::math::corr_constrain(x);
  double xcf = stan::math::corr_free(xc);
  EXPECT_FLOAT_EQ(x,xcf);
  double xcfc = stan::math::corr_constrain(xcf);
  EXPECT_FLOAT_EQ(xc,xcfc);
}


TEST(prob_transform,ordered) {
  Matrix<double,Dynamic,1> x(3);
  x << -15.0, -2.0, -5.0;
  Matrix<double,Dynamic,1> y = stan::math::ordered_constrain(x);
  EXPECT_EQ(x.size(), y.size());
  EXPECT_EQ(-15.0, y[0]);
  EXPECT_EQ(-15.0 + exp(-2.0), y[1]);
  EXPECT_EQ(-15.0 + exp(-2.0) + exp(-5.0), y[2]);
}
TEST(prob_transform,ordered_j) {
  Matrix<double,Dynamic,1> x(3);
  x << 1.0, -2.0, -5.0;
  double lp = -152.1;
  Matrix<double,Dynamic,1> y = stan::math::ordered_constrain(x,lp);
  EXPECT_EQ(x.size(), y.size());
  EXPECT_EQ(1.0, y[0]);
  EXPECT_EQ(1.0 + exp(-2.0), y[1]);
  EXPECT_EQ(1.0 + exp(-2.0) + exp(-5.0), y[2]);
  EXPECT_EQ(-152.1 - 2.0 - 5.0,lp);
}
TEST(prob_transform,ordered_f) {
  Matrix<double,Dynamic,1> y(3);
  y << -12.0, 1.1, 172.1;
  Matrix<double,Dynamic,1> x = stan::math::ordered_free(y);
  EXPECT_EQ(y.size(),x.size());
  EXPECT_FLOAT_EQ(-12.0, x[0]);
  EXPECT_FLOAT_EQ(log(1.1 + 12.0), x[1]);
  EXPECT_FLOAT_EQ(log(172.1 - 1.1), x[2]);
}
TEST(prob_transform,ordered_f_exception) {
  Matrix<double,Dynamic,1> y(3);
  y << -0.1, 0.0, 1.0;
  EXPECT_NO_THROW(stan::math::ordered_free(y));
  y << 0.0, 0.0, 0.0;
  EXPECT_THROW(stan::math::ordered_free(y), std::domain_error);
  y << 0.0, 1, 0.9;
  EXPECT_THROW(stan::math::ordered_free(y), std::domain_error);
}
TEST(prob_transform,ordered_rt) {
  Matrix<double,Dynamic,1> x(3);
  x << -1.0, 8.0, -3.9;
  Matrix<double,Dynamic,1> y = stan::math::ordered_constrain(x);
  Matrix<double,Dynamic,1> xrt = stan::math::ordered_free(y);
  EXPECT_EQ(x.size(), xrt.size());
  for (int i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(x[i], xrt[i]);
  }
}

TEST(prob_transform,positive_ordered) {
  Matrix<double,Dynamic,1> x(3);
  x << -15.0, -2.0, -5.0;
  Matrix<double,Dynamic,1> y = stan::math::positive_ordered_constrain(x);
  EXPECT_EQ(x.size(), y.size());
  EXPECT_EQ(exp(-15.0), y[0]);
  EXPECT_EQ(exp(-15.0) + exp(-2.0), y[1]);
  EXPECT_EQ(exp(-15.0) + exp(-2.0) + exp(-5.0), y[2]);
}
TEST(prob_transform,positive_ordered_j) {
  Matrix<double,Dynamic,1> x(3);
  x << 1.0, -2.0, -5.0;
  double lp = -152.1;
  Matrix<double,Dynamic,1> y = stan::math::positive_ordered_constrain(x,lp);
  EXPECT_EQ(x.size(), y.size());
  EXPECT_EQ(exp(1.0), y[0]);
  EXPECT_EQ(exp(1.0) + exp(-2.0), y[1]);
  EXPECT_EQ(exp(1.0) + exp(-2.0) + exp(-5.0), y[2]);
  EXPECT_EQ(-152.1 + 1.0 - 2.0 - 5.0,lp);
}
TEST(prob_transform,positive_ordered_f) {
  Matrix<double,Dynamic,1> y(3);
  y << 0.12, 1.1, 172.1;
  Matrix<double,Dynamic,1> x = stan::math::positive_ordered_free(y);
  EXPECT_EQ(y.size(),x.size());
  EXPECT_FLOAT_EQ(log(0.12), x[0]);
  EXPECT_FLOAT_EQ(log(1.1 - 0.12), x[1]);
  EXPECT_FLOAT_EQ(log(172.1 - 1.1), x[2]);
}
TEST(prob_transform,positive_ordered_f_exception) {
  Matrix<double,Dynamic,1> y(3);
  y << -0.1, 0.0, 1.0;
  EXPECT_THROW(stan::math::positive_ordered_free(y), std::domain_error);
  y << 0.0, 0.0, 0.0;
  EXPECT_THROW(stan::math::positive_ordered_free(y), std::domain_error);
  y << 0.0, 1, 0.9;
  EXPECT_THROW(stan::math::positive_ordered_free(y), std::domain_error);
}
TEST(prob_transform,positive_ordered_rt) {
  Matrix<double,Dynamic,1> x(3);
  x << -1.0, 8.0, -3.9;
  Matrix<double,Dynamic,1> y = stan::math::positive_ordered_constrain(x);
  Matrix<double,Dynamic,1> xrt = stan::math::positive_ordered_free(y);
  EXPECT_EQ(x.size(), xrt.size());
  for (int i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(x[i], xrt[i]);
  }
}

TEST(prob_transform,corr_matrix_j) {
  size_t K = 4;
  size_t K_choose_2 = 6; 
  Matrix<double,Dynamic,1> x(K_choose_2);
  x << -1.0, 2.0, 0.0, 1.0, 3.0, -1.5;
  double lp = -12.9;
  Matrix<double,Dynamic,Dynamic> y = stan::math::corr_matrix_constrain(x,K,lp);
  Matrix<double,Dynamic,1> xrt = stan::math::corr_matrix_free(y);
  EXPECT_EQ(x.size(), xrt.size());
  for (int i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(x[i], xrt[i]);
  }
}
TEST(prob_transform,corr_matrix_j2x2) {
  // tests K=2 boundary case, which has a different implementation
  size_t K = 2;
  size_t K_choose_2 = 1; 
  Matrix<double,Dynamic,1> x(K_choose_2);
  x << -1.3;
  double lp = -12.9;
  Matrix<double,Dynamic,Dynamic> y = stan::math::corr_matrix_constrain(x,K,lp);
  Matrix<double,Dynamic,1> xrt = stan::math::corr_matrix_free(y);
  EXPECT_EQ(x.size(), xrt.size());
  for (int i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(x[i], xrt[i]);
  }
}

TEST(prob_transform,corr_matrix_constrain_exception) {
  unsigned int K = 4;
  unsigned int K_choose_2 = 6; 
  Matrix<double,Dynamic,1> x(K_choose_2-1);
  double lp = -12.9;

  EXPECT_THROW(stan::math::corr_matrix_constrain(x, K), std::invalid_argument);
  EXPECT_THROW(stan::math::corr_matrix_constrain(x, K, lp), std::invalid_argument);
  
  x.resize(K_choose_2+1);
  EXPECT_THROW(stan::math::corr_matrix_constrain(x, K), std::invalid_argument);
  EXPECT_THROW(stan::math::corr_matrix_constrain(x, K, lp), std::invalid_argument);
}
TEST(prob_transform,corr_matrix_rt) {
  unsigned int K = 4;
  unsigned int K_choose_2 = 6; 
  Matrix<double,Dynamic,1> x(K_choose_2);
  x << -1.0, 2.0, 0.0, 1.0, 3.0, -1.5;
  Matrix<double,Dynamic,Dynamic> y = stan::math::corr_matrix_constrain(x,K);
  Matrix<double,Dynamic,1> xrt = stan::math::corr_matrix_free(y);
  EXPECT_EQ(x.size(), xrt.size());
  for (int i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(x[i], xrt[i]);
  }
}
TEST(prob_transform,corr_matrix_free_exception) {
  Matrix<double,Dynamic,Dynamic> y;
  
  EXPECT_THROW(stan::math::corr_matrix_free(y), std::domain_error);
  y.resize(0,10);
  EXPECT_THROW(stan::math::corr_matrix_free(y), std::domain_error);
  y.resize(10,0);
  EXPECT_THROW(stan::math::corr_matrix_free(y), std::domain_error);
  y.resize(1,2);
  EXPECT_THROW(stan::math::corr_matrix_free(y), std::domain_error);

  y.resize(2,2);
  y << 0, 0, 0, 0;
  EXPECT_THROW(stan::math::corr_matrix_free(y), std::runtime_error);
}



TEST(prob_transform,lkj_cov_matrix_rt) {
  unsigned int K = 4;
  unsigned int K_choose_2 = 6; 
  Matrix<double,Dynamic,1> x(K_choose_2 + K);
  x << -1.0, 2.0, 0.0, 1.0, 3.0, -1.5,
    1.0, 2.0, -1.5, 2.5;
  Matrix<double,Dynamic,Dynamic> y = stan::math::cov_matrix_constrain_lkj(x,K);
  Matrix<double,Dynamic,1> xrt = stan::math::cov_matrix_free_lkj(y);
  EXPECT_EQ(x.size(), xrt.size());
  for (int i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(x[i], xrt[i]);
  }
}
TEST(prob_transform,lkj_cov_matrix_free_exception) {
  Matrix<double,Dynamic,Dynamic> y(0,0);
  
  EXPECT_THROW(stan::math::cov_matrix_free_lkj(y), std::domain_error);
  y.resize(0,10);
  EXPECT_THROW(stan::math::cov_matrix_free_lkj(y), std::domain_error);
  y.resize(10,0);
  EXPECT_THROW(stan::math::cov_matrix_free_lkj(y), std::domain_error);
  y.resize(1,2);
  EXPECT_THROW(stan::math::cov_matrix_free_lkj(y), std::domain_error);

  y.resize(2,2);
  y << 0, 0, 0, 0;
  EXPECT_THROW(stan::math::cov_matrix_free_lkj(y), std::runtime_error);
}

TEST(prob_transform,cov_matrix_rt) {
  unsigned int K = 4;
  unsigned int K_choose_2 = 6; 
  Matrix<double,Dynamic,1> x(K_choose_2 + K);
  x << -1.0, 2.0, 0.0, 1.0, 3.0, -1.5,
    1.0, 2.0, -1.5, 2.5;
  Matrix<double,Dynamic,Dynamic> y = stan::math::cov_matrix_constrain(x,K);
  Matrix<double,Dynamic,1> xrt = stan::math::cov_matrix_free(y);
  EXPECT_EQ(x.size(), xrt.size());
  for (int i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(x[i], xrt[i]);
  }
}
TEST(prob_transform,cov_matrix_constrain_exception) {
  Matrix<double,Dynamic,1> x(7);
  int K = 12;
  EXPECT_THROW(stan::math::cov_matrix_constrain(x,K), std::domain_error);
}
TEST(prob_transform,cov_matrix_free_exception) {
  Matrix<double,Dynamic,Dynamic> y(0,0);
  
  EXPECT_THROW(stan::math::cov_matrix_free(y), std::domain_error);
  y.resize(0,10);
  EXPECT_THROW(stan::math::cov_matrix_free(y), std::domain_error);
  y.resize(10,0);
  EXPECT_THROW(stan::math::cov_matrix_free(y), std::domain_error);
  y.resize(1,2);
  EXPECT_THROW(stan::math::cov_matrix_free(y), std::domain_error);

  y.resize(2,2);
  y << 0, 0, 0, 0;
  EXPECT_THROW(stan::math::cov_matrix_free(y), std::domain_error);
}

TEST(prob_transform,simplex_rt0) {
  Matrix<double,Dynamic,1> x(4);
  x << 0.0, 0.0, 0.0, 0.0;
  Matrix<double,Dynamic,1> y = stan::math::simplex_constrain(x);
  EXPECT_FLOAT_EQ(1.0 / 5.0, y(0));
  EXPECT_FLOAT_EQ(1.0 / 5.0, y(1));
  EXPECT_FLOAT_EQ(1.0 / 5.0, y(2));
  EXPECT_FLOAT_EQ(1.0 / 5.0, y(3));
  EXPECT_FLOAT_EQ(1.0 / 5.0, y(4));

  Matrix<double,Dynamic,1> xrt = stan::math::simplex_free(y);
  EXPECT_EQ(x.size()+1,y.size());
  EXPECT_EQ(x.size(),xrt.size());
  for (int i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(x[i],xrt[i],1E-10);
  }
}
TEST(prob_transform,simplex_rt) {
  Matrix<double,Dynamic,1> x(3);
  x << 1.0, -1.0, 2.0;
  Matrix<double,Dynamic,1> y = stan::math::simplex_constrain(x);
  Matrix<double,Dynamic,1> xrt = stan::math::simplex_free(y);
  EXPECT_EQ(x.size()+1,y.size());
  EXPECT_EQ(x.size(),xrt.size());
  for (int i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(x[i],xrt[i]);
  }
}
TEST(prob_transform,simplex_match) {
  Matrix<double,Dynamic,1> x(3);
  x << 1.0, -1.0, 2.0;
  double lp;
  Matrix<double,Dynamic,1> y = stan::math::simplex_constrain(x);
  Matrix<double,Dynamic,1> y2 = stan::math::simplex_constrain(x,lp);

  EXPECT_EQ(4,y.size());
  EXPECT_EQ(4,y2.size());
  for (int i = 0; i < x.size(); ++i)
    EXPECT_FLOAT_EQ(y[i],y2[i]);
}
TEST(prob_transform,simplex_f_exception) {
  Matrix<double,Dynamic,1> y(2);
  y << 0.5, 0.55;
  EXPECT_THROW(stan::math::simplex_free(y), std::domain_error);
  y << 1.1, -0.1;
  EXPECT_THROW(stan::math::simplex_free(y), std::domain_error);
}

TEST(prob_transform,unit_vector_rt0) {
  Matrix<double,Dynamic,1> x(4);
  x << 0.0, 0.0, 0.0, 0.0;
  Matrix<double,Dynamic,1> y = stan::math::unit_vector_constrain(x);
  EXPECT_NEAR(0, y(0), 1e-8);
  EXPECT_NEAR(0, y(1), 1e-8);
  EXPECT_NEAR(0, y(2), 1e-8);
  EXPECT_NEAR(0, y(3), 1e-8);
  EXPECT_NEAR(1.0, y(4), 1e-8);

  Matrix<double,Dynamic,1> xrt = stan::math::unit_vector_free(y);
  EXPECT_EQ(x.size()+1,y.size());
  EXPECT_EQ(x.size(),xrt.size());
  for (int i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(x[i],xrt[i],1E-10);
  }
}
TEST(prob_transform,unit_vector_rt) {
  Matrix<double,Dynamic,1> x(3);
  x << 1.0, -1.0, 1.0;
  Matrix<double,Dynamic,1> y = stan::math::unit_vector_constrain(x);
  Matrix<double,Dynamic,1> xrt = stan::math::unit_vector_free(y);
  EXPECT_EQ(x.size()+1,y.size());
  EXPECT_EQ(x.size(),xrt.size());
  for (int i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(x[i],xrt[i]) << "error in component " << i;
  }
}
TEST(prob_transform,unit_vector_match) {
  Matrix<double,Dynamic,1> x(3);
  x << 1.0, -1.0, 2.0;
  double lp;
  Matrix<double,Dynamic,1> y = stan::math::unit_vector_constrain(x);
  Matrix<double,Dynamic,1> y2 = stan::math::unit_vector_constrain(x,lp);

  EXPECT_EQ(4,y.size());
  EXPECT_EQ(4,y2.size());
  for (int i = 0; i < x.size(); ++i)
    EXPECT_FLOAT_EQ(y[i],y2[i]) << "error in component " << i;
}
TEST(prob_transform,unit_vector_f_exception) {
  Matrix<double,Dynamic,1> y(2);
  y << 0.5, 0.55;
  EXPECT_THROW(stan::math::unit_vector_free(y), std::domain_error);
  y << 1.1, -0.1;
  EXPECT_THROW(stan::math::unit_vector_free(y), std::domain_error);
}


TEST(ProbTransform,choleskyFactor) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::cholesky_factor_constrain;
  using stan::math::cholesky_factor_free;
  
  Matrix<double,Dynamic,1> x(3);
  x << 1, 2, 3;
  
  Matrix<double,Dynamic,Dynamic> y
    = cholesky_factor_constrain(x,2,2);

  Matrix<double,Dynamic,1> x2
    = cholesky_factor_free(y);
  
  EXPECT_EQ(x2.size(), x.size());
  EXPECT_EQ(x2.rows(), x.rows());
  EXPECT_EQ(x2.cols(), x.cols());
  for (int i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(x(i), x2(i));
}
TEST(ProbTransform,choleskyFactorLogJacobian) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::cholesky_factor_constrain;

  double lp;
  Matrix<double,Dynamic,1> x(3);

  x.resize(1);
  x << 2.3;
  lp = 1.9;
  cholesky_factor_constrain(x,1,1,lp);
  EXPECT_FLOAT_EQ(1.9 + 2.3, lp);
  
  x.resize(3);
  x << 
    1, 
    2, 3;
  lp = 7.2;
  cholesky_factor_constrain(x,2,2,lp);
  EXPECT_FLOAT_EQ(7.2 + 1 + 3, lp);

  x.resize(6);
  x << 
    1.001,
    2, 3.01,
    4, 5, 6.1;
  lp = 1.2;
  cholesky_factor_constrain(x,3,3,lp);
  EXPECT_FLOAT_EQ(1.2 + 1.001 + 3.01 + 6.1, lp);

  x.resize(9);
  lp = 1.2;
  x << 
    1.001,
    2, 3.01,
    4, 5, 6.1,
    7, 8, 9;
  cholesky_factor_constrain(x,4,3,lp);
  EXPECT_FLOAT_EQ(1.2 + 1.001 + 3.01 + 6.1, lp);

}
TEST(ProbTransform,choleskyFactorConstrainError) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::cholesky_factor_constrain;

  Matrix<double,Dynamic,1> x(3);
  x << 1, 2, 3;
  EXPECT_THROW(cholesky_factor_constrain(x,9,9),std::domain_error);
  double lp = 0;
  EXPECT_THROW(cholesky_factor_constrain(x,9,9,lp),std::domain_error);
}
TEST(ProbTransform,choleskyFactorFreeError) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::cholesky_factor_free;

  Matrix<double,Dynamic,Dynamic> y(1,1);
  y.resize(1,1);
  y << -2;
  EXPECT_THROW(cholesky_factor_free(y),std::domain_error);

  y.resize(2,2);
  y << 1, 2, 3, 4;
  EXPECT_THROW(cholesky_factor_free(y),std::domain_error);

  y.resize(2,3);
  y << 1, 0, 0,
    2, 3, 0;
  EXPECT_THROW(cholesky_factor_free(y),std::domain_error);
}


TEST(ProbTransform,CholeskyCorrelation4) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  Matrix<double,Dynamic,Dynamic> L(4,4);
  L << 
    1, 0, 0, 0,
    -0.2, 0.9797959, 0, 0,
    0.5, -0.3, 0.8124038, 0,
    0.7, -0.2, 0.6, 0.3316625;

  Matrix<double,Dynamic,1> y
    = stan::math::cholesky_corr_free(L);

  Matrix<double,Dynamic,Dynamic> x
    = stan::math::cholesky_corr_constrain(y,4);
  
  Matrix<double,Dynamic,1> yrt
    = stan::math::cholesky_corr_free(x);

  EXPECT_EQ(y.size(), yrt.size());
  for (int i = 0; i < yrt.size(); ++i)
    EXPECT_FLOAT_EQ(y(i), yrt(i));

  for (int m = 0; m < 4; ++m)
    for (int n = 0; n < 4; ++n)
      EXPECT_FLOAT_EQ(L(m,n), x(m,n));
}

void 
test_cholesky_correlation_values(const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& L) {
  using std::vector;
  using stan::math::cholesky_corr_constrain;
  using stan::math::cholesky_corr_free;
  int K = L.rows();
  int K_choose_2 = (K * (K - 1)) / 2;

  // test number of free parameters
  Matrix<double,Dynamic,1> y
    = stan::math::cholesky_corr_free(L);
  EXPECT_EQ(K_choose_2, y.size());

  // test transform roundtrip without Jacobian
  Matrix<double,Dynamic,Dynamic> x
    = stan::math::cholesky_corr_constrain(y,K);
  
  Matrix<double,Dynamic,1> yrt
    = stan::math::cholesky_corr_free(x);

  EXPECT_EQ(y.size(), yrt.size());
  for (int i = 0; i < yrt.size(); ++i)
    EXPECT_FLOAT_EQ(y(i), yrt(i));

  for (int m = 0; m < K; ++m)
    for (int n = 0; n < K; ++n)
      EXPECT_FLOAT_EQ(L(m,n), x(m,n));


  // test transform roundtrip with Jacobian (Jacobian itself tested above)
  double lp;
  Matrix<double,Dynamic,Dynamic> x2
    = stan::math::cholesky_corr_constrain(y,K,lp);
  
  Matrix<double,Dynamic,1> yrt2
    = stan::math::cholesky_corr_free(x2);

  EXPECT_EQ(y.size(), yrt2.size());
  for (int i = 0; i < yrt2.size(); ++i)
    EXPECT_FLOAT_EQ(y(i), yrt2(i));

  for (int m = 0; m < K; ++m)
    for (int n = 0; n < K; ++n)
      EXPECT_FLOAT_EQ(L(m,n), x2(m,n));
}

TEST(ProbTransform,CholeskyCorrelationRoundTrips) {
  using Eigen::Matrix;
  using Eigen::Dynamic;

  Matrix<double,Dynamic,Dynamic> L1(1,1);
  L1 << 1;
  test_cholesky_correlation_values(L1);

  Matrix<double,Dynamic,Dynamic> L2(2,2);
  L2 << 
    1, 0,
    -0.5, 0.8660254;
  test_cholesky_correlation_values(L2);
    
  Matrix<double,Dynamic,Dynamic> L4(4,4);
  L4 << 
    1, 0, 0, 0,
    -0.2, 0.9797959, 0, 0,
    0.5, -0.3, 0.8124038, 0,
    0.7, -0.2, 0.6, 0.3316625;
  test_cholesky_correlation_values(L4);
}




TEST(probTransform,factorU) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using Eigen::Array;
  using stan::math::factor_U;
  int K = 3;
  Matrix<double,Dynamic,Dynamic> U(K,K);
  U << 
    1.0, -0.25, 0.75,
    0.0,  1.0,  0.487950036474267,
    0.0,  0.0,  1.0;
  Eigen::Array<double,Dynamic,1> CPCs( (K * (K - 1)) / 2);
  CPCs << 10, 100, 1000;
  factor_U(U, CPCs);
  // test that function doesn't resize itself
  EXPECT_EQ( (K * (K - 1)) / 2, CPCs.size());
  for (int i = 0; i < CPCs.size(); ++i)
    EXPECT_LE(std::tanh(std::fabs(CPCs(i))), 1.0) << CPCs(i);
}
TEST(probTransform, factorCovMatrix) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using Eigen::Array;
  using stan::math::factor_cov_matrix;

  Matrix<double,Dynamic,Dynamic> L(3,3);
  L <<
    1.7, 0, 0,
    -2.9, 14.2, 0, 
    .2, -.5, 1.3;
    
  Matrix<double,Dynamic,Dynamic> Sigma
    = L.transpose() * L;

  Array<double,Dynamic,1> CPCs(3);   // must be sized coming in
  Array<double,Dynamic,1> sds(3);    // must be sized coming in

  // just check it doesn't bomb
  factor_cov_matrix(Sigma, CPCs, sds);

  // example of sizing for K=2
  L.resize(2,2);
  L << 1.7, 0,
    -2.3, 0.5;
  Sigma = L.transpose() * L;
  CPCs.resize(1);
  sds.resize(2);
  factor_cov_matrix(Sigma, CPCs, sds);
}

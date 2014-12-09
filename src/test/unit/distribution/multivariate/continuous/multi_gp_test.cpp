#include <gtest/gtest.h>
#include <stan/agrad/rev/matrix.hpp>
#include <stan/prob/distributions/multivariate/continuous/multi_gp.hpp>

// UTILITY FUNCTIONS FOR TESTING
#include <vector>
#include <test/unit/distribution/expect_eq_diffs.hpp>
#include <test/unit/distribution/multivariate/continuous/test_gradients.hpp>
#include <test/unit/distribution/multivariate/continuous/agrad_distributions_multi_gp.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;

template <typename T_y, typename T_scale, typename T_w> 
void expect_propto(T_y y1, T_scale sigma1, T_w w1, 
                   T_y y2, T_scale sigma2, T_w w2, 
                   std::string message = "") {
  expect_eq_diffs(stan::prob::multi_gp_log<false>(y1,sigma1,w1),
                  stan::prob::multi_gp_log<false>(y2,sigma2,w2),
                  stan::prob::multi_gp_log<true>(y1,sigma1,w1),
                  stan::prob::multi_gp_log<true>(y2,sigma2,w2),
                  message);
}

using stan::agrad::var;
using stan::agrad::to_var;


TEST_F(agrad_distributions_multi_gp,Propto) {
  expect_propto(to_var(y),to_var(Sigma),to_var(w),
                to_var(y2),to_var(Sigma2),to_var(w2),
                "All vars: y, w, sigma");
}
TEST_F(agrad_distributions_multi_gp,ProptoY) {
  expect_propto(to_var(y),Sigma,w,
                to_var(y2),Sigma,w,
                "var: y");

}
TEST_F(agrad_distributions_multi_gp,ProptoYMu) {
  expect_propto(to_var(y),Sigma,to_var(w),
                to_var(y2),Sigma,to_var(w2),
                "var: y and w");
}
TEST_F(agrad_distributions_multi_gp,ProptoYSigma) {
  expect_propto(to_var(y),to_var(Sigma),w,
                to_var(y2),to_var(Sigma2),w,
                "var: y and sigma");
}
TEST_F(agrad_distributions_multi_gp,ProptoMu) {
  expect_propto(y,Sigma,to_var(w),
                y,Sigma,to_var(w2),
                "var: w");
}
TEST_F(agrad_distributions_multi_gp,ProptoMuSigma) {
  expect_propto(y,to_var(Sigma),to_var(w),
                y,to_var(Sigma2),to_var(w2),
                "var: w and sigma");
}
TEST_F(agrad_distributions_multi_gp,ProptoSigma) {
  expect_propto(y,to_var(Sigma),w,
                y,to_var(Sigma2),w,
                "var: sigma");
}


TEST(ProbDistributionsMultiGP,MultiGPVar) {
  using stan::agrad::var;
  Matrix<var,Dynamic,Dynamic> y(3,3);
  y <<  2.0, -2.0, 11.0,
       -4.0, 0.0, 2.0,
        1.0, 5.0, 3.3;
  Matrix<var,Dynamic,1> w(3,1);
  w << 1.0, 0.5, 3.0;
  Matrix<var,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
          -3.0,  4.0, 0.0,
           0.0, 0.0, 5.0;
  EXPECT_FLOAT_EQ(-46.087162, stan::prob::multi_gp_log(y,Sigma,w).val());
}

TEST(ProbDistributionsMultiGP,MultiGPGradientUnivariate) {
  using stan::agrad::var;
  using std::vector;
  using Eigen::VectorXd;
  using stan::prob::multi_gp_log;
  
  Matrix<var,Dynamic,Dynamic> y_var(1,1);
  y_var << 2.0;

  Matrix<var,Dynamic,1> w_var(1,1);
  w_var << 1.0;

  Matrix<var,Dynamic,Dynamic> Sigma_var(1,1);
  Sigma_var(0,0) = 9.0;

  std::vector<var> x;
  x.push_back(y_var(0));
  x.push_back(w_var(0));
  x.push_back(Sigma_var(0,0));

  var lp = stan::prob::multi_gp_log(y_var,Sigma_var,w_var);
  vector<double> grad;
  lp.grad(x,grad);

  // ===================================


  Matrix<double,Dynamic,Dynamic> y(1,1);
  y << 2.0;

  Matrix<double,Dynamic,1> w(1,1);
  w << 1.0;

  Matrix<double,Dynamic,Dynamic> Sigma(1,1);
  Sigma << 9.0;

  double epsilon = 1e-6;


  Matrix<double,Dynamic,Dynamic> y_m(1,1);
  Matrix<double,Dynamic,Dynamic> y_p(1,1);
  y_p(0) = y(0) + epsilon;
  y_m(0) = y(0) - epsilon;
  double grad_diff 
    =  (multi_gp_log(y_p,Sigma,w) - multi_gp_log(y_m,Sigma,w)) 
    / (2 * epsilon);
  EXPECT_FLOAT_EQ(grad_diff, grad[0]);

  Matrix<double,Dynamic,1> w_m(1,1);
  Matrix<double,Dynamic,1> w_p(1,1);
  w_p[0] = w[0] + epsilon;
  w_m[0] = w[0] - epsilon;
  grad_diff 
    =  (multi_gp_log(y,Sigma,w_p) - multi_gp_log(y,Sigma,w_m)) 
    / (2 * epsilon);
  EXPECT_FLOAT_EQ(grad_diff, grad[1]);

  Matrix<double,Dynamic,Dynamic> Sigma_m(1,1);
  Matrix<double,Dynamic,Dynamic> Sigma_p(1,1);
  Sigma_p(0) = Sigma(0) + epsilon;
  Sigma_m(0) = Sigma(0) - epsilon;
  grad_diff 
    =  (multi_gp_log(y,Sigma_p,w) - multi_gp_log(y,Sigma_m,w)) 
    / (2 * epsilon);
  EXPECT_FLOAT_EQ(grad_diff, grad[2]);
}


struct multi_gp_fun {
  const int K_, N_;

  multi_gp_fun(int K, int N) : K_(K), N_(N) { }

  template <typename T>
  T operator()(const std::vector<T>& x) const {
    using Eigen::Matrix;
    using Eigen::Dynamic;
    using stan::agrad::var;
    Matrix<T,Dynamic,Dynamic> y(K_,N_);
    Matrix<T,Dynamic,Dynamic> Sigma(N_,N_);
    Matrix<T,Dynamic,1> w(K_);

    int pos = 0;
    for (int j = 0; j < N_; ++j)
      for (int i = 0; i < K_; ++i)
        y(i,j) = x[pos++];
    for (int j = 0; j < N_; ++j) {
      for (int i = 0; i <= j; ++i) {
        Sigma(i,j) = x[pos++];
        Sigma(j,i) = Sigma(i,j);
      }
    }
    for (int i = 0; i < K_; ++i)
      w(i) = x[pos++];
    return stan::prob::multi_gp_log<false>(y,Sigma,w);
  }
};

TEST(MultiGP, TestGradFunctional) {
  std::vector<double> x(3*2 + 3 + 3);
  // y
  x[0] = 1.0;
  x[1] = 2.0;
  x[2] = -3.0;

  x[3] = 0.0;
  x[4] = -2.0;
  x[5] = -3.0;

  // Sigma
  x[6] = 1;
  x[7] = -1;
  x[8] = 10;

  // w
  x[9] = 1;
  x[10] = 10;
  x[11] = 5;



  test_grad(multi_gp_fun(3,2), x);

  std::vector<double> u(3);
  u[0] = 1.9;
  u[1] = 0.48;
  u[2] = 2.7;
  
  test_grad(multi_gp_fun(1,1), u);
}

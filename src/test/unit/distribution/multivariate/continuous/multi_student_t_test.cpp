#include <gtest/gtest.h>

#include <stan/prob/distributions/multivariate/continuous/multi_student_t.hpp>

// UTILITY FUNCTIONS FOR TESTING
#include <vector>
#include <test/unit/distribution/expect_eq_diffs.hpp>
#include <test/unit/distribution/multivariate/continuous/test_gradients.hpp>
#include <test/unit/distribution/multivariate/continuous/agrad_distributions_multi_student_t.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;

template <typename T_y, typename T_dof, typename T_loc, typename T_scale>
void expect_propto(T_y y1, T_dof nu1, T_loc mu1, T_scale sigma1,
                   T_y y2, T_dof nu2, T_loc mu2, T_scale sigma2,
                   std::string message = "") {
  expect_eq_diffs(stan::prob::multi_student_t_log<false>(y1,nu1,mu1,sigma1),
                  stan::prob::multi_student_t_log<false>(y2,nu2,mu2,sigma2),
                  stan::prob::multi_student_t_log<true>(y1,nu1,mu1,sigma1),
                  stan::prob::multi_student_t_log<true>(y2,nu2,mu2,sigma2),
                  message);
}

using stan::agrad::var;
using stan::agrad::to_var;


TEST_F(agrad_distributions_multi_student_t,Propto) {
  expect_propto(to_var(y),to_var(nu),to_var(mu),to_var(Sigma),
                to_var(y2),to_var(nu),to_var(mu2),to_var(Sigma2),
                "All vars: y, nu, mu, sigma");
}
TEST_F(agrad_distributions_multi_student_t,ProptoY) {
  expect_propto(to_var(y),nu,mu,Sigma,
                to_var(y2),nu,mu,Sigma,
                "var: y");
}
TEST_F(agrad_distributions_multi_student_t,ProptoYMu) {
  expect_propto(to_var(y),nu,to_var(mu),Sigma,
                to_var(y2),nu,to_var(mu2),Sigma,
                "var: y and mu");
}
TEST_F(agrad_distributions_multi_student_t,ProptoYSigma) {
  expect_propto(to_var(y),nu,mu,to_var(Sigma),
                to_var(y2),nu,mu,to_var(Sigma2),
                "var: y and sigma");
}
TEST_F(agrad_distributions_multi_student_t,ProptoMu) {
  expect_propto(y,nu,to_var(mu),Sigma,
                y,nu,to_var(mu2),Sigma,
                "var: mu");
}
TEST_F(agrad_distributions_multi_student_t,ProptoMuSigma) {
  expect_propto(y,nu,to_var(mu),to_var(Sigma),
                y,nu,to_var(mu2),to_var(Sigma2),
                "var: mu and sigma");
}
TEST_F(agrad_distributions_multi_student_t,ProptoSigma) {
  expect_propto(y,nu,mu,to_var(Sigma),
                y,nu,mu,to_var(Sigma2),
                "var: sigma");
}


TEST(ProbDistributionsMultiStudentT,MultiStudentTVar) {
  using stan::agrad::var;
  var nu(5);
  Matrix<var,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<var,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<var,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  EXPECT_FLOAT_EQ(-10.213695, stan::prob::multi_student_t_log(y,nu,mu,Sigma).val());
}
TEST(ProbDistributionsMultiStudentT,MultiStudentTGradientUnivariate) {
  using stan::agrad::var;
  using std::vector;
  using Eigen::VectorXd;
  using stan::prob::multi_student_t_log;
  
  var nu_var(5);

  Matrix<var,Dynamic,1> y_var(1,1);
  y_var << 2.0;

  Matrix<var,Dynamic,1> mu_var(1,1);
  mu_var << 1.0;

  Matrix<var,Dynamic,Dynamic> Sigma_var(1,1);
  Sigma_var(0,0) = 9.0;

  std::vector<var> x;
  x.push_back(y_var(0));
  x.push_back(mu_var(0));
  x.push_back(Sigma_var(0,0));
  x.push_back(nu_var);

  var lp = stan::prob::multi_student_t_log(y_var,nu_var,mu_var,Sigma_var);
  vector<double> grad;
  lp.grad(x,grad);

  // ===================================

  double nu(5);

  Matrix<double,Dynamic,1> y(1,1);
  y << 2.0;

  Matrix<double,Dynamic,1> mu(1,1);
  mu << 1.0;

  Matrix<double,Dynamic,Dynamic> Sigma(1,1);
  Sigma << 9.0;

  double epsilon = 1e-6;

  Matrix<double,Dynamic,1> y_m(1,1);
  Matrix<double,Dynamic,1> y_p(1,1);
  y_p[0] = y[0] + epsilon;
  y_m[0] = y[0] - epsilon;
  double grad_diff 
    =  (multi_student_t_log(y_p,nu,mu,Sigma) - multi_student_t_log(y_m,nu,mu,Sigma)) 
    / (2 * epsilon);
  EXPECT_FLOAT_EQ(grad_diff, grad[0]);

  Matrix<double,Dynamic,1> mu_m(1,1);
  Matrix<double,Dynamic,1> mu_p(1,1);
  mu_p[0] = mu[0] + epsilon;
  mu_m[0] = mu[0] - epsilon;
  grad_diff 
    =  (multi_student_t_log(y,nu,mu_p,Sigma) - multi_student_t_log(y,nu,mu_m,Sigma)) 
    / (2 * epsilon);
  EXPECT_FLOAT_EQ(grad_diff, grad[1]);

  Matrix<double,Dynamic,Dynamic> Sigma_m(1,1);
  Matrix<double,Dynamic,Dynamic> Sigma_p(1,1);
  Sigma_p(0) = Sigma(0) + epsilon;
  Sigma_m(0) = Sigma(0) - epsilon;
  grad_diff 
    =  (multi_student_t_log(y,nu,mu,Sigma_p) - multi_student_t_log(y,nu,mu,Sigma_m)) 
    / (2 * epsilon);
  EXPECT_FLOAT_EQ(grad_diff, grad[2]);

  double nu_p(nu + epsilon);
  double nu_m(nu - epsilon);
  grad_diff 
    =  (multi_student_t_log(y,nu_p,mu,Sigma) - multi_student_t_log(y,nu_m,mu,Sigma)) 
    / (2 * epsilon);
  EXPECT_FLOAT_EQ(grad_diff, grad[3]);
}


struct multi_student_t_fun {
  const int K_;

  multi_student_t_fun(int K) : K_(K) { }

  template <typename T>
  T operator()(const std::vector<T>& x) const {
    using Eigen::Matrix;
    using Eigen::Dynamic;
    using stan::agrad::var;
    T nu;
    Matrix<T,Dynamic,1> y(K_);
    Matrix<T,Dynamic,1> mu(K_);
    Matrix<T,Dynamic,Dynamic> Sigma(K_,K_);
    int pos = 0;
    for (int i = 0; i < K_; ++i)
      y(i) = x[pos++];
    for (int i = 0; i < K_; ++i)
      mu(i) = x[pos++];
    for (int j = 0; j < K_; ++j) {
      for (int i = 0; i <= j; ++i) {
        Sigma(i,j) = x[pos++];
        Sigma(j,i) = Sigma(i,j);
      }
    }
    nu = x[pos++];
    return stan::prob::multi_student_t_log<false>(y,nu,mu,Sigma);
  }
};

TEST(MultiStudentT, TestGradFunctional) {
  std::vector<double> x(3 + 3 + 3 * 3 + 1);
  // y
  x[0] = 1.0;
  x[1] = 2.0;
  x[2] = -3.0;
  // mu
  x[3] = 0.0;
  x[4] = -2.0;
  x[5] = -3.0;
  // Sigma
  x[6] = 1;
  x[7] = -1;
  x[8] = 10;
  x[9] = -2;
  x[10] = 20;
  x[11] = 56;
  // nu
  x[12] = 5;

  test_grad(multi_student_t_fun(3), x);

  std::vector<double> u(4);
  u[0] = 1.9;
  u[1] = -2.7;
  u[2] = 0.48;
  u[3] = 5;
  
  test_grad(multi_student_t_fun(1), u);
}

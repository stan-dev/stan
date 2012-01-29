#include <gtest/gtest.h>
#include <test/agrad/distributions/expect_eq_diffs.hpp>
#include <stan/agrad/agrad.hpp>
#include <stan/agrad/special_functions.hpp>
#include <stan/prob/distributions/univariate/continuous/student_t.hpp>


template <typename T_y, typename T_dof, typename T_loc, typename T_scale>
void expect_propto(T_y y1, T_dof nu1, T_loc mu1, T_scale sigma1,
                   T_y y2, T_dof nu2, T_loc mu2, T_scale sigma2,
                   std::string message) {
  expect_eq_diffs(stan::prob::student_t_log<false>(y1,nu1,mu1,sigma1),
                  stan::prob::student_t_log<false>(y2,nu2,mu2,sigma2),
                  stan::prob::student_t_log<true>(y1,nu1,mu1,sigma1),
                  stan::prob::student_t_log<true>(y2,nu2,mu2,sigma2),
                  message);
}

using stan::agrad::var;

class AgradDistributionsStudentT : public ::testing::Test {
protected:
  virtual void SetUp() {
    y1 = -3.0;
    y2 = 2.3;
    
    nu1 = 2.0;
    nu2 = 20.0;
    
    sigma1 = 1.5;
    sigma2 = 10.0;
  }
  double y1;
  double y2;
  
  double nu1;
  double nu2;
  
  double mu1;
  double mu2;
  
  double sigma1;
  double sigma2;
};

TEST_F(AgradDistributionsStudentT,Propto) {
  expect_propto<var,var,var,var>(y1, nu1, mu1, sigma1,
                                 y2, nu2, mu2, sigma2, 
                                 "all var: y, nu, mu, sigma");
}
TEST_F(AgradDistributionsStudentT,ProptoY) {
  expect_propto<var,double,double,double>(y1, nu1, mu1, sigma1,
                                          y2, nu1, mu1, sigma1, 
                                          "var: y");
}
TEST_F(AgradDistributionsStudentT,ProptoYNu) {
  expect_propto<var,var,double,double>(y1, nu1, mu1, sigma1,
                                       y2, nu2, mu1, sigma1, 
                                       "var: y and nu");
}
TEST_F(AgradDistributionsStudentT,ProptoYNuMu) {
  expect_propto<var,var,var,double>(y1, nu1, mu1, sigma1,
                                    y2, nu2, mu2, sigma1, 
                                    "var: y, nu, and mu");
}
TEST_F(AgradDistributionsStudentT,ProptoYNuSigma) {
  expect_propto<var,var,double,var>(y1, nu1, mu1, sigma1,
                                    y2, nu2, mu1, sigma2, 
                                    "var: y, mu, and sigma");
}
TEST_F(AgradDistributionsStudentT,ProptoYMu) {
  expect_propto<var,double,var,double>(y1, nu1, mu1, sigma1,
                                       y2, nu1, mu2, sigma1, 
                                       "var: y and mu");
}
TEST_F(AgradDistributionsStudentT,ProptoYMuSigma) {
  expect_propto<var,double,var,var>(y1, nu1, mu1, sigma1,
                                    y2, nu1, mu2, sigma2, 
                                    "var: y, mu, and sigma");
}
TEST_F(AgradDistributionsStudentT,ProptoYSigma) {
  expect_propto<var,double,double,var>(y1, nu1, mu1, sigma1,
                                       y2, nu1, mu1, sigma2, 
                                       "var: y and sigma");
}
TEST_F(AgradDistributionsStudentT,ProptoNu) {
  expect_propto<double,var,double,double>(y1, nu1, mu1, sigma1,
                                          y1, nu2, mu1, sigma1, 
                                          "var: nu");
}
TEST_F(AgradDistributionsStudentT,ProptoNuMu) {
  expect_propto<double,var,var,double>(y1, nu1, mu1, sigma1,
                                       y1, nu2, mu2, sigma1, 
                                       "var: nu and mu");
}
TEST_F(AgradDistributionsStudentT,ProptoNuMuSigma) {
  expect_propto<double,var,var,var>(y1, nu1, mu1, sigma1,
                                    y1, nu2, mu2, sigma2, 
                                    "var: nu, mu, and sigma");
}
TEST_F(AgradDistributionsStudentT,ProptoNuSigma) {
  expect_propto<double,var,double,var>(y1, nu1, mu1, sigma1,
                                       y1, nu2, mu1, sigma2, 
                                       "var: nu and sigma");
}
TEST_F(AgradDistributionsStudentT,ProptoMu) {
  expect_propto<double,double,var,double>(y1, nu1, mu1, sigma1,
                                          y1, nu1, mu2, sigma1, 
                                          "var: mu");
}
TEST_F(AgradDistributionsStudentT,ProptoMuSigma) {
  expect_propto<double,double,var,var>(y1, nu1, mu1, sigma1,
                                       y1, nu1, mu2, sigma2, 
                                       "var: mu and sigma");
}
TEST_F(AgradDistributionsStudentT,ProptoSigma) {
  expect_propto<double,double,double,var>(y1, nu1, mu1, sigma1,
                                          y1, nu1, mu1, sigma2, 
                                          "var: sigma");
}

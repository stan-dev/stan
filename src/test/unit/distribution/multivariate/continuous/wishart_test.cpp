#include <gtest/gtest.h>
#include <test/unit/distribution/expect_eq_diffs.hpp>
#include <stan/prob/distributions/multivariate/continuous/wishart.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/meta/traits.hpp>
#include <stan/agrad/rev/matrix.hpp>


template <typename T_y, typename T_dof, typename T_scale>
void expect_propto(T_y W1, T_dof nu1, T_scale S1,
                   T_y W2, T_dof nu2, T_scale S2,
                   std::string message) {
  expect_eq_diffs(stan::prob::wishart_log<false>(W1,nu1,S1),
                  stan::prob::wishart_log<false>(W2,nu2,S2),
                  stan::prob::wishart_log<true>(W1,nu1,S1),
                  stan::prob::wishart_log<true>(W2,nu2,S2),
                  message);
}

using Eigen::Dynamic;
using Eigen::Matrix;
using stan::agrad::var;
using stan::agrad::to_var;

class AgradDistributionsWishart : public ::testing::Test {
protected:
  virtual void SetUp() {
    Y1.resize(2,2);
    Y1 <<  2.011108, -11.20661,
      -11.206611, 112.94139;
    Y2.resize(2,2);
    Y2 <<  13.4, 12.2,
      12.2, 11.5;
  
    nu1 = 3;
    nu2 = 5.3;
  
    S1.resize(2,2);
    S1 << 1.848220, 1.899623, 
      1.899623, 12.751941;
    S2.resize(2,2);
    S2 << 3.0, 1.4,
      1.4, 7.0;
  }
  Matrix<double,Dynamic,Dynamic> Y1;
  Matrix<double,Dynamic,Dynamic> Y2;
  double nu1;
  double nu2;
  Matrix<double,Dynamic,Dynamic> S1;
  Matrix<double,Dynamic,Dynamic> S2;
};

TEST_F(AgradDistributionsWishart,Propto) {
  expect_propto(to_var(Y1),to_var(nu1),to_var(S1),
                to_var(Y2),to_var(nu2),to_var(S2),
                "var: y, nu, and sigma");
}
TEST_F(AgradDistributionsWishart,ProptoY) {
  expect_propto(to_var(Y1),nu1,S1,
                to_var(Y2),nu1,S1,
                "var: y");
}
TEST_F(AgradDistributionsWishart,ProptoYNu) {
  expect_propto(to_var(Y1),to_var(nu1),S1,
                to_var(Y2),to_var(nu2),S1,
                "var: y, and nu");
}
TEST_F(AgradDistributionsWishart,ProptoYSigma) {
  expect_propto(to_var(Y1),nu1,to_var(S1),
                to_var(Y2),nu1,to_var(S2),
                "var: y and sigma");
}
TEST_F(AgradDistributionsWishart,ProptoNu) {
  expect_propto(Y1,to_var(nu1),S1,
                Y1,to_var(nu2),S1,
                "var: nu");
}
TEST_F(AgradDistributionsWishart,ProptoNuSigma) {
  expect_propto(Y1,to_var(nu1),to_var(S1),
                Y1,to_var(nu2),to_var(S2),
                "var: nu and sigma");
}
TEST_F(AgradDistributionsWishart,ProptoSigma) {
  expect_propto(Y1,nu1,to_var(S1),
                Y1,nu1,to_var(S2),
                "var: sigma");
}


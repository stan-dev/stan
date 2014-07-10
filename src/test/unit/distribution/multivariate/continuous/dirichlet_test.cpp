#include <gtest/gtest.h>
#include <test/unit/distribution/expect_eq_diffs.hpp>
#include <stan/prob/distributions/multivariate/continuous/dirichlet.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/meta/traits.hpp>
#include <stan/agrad/rev/matrix.hpp>

template <typename T_prob, typename T_prior_sample_size>
void expect_propto(T_prob theta, T_prior_sample_size alpha,
                   T_prob theta2, T_prior_sample_size alpha2,
                   std::string message) {
  expect_eq_diffs(stan::prob::dirichlet_log<false>(theta, alpha),
                  stan::prob::dirichlet_log<false>(theta2, alpha2),
                  stan::prob::dirichlet_log<true>(theta, alpha),
                  stan::prob::dirichlet_log<true>(theta2, alpha2),
                  message);
}

using Eigen::Dynamic;
using Eigen::Matrix;
using stan::agrad::var;
using stan::agrad::to_var;

class AgradDistributionsDirichlet : public ::testing::Test {
protected:
  virtual void SetUp() {
    theta.resize(3,1);
    theta << 0.2, 0.3, 0.5;
    theta2.resize(3,1);
    theta2 << 0.1, 0.2, 0.7;
    alpha.resize(3,1);
    alpha << 1.0, 1.4, 3.2;
    alpha2.resize(3,1);
    alpha2 << 13.1, 12.9, 10.1;
  }
  Matrix<double,Dynamic,1> theta;
  Matrix<double,Dynamic,1> theta2;
  Matrix<double,Dynamic,1> alpha;
  Matrix<double,Dynamic,1> alpha2;
};

TEST_F(AgradDistributionsDirichlet,Propto) {
  expect_propto(to_var(theta),to_var(alpha),
                to_var(theta2),to_var(alpha2),
                "var: theta and alpha");
}
TEST_F(AgradDistributionsDirichlet,ProptoTheta) {
  expect_propto(to_var(theta), alpha,
                to_var(theta2), alpha,
                "var: theta");
}
TEST_F(AgradDistributionsDirichlet,ProptoAlpha) {
  expect_propto(theta, to_var(alpha), 
                theta, to_var(alpha2), 
                "var: alpha");
}
TEST_F(AgradDistributionsDirichlet,Bounds) {
  using stan::prob::dirichlet_log;
  Matrix<double,Dynamic,1> good_alpha(2,1), bad_alpha(2,1);
  Matrix<double,Dynamic,1> good_theta(2,1), bad_theta(2,1);

  good_theta << 0.25, 0.75;
  good_alpha << 2, 3;
  EXPECT_NO_THROW(dirichlet_log(to_var(good_theta),to_var(good_alpha)));
  EXPECT_NO_THROW(dirichlet_log(to_var(good_theta),good_alpha));
  EXPECT_NO_THROW(dirichlet_log(good_theta,to_var(good_alpha)));

  good_theta << 1.0, 0.0;
  good_alpha << 2, 3;
  EXPECT_NO_THROW(dirichlet_log(to_var(good_theta),to_var(good_alpha)))
    << "elements of theta can be 0";
  EXPECT_NO_THROW(dirichlet_log(to_var(good_theta),good_alpha))
    << "elements of theta can be 0";
  EXPECT_NO_THROW(dirichlet_log(good_theta,to_var(good_alpha)))
    << "elements of theta can be 0";


  bad_theta << 0.25, 0.25;
  EXPECT_THROW(dirichlet_log(to_var(bad_theta),to_var(good_alpha)),
               std::domain_error)
    << "sum of theta is not 1";
  EXPECT_THROW(dirichlet_log(to_var(bad_theta),good_alpha),
               std::domain_error)
    << "sum of theta is not 1";
  EXPECT_THROW(dirichlet_log(bad_theta,to_var(good_alpha)),
               std::domain_error)
    << "sum of theta is not 1";

  bad_theta << -0.25, 1.25;
  EXPECT_THROW(dirichlet_log(to_var(bad_theta),to_var(good_alpha)),
               std::domain_error)
    << "theta has element less than 0";
  EXPECT_THROW(dirichlet_log(to_var(bad_theta),good_alpha),
               std::domain_error)
    << "theta has element less than 0";
  EXPECT_THROW(dirichlet_log(bad_theta,to_var(good_alpha)),
               std::domain_error)
    << "theta has element less than 0";

  bad_theta << -0.25, 1.25;
  EXPECT_THROW(dirichlet_log(to_var(bad_theta),to_var(good_alpha)),
               std::domain_error)
    << "theta has element less than 0";
  EXPECT_THROW(dirichlet_log(to_var(bad_theta),good_alpha),
               std::domain_error)
    << "theta has element less than 0";
  EXPECT_THROW(dirichlet_log(bad_theta,to_var(good_alpha)),
               std::domain_error)
    << "theta has element less than 0";

  bad_alpha << 0.0, 1.0;
  EXPECT_THROW(dirichlet_log(to_var(good_theta),to_var(bad_alpha)),
               std::domain_error)
    << "alpha has element equal to 0";
  EXPECT_THROW(dirichlet_log(to_var(good_theta),bad_alpha),
               std::domain_error)
    << "alpha has element equal to 0";
  EXPECT_THROW(dirichlet_log(good_theta,to_var(bad_alpha)),
               std::domain_error)
    << "alpha has element equal to 0";

  bad_alpha << -0.5, 1.0;
  EXPECT_THROW(dirichlet_log(to_var(good_theta),to_var(bad_alpha)),
               std::domain_error)
    << "alpha has element less than 0";
  EXPECT_THROW(dirichlet_log(to_var(good_theta),bad_alpha),
               std::domain_error)
    << "alpha has element less than 0";
  EXPECT_THROW(dirichlet_log(good_theta,to_var(bad_alpha)),
               std::domain_error)
    << "alpha has element less than 0";

  bad_alpha = Matrix<double,Dynamic,1>(4,1);
  bad_alpha << 1, 2, 3, 4;
  EXPECT_THROW(dirichlet_log(to_var(good_theta),to_var(bad_alpha)),
               std::domain_error)
    << "size mismatch: theta is a 2-vector, alpha is a 4-vector";
  EXPECT_THROW(dirichlet_log(to_var(good_theta),bad_alpha),
               std::domain_error)
    << "size mismatch: theta is a 2-vector, alpha is a 4-vector";
  EXPECT_THROW(dirichlet_log(good_theta,to_var(bad_alpha)),
               std::domain_error)
    << "size mismatch: theta is a 2-vector, alpha is a 4-vector";
}

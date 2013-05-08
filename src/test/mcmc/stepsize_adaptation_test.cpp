#include <stan/mcmc/stepsize_adaptation.hpp>
#include <gtest/gtest.h>

TEST(McmcStepsizeAdaptation, set_mu) {
  stan::mcmc::stepsize_adaptation adaptation;
  
  double old_mu = 0.5;
  adaptation.set_mu(old_mu);
  EXPECT_EQ(old_mu, adaptation.get_mu());
  
}

TEST(McmcStepsizeAdaptation, set_delta) {
  stan::mcmc::stepsize_adaptation adaptation;
  
  double old_delta = 0.5;
  adaptation.set_delta(old_delta);
  EXPECT_EQ(old_delta, adaptation.get_delta());
  
  adaptation.set_delta(-0.1);
  EXPECT_EQ(old_delta, adaptation.get_delta());
  
  adaptation.set_delta(1.1);
  EXPECT_EQ(old_delta, adaptation.get_delta());
  
}

TEST(McmcStepsizeAdaptation, set_gamma) {
  stan::mcmc::stepsize_adaptation adaptation;
  
  double old_gamma = 0.05;
  adaptation.set_gamma(old_gamma);
  EXPECT_EQ(old_gamma, adaptation.get_gamma());
  
  adaptation.set_gamma(-0.1);
  EXPECT_EQ(old_gamma, adaptation.get_gamma());
  
}

TEST(McmcStepsizeAdaptation, set_kappa) {
  stan::mcmc::stepsize_adaptation adaptation;
  
  double old_kappa = 0.75;
  adaptation.set_kappa(old_kappa);
  EXPECT_EQ(old_kappa, adaptation.get_kappa());
  
  adaptation.set_kappa(-0.1);
  EXPECT_EQ(old_kappa, adaptation.get_kappa());
  
}

TEST(McmcStepsizeAdaptation, set_t0) {
  stan::mcmc::stepsize_adaptation adaptation;
  
  double old_t0 = 0.05;
  adaptation.set_t0(old_t0);
  EXPECT_EQ(old_t0, adaptation.get_t0());
  
  adaptation.set_t0(-0.1);
  EXPECT_EQ(old_t0, adaptation.get_t0());
  
}

TEST(McmcStepsizeAdaptation, learn_stepsize) {
  stan::mcmc::stepsize_adaptation adaptation;
  
  double target_accept = 0.65;
  double target_epsilon = 1;
  
  adaptation.set_mu(log(target_epsilon));
  adaptation.set_delta(target_accept);
  
  double new_epsilon = 0;
  adaptation.learn_stepsize(new_epsilon, target_accept);
  
  EXPECT_EQ(target_epsilon, new_epsilon);
}

class exposed_adaptation : public stan::mcmc::stepsize_adaptation {
public:
  exposed_adaptation(const double counter,
                     const double s_bar,
                     const double x_bar,
                     const double mu,
                     const double delta,
                     const double gamma,
                     const double kappa,
                     const double t0)  {
    _counter = counter;
    _s_bar = s_bar;
    _x_bar = x_bar;
    _mu = mu;
    _delta = delta;
    _gamma = gamma;
    _kappa = kappa;
    _t0 = t0;
  }
};


TEST(McmcStepsizeAdaptation, learn_stepsize_1) {
  exposed_adaptation adaptation(273,
                                0.00513820936953773,
                                0.886351350781597,
                                2.488109515349,
                                0.5,
                                0.05,
                                0.75,
                                10);
  double epsilon, adapt_stat;
  epsilon = 2.2037632781885;
  adapt_stat = 1.0;
  adaptation.learn_stepsize(epsilon, adapt_stat);
  EXPECT_NEAR(3.95863527373545, epsilon, 1e-14);
}

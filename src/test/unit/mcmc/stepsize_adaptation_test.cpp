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
    counter_ = counter;
    s_bar_ = s_bar;
    x_bar_ = x_bar;
    mu_ = mu;
    delta_ = delta;
    gamma_ = gamma;
    kappa_ = kappa;
    t0_ = t0;
  }

  double counter() const { return counter_; }
  double s_bar() const { return s_bar_; }
  double x_bar() const { return x_bar_; }
  double mu() const { return mu_; }
  double delta() const { return delta_; }
  double gamma() const { return gamma_; }
  double kappa() const { return kappa_;}
  double t0() const { return t0_; }
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

TEST(McmcStepsizeAdaptation, learn_stepsize_2) {
  exposed_adaptation adaptation(79,
                                0.0209457573689391,
                                0.821976337071218,
                                4.13018724667137,
                                0.5,
                                0.05,
                                0.75,
                                10);
  double epsilon, adapt_stat;
  epsilon = 1.50198560950968;
  adapt_stat = 0.728285153733299;
  adaptation.learn_stepsize(epsilon, adapt_stat);
  EXPECT_NEAR(2.40769920051673, epsilon, 1e-14);
  EXPECT_NEAR(80, adaptation.counter(), 1e-14);
  EXPECT_NEAR(0.0181765250233587, adaptation.s_bar(), 1e-14);
  EXPECT_NEAR(0.824095816988227, adaptation.x_bar(), 1e-14);
  EXPECT_NEAR(4.13018724667137, adaptation.mu(), 1e-14);
  EXPECT_NEAR(0.5, adaptation.delta(), 1e-14);
  EXPECT_NEAR(0.05, adaptation.gamma(), 1e-14);
  EXPECT_NEAR(0.75, adaptation.kappa(), 1e-14);
  EXPECT_NEAR(10, adaptation.t0(), 1e-14);
}

TEST(McmcStepsizeAdaptation, learn_stepsize_3) {
  exposed_adaptation adaptation(79,
                                0.0206811620891896,
                                0.825842987938587,
                                4.13018305456267,
                                0.5,
                                0.05,
                                0.75,
                                10);
  double epsilon, adapt_stat;
  epsilon = 1.574313439903;
  adapt_stat = 0.684248188546772;
  adaptation.learn_stepsize(epsilon, adapt_stat);
  EXPECT_NEAR(2.31161210908631, epsilon, 5e-14);
  EXPECT_NEAR(80, adaptation.counter(), 1e-14);
  EXPECT_NEAR(0.0184041693043455, adaptation.s_bar(), 1e-14);
  EXPECT_NEAR(0.826295412288499, adaptation.x_bar(), 1e-14);
  EXPECT_NEAR(4.13018305456267, adaptation.mu(), 1e-14);
  EXPECT_NEAR(0.5, adaptation.delta(), 1e-14);
  EXPECT_NEAR(0.05, adaptation.gamma(), 1e-14);
  EXPECT_NEAR(0.75, adaptation.kappa(), 1e-14);
  EXPECT_NEAR(10, adaptation.t0(), 1e-14);
}

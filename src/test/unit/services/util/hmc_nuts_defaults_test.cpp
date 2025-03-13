#include <stan/services/util/hmc_nuts_defaults.hpp>
#include <gtest/gtest.h>
#include <cmath>

// Test all the NUTS-HMC default values and validations

TEST(hmc_nuts_defaults, stepsize) {
  using stan::services::util::stepsize;
  EXPECT_EQ("Step size for discrete evolution.", stepsize::description());

  EXPECT_NO_THROW(stepsize::validate(stepsize::default_value()));
  EXPECT_NO_THROW(stepsize::validate(0.1));
  EXPECT_THROW(stepsize::validate(0.0), std::invalid_argument);
  EXPECT_THROW(stepsize::validate(-0.5), std::invalid_argument);

  EXPECT_FLOAT_EQ(1.0, stepsize::default_value());
}

TEST(hmc_nuts_defaults, stepsize_jitter) {
  using stan::services::util::stepsize_jitter;
  EXPECT_EQ("Uniformly random jitter of the stepsize, in percent.", 
            stepsize_jitter::description());

  EXPECT_NO_THROW(stepsize_jitter::validate(stepsize_jitter::default_value()));
  EXPECT_NO_THROW(stepsize_jitter::validate(0.0));
  EXPECT_NO_THROW(stepsize_jitter::validate(0.5));
  EXPECT_NO_THROW(stepsize_jitter::validate(1.0));
  EXPECT_THROW(stepsize_jitter::validate(-0.1), std::invalid_argument);
  EXPECT_THROW(stepsize_jitter::validate(1.1), std::invalid_argument);

  EXPECT_FLOAT_EQ(0.0, stepsize_jitter::default_value());
}

TEST(hmc_nuts_defaults, max_depth) {
  using stan::services::util::max_depth;
  EXPECT_EQ("Maximum tree depth.", max_depth::description());

  EXPECT_NO_THROW(max_depth::validate(max_depth::default_value()));
  EXPECT_NO_THROW(max_depth::validate(1));
  EXPECT_NO_THROW(max_depth::validate(15));
  EXPECT_THROW(max_depth::validate(0), std::invalid_argument);
  EXPECT_THROW(max_depth::validate(-1), std::invalid_argument);

  EXPECT_EQ(10, max_depth::default_value());
}

TEST(hmc_nuts_defaults, delta) {
  using stan::services::util::delta;
  EXPECT_EQ("Adaptation target acceptance statistic.", delta::description());

  EXPECT_NO_THROW(delta::validate(delta::default_value()));
  EXPECT_NO_THROW(delta::validate(0.1));
  EXPECT_NO_THROW(delta::validate(0.9));
  EXPECT_THROW(delta::validate(0.0), std::invalid_argument);
  EXPECT_THROW(delta::validate(1.0), std::invalid_argument);
  EXPECT_THROW(delta::validate(-0.5), std::invalid_argument);

  EXPECT_FLOAT_EQ(0.8, delta::default_value());
}

TEST(hmc_nuts_defaults, gamma) {
  using stan::services::util::gamma;
  EXPECT_EQ("Adaptation regularization scale.", gamma::description());

  EXPECT_NO_THROW(gamma::validate(gamma::default_value()));
  EXPECT_NO_THROW(gamma::validate(0.1));
  EXPECT_THROW(gamma::validate(0.0), std::invalid_argument);
  EXPECT_THROW(gamma::validate(-0.5), std::invalid_argument);

  EXPECT_FLOAT_EQ(0.05, gamma::default_value());
}

TEST(hmc_nuts_defaults, kappa) {
  using stan::services::util::kappa;
  EXPECT_EQ("Adaptation relaxation exponent.", kappa::description());

  EXPECT_NO_THROW(kappa::validate(kappa::default_value()));
  EXPECT_NO_THROW(kappa::validate(0.1));
  EXPECT_THROW(kappa::validate(0.0), std::invalid_argument);
  EXPECT_THROW(kappa::validate(-0.5), std::invalid_argument);

  EXPECT_FLOAT_EQ(0.75, kappa::default_value());
}

TEST(hmc_nuts_defaults, t0) {
  using stan::services::util::t0;
  EXPECT_EQ("Adaptation iteration offset.", t0::description());

  EXPECT_NO_THROW(t0::validate(t0::default_value()));
  EXPECT_NO_THROW(t0::validate(0.1));
  EXPECT_THROW(t0::validate(0.0), std::invalid_argument);
  EXPECT_THROW(t0::validate(-0.5), std::invalid_argument);

  EXPECT_FLOAT_EQ(10.0, t0::default_value());
}

TEST(hmc_nuts_defaults, init_buffer) {
  using stan::services::util::init_buffer;
  EXPECT_EQ("Width of initial fast adaptation interval.", 
            init_buffer::description());

  EXPECT_NO_THROW(init_buffer::validate(init_buffer::default_value()));
  EXPECT_NO_THROW(init_buffer::validate(0));
  EXPECT_NO_THROW(init_buffer::validate(100));

  EXPECT_EQ(75, init_buffer::default_value());
}

TEST(hmc_nuts_defaults, term_buffer) {
  using stan::services::util::term_buffer;
  EXPECT_EQ("Width of final fast adaptation interval.", 
            term_buffer::description());

  EXPECT_NO_THROW(term_buffer::validate(term_buffer::default_value()));
  EXPECT_NO_THROW(term_buffer::validate(0));
  EXPECT_NO_THROW(term_buffer::validate(100));

  EXPECT_EQ(50, term_buffer::default_value());
}

TEST(hmc_nuts_defaults, window) {
  using stan::services::util::window;
  EXPECT_EQ("Initial width of slow adaptation interval.", 
            window::description());

  EXPECT_NO_THROW(window::validate(window::default_value()));
  EXPECT_NO_THROW(window::validate(0));
  EXPECT_NO_THROW(window::validate(100));

  EXPECT_EQ(25, window::default_value());
}

TEST(hmc_nuts_defaults, adaptation_engaged) {
  using stan::services::util::adaptation_engaged;
  EXPECT_EQ("Indicates whether adaptation is engaged.", 
            adaptation_engaged::description());

  EXPECT_NO_THROW(adaptation_engaged::validate(adaptation_engaged::default_value()));
  EXPECT_NO_THROW(adaptation_engaged::validate(false));
  EXPECT_NO_THROW(adaptation_engaged::validate(true));

  EXPECT_TRUE(adaptation_engaged::default_value());
}

TEST(hmc_nuts_defaults, num_warmup) {
  using stan::services::util::num_warmup;
  EXPECT_EQ("Number of warmup iterations.", num_warmup::description());

  EXPECT_NO_THROW(num_warmup::validate(num_warmup::default_value()));
  EXPECT_NO_THROW(num_warmup::validate(0));
  EXPECT_THROW(num_warmup::validate(-1), std::invalid_argument);

  EXPECT_EQ(1000, num_warmup::default_value());
}

TEST(hmc_nuts_defaults, num_samples) {
  using stan::services::util::num_samples;
  EXPECT_EQ("Number of sampling iterations.", num_samples::description());

  EXPECT_NO_THROW(num_samples::validate(num_samples::default_value()));
  EXPECT_NO_THROW(num_samples::validate(0));
  EXPECT_THROW(num_samples::validate(-1), std::invalid_argument);

  EXPECT_EQ(1000, num_samples::default_value());
}

TEST(hmc_nuts_defaults, save_warmup) {
  using stan::services::util::save_warmup;
  EXPECT_EQ("Save warmup iterations to output.", save_warmup::description());

  EXPECT_NO_THROW(save_warmup::validate(save_warmup::default_value()));
  EXPECT_NO_THROW(save_warmup::validate(false));
  EXPECT_NO_THROW(save_warmup::validate(true));

  EXPECT_FALSE(save_warmup::default_value());
}

TEST(hmc_nuts_defaults, thin) {
  using stan::services::util::thin;
  EXPECT_EQ("Period between saved samples.", thin::description());

  EXPECT_NO_THROW(thin::validate(thin::default_value()));
  EXPECT_NO_THROW(thin::validate(1));
  EXPECT_THROW(thin::validate(0), std::invalid_argument);
  EXPECT_THROW(thin::validate(-1), std::invalid_argument);

  EXPECT_EQ(1, thin::default_value());
}

TEST(hmc_nuts_defaults, refresh) {
  using stan::services::util::refresh;
  EXPECT_EQ("Period between status output messages.", refresh::description());

  EXPECT_NO_THROW(refresh::validate(refresh::default_value()));
  EXPECT_NO_THROW(refresh::validate(0));
  EXPECT_NO_THROW(refresh::validate(-1)); // Negative values disable output

  EXPECT_EQ(100, refresh::default_value());
}

TEST(hmc_nuts_defaults, int_time) {
  using stan::services::util::int_time;
  EXPECT_EQ("Total integration time for Hamiltonian evolution.", 
            int_time::description());

  EXPECT_NO_THROW(int_time::validate(int_time::default_value()));
  EXPECT_NO_THROW(int_time::validate(1.0));
  EXPECT_THROW(int_time::validate(0.0), std::invalid_argument);
  EXPECT_THROW(int_time::validate(-1.0), std::invalid_argument);

  EXPECT_FLOAT_EQ(M_PI * 2, int_time::default_value());
}

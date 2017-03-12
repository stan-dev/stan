#include <stan/services/sample/defaults.hpp>
#include <gtest/gtest.h>

TEST(sample_defaults, num_samples) {
  using stan::services::sample::num_samples;
  EXPECT_EQ("Number of sampling iterations.",
            num_samples::description());

  EXPECT_NO_THROW(num_samples::validate(num_samples::default_value()));
  EXPECT_NO_THROW(num_samples::validate(0));
  EXPECT_THROW(num_samples::validate(-1),
               std::invalid_argument);

  EXPECT_EQ(1000, num_samples::default_value());
}

TEST(sample_defaults, num_warmup) {
  using stan::services::sample::num_warmup;
  EXPECT_EQ("Number of warmup iterations.",
            num_warmup::description());

  EXPECT_NO_THROW(num_warmup::validate(num_warmup::default_value()));
  EXPECT_NO_THROW(num_warmup::validate(0));
  EXPECT_THROW(num_warmup::validate(-1),
               std::invalid_argument);

  EXPECT_EQ(1000, num_warmup::default_value());
}

TEST(sample_defaults, save_warmup) {
  using stan::services::sample::save_warmup;
  EXPECT_EQ("Save warmup iterations to output.",
            save_warmup::description());

  EXPECT_NO_THROW(save_warmup::validate(save_warmup::default_value()));
  EXPECT_NO_THROW(save_warmup::validate(false));
  EXPECT_NO_THROW(save_warmup::validate(true));

  EXPECT_FALSE(save_warmup::default_value());
}

TEST(sample_defaults, thin) {
  using stan::services::sample::thin;
  EXPECT_EQ("Period between saved samples.",
            thin::description());

  EXPECT_NO_THROW(thin::validate(thin::default_value()));
  EXPECT_NO_THROW(thin::validate(1));
  EXPECT_THROW(thin::validate(0),
               std::invalid_argument);

  EXPECT_EQ(1, thin::default_value());
}

TEST(sample_defaults, adaptation_engaged) {
  using stan::services::sample::adaptation_engaged;
  EXPECT_EQ("Indicates whether adaptation is engaged.",
            adaptation_engaged::description());

  EXPECT_NO_THROW(adaptation_engaged::validate(adaptation_engaged::default_value()));
  EXPECT_NO_THROW(adaptation_engaged::validate(false));
  EXPECT_NO_THROW(adaptation_engaged::validate(true));

  EXPECT_TRUE(adaptation_engaged::default_value());
}

TEST(sample_defaults, gamma) {
  using stan::services::sample::gamma;
  EXPECT_EQ("Adaptation regularization scale.",
            gamma::description());

  EXPECT_NO_THROW(gamma::validate(gamma::default_value()));
  EXPECT_NO_THROW(gamma::validate(1.0));
  EXPECT_THROW(gamma::validate(0.0),
               std::invalid_argument);

  EXPECT_FLOAT_EQ(0.05, gamma::default_value());
}

TEST(sample_defaults, kappa) {
  using stan::services::sample::kappa;
  EXPECT_EQ("Adaptation relaxation exponent.",
            kappa::description());

  EXPECT_NO_THROW(kappa::validate(kappa::default_value()));
  EXPECT_NO_THROW(kappa::validate(1.0));
  EXPECT_THROW(kappa::validate(0.0),
               std::invalid_argument);

  EXPECT_FLOAT_EQ(0.75, kappa::default_value());
}

TEST(sample_defaults, t0) {
  using stan::services::sample::t0;
  EXPECT_EQ("Adaptation iteration offset.",
            t0::description());

  EXPECT_NO_THROW(t0::validate(t0::default_value()));
  EXPECT_NO_THROW(t0::validate(1.0));
  EXPECT_THROW(t0::validate(0.0),
               std::invalid_argument);

  EXPECT_FLOAT_EQ(10.0, t0::default_value());
}

TEST(sample_defaults, init_buffer) {
  using stan::services::sample::init_buffer;
  EXPECT_EQ("Width of initial fast adaptation interval.",
            init_buffer::description());

  EXPECT_NO_THROW(init_buffer::validate(init_buffer::default_value()));
  EXPECT_NO_THROW(init_buffer::validate(0));
  EXPECT_NO_THROW(init_buffer::validate(1000));

  EXPECT_EQ(75, init_buffer::default_value());
}

TEST(sample_defaults, term_buffer) {
  using stan::services::sample::term_buffer;
  EXPECT_EQ("Width of final fast adaptation interval.",
            term_buffer::description());

  EXPECT_NO_THROW(term_buffer::validate(term_buffer::default_value()));
  EXPECT_NO_THROW(term_buffer::validate(0));
  EXPECT_NO_THROW(term_buffer::validate(1000));

  EXPECT_EQ(50, term_buffer::default_value());
}

TEST(sample_defaults, window) {
  using stan::services::sample::window;
  EXPECT_EQ("Initial width of slow adaptation interval.",
            window::description());

  EXPECT_NO_THROW(window::validate(window::default_value()));
  EXPECT_NO_THROW(window::validate(0));
  EXPECT_NO_THROW(window::validate(1000));

  EXPECT_EQ(25, window::default_value());
}

TEST(sample_defaults, int_time) {
  using stan::services::sample::int_time;
  EXPECT_EQ("Total integration time for Hamiltonian evolution.",
            int_time::description());

  EXPECT_NO_THROW(int_time::validate(int_time::default_value()));
  EXPECT_NO_THROW(int_time::validate(1.0));
  EXPECT_THROW(int_time::validate(0.0),
               std::invalid_argument);

  EXPECT_FLOAT_EQ(6.28318530717959, int_time::default_value());
}

TEST(sample_defaults, max_depth) {
  using stan::services::sample::max_depth;
  EXPECT_EQ("Maximum tree depth.",
            max_depth::description());

  EXPECT_NO_THROW(max_depth::validate(max_depth::default_value()));
  EXPECT_NO_THROW(max_depth::validate(1));
  EXPECT_THROW(max_depth::validate(0),
               std::invalid_argument);

  EXPECT_EQ(10, max_depth::default_value());
}

TEST(sample_defaults, stepsize) {
  using stan::services::sample::stepsize;
  EXPECT_EQ("Step size for discrete evolution.",
            stepsize::description());

  EXPECT_NO_THROW(stepsize::validate(stepsize::default_value()));
  EXPECT_NO_THROW(stepsize::validate(1.0));
  EXPECT_THROW(stepsize::validate(0.0),
               std::invalid_argument);

  EXPECT_FLOAT_EQ(1.0, stepsize::default_value());
}

TEST(sample_defaults, stepsize_jitter) {
  using stan::services::sample::stepsize_jitter;
  EXPECT_EQ("Uniformly random jitter of the stepsize, in percent.",
            stepsize_jitter::description());

  EXPECT_NO_THROW(stepsize_jitter::validate(stepsize_jitter::default_value()));
  EXPECT_NO_THROW(stepsize_jitter::validate(0.0));
  EXPECT_NO_THROW(stepsize_jitter::validate(1.0));
  EXPECT_THROW(stepsize_jitter::validate(-0.001),
               std::invalid_argument);
  EXPECT_THROW(stepsize_jitter::validate(1.0001),
               std::invalid_argument);

  EXPECT_FLOAT_EQ(0.0, stepsize_jitter::default_value());
}

#include <stan/services/experimental/advi/defaults.hpp>
#include <gtest/gtest.h>

TEST(experimental_advi_defaults, gradient_samples) {
  using stan::services::experimental::advi::gradient_samples;
  EXPECT_EQ("Number of Monte Carlo draws for computing the gradient.",
            gradient_samples::description());

  EXPECT_NO_THROW(gradient_samples::validate(gradient_samples::default_value()));
  EXPECT_NO_THROW(gradient_samples::validate(1));
  EXPECT_THROW(gradient_samples::validate(0), std::invalid_argument);

  EXPECT_EQ(1, gradient_samples::default_value());
}

TEST(experimental_advi_defaults, elbo_samples) {
  using stan::services::experimental::advi::elbo_samples;
  EXPECT_EQ("Number of Monte Carlo draws for estimate of ELBO.",
            elbo_samples::description());

  EXPECT_NO_THROW(elbo_samples::validate(elbo_samples::default_value()));
  EXPECT_NO_THROW(elbo_samples::validate(1));
  EXPECT_THROW(elbo_samples::validate(0), std::invalid_argument);

  EXPECT_EQ(100, elbo_samples::default_value());
}

TEST(experimental_advi_defaults, max_iterations) {
  using stan::services::experimental::advi::max_iterations;
  EXPECT_EQ("Maximum number of ADVI iterations.",
            max_iterations::description());

  EXPECT_NO_THROW(max_iterations::validate(max_iterations::default_value()));
  EXPECT_NO_THROW(max_iterations::validate(1));
  EXPECT_THROW(max_iterations::validate(0), std::invalid_argument);

  EXPECT_EQ(10000, max_iterations::default_value());
}

TEST(experimental_advi_defaults, tol_rel_obj) {
  using stan::services::experimental::advi::tol_rel_obj;
  EXPECT_EQ("Relative tolerance parameter for convergence.",
            tol_rel_obj::description());

  EXPECT_NO_THROW(tol_rel_obj::validate(tol_rel_obj::default_value()));
  EXPECT_NO_THROW(tol_rel_obj::validate(1.0));
  EXPECT_THROW(tol_rel_obj::validate(0.0), std::invalid_argument);

  EXPECT_FLOAT_EQ(0.01, tol_rel_obj::default_value());
}

TEST(experimental_advi_defaults, eta) {
  using stan::services::experimental::advi::eta;
  EXPECT_EQ("Stepsize scaling parameter.",
            eta::description());

  EXPECT_NO_THROW(eta::validate(eta::default_value()));
  EXPECT_NO_THROW(eta::validate(1.0));
  EXPECT_THROW(eta::validate(0.0), std::invalid_argument);

  EXPECT_FLOAT_EQ(1.0, eta::default_value());
}

TEST(experimental_advi_defaults, adapt_engaged) {
  using stan::services::experimental::advi::adapt_engaged;
  EXPECT_EQ("Boolean flag for eta adaptation.",
            adapt_engaged::description());

  EXPECT_NO_THROW(adapt_engaged::validate(adapt_engaged::default_value()));
  EXPECT_NO_THROW(adapt_engaged::validate(true));
  EXPECT_NO_THROW(adapt_engaged::validate(false));

  EXPECT_TRUE(adapt_engaged::default_value());
}

TEST(experimental_advi_defaults, adapt_iterations) {
  using stan::services::experimental::advi::adapt_iterations;
  EXPECT_EQ("Number of iterations for eta adaptation.",
            adapt_iterations::description());

  EXPECT_NO_THROW(adapt_iterations::validate(adapt_iterations::default_value()));
  EXPECT_NO_THROW(adapt_iterations::validate(1));
  EXPECT_THROW(adapt_iterations::validate(0), std::invalid_argument);

  EXPECT_EQ(50, adapt_iterations::default_value());
}

TEST(experimental_advi_defaults, eval_elbo) {
  using stan::services::experimental::advi::eval_elbo;
  EXPECT_EQ("Number of interations between ELBO evaluations",
            eval_elbo::description());

  EXPECT_NO_THROW(eval_elbo::validate(eval_elbo::default_value()));
  EXPECT_NO_THROW(eval_elbo::validate(1));
  EXPECT_THROW(eval_elbo::validate(0), std::invalid_argument);

  EXPECT_EQ(100, eval_elbo::default_value());
}

TEST(experimental_advi_defaults, output_draws) {
  using stan::services::experimental::advi::output_draws;
  EXPECT_EQ("Number of approximate posterior output draws to save.",
            output_draws::description());

  EXPECT_NO_THROW(output_draws::validate(output_draws::default_value()));
  EXPECT_NO_THROW(output_draws::validate(0));
  EXPECT_THROW(output_draws::validate(-1), std::invalid_argument);

  EXPECT_EQ(1000, output_draws::default_value());
}

#include <stan/services/optimize/defaults.hpp>
#include <gtest/gtest.h>

TEST(optimize_defaults, init_alpha) {
  using stan::services::optimize::init_alpha;
  EXPECT_EQ("Line search step size for first iteration.",
            init_alpha::description());

  EXPECT_NO_THROW(init_alpha::validate(init_alpha::default_value()));
  EXPECT_NO_THROW(init_alpha::validate(1.0));
  EXPECT_THROW(init_alpha::validate(0.0),
               std::invalid_argument);

  EXPECT_FLOAT_EQ(0.001, init_alpha::default_value());
}

TEST(optimize_defaults, tol_obj) {
  using stan::services::optimize::tol_obj;
  EXPECT_EQ("Convergence tolerance on absolute changes in objective function value.",
            tol_obj::description());

  EXPECT_NO_THROW(tol_obj::validate(tol_obj::default_value()));
  EXPECT_NO_THROW(tol_obj::validate(1.0));
  EXPECT_NO_THROW(tol_obj::validate(0.0));
  EXPECT_THROW(tol_obj::validate(-0.0001),
               std::invalid_argument);

  EXPECT_FLOAT_EQ(1e-12, tol_obj::default_value());
}

TEST(optimize_defaults, tol_rel_obj) {
  using stan::services::optimize::tol_rel_obj;
  EXPECT_EQ("Convergence tolerance on relative changes in objective function value.",
            tol_rel_obj::description());

  EXPECT_NO_THROW(tol_rel_obj::validate(tol_rel_obj::default_value()));
  EXPECT_NO_THROW(tol_rel_obj::validate(1.0));
  EXPECT_NO_THROW(tol_rel_obj::validate(0.0));
  EXPECT_THROW(tol_rel_obj::validate(-0.0001),
               std::invalid_argument);

  EXPECT_FLOAT_EQ(10000, tol_rel_obj::default_value());
}

TEST(optimize_defaults, tol_grad) {
  using stan::services::optimize::tol_grad;
  EXPECT_EQ("Convergence tolerance on the norm of the gradient.",
            tol_grad::description());

  EXPECT_NO_THROW(tol_grad::validate(tol_grad::default_value()));
  EXPECT_NO_THROW(tol_grad::validate(1.0));
  EXPECT_NO_THROW(tol_grad::validate(0.0));
  EXPECT_THROW(tol_grad::validate(-0.0001),
               std::invalid_argument);

  EXPECT_FLOAT_EQ(1e-8, tol_grad::default_value());
}

TEST(optimize_defaults, tol_rel_grad) {
  using stan::services::optimize::tol_rel_grad;
  EXPECT_EQ("Convergence tolerance on the relative norm of the gradient.",
            tol_rel_grad::description());

  EXPECT_NO_THROW(tol_rel_grad::validate(tol_rel_grad::default_value()));
  EXPECT_NO_THROW(tol_rel_grad::validate(1.0));
  EXPECT_NO_THROW(tol_rel_grad::validate(0.0));
  EXPECT_THROW(tol_rel_grad::validate(-0.0001),
               std::invalid_argument);

  EXPECT_FLOAT_EQ(10000000, tol_rel_grad::default_value());
}

TEST(optimize_defaults, tol_param) {
  using stan::services::optimize::tol_param;
  EXPECT_EQ("Convergence tolerance on changes in parameter value.",
            tol_param::description());

  EXPECT_NO_THROW(tol_param::validate(tol_param::default_value()));
  EXPECT_NO_THROW(tol_param::validate(1.0));
  EXPECT_NO_THROW(tol_param::validate(0.0));
  EXPECT_THROW(tol_param::validate(-0.0001),
               std::invalid_argument);

  EXPECT_FLOAT_EQ(1e-8, tol_param::default_value());
}

TEST(optimize_defaults, history_size) {
  using stan::services::optimize::history_size;
  EXPECT_EQ("Amount of history to keep for L-BFGS.",
            history_size::description());

  EXPECT_NO_THROW(history_size::validate(history_size::default_value()));
  EXPECT_NO_THROW(history_size::validate(1));
  EXPECT_THROW(history_size::validate(0),
               std::invalid_argument);

  EXPECT_EQ(5, history_size::default_value());
}

TEST(optimize_defaults, iter) {
  using stan::services::optimize::iter;
  EXPECT_EQ("Total number of iterations.",
            iter::description());

  EXPECT_NO_THROW(iter::validate(iter::default_value()));
  EXPECT_NO_THROW(iter::validate(1));
  EXPECT_THROW(iter::validate(0),
               std::invalid_argument);

  EXPECT_FLOAT_EQ(2000, iter::default_value());
}

TEST(optimize_defaults, save_iterations) {
  using stan::services::optimize::save_iterations;
  EXPECT_EQ("Save optimization interations to output.",
            save_iterations::description());

  EXPECT_NO_THROW(save_iterations::validate(save_iterations::default_value()));
  EXPECT_NO_THROW(save_iterations::validate(false));
  EXPECT_NO_THROW(save_iterations::validate(true));
  
  EXPECT_FALSE(save_iterations::default_value());
}

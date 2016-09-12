#include <stan/services/diagnose/defaults.hpp>
#include <gtest/gtest.h>

TEST(diagnose_defaults, epsilon) {
  using stan::services::diagnose::epsilon;
  EXPECT_EQ("Finite difference stepsize.", epsilon::description());

  EXPECT_NO_THROW(epsilon::validate(epsilon::default_value()));
  EXPECT_NO_THROW(epsilon::validate(1.0));
  EXPECT_THROW(epsilon::validate(0.0), std::invalid_argument);

  EXPECT_FLOAT_EQ(1e-6, epsilon::default_value());
}


TEST(diagnose_defaults, error) {
  using stan::services::diagnose::error;
  EXPECT_EQ("Absolute error threshold.", error::description());

  EXPECT_NO_THROW(error::validate(error::default_value()));
  EXPECT_NO_THROW(error::validate(1.0));
  EXPECT_THROW(error::validate(0.0), std::invalid_argument);

  EXPECT_FLOAT_EQ(1e-6, error::default_value());
}

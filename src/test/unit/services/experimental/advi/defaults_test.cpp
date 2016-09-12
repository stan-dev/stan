#include <stan/services/experimental/advi/defaults.hpp>
#include <gtest/gtest.h>

// TEST(experimental_advi, epsilon) {
//   EXPECT_EQ("finite difference stepsize", stan::services::diagnose::epsilon::description());

//   EXPECT_NO_THROW(stan::services::diagnose::epsilon::validate(1.0));
//   EXPECT_THROW(stan::services::diagnose::epsilon::validate(0.0), std::invalid_argument);

//   EXPECT_FLOAT_EQ(1e-6, stan::services::diagnose::epsilon::default_value());
// }

// TEST(experimental_advi, error) {
//   EXPECT_EQ("absolute error threshold", stan::services::diagnose::error::description());

//   EXPECT_NO_THROW(stan::services::diagnose::error::validate(1.0));
//   EXPECT_THROW(stan::services::diagnose::error::validate(0.0), std::invalid_argument);

//   EXPECT_FLOAT_EQ(1e-6, stan::services::diagnose::error::default_value());
// }

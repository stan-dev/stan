#include <stan/services/util/defaults.hpp>
#include <gtest/gtest.h>

// Test that the generic option template has correct fallbacks
TEST(defaults, generic_option_defaults) {
  struct test_int_tag {};
  struct test_double_tag {};
  struct test_bool_tag {};
  struct test_custom_tag {};
  
  // Test integer default
  EXPECT_EQ(0, (stan::services::util::option<int, test_int_tag>::default_value()));
  
  // Test double default
  EXPECT_FLOAT_EQ(0.0, (stan::services::util::option<double, test_double_tag>::default_value()));
  
  // Test bool default
  EXPECT_FALSE((stan::services::util::option<bool, test_bool_tag>::default_value()));
  
  // Test generic description
  EXPECT_EQ("Option description not available.", 
            (stan::services::util::option<int, test_int_tag>::description()));
            
  // Test generic validation (no-op)
  EXPECT_NO_THROW((stan::services::util::option<int, test_int_tag>::validate(123)));
  EXPECT_NO_THROW((stan::services::util::option<double, test_double_tag>::validate(-3.14)));
  EXPECT_NO_THROW((stan::services::util::option<bool, test_bool_tag>::validate(true)));
}

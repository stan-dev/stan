#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(lang_parser, integrate_1d_good) {
  test_parsable("integrate_1d_good");
}
TEST(lang_parser, integrate_1d_bad) {
  test_throws("ode/bad_fun_type",
      "first argument to integrate_ode must be the name of a function with signature");
}

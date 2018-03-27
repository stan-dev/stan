#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(langParserMapRect, good) {
  test_parsable("map_rect");
  expect_match("map_rect", "map_rect<");
  expect_match("map_rect",
               "STAN_REGISTER_MAP_RECT(1, map_rect_namespace::foo_functor__)");
  expect_match("map_rect",
               "STAN_REGISTER_MAP_RECT(16, map_rect_namespace::foo_functor__)");
}

TEST(langParserMapRect, badFunShape) {
  test_throws("map_rect/bad_fun_type",
              "first argument to map_rect must be"
              " the name of a function with signature"
              " (vector, vector, real[], int[]) : vector");
}
TEST(langParserMapRect, badSharedParamsShape) {
  test_throws("map_rect/bad_shared_params_type",
              "second argument to map_rect must be of type vector");
}
TEST(langParserMapRect, badJobParamsShape) {
  test_throws("map_rect/bad_job_params_type",
              "third argument to map_rect must be of type vector[]");
}
TEST(langParserMapRect, badDataRealShape) {
  test_throws("map_rect/bad_data_r_type",
              "fourth argument to map_rect must be of type real[ , ]");
}
TEST(langParserMapRect, badDataIntShape) {
  test_throws("map_rect/bad_data_i_type",
              "fifth argument to map_rect must be of type int[ , ]");
}

TEST(langParserMapRect, badDataRealConst) {
  test_throws("map_rect/bad_data_real_const",
              "fourth argment to map_rect must be data only");
}

TEST(langParserMapRect, badDataIntConst) {
  test_throws("map_rect/bad_data_int_const",
              "fifth argument to map_rect must be data only");
}

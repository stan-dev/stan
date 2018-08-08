#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(lang_parser, integrate_dae_good) {
  test_parsable("dae_good");
}
TEST(lang_parser, integrate_dae_bad) {
  test_throws("dae/bad_fun_type",
              "wrong signature ");
  test_throws("dae/bad_yy_type",
              "second argument to integrate_dae must");
  test_throws("dae/bad_yp_type",
              "third argument to integrate_dae must");
  test_throws("dae/bad_t_type",
              "fourth argument to integrate_dae must");
  test_throws("dae/bad_ts_type",
              "fifth argument to integrate_dae must");
  test_throws("dae/bad_theta_type",
              "sixth argument to integrate_dae must");
  test_throws("dae/bad_x_type",
              "seventh argument to integrate_dae must");
  test_throws("dae/bad_x_int_type",
              "eighth argument to integrate_dae must");
  test_throws("dae/bad_rtol_type",
              "ninth argument to integrate_dae");
  test_throws("dae/bad_atol_type",
              "tenth argument to integrate_dae");
  test_throws("dae/bad_max_step_type",
              "eleventh argument to integrate_dae");

  // check data-only types
  test_throws("dae/bad_t0_var_type",
              "fourth argument to integrate_dae");
  test_throws("dae/bad_ts_var_type",
              "fifth argument to integrate_dae");
  test_throws("dae/bad_x_var_type",
              "seventh argument to integrate_dae");
}

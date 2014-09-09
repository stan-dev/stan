#include <gtest/gtest.h>
#include <test/unit/gm/utility.hpp>

TEST(gm_parser, integrate_ode_good) {
  test_parsable("ode_good");
}
TEST(gm_parser, integrate_ode_bad) {
  test_throws("ode_bad_fun_type",
              "first argument to integrate_ode must be a function with signature");

  test_throws("ode_bad_y_type",
              "second argument to integrate_ode must be type real[]");
  test_throws("ode_bad_t_type",
              "third argument to integrate_ode must be type real or int");
  test_throws("ode_bad_ts_type",
              "fourth argument to integrate_ode must be type real[]");
  test_throws("ode_bad_theta_type",
              "fifth argument to integrate_ode must be type real[]");
  test_throws("ode_bad_x_type",
              "sixth argument to integrate_ode must be type real[]");
  test_throws("ode_bad_x_int_type",
              "seventh argument to integrate_ode must be type int[]");

  test_throws("ode_bad_t0_var_type",
              "third argument to integrate_ode (initial times) must be data only");
  test_throws("ode_bad_ts_var_type",
              "fourth argument to integrate_ode (solution times) must be data only");
  test_throws("ode_bad_x_var_type",
              "fifth argument to integrate_ode (real data) must be data only");


}

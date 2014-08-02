#include <gtest/gtest.h>
#include <test/unit/gm/utility.hpp>

TEST(gm_parser, solve_ode_good) {
  test_parsable("ode_good");
}
TEST(gm_parser, solve_ode_bad) {
  test_throws("ode_bad_fun_type",
              "first argument to solve_ode must be a function with signature");

  test_throws("ode_bad_y_type",
              "second argument to solve_ode must be type real[]");
  test_throws("ode_bad_t_type",
              "third argument to solve_ode must be type real or int");
  test_throws("ode_bad_ts_type",
              "fourth argument to solve_ode must be type real[]");
  test_throws("ode_bad_theta_type",
              "fifth argument to solve_ode must be type real[]");
  test_throws("ode_bad_x_type",
              "sixth argument to solve_ode must be type real[]");
  test_throws("ode_bad_x_int_type",
              "seventh argument to solve_ode must be type int[]");

  test_throws("ode_bad_t0_var_type",
              "third argument to solve_ode (initial times) must be data only");
  test_throws("ode_bad_ts_var_type",
              "fourth argument to solve_ode (solution times) must be data only");
  test_throws("ode_bad_x_var_type",
              "fifth argument to solve_ode (real data) must be data only");


}

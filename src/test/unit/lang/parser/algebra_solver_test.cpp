#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(lang_parser, algebra_solver_good) {
  test_parsable("algebra_solver_good");
}

TEST(lang_parser, algebra_solver_bad) {
  test_throws("algebra_solver/bad_fun_type",
              "first argument to algebra_solver must be the name of a function with signature");
  test_throws("algebra_solver/bad_y_type",
              "second argument to algebra_solver must have type vector");
  test_throws("algebra_solver/bad_theta_type",
              "third argument to algebra_solver must have type vector");
  test_throws("algebra_solver/bad_x_r_type",
              "fourth argument to algebra_solver must have type real[]");
  test_throws("algebra_solver/bad_x_i_type",
              "fifth argument to algebra_solver must have type int[]");

  test_throws("algebra_solver/bad_y_var_type",
              "second argument to algebra_solver (initial guess) must be data only");
  test_throws("algebra_solver/bad_x_r_var_type",
              "fourth argument to algebra_solver (real data) must be data only");
}

TEST(lang_parser, algebra_solver_control_bad) {
  test_throws("algebra_solver/bad_fun_type_control",
              "first argument to algebra_solver must be the name of a function with signature");
  test_throws("algebra_solver/bad_y_type_control",
              "second argument to algebra_solver must have type vector");
  test_throws("algebra_solver/bad_theta_type_control",
              "third argument to algebra_solver must have type vector");
  test_throws("algebra_solver/bad_x_r_type_control",
              "fourth argument to algebra_solver must have type real[]");
  test_throws("algebra_solver/bad_x_i_type_control",
              "fifth argument to algebra_solver must have type int[]");

  test_throws("algebra_solver/bad_y_var_type_control",
              "second argument to algebra_solver (initial guess) must be data only");
  test_throws("algebra_solver/bad_x_r_var_type_control",
              "fourth argument to algebra_solver (real data) must be data only");
}

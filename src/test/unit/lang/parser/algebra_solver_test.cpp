#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(lang_parser, algebra_solver_good) {
  test_parsable("algebra_solver_good");
}

TEST(lang_parser, algebra_solver_bad) {
  test_throws("algebra_solver/bad_fun_type",
              "Wrong signature ");
  test_throws("algebra_solver/bad_y_type",
              "Second argument to algebra_solver must have type");
  test_throws("algebra_solver/bad_theta_type",
              "Third argument to algebra_solver must have type");
  test_throws("algebra_solver/bad_x_r_type",
              "Fourth argument to algebra_solver must have type");
  test_throws("algebra_solver/bad_x_i_type",
              "Fifth argument to algebra_solver must have type");

  test_throws("algebra_solver/bad_x_r_var_type",
              "Fourth argument to algebra_solver must be data only");
}

TEST(lang_parser, algebra_solver_control_bad) {
  test_throws("algebra_solver/bad_fun_type_control",
              "Wrong signature ");
  test_throws("algebra_solver/bad_y_type_control",
              "Second argument to algebra_solver must have type");
  test_throws("algebra_solver/bad_theta_type_control",
              "Third argument to algebra_solver must have type");
  test_throws("algebra_solver/bad_x_r_type_control",
              "Fourth argument to algebra_solver must have type");
  test_throws("algebra_solver/bad_x_i_type_control",
              "Fifth argument to algebra_solver must have type");

  test_throws("algebra_solver/bad_x_r_var_type_control",
              "Fourth argument to algebra_solver must be data only");

}

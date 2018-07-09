#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(lang_parser, integrate_ode_good) {
  test_parsable("ode_good");
}
TEST(lang_parser, integrate_ode_bad) {
  test_throws("ode/bad_fun_type",
              "Wrong signature ");
  test_throws("ode/bad_y_type",
              "Second argument to integrate_ode must");
  test_throws("ode/bad_t_type",
              "Third argument to integrate_ode must");
  test_throws("ode/bad_ts_type",
              "Fourth argument to integrate_ode must");
  test_throws("ode/bad_theta_type",
              "Fifth argument to integrate_ode must");
  test_throws("ode/bad_x_type",
              "Sixth argument to integrate_ode must");
  test_throws("ode/bad_x_int_type",
              "Seventh argument to integrate_ode must");
  test_throws("ode/bad_t0_var_type",
              "Third argument to integrate_ode (initial times)");
  test_throws("ode/bad_ts_var_type",
              "Fourth argument to integrate_ode (solution times)");
  test_throws("ode/bad_x_var_type",
              "Sixth argument to integrate_ode (real data)");
}
TEST(lang_parser, integrate_ode_rk45_bad) {
  test_throws("ode/bad_fun_type_rk45",
              "Wrong signature for function integrate_ode_rk45");
  test_throws("ode/bad_y_type_rk45",
              "Second argument to integrate_ode_rk45 must");
  test_throws("ode/bad_t_type_rk45",
              "Third argument to integrate_ode_rk45 must");
  test_throws("ode/bad_ts_type_rk45",
              "Fourth argument to integrate_ode_rk45 must");
  test_throws("ode/bad_theta_type_rk45",
              "Fifth argument to integrate_ode_rk45 must");
  test_throws("ode/bad_x_type_rk45",
              "Sixth argument to integrate_ode_rk45 must");
  test_throws("ode/bad_x_int_type_rk45",
              "Seventh argument to integrate_ode_rk45 must");
  test_throws("ode/bad_t0_var_type_rk45",
              "Third argument to integrate_ode_rk45 (initial times)");
  test_throws("ode/bad_ts_var_type_rk45",
              "Fourth argument to integrate_ode_rk45 (solution times)");
  test_throws("ode/bad_x_var_type_rk45",
              "Sixth argument to integrate_ode_rk45 (real data)");
}
TEST(lang_parser, integrate_ode_bdf_bad) {
  test_throws("ode/bad_bdf_control_function_return",
              "Wrong signature ");
  test_throws("ode/bad_fun_type_bdf",
              "Wrong signature ");
  test_throws("ode/bad_y_type_bdf",
              "Second argument to integrate_ode_bdf must");
  test_throws("ode/bad_t_type_bdf",
              "Third argument to integrate_ode_bdf must have");
  test_throws("ode/bad_ts_type_bdf",
              "Fourth argument to integrate_ode_bdf must");
  test_throws("ode/bad_theta_type_bdf",
              "Fifth argument to integrate_ode_bdf must");
  test_throws("ode/bad_x_type_bdf",
              "Sixth argument to integrate_ode_bdf must");
  test_throws("ode/bad_x_int_type_bdf",
              "Seventh argument to integrate_ode_bdf must");
  test_throws("ode/bad_t0_var_type_bdf",
              "Third argument to integrate_ode_bdf (initial times)");
  test_throws("ode/bad_ts_var_type_bdf",
              "Fourth argument to integrate_ode_bdf (solution times)");
  test_throws("ode/bad_x_var_type_bdf",
              "Sixth argument to integrate_ode_bdf (real data)");
}



TEST(lang_parser, integrate_ode_rk45_control_bad) {
  test_throws("ode/bad_fun_type_rk45_control",
              "Wrong signature ");
  test_throws("ode/bad_y_type_rk45_control",
              "Second argument to integrate_ode_rk45 must");
  test_throws("ode/bad_t_type_rk45_control",
              "Third argument to integrate_ode_rk45 must");
  test_throws("ode/bad_ts_type_rk45_control",
              "Fourth argument to integrate_ode_rk45 must");
  test_throws("ode/bad_theta_type_rk45_control",
              "Fifth argument to integrate_ode_rk45 must");
  test_throws("ode/bad_x_type_rk45_control",
              "Sixth argument to integrate_ode_rk45 must");
  test_throws("ode/bad_x_int_type_rk45_control",
              "Seventh argument to integrate_ode_rk45 must");
  test_throws("ode/bad_t0_var_type_rk45_control",
              "Third argument to integrate_ode_rk45 (initial times)");
  test_throws("ode/bad_ts_var_type_rk45_control",
              "Fourth argument to integrate_ode_rk45 (solution times)");
  test_throws("ode/bad_x_var_type_rk45_control",
              "Sixth argument to integrate_ode_rk45 (real data)");
}
TEST(lang_parser, integrate_ode_bdf_control_bad) {
  test_throws("ode/bad_fun_type_bdf_control",
              "Wrong signature ");
  test_throws("ode/bad_y_type_bdf_control",
              "Second argument to integrate_ode_bdf must");
  test_throws("ode/bad_t_type_bdf_control",
              "Third argument to integrate_ode_bdf must");
  test_throws("ode/bad_ts_type_bdf_control",
              "Fourth argument to integrate_ode_bdf must");
  test_throws("ode/bad_theta_type_bdf_control",
              "Fifth argument to integrate_ode_bdf must");
  test_throws("ode/bad_x_type_bdf_control",
              "Sixth argument to integrate_ode_bdf must");
  test_throws("ode/bad_x_int_type_bdf_control",
              "Seventh argument to integrate_ode_bdf must");
  test_throws("ode/bad_t0_var_type_bdf_control",
              "Third argument to integrate_ode_bdf (initial times)");
  test_throws("ode/bad_ts_var_type_bdf_control",
              "Fourth argument to integrate_ode_bdf (solution times)");
  test_throws("ode/bad_x_var_type_bdf_control",
              "Sixth argument to integrate_ode_bdf (real data)");
}

TEST(lang_parser, integrate_ode_adams_bad) {
  test_throws("ode/adams/bad_adams_control_function_return",
      "first argument to integrate_ode_adams must be the name of a function with signature");
  test_throws("ode/adams/bad_fun_type",
      "first argument to integrate_ode_adams must be the name of a function with signature");
  test_throws("ode/adams/bad_y_type",
              "second argument to integrate_ode_adams must have type real[]");
  test_throws("ode/adams/bad_t_type",
          "third argument to integrate_ode_adams must have type real or int");
  test_throws("ode/adams/bad_ts_type",
              "fourth argument to integrate_ode_adams must have type real[]");
  test_throws("ode/adams/bad_theta_type",
              "fifth argument to integrate_ode_adams must have type real[]");
  test_throws("ode/adams/bad_x_type",
              "sixth argument to integrate_ode_adams must have type real[]");
  test_throws("ode/adams/bad_x_int_type",
              "seventh argument to integrate_ode_adams must have type int[]");
  test_throws("ode/adams/bad_t0_var_type",
      "third argument to integrate_ode_adams (initial times) must be data only");
  test_throws("ode/adams/bad_x_var_type",
      "sixth argument to integrate_ode_adams (real data) must be data only");
}

TEST(lang_parser, integrate_ode_adams_control_bad) {
  test_throws("ode/adams/bad_fun_type_control",
      "first argument to integrate_ode_adams must be the name of a function with signature");
  test_throws("ode/adams/bad_y_type_control",
              "second argument to integrate_ode_adams must have type real[]");
  test_throws("ode/adams/bad_t_type_control",
          "third argument to integrate_ode_adams must have type real or int");
  test_throws("ode/adams/bad_ts_type_control",
              "fourth argument to integrate_ode_adams must have type real[]");
  test_throws("ode/adams/bad_theta_type_control",
              "fifth argument to integrate_ode_adams must have type real[]");
  test_throws("ode/adams/bad_x_type_control",
              "sixth argument to integrate_ode_adams must have type real[]");
  test_throws("ode/adams/bad_x_int_type_control",
              "seventh argument to integrate_ode_adams must have type int[]");
  test_throws("ode/adams/bad_t0_var_type_control",
      "third argument to integrate_ode_adams (initial times) must be data only");
  test_throws("ode/adams/bad_ts_var_type_control",
    "fourth argument to integrate_ode_adams (solution times) must be data only");
  test_throws("ode/adams/bad_x_var_type_adams_control",
      "sixth argument to integrate_ode_adams (real data) must be data only");
}

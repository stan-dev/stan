#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(lang_parser, integrate_ode_good) {
  test_parsable("ode_good");
}
TEST(lang_parser, integrate_ode_bad) {
  test_throws("ode/bad_fun_type",
              "wrong signature ");
  test_throws("ode/bad_y_type",
              "second argument to integrate_ode must");
  test_throws("ode/bad_t_type",
              "third argument to integrate_ode must");
  test_throws("ode/bad_ts_type",
              "fourth argument to integrate_ode must");
  test_throws("ode/bad_theta_type",
              "fifth argument to integrate_ode must");
  test_throws("ode/bad_x_type",
              "sixth argument to integrate_ode must");
  test_throws("ode/bad_x_int_type",
              "seventh argument to integrate_ode must");
  test_throws("ode/bad_t0_var_type",
              "third argument to integrate_ode (initial times)");
  test_throws("ode/bad_ts_var_type",
              "fourth argument to integrate_ode (solution times)");
  test_throws("ode/bad_x_var_type",
              "sixth argument to integrate_ode (real data)");
}
TEST(lang_parser, integrate_ode_rk45_bad) {
  test_throws("ode/bad_fun_type_rk45",
              "wrong signature for function integrate_ode_rk45");
  test_throws("ode/bad_y_type_rk45",
              "second argument to integrate_ode_rk45 must");
  test_throws("ode/bad_t_type_rk45",
              "third argument to integrate_ode_rk45 must");
  test_throws("ode/bad_ts_type_rk45",
              "fourth argument to integrate_ode_rk45 must");
  test_throws("ode/bad_theta_type_rk45",
              "fifth argument to integrate_ode_rk45 must");
  test_throws("ode/bad_x_type_rk45",
              "sixth argument to integrate_ode_rk45 must");
  test_throws("ode/bad_x_int_type_rk45",
              "seventh argument to integrate_ode_rk45 must");
  test_throws("ode/bad_t0_var_type_rk45",
              "third argument to integrate_ode_rk45 (initial times)");
  test_throws("ode/bad_ts_var_type_rk45",
              "fourth argument to integrate_ode_rk45 (solution times)");
  test_throws("ode/bad_x_var_type_rk45",
              "sixth argument to integrate_ode_rk45 (real data)");
}
TEST(lang_parser, integrate_ode_bdf_bad) {
  test_throws("ode/bad_bdf_control_function_return",
              "wrong signature ");
  test_throws("ode/bad_fun_type_bdf",
              "wrong signature ");
  test_throws("ode/bad_y_type_bdf",
              "second argument to integrate_ode_bdf must");
  test_throws("ode/bad_t_type_bdf",
              "third argument to integrate_ode_bdf must have");
  test_throws("ode/bad_ts_type_bdf",
              "fourth argument to integrate_ode_bdf must");
  test_throws("ode/bad_theta_type_bdf",
              "fifth argument to integrate_ode_bdf must");
  test_throws("ode/bad_x_type_bdf",
              "sixth argument to integrate_ode_bdf must");
  test_throws("ode/bad_x_int_type_bdf",
              "seventh argument to integrate_ode_bdf must");
  test_throws("ode/bad_t0_var_type_bdf",
              "third argument to integrate_ode_bdf (initial times)");
  test_throws("ode/bad_ts_var_type_bdf",
              "fourth argument to integrate_ode_bdf (solution times)");
  test_throws("ode/bad_x_var_type_bdf",
              "sixth argument to integrate_ode_bdf (real data)");
}
TEST(lang_parser, integrate_ode_rk45_control_bad) {
  test_throws("ode/bad_fun_type_rk45_control",
              "wrong signature ");
  test_throws("ode/bad_y_type_rk45_control",
              "second argument to integrate_ode_rk45 must");
  test_throws("ode/bad_t_type_rk45_control",
              "third argument to integrate_ode_rk45 must");
  test_throws("ode/bad_ts_type_rk45_control",
              "fourth argument to integrate_ode_rk45 must");
  test_throws("ode/bad_theta_type_rk45_control",
              "fifth argument to integrate_ode_rk45 must");
  test_throws("ode/bad_x_type_rk45_control",
              "sixth argument to integrate_ode_rk45 must");
  test_throws("ode/bad_x_int_type_rk45_control",
              "seventh argument to integrate_ode_rk45 must");
  test_throws("ode/bad_t0_var_type_rk45_control",
              "third argument to integrate_ode_rk45 (initial times)");
  test_throws("ode/bad_ts_var_type_rk45_control",
              "fourth argument to integrate_ode_rk45 (solution times)");
  test_throws("ode/bad_x_var_type_rk45_control",
              "sixth argument to integrate_ode_rk45 (real data)");
}
TEST(lang_parser, integrate_ode_bdf_control_bad) {
  test_throws("ode/bad_fun_type_bdf_control",
              "wrong signature ");
  test_throws("ode/bad_y_type_bdf_control",
              "second argument to integrate_ode_bdf must");
  test_throws("ode/bad_t_type_bdf_control",
              "third argument to integrate_ode_bdf must");
  test_throws("ode/bad_ts_type_bdf_control",
              "fourth argument to integrate_ode_bdf must");
  test_throws("ode/bad_theta_type_bdf_control",
              "fifth argument to integrate_ode_bdf must");
  test_throws("ode/bad_x_type_bdf_control",
              "sixth argument to integrate_ode_bdf must");
  test_throws("ode/bad_x_int_type_bdf_control",
              "seventh argument to integrate_ode_bdf must");
  test_throws("ode/bad_t0_var_type_bdf_control",
              "third argument to integrate_ode_bdf (initial times)");
  test_throws("ode/bad_ts_var_type_bdf_control",
              "fourth argument to integrate_ode_bdf (solution times)");
  test_throws("ode/bad_x_var_type_bdf_control",
              "sixth argument to integrate_ode_bdf (real data)");
}
TEST(lang_parser, integrate_ode_adams_bad) {
  test_throws("ode/adams/bad_adams_control_function_return",
      "wrong signature ");
  test_throws("ode/adams/bad_y_type",
              "second argument to integrate_ode_adams must have type real[ ]");
  test_throws("ode/adams/bad_t_type",
              "third argument to integrate_ode_adams must have type real");
  test_throws("ode/adams/bad_ts_type",
              "fourth argument to integrate_ode_adams must have type real[ ]");
  test_throws("ode/adams/bad_theta_type",
              "fifth argument to integrate_ode_adams must have type real[ ]");
  test_throws("ode/adams/bad_x_type",
              "sixth argument to integrate_ode_adams must have type data real[ ]");
  test_throws("ode/adams/bad_x_int_type",
              "seventh argument to integrate_ode_adams must have type data int[ ]");
  test_throws("ode/adams/bad_t0_var_type",
      "third argument to integrate_ode_adams (initial times) must be data only");
  test_throws("ode/adams/bad_x_var_type",
      "sixth argument to integrate_ode_adams (real data) must be data only");
}
TEST(lang_parser, integrate_ode_adams_control_bad) {
  test_throws("ode/adams/bad_fun_type_control",
      "wrong signature ");
  test_throws("ode/adams/bad_y_type_control",
              "second argument to integrate_ode_adams must have type real[ ]");
  test_throws("ode/adams/bad_t_type_control",
          "third argument to integrate_ode_adams must have type real");
  test_throws("ode/adams/bad_ts_type_control",
              "fourth argument to integrate_ode_adams must have type real[ ]");
  test_throws("ode/adams/bad_theta_type_control",
              "fifth argument to integrate_ode_adams must have type real[ ]");
  test_throws("ode/adams/bad_x_type_control",
              "sixth argument to integrate_ode_adams must have type data real[ ]");
  test_throws("ode/adams/bad_x_int_type_control",
              "seventh argument to integrate_ode_adams must have type data int[ ]");
  test_throws("ode/adams/bad_t0_var_type_control",
      "third argument to integrate_ode_adams (initial times) must be data only");
  test_throws("ode/adams/bad_ts_var_type_control",
    "fourth argument to integrate_ode_adams (solution times) must be data only");
  test_throws("ode/adams/bad_x_var_type_adams_control",
      "sixth argument to integrate_ode_adams (real data) must be data only");
}

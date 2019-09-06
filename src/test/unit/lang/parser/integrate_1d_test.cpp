#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(lang_parser, integrate_1d_good) {
  test_parsable("integrate_1d_good");
}
TEST(lang_parser, integrate_1d_bad) {
  test_throws("integrate_1d/bad_fun_sig",
              "first argument to integrate_1d must be the name"
              " of a function with signature (real, real,"
              " real[], real[], int[]) : real");
  test_throws("integrate_1d/bad_lb_type",
              "second argument to integrate_1d, the lower bound of"
              " integration, must have type int or real; found type = real[ ]");
  test_throws("integrate_1d/bad_ub_type",
              "third argument to integrate_1d, the upper bound of"
              " integration, must have type int or real; found type = real[ ]");
  test_throws("integrate_1d/bad_parameters_type",
              "fourth argument to integrate_1d, the parameters, must have"
              " type real[]; found type = int.");
  test_throws("integrate_1d/bad_real_data_type",
              "fifth argument to integrate_1d, the real data, must have"
              " type real[]; found type = real.");
  test_throws("integrate_1d/bad_int_data_type",
              "sixth argument to integrate_1d, the integer data, must have"
              " type int[]; found type = real[ ].");
  test_throws("integrate_1d/bad_rel_tol_type",
              "seventh argument to integrate_1d, relative tolerance,"
              " must be of type int or real;  found type = real[ ].");

  test_throws("integrate_1d/bad_real_data_data",
              "fifth argument to integrate_1d, the real data, must"
              " be data only and not reference parameters.");
  test_throws("integrate_1d/bad_rel_tol_data",
              "seventh argument to integrate_1d, relative tolerance,"
              " must be data only and not reference parameters.");
  test_throws("integrate_1d/bad_fun_name",
              "integrated function may not be an _rng function,"
              " found function name: normal_rng");
}


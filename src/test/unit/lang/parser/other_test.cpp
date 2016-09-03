#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <istream>
#include <sstream>
#include <exception>
#include <stdexcept>
#include <test/unit/lang/utility.hpp>

TEST(lang_parser,good_trunc) {
  EXPECT_TRUE(is_parsable("src/test/test-models/bad/lang/good_trunc.stan"));
}

TEST(lang_parser,good_vec_constraints) {
  EXPECT_TRUE(is_parsable("src/test/test-models/bad/lang/good_trunc.stan"));
}

TEST(lang_parser,good_const) {
  EXPECT_TRUE(is_parsable("src/test/test-models/bad/lang/good_const.stan"));
}

TEST(lang_parser,good_matrix_ops) {
  EXPECT_TRUE(is_parsable("src/test/test-models/bad/lang/good_matrix_ops.stan"));
}

TEST(lang_parser,good_funs) {
  EXPECT_TRUE(is_parsable("src/test/test-models/bad/lang/good_funs.stan"));
}


TEST(lang_parser,good_vars) {
  EXPECT_TRUE(is_parsable("src/test/test-models/bad/lang/good_vars.stan"));
}

TEST(lang_parser,good_intercept_var) {
  EXPECT_TRUE(is_parsable("src/test/test-models/bad/lang/good_intercept_var.stan"));
}

TEST(lang_parser,good_cov) {
  EXPECT_TRUE(is_parsable("src/test/test-models/bad/lang/good_cov.stan"));
}

TEST(lang_parser,good_local_var_array_size) {
  EXPECT_TRUE(is_parsable("src/test/test-models/bad/lang/good_local_var_array_size.stan"));
}

TEST(lang_parser,parsable_test_bad1) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad1.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad2) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad2.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad3) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad3.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad4) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad4.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad5) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad5.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad6) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad6.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad7) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad7.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad8) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad8.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad9) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad9.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad10) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad10.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad11) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad11.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad_fun_name) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_fun_name.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad_cov_exp_quad_vec_rvec_data) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_cov_exp_quad_vec_rvec_data.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad_cov_exp_quad_vec_arr_data) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_cov_exp_quad_vec_arr_data.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad_cov_exp_quad_rvec_vec_data) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_cov_exp_quad_rvec_vec_data.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad_cov_exp_quad_rvec_arr_data) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_cov_exp_quad_rvec_arr_data.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad_cov_exp_quad_arr_vec_data) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_cov_exp_quad_arr_vec_data.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad_cov_exp_quad_arr_rvec_data) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_cov_exp_quad_arr_rvec_data.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad_cov_exp_quad_vec_rvec_param) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_cov_exp_quad_vec_rvec_param.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad_cov_exp_quad_vec_arr_param) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_cov_exp_quad_vec_arr_param.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad_cov_exp_quad_rvec_vec_param) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_cov_exp_quad_rvec_vec_param.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad_cov_exp_quad_rvec_arr_param) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_cov_exp_quad_rvec_arr_param.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad_cov_exp_quad_arr_vec_param) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_cov_exp_quad_arr_vec_param.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad_cov_exp_quad_arr_rvec_param) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_cov_exp_quad_arr_rvec_param.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad_cov_exp_quad_vec_data) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_cov_exp_quad_vec_data.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad_cov_exp_quad_rvec_data) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_cov_exp_quad_rvec_data.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad_cov_exp_quad_sigma_data) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_cov_exp_quad_sigma_data.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad_cov_exp_quad_len_data) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_cov_exp_quad_len_data.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad_cov_exp_quad_sigma_param) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_cov_exp_quad_sigma_param.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad_cov_exp_quad_len_param) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_cov_exp_quad_len_param.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad_cov_exp_quad_sigma_vec_param) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_cov_exp_quad_sigma_vec_param.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad_cov_exp_quad_len_vec_param) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_cov_exp_quad_len_vec_param.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad_cov_exp_quad_sigma_rvec_param) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_cov_exp_quad_sigma_rvec_param.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_bad_cov_exp_quad_len_rvec_param) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_cov_exp_quad_len_rvec_param.stan"),
               std::invalid_argument);
}

TEST(lang_parser,parsable_test_good_fun_name) {
  EXPECT_TRUE(is_parsable("src/test/test-models/bad/lang/good_fun_name.stan"));
}

TEST(langParser,parsableBadPeriods) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_periods_data.stan"),
               std::invalid_argument);
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_periods_tdata.stan"),
               std::invalid_argument);
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_periods_params.stan"),
               std::invalid_argument);
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_periods_tparams.stan"),
               std::invalid_argument);
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_periods_gqs.stan"),
               std::invalid_argument);
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_periods_local.stan"),
               std::invalid_argument);
}

TEST(langParser,declareVarWithSameNameAsModel) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/lang/bad_model_name_var.stan"),
               std::invalid_argument);
}

TEST(lang_parser, infVariableName) {
  test_parsable("good_inf");
}

TEST(lang_parser, declarations_funciton_signatures) {
  test_parsable("declarations");
}

TEST(lang_parser, illegal_generated_quantities) {
  EXPECT_THROW(is_parsable("illegal_generated_quantities"),
               std::invalid_argument);
}

TEST(lang_parser, illegal_transformed_data) {
  EXPECT_THROW(is_parsable("illegal_transformed_data"),
               std::invalid_argument);
}

TEST(lang_parser, illegal_transformed_parameters) {
  EXPECT_THROW(is_parsable("illegal_transformed_parameters"),
               std::invalid_argument);
}

TEST(lang_parser, increment_log_prob) {
  test_parsable("increment_log_prob");
}

TEST(lang_parser, intFun) {
  test_parsable("int_fun");
}

TEST(lang_parser, print_chars) {
  test_parsable("print_chars");
}

TEST(lang_parser, print_indexing) {
  test_parsable("print_indexing");
}


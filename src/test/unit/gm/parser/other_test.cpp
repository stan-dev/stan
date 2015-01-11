#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <istream>
#include <sstream>
#include <exception>
#include <stdexcept>
#include <test/unit/gm/utility.hpp>

TEST(gm_parser,good_trunc) {
  EXPECT_TRUE(is_parsable("src/test/test-models/bad/gm/good_trunc.stan"));
}

TEST(gm_parser,good_vec_constraints) {
  EXPECT_TRUE(is_parsable("src/test/test-models/bad/gm/good_trunc.stan"));
}

TEST(gm_parser,good_const) {
  EXPECT_TRUE(is_parsable("src/test/test-models/bad/gm/good_const.stan"));
}

TEST(gm_parser,good_matrix_ops) {
  EXPECT_TRUE(is_parsable("src/test/test-models/bad/gm/good_matrix_ops.stan"));
}

TEST(gm_parser,good_funs) {
  EXPECT_TRUE(is_parsable("src/test/test-models/bad/gm/good_funs.stan"));
}


TEST(gm_parser,good_vars) {
  EXPECT_TRUE(is_parsable("src/test/test-models/bad/gm/good_vars.stan"));
}

TEST(gm_parser,good_intercept_var) {
  EXPECT_TRUE(is_parsable("src/test/test-models/bad/gm/good_intercept_var.stan"));
}

TEST(gm_parser,good_cov) {
  EXPECT_TRUE(is_parsable("src/test/test-models/bad/gm/good_cov.stan"));
}

TEST(gm_parser,good_local_var_array_size) {
  EXPECT_TRUE(is_parsable("src/test/test-models/bad/gm/good_local_var_array_size.stan"));
}

TEST(gm_parser,parsable_test_bad1) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/gm/bad1.stan"),
               std::invalid_argument);
}

TEST(gm_parser,parsable_test_bad2) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/gm/bad2.stan"),
               std::invalid_argument);
}

TEST(gm_parser,parsable_test_bad3) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/gm/bad3.stan"),
               std::invalid_argument);
}

TEST(gm_parser,parsable_test_bad4) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/gm/bad4.stan"),
               std::invalid_argument);
}

TEST(gm_parser,parsable_test_bad5) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/gm/bad5.stan"),
               std::invalid_argument);
}

TEST(gm_parser,parsable_test_bad6) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/gm/bad6.stan"),
               std::invalid_argument);
}

TEST(gm_parser,parsable_test_bad7) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/gm/bad7.stan"),
               std::invalid_argument);
}

TEST(gm_parser,parsable_test_bad8) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/gm/bad8.stan"),
               std::invalid_argument);
}

TEST(gm_parser,parsable_test_bad9) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/gm/bad9.stan"),
               std::invalid_argument);
}

TEST(gm_parser,parsable_test_bad10) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/gm/bad10.stan"),
               std::invalid_argument);
}

TEST(gm_parser,parsable_test_bad11) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/gm/bad11.stan"),
               std::invalid_argument);
}

TEST(gm_parser,parsable_test_bad_fun_name) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/gm/bad_fun_name.stan"),
               std::invalid_argument);
}

TEST(gm_parser,parsable_test_good_fun_name) {
  EXPECT_TRUE(is_parsable("src/test/test-models/bad/gm/good_fun_name.stan"));
}

TEST(gmParser,parsableBadPeriods) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/gm/bad_periods_data.stan"),
               std::invalid_argument);
  EXPECT_THROW(is_parsable("src/test/test-models/bad/gm/bad_periods_tdata.stan"),
               std::invalid_argument);
  EXPECT_THROW(is_parsable("src/test/test-models/bad/gm/bad_periods_params.stan"),
               std::invalid_argument);
  EXPECT_THROW(is_parsable("src/test/test-models/bad/gm/bad_periods_tparams.stan"),
               std::invalid_argument);
  EXPECT_THROW(is_parsable("src/test/test-models/bad/gm/bad_periods_gqs.stan"),
               std::invalid_argument);
  EXPECT_THROW(is_parsable("src/test/test-models/bad/gm/bad_periods_local.stan"),
               std::invalid_argument);
}

TEST(gmParser,declareVarWithSameNameAsModel) {
  EXPECT_THROW(is_parsable("src/test/test-models/bad/gm/bad_model_name_var.stan"),
               std::invalid_argument);
}

TEST(gm_parser, infVariableName) {
  test_parsable("good_inf");
}

TEST(gm_parser, declarations_funciton_signatures) {
  test_parsable("declarations");
}

TEST(gm_parser, illegal_generated_quantities) {
  EXPECT_THROW(is_parsable("illegal_generated_quantities"),
               std::invalid_argument);
}

TEST(gm_parser, illegal_transformed_data) {
  EXPECT_THROW(is_parsable("illegal_transformed_data"),
               std::invalid_argument);
}

TEST(gm_parser, illegal_transformed_parameters) {
  EXPECT_THROW(is_parsable("illegal_transformed_parameters"),
               std::invalid_argument);
}

TEST(gm_parser, increment_log_prob) {
  test_parsable("increment_log_prob");
}

TEST(gm_parser, intFun) {
  test_parsable("int_fun");
}

TEST(gm_parser, print_chars) {
  test_parsable("print_chars");
}

TEST(gm_parser, print_indexing) {
  test_parsable("print_indexing");
}


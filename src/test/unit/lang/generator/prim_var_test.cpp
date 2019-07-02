#include <stan/lang/ast_def.cpp>

#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <string>
#include <test/unit/lang/utility.hpp>
#include <test/unit/util.hpp>

TEST(lang, data_block_var_ast) {
  std::string m1("data {\n"
                 "  int<lower=0, upper=1> p1;\n"
                 "  real p2;\n"
                 "  int ar_p1[3];\n"
                 "  real<lower=0, upper=1> ar_p2[4];\n"
                 "}\n");
  stan::lang::program prog = model_to_ast("data_prim", m1);

  EXPECT_EQ(4, prog.data_decl_.size());
  stan::lang::block_var_decl bvd1 = prog.data_decl_[0];
  stan::lang::block_var_decl bvd2 = prog.data_decl_[1];
  stan::lang::block_var_decl bvd3 = prog.data_decl_[2];
  stan::lang::block_var_decl bvd4 = prog.data_decl_[3];
  EXPECT_EQ("p1", bvd1.name());
  EXPECT_EQ("p2", bvd2.name());
  EXPECT_EQ("ar_p1", bvd3.name());
  EXPECT_EQ("ar_p2", bvd4.name());

  std::stringstream ss;
  write_bare_expr_type(ss, bvd1.type().bare_type());
  EXPECT_EQ("int", ss.str());
  ss.str(std::string());
  write_bare_expr_type(ss, bvd2.type().bare_type());
  EXPECT_EQ("real", ss.str());
  ss.str(std::string());
  write_bare_expr_type(ss, bvd3.type().bare_type());
  EXPECT_EQ("int[ ]", ss.str());
  ss.str(std::string());
  write_bare_expr_type(ss, bvd4.type().bare_type());
  EXPECT_EQ("real[ ]", ss.str());
  ss.str(std::string());

  EXPECT_TRUE(bvd1.type().innermost_type().is_constrained());
  stan::lang::range bounds = bvd1.type().innermost_type().bounds();
  EXPECT_EQ("0", bounds.low_.to_string());
  EXPECT_EQ("1", bounds.high_.to_string());
}

TEST(lang, data_block_var_hpp_class_member_vars) {
  std::string m1("data {\n"
                 "  int<lower=0, upper=1> p1;\n"
                 "  real p2;\n"
                 "  int ar_p1[3];\n"
                 "  real<lower=0, upper=1> ar_p2[4];\n"
                 "}\n");
  std::string hpp = model_to_hpp("data_prim", m1);

  std::string expected("private:\n"
                       "        int p1;\n"
                       "        double p2;\n"
                       "        std::vector<int> ar_p1;\n"
                       "        std::vector<double> ar_p2;\n");

  EXPECT_EQ(1, count_matches(expected, hpp));
}

TEST(lang, data_block_var_hpp_ctor) {
  std::string m1("data {\n"
                 "  int<lower=0, upper=1> p1;\n"
                 "  real p2;\n"
                 "  int ar_p1[3];\n"
                 "  real<lower=0, upper=1> ar_p2[4];\n"
                 "  real<offset=1, multiplier=2> ar_p3[5];\n"
                 "}\n");
  std::string hpp = model_to_hpp("data_prim", m1);

  std::string expected(
      "            // initialize data block variables from context__\n"
      "            current_statement_begin__ = 2;\n"
      "            context__.validate_dims(\"data initialization\", \"p1\", "
      "\"int\", context__.to_vec());\n"
      "            p1 = int(0);\n"
      "            vals_i__ = context__.vals_i(\"p1\");\n"
      "            pos__ = 0;\n"
      "            p1 = vals_i__[pos__++];\n"
      "            check_greater_or_equal(function__, \"p1\", p1, 0);\n"
      "            check_less_or_equal(function__, \"p1\", p1, 1);\n"
      "\n"
      "            current_statement_begin__ = 3;\n"
      "            context__.validate_dims(\"data initialization\", \"p2\", "
      "\"double\", context__.to_vec());\n"
      "            p2 = double(0);\n"
      "            vals_r__ = context__.vals_r(\"p2\");\n"
      "            pos__ = 0;\n"
      "            p2 = vals_r__[pos__++];\n"
      "\n"
      "            current_statement_begin__ = 4;\n"
      "            validate_non_negative_index(\"ar_p1\", \"3\", 3);\n"
      "            context__.validate_dims(\"data initialization\", \"ar_p1\", "
      "\"int\", context__.to_vec(3));\n"
      "            ar_p1 = std::vector<int>(3, int(0));\n"
      "            vals_i__ = context__.vals_i(\"ar_p1\");\n"
      "            pos__ = 0;\n"
      "            size_t ar_p1_k_0_max__ = 3;\n"
      "            for (size_t k_0__ = 0; k_0__ < ar_p1_k_0_max__; ++k_0__) {\n"
      "                ar_p1[k_0__] = vals_i__[pos__++];\n"
      "            }\n"
      "\n"
      "            current_statement_begin__ = 5;\n"
      "            validate_non_negative_index(\"ar_p2\", \"4\", 4);\n"
      "            context__.validate_dims(\"data initialization\", \"ar_p2\", "
      "\"double\", context__.to_vec(4));\n"
      "            ar_p2 = std::vector<double>(4, double(0));\n"
      "            vals_r__ = context__.vals_r(\"ar_p2\");\n"
      "            pos__ = 0;\n"
      "            size_t ar_p2_k_0_max__ = 4;\n"
      "            for (size_t k_0__ = 0; k_0__ < ar_p2_k_0_max__; ++k_0__) {\n"
      "                ar_p2[k_0__] = vals_r__[pos__++];\n"
      "            }\n"
      "            size_t ar_p2_i_0_max__ = 4;\n"
      "            for (size_t i_0__ = 0; i_0__ < ar_p2_i_0_max__; ++i_0__) {\n"
      "                check_greater_or_equal(function__, \"ar_p2[i_0__]\", "
      "ar_p2[i_0__], 0);\n"
      "                check_less_or_equal(function__, \"ar_p2[i_0__]\", "
      "ar_p2[i_0__], 1);\n"
      "            }\n"
      "\n"
      "            current_statement_begin__ = 6;\n"
      "            validate_non_negative_index(\"ar_p3\", \"5\", 5);\n"
      "            context__.validate_dims(\"data initialization\", \"ar_p3\", "
      "\"double\", context__.to_vec(5));\n"
      "            ar_p3 = std::vector<double>(5, double(0));\n"
      "            vals_r__ = context__.vals_r(\"ar_p3\");\n"
      "            pos__ = 0;\n"
      "            size_t ar_p3_k_0_max__ = 5;\n"
      "            for (size_t k_0__ = 0; k_0__ < ar_p3_k_0_max__; ++k_0__) {\n"
      "                ar_p3[k_0__] = vals_r__[pos__++];\n"
      "            }\n"
      "\n");

  EXPECT_EQ(1, count_matches(expected, hpp));
}

TEST(lang, transformed_data_block_var_ast) {
  std::string m1("transformed data {\n"
                 "  int<lower=0, upper=1> p1;\n"
                 "  real p2;\n"
                 "  int ar_p1[3];\n"
                 "  real<lower=0, upper=1> ar_p2[4];\n"
                 "}\n");
  stan::lang::program prog = model_to_ast("transformed_data_prim", m1);

  EXPECT_EQ(4, prog.derived_data_decl_.first.size());
  stan::lang::block_var_decl bvd1 = prog.derived_data_decl_.first[0];
  stan::lang::block_var_decl bvd2 = prog.derived_data_decl_.first[1];
  stan::lang::block_var_decl bvd3 = prog.derived_data_decl_.first[2];
  stan::lang::block_var_decl bvd4 = prog.derived_data_decl_.first[3];
  EXPECT_EQ("p1", bvd1.name());
  EXPECT_EQ("p2", bvd2.name());
  EXPECT_EQ("ar_p1", bvd3.name());
  EXPECT_EQ("ar_p2", bvd4.name());
}

TEST(lang, transformed_data_block_var_hpp_ctor) {
  std::string m1("transformed data {\n"
                 "  int<lower=0, upper=1> p1;\n"
                 "  real p2;\n"
                 "  int ar_p1[3];\n"
                 "  real<lower=0, upper=1> ar_p2[4];\n"
                 "}\n");
  std::string hpp = model_to_hpp("transformed_data_prim", m1);

  std::string expected_1(
      "            // initialize transformed data variables\n"
      "            current_statement_begin__ = 2;\n"
      "            p1 = int(0);\n"
      "            stan::math::fill(p1, std::numeric_limits<int>::min());\n"
      "\n"
      "            current_statement_begin__ = 3;\n"
      "            p2 = double(0);\n"
      "            stan::math::fill(p2, DUMMY_VAR__);\n"
      "\n"
      "            current_statement_begin__ = 4;\n"
      "            validate_non_negative_index(\"ar_p1\", \"3\", 3);\n"
      "            ar_p1 = std::vector<int>(3, int(0));\n"
      "            stan::math::fill(ar_p1, std::numeric_limits<int>::min());\n"
      "\n"
      "            current_statement_begin__ = 5;\n"
      "            validate_non_negative_index(\"ar_p2\", \"4\", 4);\n"
      "            ar_p2 = std::vector<double>(4, double(0));\n"
      "            stan::math::fill(ar_p2, DUMMY_VAR__);\n");
  EXPECT_EQ(1, count_matches(expected_1, hpp));

  std::string expected_2(
      "            // validate transformed data\n"
      "            current_statement_begin__ = 2;\n"
      "            check_greater_or_equal(function__, \"p1\", p1, 0);\n"
      "            check_less_or_equal(function__, \"p1\", p1, 1);\n"
      "\n"
      "            current_statement_begin__ = 5;\n"
      "            size_t ar_p2_i_0_max__ = 4;\n"
      "            for (size_t i_0__ = 0; i_0__ < ar_p2_i_0_max__; ++i_0__) {\n"
      "                check_greater_or_equal(function__, \"ar_p2[i_0__]\", "
      "ar_p2[i_0__], 0);\n"
      "                check_less_or_equal(function__, \"ar_p2[i_0__]\", "
      "ar_p2[i_0__], 1);\n"
      "            }\n");
  EXPECT_EQ(1, count_matches(expected_2, hpp));
}

TEST(lang, params_block_var_ast) {
  std::string m1("parameters {\n"
                 "  real p2;\n"
                 "  real<lower=0, upper=1> ar_p2[4];\n"
                 "  real<offset=1, multiplier=2> ar_p3[5];\n"
                 "}\n");
  stan::lang::program prog = model_to_ast("parameters_prim", m1);
  EXPECT_EQ(3, prog.parameter_decl_.size());
  stan::lang::block_var_decl bvd1 = prog.parameter_decl_[0];
  stan::lang::block_var_decl bvd2 = prog.parameter_decl_[1];
  stan::lang::block_var_decl bvd3 = prog.parameter_decl_[2];
  EXPECT_EQ("p2", bvd1.name());
  EXPECT_EQ("ar_p2", bvd2.name());
  EXPECT_EQ("ar_p3", bvd3.name());
}

TEST(lang, params_block_var_hpp_ctor) {
  std::string m1("parameters {\n"
                 "  real p2;\n"
                 "  real<lower=0, upper=1> ar_p2[4];\n"
                 "  real<offset=1, multiplier=2> ar_p3[5];\n"
                 "}\n");
  std::string hpp = model_to_hpp("parameters_prim", m1);

  std::string expected(
      "            // validate, set parameter ranges\n"
      "            num_params_r__ = 0U;\n"
      "            param_ranges_i__.clear();\n"
      "            current_statement_begin__ = 2;\n"
      "            num_params_r__ += 1;\n"
      "            current_statement_begin__ = 3;\n"
      "            validate_non_negative_index(\"ar_p2\", \"4\", 4);\n"
      "            num_params_r__ += (1 * 4);\n"
      "            current_statement_begin__ = 4;\n"
      "            validate_non_negative_index(\"ar_p3\", \"5\", 5);\n"
      "            num_params_r__ += (1 * 5);\n");
  EXPECT_EQ(1, count_matches(expected, hpp));
}

TEST(lang, params_block_var_hpp_xform_inits) {
  // transform_inits block has parameter initialization
  std::string m1("parameters {\n"
                 "  real p2;\n"
                 "  real<lower=0, upper=1> ar_p2[4];\n"
                 "  real<offset=1, multiplier=2> ar_p3[5];\n"
                 "}\n");
  std::string hpp = model_to_hpp("parameters_prim", m1);

  std::string expected(
      "        current_statement_begin__ = 2;\n"
      "        if (!(context__.contains_r(\"p2\")))\n"
      "            "
      "stan::lang::rethrow_located(std::runtime_error(std::string(\"Variable "
      "p2 missing\")), current_statement_begin__, prog_reader__());\n"
      "        vals_r__ = context__.vals_r(\"p2\");\n"
      "        pos__ = 0U;\n"
      "        context__.validate_dims(\"parameter initialization\", \"p2\", "
      "\"double\", context__.to_vec());\n"
      "        double p2(0);\n"
      "        p2 = vals_r__[pos__++];\n"
      "        try {\n"
      "            writer__.scalar_unconstrain(p2);\n"
      "        } catch (const std::exception& e) {\n"
      "            "
      "stan::lang::rethrow_located(std::runtime_error(std::string(\"Error "
      "transforming variable p2: \") + e.what()), current_statement_begin__, "
      "prog_reader__());\n"
      "        }\n"
      "\n"
      "        current_statement_begin__ = 3;\n"
      "        if (!(context__.contains_r(\"ar_p2\")))\n"
      "            "
      "stan::lang::rethrow_located(std::runtime_error(std::string(\"Variable "
      "ar_p2 missing\")), current_statement_begin__, prog_reader__());\n"
      "        vals_r__ = context__.vals_r(\"ar_p2\");\n"
      "        pos__ = 0U;\n"
      "        validate_non_negative_index(\"ar_p2\", \"4\", 4);\n"
      "        context__.validate_dims(\"parameter initialization\", "
      "\"ar_p2\", \"double\", context__.to_vec(4));\n"
      "        std::vector<double> ar_p2(4, double(0));\n"
      "        size_t ar_p2_k_0_max__ = 4;\n"
      "        for (size_t k_0__ = 0; k_0__ < ar_p2_k_0_max__; ++k_0__) {\n"
      "            ar_p2[k_0__] = vals_r__[pos__++];\n"
      "        }\n"
      "        size_t ar_p2_i_0_max__ = 4;\n"
      "        for (size_t i_0__ = 0; i_0__ < ar_p2_i_0_max__; ++i_0__) {\n"
      "            try {\n"
      "                writer__.scalar_lub_unconstrain(0, 1, ar_p2[i_0__]);\n"
      "            } catch (const std::exception& e) {\n"
      "                "
      "stan::lang::rethrow_located(std::runtime_error(std::string(\"Error "
      "transforming variable ar_p2: \") + e.what()), "
      "current_statement_begin__, prog_reader__());\n"
      "            }\n"
      "        }\n"
      "\n"
      "        current_statement_begin__ = 4;\n"
      "        if (!(context__.contains_r(\"ar_p3\")))\n"
      "            "
      "stan::lang::rethrow_located(std::runtime_error(std::string(\"Variable "
      "ar_p3 missing\")), current_statement_begin__, prog_reader__());\n"
      "        vals_r__ = context__.vals_r(\"ar_p3\");\n"
      "        pos__ = 0U;\n"
      "        validate_non_negative_index(\"ar_p3\", \"5\", 5);\n"
      "        context__.validate_dims(\"parameter initialization\", "
      "\"ar_p3\", \"double\", context__.to_vec(5));\n"
      "        std::vector<double> ar_p3(5, double(0));\n"
      "        size_t ar_p3_k_0_max__ = 5;\n"
      "        for (size_t k_0__ = 0; k_0__ < ar_p3_k_0_max__; ++k_0__) {\n"
      "            ar_p3[k_0__] = vals_r__[pos__++];\n"
      "        }\n"
      "        size_t ar_p3_i_0_max__ = 5;\n"
      "        for (size_t i_0__ = 0; i_0__ < ar_p3_i_0_max__; ++i_0__) {\n"
      "            try {\n"
      "                writer__.scalar_offset_multiplier_unconstrain(1, 2, "
      "ar_p3[i_0__]);\n"
      "            } catch (const std::exception& e) {\n"
      "                "
      "stan::lang::rethrow_located(std::runtime_error(std::string(\"Error "
      "transforming variable ar_p3: \") + e.what()), "
      "current_statement_begin__, prog_reader__());\n"
      "            }\n"
      "        }\n");
  EXPECT_EQ(1, count_matches(expected, hpp));
}

TEST(lang, params_block_var_hpp_log_prob) {
  // log_prob checks constraints on model param
  std::string m1("parameters {\n"
                 "  real p2;\n"
                 "  real<lower=0, upper=1> ar_p2[4];\n"
                 "  real<offset=1, multiplier=2> ar_p3[5];\n"
                 "}\n");
  std::string hpp = model_to_hpp("parameters_prim", m1);

  std::string expected(
      "            // model parameters\n"
      "            current_statement_begin__ = 2;\n"
      "            local_scalar_t__ p2;\n"
      "            (void) p2;  // dummy to suppress unused var warning\n"
      "            if (jacobian__)\n"
      "                p2 = in__.scalar_constrain(lp__);\n"
      "            else\n"
      "                p2 = in__.scalar_constrain();\n"
      "\n"
      "            current_statement_begin__ = 3;\n"
      "            std::vector<local_scalar_t__> ar_p2;\n"
      "            size_t ar_p2_d_0_max__ = 4;\n"
      "            ar_p2.reserve(ar_p2_d_0_max__);\n"
      "            for (size_t d_0__ = 0; d_0__ < ar_p2_d_0_max__; ++d_0__) {\n"
      "                if (jacobian__)\n"
      "                    ar_p2.push_back(in__.scalar_lub_constrain(0, 1, "
      "lp__));\n"
      "                else\n"
      "                    ar_p2.push_back(in__.scalar_lub_constrain(0, 1));\n"
      "            }\n"
      "\n"
      "            current_statement_begin__ = 4;\n"
      "            std::vector<local_scalar_t__> ar_p3;\n"
      "            size_t ar_p3_d_0_max__ = 5;\n"
      "            ar_p3.reserve(ar_p3_d_0_max__);\n"
      "            for (size_t d_0__ = 0; d_0__ < ar_p3_d_0_max__; ++d_0__) {\n"
      "                if (jacobian__)\n"
      "                    "
      "ar_p3.push_back(in__.scalar_offset_multiplier_constrain(1, 2, lp__));\n"
      "                else\n"
      "                    "
      "ar_p3.push_back(in__.scalar_offset_multiplier_constrain(1, 2));\n"
      "            }\n");
  EXPECT_EQ(1, count_matches(expected, hpp));
}

TEST(lang, params_block_var_hpp_get_dims) {
  // get_dims gets all dimensions
  std::string m1("parameters {\n"
                 "  real p2;\n"
                 "  real<lower=0, upper=1> ar_p2[4];\n"
                 "}\n");
  std::string hpp = model_to_hpp("parameters_prim", m1);

  std::string expected("        dimss__.resize(0);\n"
                       "        std::vector<size_t> dims__;\n"
                       "        dims__.resize(0);\n"
                       "        dimss__.push_back(dims__);\n"
                       "        dims__.resize(0);\n"
                       "        dims__.push_back(4);\n"
                       "        dimss__.push_back(dims__);\n");

  EXPECT_EQ(1, count_matches(expected, hpp));
}

TEST(lang, params_block_var_hpp_write_array) {
  std::string m1("parameters {\n"
                 "  real p2;\n"
                 "  real<lower=0, upper=1> ar_p2[4];\n"
                 "}\n");
  std::string hpp = model_to_hpp("parameters_prim", m1);
  std::string expected(
      "        // read-transform, write parameters\n"
      "        double p2 = in__.scalar_constrain();\n"
      "        vars__.push_back(p2);\n"
      "\n"
      "        std::vector<double> ar_p2;\n"
      "        size_t ar_p2_d_0_max__ = 4;\n"
      "        ar_p2.reserve(ar_p2_d_0_max__);\n"
      "        for (size_t d_0__ = 0; d_0__ < ar_p2_d_0_max__; ++d_0__) {\n"
      "            ar_p2.push_back(in__.scalar_lub_constrain(0, 1));\n"
      "        }\n"
      "        size_t ar_p2_k_0_max__ = 4;\n"
      "        for (size_t k_0__ = 0; k_0__ < ar_p2_k_0_max__; ++k_0__) {\n"
      "            vars__.push_back(ar_p2[k_0__]);\n"
      "        }\n");
  EXPECT_EQ(1, count_matches(expected, hpp));
}

TEST(lang, params_block_var_hpp_param_names) {
  std::string m1("parameters {\n"
                 "  real p2;\n"
                 "  real<lower=0, upper=1> ar_p2[4];\n"
                 "}\n");
  std::string hpp = model_to_hpp("parameters_prim", m1);

  std::string expected(
      "        std::stringstream param_name_stream__;\n"
      "        param_name_stream__.str(std::string());\n"
      "        param_name_stream__ << \"p2\";\n"
      "        param_names__.push_back(param_name_stream__.str());\n"
      "        size_t ar_p2_k_0_max__ = 4;\n"
      "        for (size_t k_0__ = 0; k_0__ < ar_p2_k_0_max__; ++k_0__) {\n"
      "            param_name_stream__.str(std::string());\n"
      "            param_name_stream__ << \"ar_p2\" << '.' << k_0__ + 1;\n"
      "            param_names__.push_back(param_name_stream__.str());\n"
      "        }\n");
  EXPECT_EQ(2, count_matches(expected, hpp));  // matches 2 methods:
                                              // constrained_param_names,
                                              // unconstrained_param_names
}

TEST(lang, transformed_params_block_var_ast) {
  std::string m1("transformed parameters {\n"
                 "  real p2;\n"
                 "  real<lower=0, upper=1> ar_p2[4];\n"
                 "}\n");
  stan::lang::program prog = model_to_ast("xformed_parameters_prim", m1);
  EXPECT_EQ(2, prog.derived_decl_.first.size());
  stan::lang::block_var_decl bvd1 = prog.derived_decl_.first[0];
  stan::lang::block_var_decl bvd2 = prog.derived_decl_.first[1];
  EXPECT_EQ("p2", bvd1.name());
  EXPECT_EQ("ar_p2", bvd2.name());
}

TEST(lang, transformed_params_block_var_hpp_log_prob) {
  std::string m1("transformed parameters {\n"
                 "  real p2;\n"
                 "  real<lower=0, upper=1> ar_p2[4];\n"
                 "}\n");
  std::string hpp = model_to_hpp("xformed_parameters_prim", m1);

  // declare
  std::string expected_1(
      "            // transformed parameters\n"
      "            current_statement_begin__ = 2;\n"
      "            local_scalar_t__ p2;\n"
      "            (void) p2;  // dummy to suppress unused var warning\n"
      "            stan::math::initialize(p2, DUMMY_VAR__);\n"
      "            stan::math::fill(p2, DUMMY_VAR__);\n"
      "\n"
      "            current_statement_begin__ = 3;\n"
      "            validate_non_negative_index(\"ar_p2\", \"4\", 4);\n"
      "            std::vector<local_scalar_t__> ar_p2(4, "
      "local_scalar_t__(0));\n"
      "            stan::math::initialize(ar_p2, DUMMY_VAR__);\n"
      "            stan::math::fill(ar_p2, DUMMY_VAR__);\n");

  EXPECT_EQ(1, count_matches(expected_1, hpp));

  // validate
  std::string expected_2(
      "            // validate transformed parameters\n"
      "            const char* function__ = \"validate transformed params\";\n"
      "            (void) function__;  // dummy to suppress unused var "
      "warning\n"
      "\n"
      "            current_statement_begin__ = 2;\n"
      "            if (stan::math::is_uninitialized(p2)) {\n"
      "                std::stringstream msg__;\n"
      "                msg__ << \"Undefined transformed parameter: p2\";\n"
      "                "
      "stan::lang::rethrow_located(std::runtime_error(std::string(\"Error "
      "initializing variable p2: \") + msg__.str()), "
      "current_statement_begin__, prog_reader__());\n"
      "            }\n"
      "            current_statement_begin__ = 3;\n"
      "            size_t ar_p2_k_0_max__ = 4;\n"
      "            for (size_t k_0__ = 0; k_0__ < ar_p2_k_0_max__; ++k_0__) {\n"
      "                if (stan::math::is_uninitialized(ar_p2[k_0__])) {\n"
      "                    std::stringstream msg__;\n"
      "                    msg__ << \"Undefined transformed parameter: ar_p2\" "
      "<< \"[\" << k_0__ << \"]\";\n"
      "                    "
      "stan::lang::rethrow_located(std::runtime_error(std::string(\"Error "
      "initializing variable ar_p2: \") + msg__.str()), "
      "current_statement_begin__, prog_reader__());\n"
      "                }\n"
      "            }\n"
      "            size_t ar_p2_i_0_max__ = 4;\n"
      "            for (size_t i_0__ = 0; i_0__ < ar_p2_i_0_max__; ++i_0__) {\n"
      "                check_greater_or_equal(function__, \"ar_p2[i_0__]\", "
      "ar_p2[i_0__], 0);\n"
      "                check_less_or_equal(function__, \"ar_p2[i_0__]\", "
      "ar_p2[i_0__], 1);\n"
      "            }\n");

  EXPECT_EQ(1, count_matches(expected_2, hpp));
}

TEST(lang, transformed_params_block_var_hpp_get_dims) {
  std::string m1("transformed parameters {\n"
                 "  real p2;\n"
                 "  real<lower=0, upper=1> ar_p2[4];\n"
                 "}\n");
  std::string hpp = model_to_hpp("xformed_parameters_prim", m1);

  std::string expected("        dimss__.resize(0);\n"
                       "        std::vector<size_t> dims__;\n"
                       "        dims__.resize(0);\n"
                       "        dimss__.push_back(dims__);\n"
                       "        dims__.resize(0);\n"
                       "        dims__.push_back(4);\n"
                       "        dimss__.push_back(dims__);\n");
  EXPECT_EQ(1, count_matches(expected, hpp));
}

TEST(lang, transformed_params_block_var_hpp_write_array) {
  std::string m1("transformed parameters {\n"
                 "  real p2;\n"
                 "  real<lower=0, upper=1> ar_p2[4];\n"
                 "}\n");
  std::string hpp = model_to_hpp("xformed_parameters_prim", m1);

  std::string expected_1(
      "            // declare and define transformed parameters\n"
      "            current_statement_begin__ = 2;\n"
      "            double p2;\n"
      "            (void) p2;  // dummy to suppress unused var warning\n"
      "            stan::math::initialize(p2, DUMMY_VAR__);\n"
      "            stan::math::fill(p2, DUMMY_VAR__);\n"
      "\n"
      "            current_statement_begin__ = 3;\n"
      "            validate_non_negative_index(\"ar_p2\", \"4\", 4);\n"
      "            std::vector<double> ar_p2(4, double(0));\n"
      "            stan::math::initialize(ar_p2, DUMMY_VAR__);\n"
      "            stan::math::fill(ar_p2, DUMMY_VAR__);\n");
  EXPECT_EQ(1, count_matches(expected_1, hpp));

  std::string expected_2(
      "            // validate transformed parameters\n"
      "            const char* function__ = \"validate transformed params\";\n"
      "            (void) function__;  // dummy to suppress unused var "
      "warning\n"
      "\n"
      "            current_statement_begin__ = 3;\n"
      "            size_t ar_p2_i_0_max__ = 4;\n"
      "            for (size_t i_0__ = 0; i_0__ < ar_p2_i_0_max__; ++i_0__) {\n"
      "                check_greater_or_equal(function__, \"ar_p2[i_0__]\", "
      "ar_p2[i_0__], 0);\n"
      "                check_less_or_equal(function__, \"ar_p2[i_0__]\", "
      "ar_p2[i_0__], 1);\n"
      "            }\n"
      "\n"
      "            // write transformed parameters\n"
      "            if (include_tparams__) {\n"
      "                vars__.push_back(p2);\n"
      "                size_t ar_p2_k_0_max__ = 4;\n"
      "                for (size_t k_0__ = 0; k_0__ < ar_p2_k_0_max__; "
      "++k_0__) {\n"
      "                    vars__.push_back(ar_p2[k_0__]);\n"
      "                }\n"
      "            }\n");
  EXPECT_EQ(1, count_matches(expected_2, hpp));
}

TEST(lang, transformed_params_block_var_hpp_param_names) {
  std::string m1("transformed parameters {\n"
                 "  real p2;\n"
                 "  real<lower=0, upper=1> ar_p2[4];\n"
                 "}\n");
  std::string hpp = model_to_hpp("xformed_parameters_prim", m1);

  std::string expected(
      "        if (include_tparams__) {\n"
      "            param_name_stream__.str(std::string());\n"
      "            param_name_stream__ << \"p2\";\n"
      "            param_names__.push_back(param_name_stream__.str());\n"
      "            size_t ar_p2_k_0_max__ = 4;\n"
      "            for (size_t k_0__ = 0; k_0__ < ar_p2_k_0_max__; ++k_0__) {\n"
      "                param_name_stream__.str(std::string());\n"
      "                param_name_stream__ << \"ar_p2\" << '.' << k_0__ + 1;\n"
      "                param_names__.push_back(param_name_stream__.str());\n"
      "            }\n"
      "        }\n");
  EXPECT_EQ(2, count_matches(expected, hpp));
}

TEST(lang, generated_quantities_block_var_ast) {
  std::string m1("generated quantities {\n"
                 "  int<lower=0, upper=1> p1;\n"
                 "  real p2;\n"
                 "  int ar_p1[3];\n"
                 "  real<lower=0, upper=1> ar_p2[4];\n"
                 "}\n");
  stan::lang::program prog = model_to_ast("gqs_prim", m1);
  EXPECT_EQ(4, prog.generated_decl_.first.size());
  stan::lang::block_var_decl bvd1 = prog.generated_decl_.first[0];
  stan::lang::block_var_decl bvd2 = prog.generated_decl_.first[1];
  stan::lang::block_var_decl bvd3 = prog.generated_decl_.first[2];
  stan::lang::block_var_decl bvd4 = prog.generated_decl_.first[3];
  EXPECT_EQ("p1", bvd1.name());
  EXPECT_EQ("p2", bvd2.name());
  EXPECT_EQ("ar_p1", bvd3.name());
  EXPECT_EQ("ar_p2", bvd4.name());
}

TEST(lang, generated_quantities_block_var_hpp_get_dims) {
  std::string m1("generated quantities {\n"
                 "  int<lower=0, upper=1> p1;\n"
                 "  real p2;\n"
                 "  int ar_p1[3];\n"
                 "  real<lower=0, upper=1> ar_p2[4];\n"
                 "}\n");
  std::string hpp = model_to_hpp("gqs_prim", m1);

  std::string expected("        dimss__.resize(0);\n"
                       "        std::vector<size_t> dims__;\n"
                       "        dims__.resize(0);\n"
                       "        dimss__.push_back(dims__);\n"
                       "        dims__.resize(0);\n"
                       "        dimss__.push_back(dims__);\n"
                       "        dims__.resize(0);\n"
                       "        dims__.push_back(3);\n"
                       "        dimss__.push_back(dims__);\n"
                       "        dims__.resize(0);\n"
                       "        dims__.push_back(4);\n"
                       "        dimss__.push_back(dims__);\n");
  EXPECT_EQ(1, count_matches(expected, hpp));
}

TEST(lang, generated_quantities_block_var_hpp_write_array) {
  std::string m1("generated quantities {\n"
                 "  int<lower=0, upper=1> p1;\n"
                 "  real p2;\n"
                 "  int ar_p1[3];\n"
                 "  real<lower=0, upper=1> ar_p2[4];\n"
                 "}\n");
  std::string hpp = model_to_hpp("gqs_prim", m1);
  // decl-def
  std::string expected_1(
      "            if (!include_gqs__) return;\n"
      "            // declare and define generated quantities\n"
      "            current_statement_begin__ = 2;\n"
      "            int p1;\n"
      "            (void) p1;  // dummy to suppress unused var warning\n"
      "            stan::math::fill(p1, std::numeric_limits<int>::min());\n"
      "\n"
      "            current_statement_begin__ = 3;\n"
      "            double p2;\n"
      "            (void) p2;  // dummy to suppress unused var warning\n"
      "            stan::math::initialize(p2, DUMMY_VAR__);\n"
      "            stan::math::fill(p2, DUMMY_VAR__);\n"
      "\n"
      "            current_statement_begin__ = 4;\n"
      "            validate_non_negative_index(\"ar_p1\", \"3\", 3);\n"
      "            std::vector<int> ar_p1(3, int(0));\n"
      "            stan::math::fill(ar_p1, std::numeric_limits<int>::min());\n"
      "\n"
      "            current_statement_begin__ = 5;\n"
      "            validate_non_negative_index(\"ar_p2\", \"4\", 4);\n"
      "            std::vector<double> ar_p2(4, double(0));\n"
      "            stan::math::initialize(ar_p2, DUMMY_VAR__);\n"
      "            stan::math::fill(ar_p2, DUMMY_VAR__);\n");
  EXPECT_EQ(1, count_matches(expected_1, hpp));

  // validate
  std::string expected_2(
      "            current_statement_begin__ = 2;\n"
      "            check_greater_or_equal(function__, \"p1\", p1, 0);\n"
      "            check_less_or_equal(function__, \"p1\", p1, 1);\n");
  EXPECT_EQ(1, count_matches(expected_2, hpp));

  // push_back
  std::string expected_3(
      "            vars__.push_back(p1);\n"
      "\n"
      "            current_statement_begin__ = 3;\n"
      "            vars__.push_back(p2);\n"
      "\n"
      "            current_statement_begin__ = 4;\n"
      "            size_t ar_p1_k_0_max__ = 3;\n"
      "            for (size_t k_0__ = 0; k_0__ < ar_p1_k_0_max__; ++k_0__) {\n"
      "                vars__.push_back(ar_p1[k_0__]);\n"
      "            }\n"
      "\n"
      "            current_statement_begin__ = 5;\n"
      "            size_t ar_p2_i_0_max__ = 4;\n"
      "            for (size_t i_0__ = 0; i_0__ < ar_p2_i_0_max__; ++i_0__) {\n"
      "                check_greater_or_equal(function__, \"ar_p2[i_0__]\", "
      "ar_p2[i_0__], 0);\n"
      "                check_less_or_equal(function__, \"ar_p2[i_0__]\", "
      "ar_p2[i_0__], 1);\n"
      "            }\n"
      "\n"
      "            size_t ar_p2_k_0_max__ = 4;\n"
      "            for (size_t k_0__ = 0; k_0__ < ar_p2_k_0_max__; ++k_0__) {\n"
      "                vars__.push_back(ar_p2[k_0__]);\n"
      "            }\n");
  EXPECT_EQ(1, count_matches(expected_3, hpp));
}

TEST(lang, generated_quantities_block_var_hpp_param_names) {
  std::string m1("generated quantities {\n"
                 "  int<lower=0, upper=1> p1;\n"
                 "  real p2;\n"
                 "  int ar_p1[3];\n"
                 "  real<lower=0, upper=1> ar_p2[4];\n"
                 "}\n");
  std::string hpp = model_to_hpp("gqs_prim", m1);

  std::string expected(
      "        if (!include_gqs__) return;\n"
      "        param_name_stream__.str(std::string());\n"
      "        param_name_stream__ << \"p1\";\n"
      "        param_names__.push_back(param_name_stream__.str());\n"
      "        param_name_stream__.str(std::string());\n"
      "        param_name_stream__ << \"p2\";\n"
      "        param_names__.push_back(param_name_stream__.str());\n"
      "        size_t ar_p1_k_0_max__ = 3;\n"
      "        for (size_t k_0__ = 0; k_0__ < ar_p1_k_0_max__; ++k_0__) {\n"
      "            param_name_stream__.str(std::string());\n"
      "            param_name_stream__ << \"ar_p1\" << '.' << k_0__ + 1;\n"
      "            param_names__.push_back(param_name_stream__.str());\n"
      "        }\n"
      "        size_t ar_p2_k_0_max__ = 4;\n"
      "        for (size_t k_0__ = 0; k_0__ < ar_p2_k_0_max__; ++k_0__) {\n"
      "            param_name_stream__.str(std::string());\n"
      "            param_name_stream__ << \"ar_p2\" << '.' << k_0__ + 1;\n"
      "            param_names__.push_back(param_name_stream__.str());\n"
      "        }\n");
  EXPECT_EQ(2, count_matches(expected, hpp));
}

#include <stan/lang/ast_def.cpp>

#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <test/unit/lang/utility.hpp>
#include <sstream>
#include <string>
#include <iostream>

TEST(lang, data_block_ast) {
  std::string m1("data {\n"
                 "  cholesky_factor_cov[5,4] cfcov_54;\n"
                 "  cholesky_factor_cov[3] cfcov_33;\n"
                 "}\n");
  stan::lang::program prog = model_to_ast("data_cholesky_cov_mat", m1);
  EXPECT_EQ(2,prog.data_decl_.size());
  stan::lang::block_var_decl bvd1 = prog.data_decl_[0];
  EXPECT_EQ("cfcov_54", bvd1.name());

  std::stringstream ss;
  write_bare_expr_type(ss, bvd1.bare_type());
  EXPECT_EQ("matrix", ss.str());

  stan::lang::block_var_type bvt1 = bvd1.type();
  EXPECT_FALSE(bvt1.is_array_type());
  EXPECT_TRUE(bvt1.is_constrained());
  EXPECT_EQ("5",bvt1.arg1().to_string());
  EXPECT_EQ("4",bvt1.arg2().to_string());

  stan::lang::block_var_decl bvd2 = prog.data_decl_[1];
  EXPECT_EQ("cfcov_33", bvd2.name());

  stan::lang::block_var_type bvt2 = bvd2.type();
  EXPECT_FALSE(bvt2.is_array_type());
  EXPECT_TRUE(bvt2.is_constrained());
  EXPECT_EQ("3",bvt2.arg1().to_string());
  EXPECT_EQ("3",bvt2.arg2().to_string());
}

TEST(lang, data_block_hpp_members) {
  std::string m1("data {\n"
                 "  cholesky_factor_cov[5,4] cfcov_54;\n"
                 "  cholesky_factor_cov[3] cfcov_33;\n"
                 "}\n");
  std::string hpp = model_to_hpp("data_cholesky_cov_mat", m1);

  std::string expected("private:\n"
                       "        matrix_d cfcov_54;\n"
                       "        matrix_d cfcov_33;\n");
  EXPECT_EQ(1, count_matches(expected, hpp));
}


TEST(lang, data_block_hpp_ctor) {
  std::string m1("data {\n"
                 "  cholesky_factor_cov[5,4] cfcov_54;\n"
                 "  cholesky_factor_cov[3] cfcov_33;\n"
                 "}\n");
  std::string hpp = model_to_hpp("data_cholesky_cov_mat", m1);

  std::string expected("            // initialize data block variables from context__\n"
                       "            current_statement_begin__ = 2;\n"
                       "            validate_non_negative_index(\"cfcov_54\", \"5\", 5);\n"
                       "            validate_non_negative_index(\"cfcov_54\", \"4\", 4);\n"
                       "            context__.validate_dims(\"data initialization\", \"cfcov_54\", \"matrix_d\", context__.to_vec(5,4));\n"
                       "            cfcov_54 = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(5, 4);\n"
                       "            vals_r__ = context__.vals_r(\"cfcov_54\");\n"
                       "            pos__ = 0;\n"
                       "            size_t cfcov_54_j_2_max__ = 4;\n"
                       "            size_t cfcov_54_j_1_max__ = 5;\n"
                       "            for (size_t j_2__ = 0; j_2__ < cfcov_54_j_2_max__; ++j_2__) {\n"
                       "                for (size_t j_1__ = 0; j_1__ < cfcov_54_j_1_max__; ++j_1__) {\n"
                       "                    cfcov_54(j_1__, j_2__) = vals_r__[pos__++];\n"
                       "                }\n"
                       "            }\n"
                       "            stan::math::check_cholesky_factor(function__, \"cfcov_54\", cfcov_54);\n"
                       "\n"
                       "            current_statement_begin__ = 3;\n"
                       "            validate_non_negative_index(\"cfcov_33\", \"3\", 3);\n"
                       "            validate_non_negative_index(\"cfcov_33\", \"3\", 3);\n"
                       "            context__.validate_dims(\"data initialization\", \"cfcov_33\", \"matrix_d\", context__.to_vec(3,3));\n"
                       "            cfcov_33 = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(3, 3);\n"
                       "            vals_r__ = context__.vals_r(\"cfcov_33\");\n"
                       "            pos__ = 0;\n"
                       "            size_t cfcov_33_j_2_max__ = 3;\n"
                       "            size_t cfcov_33_j_1_max__ = 3;\n"
                       "            for (size_t j_2__ = 0; j_2__ < cfcov_33_j_2_max__; ++j_2__) {\n"
                       "                for (size_t j_1__ = 0; j_1__ < cfcov_33_j_1_max__; ++j_1__) {\n"
                       "                    cfcov_33(j_1__, j_2__) = vals_r__[pos__++];\n"
                       "                }\n"
                       "            }\n"
                       "            stan::math::check_cholesky_factor(function__, \"cfcov_33\", cfcov_33);\n");

  EXPECT_EQ(1, count_matches(expected, hpp));
}

TEST(lang, transformed_data_block_ast) {
  std::string m1("transformed data {\n"
                 "  cholesky_factor_cov[5,4] cfcov_54;\n"
                 "  cholesky_factor_cov[3] cfcov_33;\n"
                 "}\n");
  stan::lang::program prog = model_to_ast("transformed_data_cholesky_cov_mat", m1);

  EXPECT_EQ(2,prog.derived_data_decl_.first.size());
  stan::lang::block_var_decl bvd1 = prog.derived_data_decl_.first[0];
  EXPECT_EQ("cfcov_54", bvd1.name());
  stan::lang::block_var_decl bvd2 = prog.derived_data_decl_.first[1];
  EXPECT_EQ("cfcov_33", bvd2.name());
}

TEST(lang, transformed_data_block_hpp_members) {
  std::string m1("transformed data {\n"
                 "  cholesky_factor_cov[5,4] cfcov_54;\n"
                 "  cholesky_factor_cov[3] cfcov_33;\n"
                 "}\n");
  std::string hpp = model_to_hpp("data_cholesky_cov_mat", m1);

  std::string expected("private:\n"
                       "        matrix_d cfcov_54;\n"
                       "        matrix_d cfcov_33;\n");
  EXPECT_EQ(1, count_matches(expected, hpp));
}


TEST(lang, transformed_data_block_hpp_ctor) {
  std::string m1("transformed data {\n"
                 "  cholesky_factor_cov[5,4] cfcov_54;\n"
                 "  cholesky_factor_cov[3] cfcov_33;\n"
                 "}\n");
  std::string hpp = model_to_hpp("transformed_data_cholesky_cov_mat", m1);

  std::string expected_1 = ("            // initialize transformed data variables\n"
                            "            current_statement_begin__ = 2;\n"
                            "            validate_non_negative_index(\"cfcov_54\", \"5\", 5);\n"
                            "            validate_non_negative_index(\"cfcov_54\", \"4\", 4);\n"
                            "            cfcov_54 = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(5, 4);\n"
                            "            stan::math::fill(cfcov_54, DUMMY_VAR__);\n"
                            "\n"
                            "            current_statement_begin__ = 3;\n"
                            "            validate_non_negative_index(\"cfcov_33\", \"3\", 3);\n"
                            "            validate_non_negative_index(\"cfcov_33\", \"3\", 3);\n"
                            "            cfcov_33 = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(3, 3);\n"
                            "            stan::math::fill(cfcov_33, DUMMY_VAR__);\n");
  EXPECT_EQ(1, count_matches(expected_1, hpp));

  std::string expected_2("            // validate transformed data\n"
                         "            current_statement_begin__ = 2;\n"
                         "            stan::math::check_cholesky_factor(function__, \"cfcov_54\", cfcov_54);\n"
                         "\n"
                         "            current_statement_begin__ = 3;\n"
                         "            stan::math::check_cholesky_factor(function__, \"cfcov_33\", cfcov_33);\n");

  EXPECT_EQ(1, count_matches(expected_2, hpp));
}

TEST(lang, params_block_ast) {
  std::string m1("parameters {\n"
                 "  cholesky_factor_cov[5,4] cfcov_54;\n"
                 "  cholesky_factor_cov[3] cfcov_33;\n"
                 "}\n");
  stan::lang::program prog = model_to_ast("parameters_cholesky_cov_mat", m1);
  EXPECT_EQ(2, prog.parameter_decl_.size());
  stan::lang::block_var_decl bvd1 = prog.parameter_decl_[0];
  EXPECT_EQ("cfcov_54", bvd1.name());
  stan::lang::block_var_decl bvd2 = prog.parameter_decl_[1];
  EXPECT_EQ("cfcov_33", bvd2.name());
}

TEST(lang, params_block_var_hpp_ctor) {
  std::string m1("parameters {\n"
                 "  cholesky_factor_cov[5,4] cfcov_54;\n"
                 "  cholesky_factor_cov[3] cfcov_33;\n"
                 "}\n");
  std::string hpp = model_to_hpp("parameters_cholesky_cov_mat", m1);
  std::string expected(
                       "            // validate, set parameter ranges\n"
                       "            num_params_r__ = 0U;\n"
                       "            param_ranges_i__.clear();\n"
                       "            current_statement_begin__ = 2;\n"
                       "            validate_non_negative_index(\"cfcov_54\", \"5\", 5);\n"
                       "            validate_non_negative_index(\"cfcov_54\", \"4\", 4);\n"
                       "            num_params_r__ += (((4 * (4 + 1)) / 2) + ((5 - 4) * 4));\n"
                       "            current_statement_begin__ = 3;\n"
                       "            validate_non_negative_index(\"cfcov_33\", \"3\", 3);\n"
                       "            validate_non_negative_index(\"cfcov_33\", \"3\", 3);\n"
                       "            num_params_r__ += (((3 * (3 + 1)) / 2) + ((3 - 3) * 3));\n");
                       
  EXPECT_EQ(1, count_matches(expected,hpp));
}

TEST(lang, params_block_var_hpp_xform_inits) {
  // transform_inits block has parameter initialization
  std::string m1("parameters {\n"
                 "  cholesky_factor_cov[5,4] cfcov_54;\n"
                 "  cholesky_factor_cov[3] cfcov_33;\n"
                 "}\n");
  std::string hpp = model_to_hpp("parameters_cholesky_cov_mat", m1);

  std::string expected_1("        current_statement_begin__ = 2;\n"
                         "        if (!(context__.contains_r(\"cfcov_54\")))\n"
                         "            stan::lang::rethrow_located(std::runtime_error(std::string(\"Variable cfcov_54 missing\")), current_statement_begin__, prog_reader__());\n"
                         "        vals_r__ = context__.vals_r(\"cfcov_54\");\n"
                         "        pos__ = 0U;\n"
                         "        validate_non_negative_index(\"cfcov_54\", \"5\", 5);\n"
                         "        validate_non_negative_index(\"cfcov_54\", \"4\", 4);\n"
                         "        context__.validate_dims(\"parameter initialization\", \"cfcov_54\", \"matrix_d\", context__.to_vec(5,4));\n"
                         "        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cfcov_54(5, 4);\n"
                         "        size_t cfcov_54_j_2_max__ = 4;\n"
                         "        size_t cfcov_54_j_1_max__ = 5;\n"
                         "        for (size_t j_2__ = 0; j_2__ < cfcov_54_j_2_max__; ++j_2__) {\n"
                         "            for (size_t j_1__ = 0; j_1__ < cfcov_54_j_1_max__; ++j_1__) {\n"
                         "                cfcov_54(j_1__, j_2__) = vals_r__[pos__++];\n"
                         "            }\n"
                         "        }\n"
                         "        try {\n"
                         "            writer__.cholesky_factor_cov_unconstrain(cfcov_54);\n"
                         "        } catch (const std::exception& e) {\n"
                         "            stan::lang::rethrow_located(std::runtime_error(std::string(\"Error transforming variable cfcov_54: \") + e.what()), current_statement_begin__, prog_reader__());\n"
                         "        }\n");
  EXPECT_EQ(1, count_matches(expected_1,hpp));

  std::string expected_2("        current_statement_begin__ = 3;\n"
                         "        if (!(context__.contains_r(\"cfcov_33\")))\n"
                         "            stan::lang::rethrow_located(std::runtime_error(std::string(\"Variable cfcov_33 missing\")), current_statement_begin__, prog_reader__());\n"
                         "        vals_r__ = context__.vals_r(\"cfcov_33\");\n"
                         "        pos__ = 0U;\n"
                         "        validate_non_negative_index(\"cfcov_33\", \"3\", 3);\n"
                         "        validate_non_negative_index(\"cfcov_33\", \"3\", 3);\n"
                         "        context__.validate_dims(\"parameter initialization\", \"cfcov_33\", \"matrix_d\", context__.to_vec(3,3));\n"
                         "        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cfcov_33(3, 3);\n"
                         "        size_t cfcov_33_j_2_max__ = 3;\n"
                         "        size_t cfcov_33_j_1_max__ = 3;\n"
                         "        for (size_t j_2__ = 0; j_2__ < cfcov_33_j_2_max__; ++j_2__) {\n"
                         "            for (size_t j_1__ = 0; j_1__ < cfcov_33_j_1_max__; ++j_1__) {\n"
                         "                cfcov_33(j_1__, j_2__) = vals_r__[pos__++];\n"
                         "            }\n"
                         "        }\n"
                         "        try {\n"
                         "            writer__.cholesky_factor_cov_unconstrain(cfcov_33);\n"
                         "        } catch (const std::exception& e) {\n"
                         "            stan::lang::rethrow_located(std::runtime_error(std::string(\"Error transforming variable cfcov_33: \") + e.what()), current_statement_begin__, prog_reader__());\n"
                         "        }\n");
  EXPECT_EQ(1, count_matches(expected_2,hpp));
}

TEST(lang, params_block_var_hpp_xform_log_prob) {
  // log_prob checks constraints on model param
  std::string m1("parameters {\n"
                 "  cholesky_factor_cov[5,4] cfcov_54;\n"
                 "  cholesky_factor_cov[3] cfcov_33;\n"
                 "}\n");
  std::string hpp = model_to_hpp("parameters_cholesky_cov_mat", m1);
  std::string expected("            // model parameters\n"
                       "            current_statement_begin__ = 2;\n"
                       "            Eigen::Matrix<local_scalar_t__, Eigen::Dynamic, Eigen::Dynamic> cfcov_54;\n"
                       "            (void) cfcov_54;  // dummy to suppress unused var warning\n"
                       "            if (jacobian__)\n"
                       "                cfcov_54 = in__.cholesky_factor_cov_constrain(5, 4, lp__);\n"
                       "            else\n"
                       "                cfcov_54 = in__.cholesky_factor_cov_constrain(5, 4);\n"
                       "\n"
                       "            current_statement_begin__ = 3;\n"
                       "            Eigen::Matrix<local_scalar_t__, Eigen::Dynamic, Eigen::Dynamic> cfcov_33;\n"
                       "            (void) cfcov_33;  // dummy to suppress unused var warning\n"
                       "            if (jacobian__)\n"
                       "                cfcov_33 = in__.cholesky_factor_cov_constrain(3, 3, lp__);\n"
                       "            else\n"
                       "                cfcov_33 = in__.cholesky_factor_cov_constrain(3, 3);\n");
  EXPECT_EQ(1, count_matches(expected,hpp));
}  

TEST(lang, params_block_var_hpp_get_dims) {
  // get_dims gets all dimensions
  std::string m1("parameters {\n"
                 "  cholesky_factor_cov[5,4] cfcov_54;\n"
                 "  cholesky_factor_cov[3] cfcov_33;\n"
                 "}\n");
  std::string hpp = model_to_hpp("parameters_cholesky_cov_mat", m1);

  std::string expected("    void get_dims(std::vector<std::vector<size_t> >& dimss__) const {\n"
                       "        dimss__.resize(0);\n"
                       "        std::vector<size_t> dims__;\n"
                       "        dims__.resize(0);\n"
                       "        dims__.push_back(5);\n"
                       "        dims__.push_back(4);\n"
                       "        dimss__.push_back(dims__);\n"
                       "        dims__.resize(0);\n"
                       "        dims__.push_back(3);\n"
                       "        dims__.push_back(3);\n"
                       "        dimss__.push_back(dims__);\n"
                       "    }\n");

  EXPECT_EQ(1, count_matches(expected,hpp));
}

TEST(lang, params_block_var_hpp_write_array) {
  // write_array writes param
  std::string m1("parameters {\n"
                 "  cholesky_factor_cov[5,4] cfcov_54;\n"
                 "  cholesky_factor_cov[3] cfcov_33;\n"
                 "}\n");
  std::string hpp = model_to_hpp("parameters_cholesky_cov_mat", m1);

  std::string expected("        // read-transform, write parameters\n"
                       "        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cfcov_54 = in__.cholesky_factor_cov_constrain(5, 4);\n"
                       "        size_t cfcov_54_j_2_max__ = 4;\n"
                       "        size_t cfcov_54_j_1_max__ = 5;\n"
                       "        for (size_t j_2__ = 0; j_2__ < cfcov_54_j_2_max__; ++j_2__) {\n"
                       "            for (size_t j_1__ = 0; j_1__ < cfcov_54_j_1_max__; ++j_1__) {\n"
                       "                vars__.push_back(cfcov_54(j_1__, j_2__));\n"
                       "            }\n"
                       "        }\n"
                       "\n"
                       "        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cfcov_33 = in__.cholesky_factor_cov_constrain(3, 3);\n"
                       "        size_t cfcov_33_j_2_max__ = 3;\n"
                       "        size_t cfcov_33_j_1_max__ = 3;\n"
                       "        for (size_t j_2__ = 0; j_2__ < cfcov_33_j_2_max__; ++j_2__) {\n"
                       "            for (size_t j_1__ = 0; j_1__ < cfcov_33_j_1_max__; ++j_1__) {\n"
                       "                vars__.push_back(cfcov_33(j_1__, j_2__));\n"
                       "            }\n"
                       "        }\n");

  EXPECT_EQ(1, count_matches(expected,hpp));
}

TEST(lang, params_block_var_hpp_constrained_param_names) {
  std::string m1("parameters {\n"
                 "  cholesky_factor_cov[5,4] cfcov_54;\n"
                 "  cholesky_factor_cov[3] cfcov_33;\n"
                 "}\n");
  std::string hpp = model_to_hpp("parameters_cholesky_cov_mat", m1);

  std::string expected("        size_t cfcov_54_j_2_max__ = 4;\n"
                       "        size_t cfcov_54_j_1_max__ = 5;\n"
                       "        for (size_t j_2__ = 0; j_2__ < cfcov_54_j_2_max__; ++j_2__) {\n"
                       "            for (size_t j_1__ = 0; j_1__ < cfcov_54_j_1_max__; ++j_1__) {\n"
                       "                param_name_stream__.str(std::string());\n"
                       "                param_name_stream__ << \"cfcov_54\" << '.' << j_1__ + 1 << '.' << j_2__ + 1;\n"
                       "                param_names__.push_back(param_name_stream__.str());\n"
                       "            }\n"
                       "        }\n"
                       "        size_t cfcov_33_j_2_max__ = 3;\n"
                       "        size_t cfcov_33_j_1_max__ = 3;\n"
                       "        for (size_t j_2__ = 0; j_2__ < cfcov_33_j_2_max__; ++j_2__) {\n"
                       "            for (size_t j_1__ = 0; j_1__ < cfcov_33_j_1_max__; ++j_1__) {\n"
                       "                param_name_stream__.str(std::string());\n"
                       "                param_name_stream__ << \"cfcov_33\" << '.' << j_1__ + 1 << '.' << j_2__ + 1;\n"
                       "                param_names__.push_back(param_name_stream__.str());\n"
                       "            }\n"
                       "        }\n");

  EXPECT_EQ(1, count_matches(expected,hpp));
}

TEST(lang, params_block_var_hpp_unconstrained_param_names) {
  std::string m1("parameters {\n"
                 "  cholesky_factor_cov[5,4] cfcov_54;\n"
                 "  cholesky_factor_cov[3] cfcov_33;\n"
                 "}\n");
  std::string hpp = model_to_hpp("parameters_cholesky_cov_mat", m1);

  std::string expected("        size_t cfcov_54_j_1_max__ = (((4 * (4 + 1)) / 2) + ((5 - 4) * 4));\n"
                       "        for (size_t j_1__ = 0; j_1__ < cfcov_54_j_1_max__; ++j_1__) {\n"
                       "            param_name_stream__.str(std::string());\n"
                       "            param_name_stream__ << \"cfcov_54\" << '.' << j_1__ + 1;\n"
                       "            param_names__.push_back(param_name_stream__.str());\n"
                       "        }\n"
                       "        size_t cfcov_33_j_1_max__ = (((3 * (3 + 1)) / 2) + ((3 - 3) * 3));\n"
                       "        for (size_t j_1__ = 0; j_1__ < cfcov_33_j_1_max__; ++j_1__) {\n"
                       "            param_name_stream__.str(std::string());\n"
                       "            param_name_stream__ << \"cfcov_33\" << '.' << j_1__ + 1;\n"
                       "            param_names__.push_back(param_name_stream__.str());\n"
                       "        }\n");
  EXPECT_EQ(1, count_matches(expected,hpp));
}

TEST(lang, transformed_parameters_block_ast) {
  std::string m1("transformed parameters {\n"
                 "  cholesky_factor_cov[5,4] cfcov_54;\n"
                 "  cholesky_factor_cov[3] cfcov_33;\n"
                 "}\n");
  stan::lang::program prog = model_to_ast("data_cholesky_cov_mat", m1);
  EXPECT_EQ(2,prog.derived_decl_.first.size());
  stan::lang::block_var_decl bvd1 = prog.derived_decl_.first[0];
  EXPECT_EQ("cfcov_54", bvd1.name());
}

TEST(lang, transformed_params_block_var_hpp_log_prob) {
  std::string m1("transformed parameters {\n"
                 "  cholesky_factor_cov[5,4] cfcov_54;\n"
                 "  cholesky_factor_cov[3] cfcov_33;\n"
                 "}\n");
  std::string hpp = model_to_hpp("xformed_parameters_cholesky_cov_mat", m1);

  // declare
  std::string expected_1("            // transformed parameters\n"
                         "            current_statement_begin__ = 2;\n"
                         "            validate_non_negative_index(\"cfcov_54\", \"5\", 5);\n"
                         "            validate_non_negative_index(\"cfcov_54\", \"4\", 4);\n"
                         "            Eigen::Matrix<local_scalar_t__, Eigen::Dynamic, Eigen::Dynamic> cfcov_54(5, 4);\n"
                         "            stan::math::initialize(cfcov_54, DUMMY_VAR__);\n"
                         "            stan::math::fill(cfcov_54, DUMMY_VAR__);\n"
                         "\n"
                         "            current_statement_begin__ = 3;\n"
                         "            validate_non_negative_index(\"cfcov_33\", \"3\", 3);\n"
                         "            validate_non_negative_index(\"cfcov_33\", \"3\", 3);\n"
                         "            Eigen::Matrix<local_scalar_t__, Eigen::Dynamic, Eigen::Dynamic> cfcov_33(3, 3);\n"
                         "            stan::math::initialize(cfcov_33, DUMMY_VAR__);\n"
                         "            stan::math::fill(cfcov_33, DUMMY_VAR__);\n");
  EXPECT_EQ(1, count_matches(expected_1,hpp));

  // validate
  std::string expected_2("            // validate transformed parameters\n"
                         "            const char* function__ = \"validate transformed params\";\n"
                         "            (void) function__;  // dummy to suppress unused var warning\n"
                         "\n"
                         "            current_statement_begin__ = 2;\n"
                         "            size_t cfcov_54_j_1_max__ = 5;\n"
                         "            size_t cfcov_54_j_2_max__ = 4;\n"
                         "            for (size_t j_1__ = 0; j_1__ < cfcov_54_j_1_max__; ++j_1__) {\n"
                         "                for (size_t j_2__ = 0; j_2__ < cfcov_54_j_2_max__; ++j_2__) {\n"
                         "                    if (stan::math::is_uninitialized(cfcov_54(j_1__, j_2__))) {\n"
                         "                        std::stringstream msg__;\n"
                         "                        msg__ << \"Undefined transformed parameter: cfcov_54\" << \"(\" << j_1__ << \", \" << j_2__ << \")\";\n"
                         "                        stan::lang::rethrow_located(std::runtime_error(std::string(\"Error initializing variable cfcov_54: \") + msg__.str()), current_statement_begin__, prog_reader__());\n"
                         "                    }\n"
                         "                }\n"
                         "            }\n"
                         "            stan::math::check_cholesky_factor(function__, \"cfcov_54\", cfcov_54);\n"
                         "\n"
                         "            current_statement_begin__ = 3;\n"
                         "            size_t cfcov_33_j_1_max__ = 3;\n"
                         "            size_t cfcov_33_j_2_max__ = 3;\n"
                         "            for (size_t j_1__ = 0; j_1__ < cfcov_33_j_1_max__; ++j_1__) {\n"
                         "                for (size_t j_2__ = 0; j_2__ < cfcov_33_j_2_max__; ++j_2__) {\n"
                         "                    if (stan::math::is_uninitialized(cfcov_33(j_1__, j_2__))) {\n"
                         "                        std::stringstream msg__;\n"
                         "                        msg__ << \"Undefined transformed parameter: cfcov_33\" << \"(\" << j_1__ << \", \" << j_2__ << \")\";\n"
                         "                        stan::lang::rethrow_located(std::runtime_error(std::string(\"Error initializing variable cfcov_33: \") + msg__.str()), current_statement_begin__, prog_reader__());\n"
                         "                    }\n"
                         "                }\n"
                         "            }\n"
                         "            stan::math::check_cholesky_factor(function__, \"cfcov_33\", cfcov_33);\n");
  EXPECT_EQ(1, count_matches(expected_2,hpp));
}

TEST(lang, transformed_params_block_var_hpp_get_dims) {
  std::string m1("transformed parameters {\n"
                 "  cholesky_factor_cov[5,4] cfcov_54;\n"
                 "  cholesky_factor_cov[3] cfcov_33;\n"
                 "}\n");
  std::string hpp = model_to_hpp("xformed_parameters_cholesky_cov_mat", m1);

  std::string expected("    void get_dims(std::vector<std::vector<size_t> >& dimss__) const {\n"
                       "        dimss__.resize(0);\n"
                       "        std::vector<size_t> dims__;\n"
                       "        dims__.resize(0);\n"
                       "        dims__.push_back(5);\n"
                       "        dims__.push_back(4);\n"
                       "        dimss__.push_back(dims__);\n"
                       "        dims__.resize(0);\n"
                       "        dims__.push_back(3);\n"
                       "        dims__.push_back(3);\n"
                       "        dimss__.push_back(dims__);\n"
                       "    }\n");
  EXPECT_EQ(1, count_matches(expected,hpp));
}

TEST(lang, transformed_params_block_var_hpp_write_array) {
  std::string m1("transformed parameters {\n"
                 "  cholesky_factor_cov[5,4] cfcov_54;\n"
                 "  cholesky_factor_cov[3] cfcov_33;\n"
                 "}\n");
  std::string hpp = model_to_hpp("xformed_parameters_cholesky_cov_mat", m1);

  std::string expected_1("        // declare and define transformed parameters\n"
                         "            current_statement_begin__ = 2;\n"
                         "            validate_non_negative_index(\"cfcov_54\", \"5\", 5);\n"
                         "            validate_non_negative_index(\"cfcov_54\", \"4\", 4);\n"
                         "            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cfcov_54(5, 4);\n"
                         "            stan::math::initialize(cfcov_54, DUMMY_VAR__);\n"
                         "            stan::math::fill(cfcov_54, DUMMY_VAR__);\n"
                         "\n"
                         "            current_statement_begin__ = 3;\n"
                         "            validate_non_negative_index(\"cfcov_33\", \"3\", 3);\n"
                         "            validate_non_negative_index(\"cfcov_33\", \"3\", 3);\n"
                         "            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cfcov_33(3, 3);\n"
                         "            stan::math::initialize(cfcov_33, DUMMY_VAR__);\n"
                         "            stan::math::fill(cfcov_33, DUMMY_VAR__);\n");
  EXPECT_EQ(1, count_matches(expected_1,hpp));

  std::string expected_2(
                         "            // validate transformed parameters\n"
                         "            const char* function__ = \"validate transformed params\";\n"
                         "            (void) function__;  // dummy to suppress unused var warning\n"
                         "\n"
                         "            current_statement_begin__ = 2;\n"
                         "            stan::math::check_cholesky_factor(function__, \"cfcov_54\", cfcov_54);\n"
                         "\n"
                         "            current_statement_begin__ = 3;\n"
                         "            stan::math::check_cholesky_factor(function__, \"cfcov_33\", cfcov_33);\n");
  EXPECT_EQ(1, count_matches(expected_2,hpp));

  std::string expected_3("            // write transformed parameters\n"
                         "            if (include_tparams__) {\n"
                         "                size_t cfcov_54_j_2_max__ = 4;\n"
                         "                size_t cfcov_54_j_1_max__ = 5;\n"
                         "                for (size_t j_2__ = 0; j_2__ < cfcov_54_j_2_max__; ++j_2__) {\n"
                         "                    for (size_t j_1__ = 0; j_1__ < cfcov_54_j_1_max__; ++j_1__) {\n"
                         "                        vars__.push_back(cfcov_54(j_1__, j_2__));\n"
                         "                    }\n"
                         "                }\n"
                         "                size_t cfcov_33_j_2_max__ = 3;\n"
                         "                size_t cfcov_33_j_1_max__ = 3;\n"
                         "                for (size_t j_2__ = 0; j_2__ < cfcov_33_j_2_max__; ++j_2__) {\n"
                         "                    for (size_t j_1__ = 0; j_1__ < cfcov_33_j_1_max__; ++j_1__) {\n"
                         "                        vars__.push_back(cfcov_33(j_1__, j_2__));\n"
                         "                    }\n"
                         "                }\n");
  EXPECT_EQ(1, count_matches(expected_3,hpp));
}

TEST(lang, xform_params_block_var_hpp_constrained_param_names) {
  std::string m1("transformed parameters {\n"
                 "  cholesky_factor_cov[5,4] cfcov_54;\n"
                 "  cholesky_factor_cov[3] cfcov_33;\n"
                 "}\n");
  std::string hpp = model_to_hpp("xform_parameters_cholesky_cov_mat", m1);

  std::string expected("        if (include_tparams__) {\n"
                       "            size_t cfcov_54_j_2_max__ = 4;\n"
                       "            size_t cfcov_54_j_1_max__ = 5;\n"
                       "            for (size_t j_2__ = 0; j_2__ < cfcov_54_j_2_max__; ++j_2__) {\n"
                       "                for (size_t j_1__ = 0; j_1__ < cfcov_54_j_1_max__; ++j_1__) {\n"
                       "                    param_name_stream__.str(std::string());\n"
                       "                    param_name_stream__ << \"cfcov_54\" << '.' << j_1__ + 1 << '.' << j_2__ + 1;\n"
                       "                    param_names__.push_back(param_name_stream__.str());\n"
                       "                }\n"
                       "            }\n"
                       "            size_t cfcov_33_j_2_max__ = 3;\n"
                       "            size_t cfcov_33_j_1_max__ = 3;\n"
                       "            for (size_t j_2__ = 0; j_2__ < cfcov_33_j_2_max__; ++j_2__) {\n"
                       "                for (size_t j_1__ = 0; j_1__ < cfcov_33_j_1_max__; ++j_1__) {\n"
                       "                    param_name_stream__.str(std::string());\n"
                       "                    param_name_stream__ << \"cfcov_33\" << '.' << j_1__ + 1 << '.' << j_2__ + 1;\n"
                       "                    param_names__.push_back(param_name_stream__.str());\n"
                       "                }\n"
                       "            }\n"
                       "        }\n");
  EXPECT_EQ(1, count_matches(expected,hpp));
}

TEST(lang, xform_params_block_var_hpp_unconstrained_param_names) {
  std::string m1("transformed parameters {\n"
                 "  cholesky_factor_cov[5,4] cfcov_54;\n"
                 "  cholesky_factor_cov[3] cfcov_33;\n"
                 "}\n");
  std::string hpp = model_to_hpp("xform_parameters_cholesky_cov_mat", m1);

  std::string expected("        if (include_tparams__) {\n"
                       "            size_t cfcov_54_j_1_max__ = (((4 * (4 + 1)) / 2) + ((5 - 4) * 4));\n"
                       "            for (size_t j_1__ = 0; j_1__ < cfcov_54_j_1_max__; ++j_1__) {\n"
                       "                param_name_stream__.str(std::string());\n"
                       "                param_name_stream__ << \"cfcov_54\" << '.' << j_1__ + 1;\n"
                       "                param_names__.push_back(param_name_stream__.str());\n"
                       "            }\n"
                       "            size_t cfcov_33_j_1_max__ = (((3 * (3 + 1)) / 2) + ((3 - 3) * 3));\n"
                       "            for (size_t j_1__ = 0; j_1__ < cfcov_33_j_1_max__; ++j_1__) {\n"
                       "                param_name_stream__.str(std::string());\n"
                       "                param_name_stream__ << \"cfcov_33\" << '.' << j_1__ + 1;\n"
                       "                param_names__.push_back(param_name_stream__.str());\n"
                       "            }\n"
                       "        }\n");
  EXPECT_EQ(1, count_matches(expected,hpp));
}

TEST(lang, generated_quantities_block_var_ast) {
  std::string m1("generated quantities {\n"
                 "  cholesky_factor_cov[5,4] cfcov_54;\n"
                 "  cholesky_factor_cov[3] cfcov_33;\n"
                 "}\n");
  stan::lang::program prog = model_to_ast("gqs_cholesky_cov_mat", m1);
  EXPECT_EQ(2, prog.generated_decl_.first.size());
  stan::lang::block_var_decl bvd = prog.generated_decl_.first[0];
  EXPECT_EQ("cfcov_54", bvd.name());
}

TEST(lang, generated_quantities_block_var_hpp_write_array) {
  std::string m1("generated quantities {\n"
                 "  cholesky_factor_cov[5,4] cfcov_54;\n"
                 "  cholesky_factor_cov[3] cfcov_33;\n"
                 "}\n");
  std::string hpp = model_to_hpp("gqs_cholesky_cov_mat", m1);

  std::string expected_1("            if (!include_gqs__) return;\n"
                         "            // declare and define generated quantities\n"
                         "            current_statement_begin__ = 2;\n"
                         "            validate_non_negative_index(\"cfcov_54\", \"5\", 5);\n"
                         "            validate_non_negative_index(\"cfcov_54\", \"4\", 4);\n"
                         "            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cfcov_54(5, 4);\n"
                         "            stan::math::initialize(cfcov_54, DUMMY_VAR__);\n"
                         "            stan::math::fill(cfcov_54, DUMMY_VAR__);\n"
                         "\n"
                         "            current_statement_begin__ = 3;\n"
                         "            validate_non_negative_index(\"cfcov_33\", \"3\", 3);\n"
                         "            validate_non_negative_index(\"cfcov_33\", \"3\", 3);\n"
                         "            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cfcov_33(3, 3);\n"
                         "            stan::math::initialize(cfcov_33, DUMMY_VAR__);\n"
                         "            stan::math::fill(cfcov_33, DUMMY_VAR__);\n");
  EXPECT_EQ(1, count_matches(expected_1,hpp));

  std::string expected_2("            // validate, write generated quantities\n"
                         "            current_statement_begin__ = 2;\n"
                         "            stan::math::check_cholesky_factor(function__, \"cfcov_54\", cfcov_54);\n"
                         "\n"
                         "            size_t cfcov_54_j_2_max__ = 4;\n"
                         "            size_t cfcov_54_j_1_max__ = 5;\n"
                         "            for (size_t j_2__ = 0; j_2__ < cfcov_54_j_2_max__; ++j_2__) {\n"
                         "                for (size_t j_1__ = 0; j_1__ < cfcov_54_j_1_max__; ++j_1__) {\n"
                         "                    vars__.push_back(cfcov_54(j_1__, j_2__));\n"
                         "                }\n"
                         "            }\n"
                         "\n"
                         "            current_statement_begin__ = 3;\n"
                         "            stan::math::check_cholesky_factor(function__, \"cfcov_33\", cfcov_33);\n"
                         "\n"
                         "            size_t cfcov_33_j_2_max__ = 3;\n"
                         "            size_t cfcov_33_j_1_max__ = 3;\n"
                         "            for (size_t j_2__ = 0; j_2__ < cfcov_33_j_2_max__; ++j_2__) {\n"
                         "                for (size_t j_1__ = 0; j_1__ < cfcov_33_j_1_max__; ++j_1__) {\n"
                         "                    vars__.push_back(cfcov_33(j_1__, j_2__));\n"
                         "                }\n"
                         "            }\n");

  EXPECT_EQ(1, count_matches(expected_2,hpp));
}

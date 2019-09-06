#include <stan/lang/ast_def.cpp>

#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <test/unit/lang/utility.hpp>
#include <sstream>
#include <string>
#include <iostream>

TEST(lang, data_block_var_ast) {
  std::string m1("data {\n"
                 "  matrix<lower=0,upper=1>[2,3] ar_mat[4,5];\n"
                 "}\n");
  stan::lang::program prog = model_to_ast("data_2d_ar_mat", m1);

  EXPECT_EQ(1,prog.data_decl_.size());
  stan::lang::block_var_decl bvd = prog.data_decl_[0];
  EXPECT_EQ("ar_mat", bvd.name());

  stan::lang::block_var_type bvt = bvd.type();
  std::stringstream ss;
  write_bare_expr_type(ss, bvt.bare_type());
  EXPECT_EQ("matrix[ , ]", ss.str());


  EXPECT_TRUE(bvt.is_array_type());
  std::vector<stan::lang::expression> array_lens = bvt.array_lens();
  EXPECT_EQ("4",array_lens[0].to_string());
  EXPECT_EQ("5",array_lens[1].to_string());

  stan::lang::block_var_type mat = bvt.innermost_type();
  EXPECT_EQ("2",mat.arg1().to_string());
  EXPECT_EQ("3",mat.arg2().to_string());

  EXPECT_TRUE(bvt.innermost_type().is_constrained());
  stan::lang::range bounds = bvt.innermost_type().bounds();
  EXPECT_EQ("0",bounds.low_.to_string());
  EXPECT_EQ("1",bounds.high_.to_string());
}

TEST(lang, data_block_var_hpp) {
  std::string m1("data {\n"
                 "  matrix<lower=0,upper=1>[2,3] ar_mat[4,5];\n"
                 "}\n");
  std::string hpp = model_to_hpp("data_2d_ar_mat", m1);

  std::string expected("// initialize data block variables from context__\n"
                       "            current_statement_begin__ = 2;\n"
                       "            validate_non_negative_index(\"ar_mat\", \"2\", 2);\n"
                       "            validate_non_negative_index(\"ar_mat\", \"3\", 3);\n"
                       "            validate_non_negative_index(\"ar_mat\", \"4\", 4);\n"
                       "            validate_non_negative_index(\"ar_mat\", \"5\", 5);\n"
                       "            context__.validate_dims(\"data initialization\", \"ar_mat\", \"matrix_d\", context__.to_vec(4,5,2,3));\n"
                       "            ar_mat = std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > >(4, std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> >(5, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(2, 3)));\n"
                       "            vals_r__ = context__.vals_r(\"ar_mat\");\n"
                       "            pos__ = 0;\n"
                       "            size_t ar_mat_j_2_max__ = 3;\n"
                       "            size_t ar_mat_j_1_max__ = 2;\n"
                       "            size_t ar_mat_k_0_max__ = 4;\n"
                       "            size_t ar_mat_k_1_max__ = 5;\n"
                       "            for (size_t j_2__ = 0; j_2__ < ar_mat_j_2_max__; ++j_2__) {\n"
                       "                for (size_t j_1__ = 0; j_1__ < ar_mat_j_1_max__; ++j_1__) {\n"
                       "                    for (size_t k_1__ = 0; k_1__ < ar_mat_k_1_max__; ++k_1__) {\n"
                       "                        for (size_t k_0__ = 0; k_0__ < ar_mat_k_0_max__; ++k_0__) {\n"
                       "                            ar_mat[k_0__][k_1__](j_1__, j_2__) = vals_r__[pos__++];\n"
                       "                        }\n"
                       "                    }\n"
                       "                }\n"
                       "            }\n"
                       "            size_t ar_mat_i_0_max__ = 4;\n"
                       "            size_t ar_mat_i_1_max__ = 5;\n"
                       "            for (size_t i_0__ = 0; i_0__ < ar_mat_i_0_max__; ++i_0__) {\n"
                       "                for (size_t i_1__ = 0; i_1__ < ar_mat_i_1_max__; ++i_1__) {\n"
                       "                    check_greater_or_equal(function__, \"ar_mat[i_0__][i_1__]\", ar_mat[i_0__][i_1__], 0);\n"
                       "                    check_less_or_equal(function__, \"ar_mat[i_0__][i_1__]\", ar_mat[i_0__][i_1__], 1);\n"
                       "                }\n"
                       "            }\n");

  EXPECT_EQ(1, count_matches(expected, hpp));
}

TEST(lang, transformed_data_block_var_ast) {
  std::string m1("transformed data {\n"
                 "  matrix<lower=0,upper=1>[2,3] ar_mat[4,5];\n"
                 "}\n");
  stan::lang::program prog = model_to_ast("transformed_data_2d_ar_mat", m1);

  EXPECT_EQ(1, prog.derived_data_decl_.first.size());
  stan::lang::block_var_decl bvd = prog.derived_data_decl_.first[0];
  EXPECT_EQ("ar_mat", bvd.name());
}


TEST(lang, transformed_data_block_var_hpp) {
  std::string m1("transformed data {\n"
                 "  matrix<lower=0,upper=1>[2,3] ar_mat[4,5];\n"
                 "}\n");
  std::string hpp = model_to_hpp("transformed_data_2d_ar_mat", m1);

  std::string expected("            // initialize transformed data variables\n"
                       "            current_statement_begin__ = 2;\n"
                       "            validate_non_negative_index(\"ar_mat\", \"2\", 2);\n"
                       "            validate_non_negative_index(\"ar_mat\", \"3\", 3);\n"
                       "            validate_non_negative_index(\"ar_mat\", \"4\", 4);\n"
                       "            validate_non_negative_index(\"ar_mat\", \"5\", 5);\n"
                       "            ar_mat = std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > >(4, std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> >(5, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(2, 3)));\n"
                       "            stan::math::fill(ar_mat, DUMMY_VAR__);\n");
  EXPECT_EQ(1, count_matches(expected, hpp));
}


TEST(lang, params_block_var_ast) {
  std::string m1("parameters {\n"
                 "  matrix<lower=0,upper=1>[2,3] ar_mat[4,5];\n"
                 "}\n");
  stan::lang::program prog = model_to_ast("parameters_2s_ar_mat", m1);
  EXPECT_EQ(1, prog.parameter_decl_.size());
  stan::lang::block_var_decl bvd = prog.parameter_decl_[0];
  EXPECT_EQ("ar_mat", bvd.name());
}

TEST(lang, params_block_var_hpp_ctor) {
  std::string m1("parameters {\n"
                 "  matrix<lower=0,upper=1>[2,3] ar_mat[4,5];\n"
                 "}\n");
  std::string hpp = model_to_hpp("parameters_2d_ar_mat", m1);

  std::string expected("            current_statement_begin__ = 2;\n"
                       "            validate_non_negative_index(\"ar_mat\", \"2\", 2);\n"
                       "            validate_non_negative_index(\"ar_mat\", \"3\", 3);\n"
                       "            validate_non_negative_index(\"ar_mat\", \"4\", 4);\n"
                       "            validate_non_negative_index(\"ar_mat\", \"5\", 5);\n"
                       "            num_params_r__ += (((2 * 3) * 4) * 5);\n");
  EXPECT_EQ(1, count_matches(expected,hpp));
}

TEST(lang, params_block_var_hpp_xform_inits) {
  // transform_inits block has parameter initialization
  std::string m1("parameters {\n"
                 "  matrix<lower=0,upper=1>[2,3] ar_mat[4,5];\n"
                 "}\n");
  std::string hpp = model_to_hpp("parameters_2d_ar_mat", m1);

  std::string expected("        current_statement_begin__ = 2;\n"
                       "        if (!(context__.contains_r(\"ar_mat\")))\n"
                       "            stan::lang::rethrow_located(std::runtime_error(std::string(\"Variable ar_mat missing\")), current_statement_begin__, prog_reader__());\n"
                       "        vals_r__ = context__.vals_r(\"ar_mat\");\n"
                       "        pos__ = 0U;\n"
                       "        validate_non_negative_index(\"ar_mat\", \"2\", 2);\n"
                       "        validate_non_negative_index(\"ar_mat\", \"3\", 3);\n"
                       "        validate_non_negative_index(\"ar_mat\", \"4\", 4);\n"
                       "        validate_non_negative_index(\"ar_mat\", \"5\", 5);\n"
                       "        context__.validate_dims(\"parameter initialization\", \"ar_mat\", \"matrix_d\", context__.to_vec(4,5,2,3));\n"
                       "        std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > > ar_mat(4, std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> >(5, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(2, 3)));\n"
                       "        size_t ar_mat_j_2_max__ = 3;\n"
                       "        size_t ar_mat_j_1_max__ = 2;\n"
                       "        size_t ar_mat_k_0_max__ = 4;\n"
                       "        size_t ar_mat_k_1_max__ = 5;\n"
                       "        for (size_t j_2__ = 0; j_2__ < ar_mat_j_2_max__; ++j_2__) {\n"
                       "            for (size_t j_1__ = 0; j_1__ < ar_mat_j_1_max__; ++j_1__) {\n"
                       "                for (size_t k_1__ = 0; k_1__ < ar_mat_k_1_max__; ++k_1__) {\n"
                       "                    for (size_t k_0__ = 0; k_0__ < ar_mat_k_0_max__; ++k_0__) {\n"
                       "                        ar_mat[k_0__][k_1__](j_1__, j_2__) = vals_r__[pos__++];\n"
                       "                    }\n"
                       "                }\n"
                       "            }\n"
                       "        }\n"
                       "        size_t ar_mat_i_0_max__ = 4;\n"
                       "        size_t ar_mat_i_1_max__ = 5;\n"
                       "        for (size_t i_0__ = 0; i_0__ < ar_mat_i_0_max__; ++i_0__) {\n"
                       "            for (size_t i_1__ = 0; i_1__ < ar_mat_i_1_max__; ++i_1__) {\n"
                       "                try {\n"
                       "                    writer__.matrix_lub_unconstrain(0, 1, ar_mat[i_0__][i_1__]);\n"
                       "                } catch (const std::exception& e) {\n"
                       "                    stan::lang::rethrow_located(std::runtime_error(std::string(\"Error transforming variable ar_mat: \") + e.what()), current_statement_begin__, prog_reader__());\n"
                       "                }\n"
                       "            }\n"
                       "        }\n");

  EXPECT_EQ(1, count_matches(expected,hpp));
}

TEST(lang, params_block_var_hpp_xform_log_prob) {
  // log_prob checks constraints on model param
  std::string m1("parameters {\n"
                 "  matrix<lower=0,upper=1>[2,3] ar_mat[4,5];\n"
                 "}\n");
  std::string hpp = model_to_hpp("parameters_2d_ar_mat", m1);
  std::string expected("            // model parameters\n"
                       "            current_statement_begin__ = 2;\n"
                       "            std::vector<std::vector<Eigen::Matrix<local_scalar_t__, Eigen::Dynamic, Eigen::Dynamic> > > ar_mat;\n"
                       "            size_t ar_mat_d_0_max__ = 4;\n"
                       "            size_t ar_mat_d_1_max__ = 5;\n"
                       "            ar_mat.resize(ar_mat_d_0_max__);\n"
                       "            for (size_t d_0__ = 0; d_0__ < ar_mat_d_0_max__; ++d_0__) {\n"
                       "                ar_mat[d_0__].reserve(ar_mat_d_1_max__);\n"
                       "                for (size_t d_1__ = 0; d_1__ < ar_mat_d_1_max__; ++d_1__) {\n"
                       "                    if (jacobian__)\n"
                       "                        ar_mat[d_0__].push_back(in__.matrix_lub_constrain(0, 1, 2, 3, lp__));\n"
                       "                    else\n"
                       "                        ar_mat[d_0__].push_back(in__.matrix_lub_constrain(0, 1, 2, 3));\n"
                       "                }\n"
                       "            }\n");

  EXPECT_EQ(1, count_matches(expected,hpp));
}  

TEST(lang, params_block_var_hpp_get_dims) {
  // get_dims gets all dimensions
  std::string m1("parameters {\n"
                 "  matrix<lower=0,upper=1>[2,3] ar_mat[4,5];\n"
                 "}\n");
  std::string hpp = model_to_hpp("parameters_2d_ar_mat", m1);

  std::string expected("        std::vector<size_t> dims__;\n"
                       "        dims__.resize(0);\n"
                       "        dims__.push_back(4);\n"
                       "        dims__.push_back(5);\n"
                       "        dims__.push_back(2);\n"
                       "        dims__.push_back(3);\n");

  EXPECT_EQ(1, count_matches(expected,hpp));
}

TEST(lang, params_block_var_hpp_write_array) {
  // write_array writes param
  std::string m1("parameters {\n"
                 "  matrix<lower=0,upper=1>[2,3] ar_mat[4,5];\n"
                 "}\n");
  std::string hpp = model_to_hpp("parameters_2d_ar_mat", m1);

  std::string expected("        // read-transform, write parameters\n"
                       "        std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > > ar_mat;\n"
                       "        size_t ar_mat_d_0_max__ = 4;\n"
                       "        size_t ar_mat_d_1_max__ = 5;\n"
                       "        ar_mat.resize(ar_mat_d_0_max__);\n"
                       "        for (size_t d_0__ = 0; d_0__ < ar_mat_d_0_max__; ++d_0__) {\n"
                       "            ar_mat[d_0__].reserve(ar_mat_d_1_max__);\n"
                       "            for (size_t d_1__ = 0; d_1__ < ar_mat_d_1_max__; ++d_1__) {\n"
                       "                ar_mat[d_0__].push_back(in__.matrix_lub_constrain(0, 1, 2, 3));\n"
                       "            }\n"
                       "        }\n"
                       "        size_t ar_mat_j_2_max__ = 3;\n"
                       "        size_t ar_mat_j_1_max__ = 2;\n"
                       "        size_t ar_mat_k_0_max__ = 4;\n"
                       "        size_t ar_mat_k_1_max__ = 5;\n"
                       "        for (size_t j_2__ = 0; j_2__ < ar_mat_j_2_max__; ++j_2__) {\n"
                       "            for (size_t j_1__ = 0; j_1__ < ar_mat_j_1_max__; ++j_1__) {\n"
                       "                for (size_t k_1__ = 0; k_1__ < ar_mat_k_1_max__; ++k_1__) {\n"
                       "                    for (size_t k_0__ = 0; k_0__ < ar_mat_k_0_max__; ++k_0__) {\n"
                       "                        vars__.push_back(ar_mat[k_0__][k_1__](j_1__, j_2__));\n"
                       "                    }\n"
                       "                }\n"
                       "            }\n"
                       "        }\n");

  EXPECT_EQ(1, count_matches(expected,hpp));
}

TEST(lang, params_block_var_hpp_param_names) {
  // constrained, unconstrained param names index order correctly
  std::string m1("parameters {\n"
                 "  matrix<lower=0,upper=1>[2,3] ar_mat[4,5];\n"
                 "}\n");
  std::string hpp = model_to_hpp("parameters_2d_ar_mat", m1);

  std::string expected("        size_t ar_mat_j_2_max__ = 3;\n"
                       "        size_t ar_mat_j_1_max__ = 2;\n"
                       "        size_t ar_mat_k_0_max__ = 4;\n"
                       "        size_t ar_mat_k_1_max__ = 5;\n"
                       "        for (size_t j_2__ = 0; j_2__ < ar_mat_j_2_max__; ++j_2__) {\n"
                       "            for (size_t j_1__ = 0; j_1__ < ar_mat_j_1_max__; ++j_1__) {\n"
                       "                for (size_t k_1__ = 0; k_1__ < ar_mat_k_1_max__; ++k_1__) {\n"
                       "                    for (size_t k_0__ = 0; k_0__ < ar_mat_k_0_max__; ++k_0__) {\n"
                       "                        param_name_stream__.str(std::string());\n"
                       "                        param_name_stream__ << \"ar_mat\" << '.' << k_0__ + 1 << '.' << k_1__ + 1 << '.' << j_1__ + 1 << '.' << j_2__ + 1;\n"
                       "                        param_names__.push_back(param_name_stream__.str());\n"
                       "                    }\n"
                       "                }\n"
                       "            }\n"
                       "        }\n");
  EXPECT_EQ(2, count_matches(expected,hpp)); // matches 2 methods:  constrained_param_names, unconstrained_param_names
}

TEST(lang, transformed_params_block_var_ast) {
  std::string m1("transformed parameters {\n"
                 "  matrix<lower=0,upper=1>[2,3] ar_mat[4,5];\n"
                 "}\n");
  stan::lang::program prog = model_to_ast("xformed_parameters_2d_ar_mat", m1);
  EXPECT_EQ(1, prog.derived_decl_.first.size());
  stan::lang::block_var_decl bvd = prog.derived_decl_.first[0];
  EXPECT_EQ("ar_mat", bvd.name());
}

TEST(lang, transformed_params_block_var_hpp_log_prob) {
  std::string m1("transformed parameters {\n"
                 "  matrix<lower=0,upper=1>[2,3] ar_mat[4,5];\n"
                 "}\n");
  std::string hpp = model_to_hpp("xformed_parameters_2d_ar_mat", m1);

  // declare
  std::string expected_1("            // transformed parameters\n"
                         "            current_statement_begin__ = 2;\n"
                         "            validate_non_negative_index(\"ar_mat\", \"2\", 2);\n"
                         "            validate_non_negative_index(\"ar_mat\", \"3\", 3);\n"
                         "            validate_non_negative_index(\"ar_mat\", \"4\", 4);\n"
                         "            validate_non_negative_index(\"ar_mat\", \"5\", 5);\n"
                         "            std::vector<std::vector<Eigen::Matrix<local_scalar_t__, Eigen::Dynamic, Eigen::Dynamic> > > ar_mat(4, std::vector<Eigen::Matrix<local_scalar_t__, Eigen::Dynamic, Eigen::Dynamic> >(5, Eigen::Matrix<local_scalar_t__, Eigen::Dynamic, Eigen::Dynamic>(2, 3)));\n"
                         "            stan::math::initialize(ar_mat, DUMMY_VAR__);\n"
                         "            stan::math::fill(ar_mat, DUMMY_VAR__);\n"
                         "\n");
  
  EXPECT_EQ(1, count_matches(expected_1,hpp));

  // validate
  std::string expected_2(
                         "            // validate transformed parameters\n"
                         "            const char* function__ = \"validate transformed params\";\n"
                         "            (void) function__;  // dummy to suppress unused var warning\n"
                         "\n"
                         "            current_statement_begin__ = 2;\n"
                         "            size_t ar_mat_k_0_max__ = 4;\n"
                         "            size_t ar_mat_k_1_max__ = 5;\n"
                         "            size_t ar_mat_j_1_max__ = 2;\n"
                         "            size_t ar_mat_j_2_max__ = 3;\n"
                         "            for (size_t k_0__ = 0; k_0__ < ar_mat_k_0_max__; ++k_0__) {\n"
                         "                for (size_t k_1__ = 0; k_1__ < ar_mat_k_1_max__; ++k_1__) {\n"
                         "                    for (size_t j_1__ = 0; j_1__ < ar_mat_j_1_max__; ++j_1__) {\n"
                         "                        for (size_t j_2__ = 0; j_2__ < ar_mat_j_2_max__; ++j_2__) {\n"
                         "                            if (stan::math::is_uninitialized(ar_mat[k_0__][k_1__](j_1__, j_2__))) {\n"
                         "                                std::stringstream msg__;\n"
                         "                                msg__ << \"Undefined transformed parameter: ar_mat\" << \"[\" << k_0__ << \"]\" << \"[\" << k_1__ << \"]\" << \"(\" << j_1__ << \", \" << j_2__ << \")\";\n"
                         "                                stan::lang::rethrow_located(std::runtime_error(std::string(\"Error initializing variable ar_mat: \") + msg__.str()), current_statement_begin__, prog_reader__());\n"
                         "                            }\n"
                         "                        }\n"
                         "                    }\n"
                         "                }\n"
                         "            }\n"
                         "            size_t ar_mat_i_0_max__ = 4;\n"
                         "            size_t ar_mat_i_1_max__ = 5;\n"
                         "            for (size_t i_0__ = 0; i_0__ < ar_mat_i_0_max__; ++i_0__) {\n"
                         "                for (size_t i_1__ = 0; i_1__ < ar_mat_i_1_max__; ++i_1__) {\n"
                         "                    check_greater_or_equal(function__, \"ar_mat[i_0__][i_1__]\", ar_mat[i_0__][i_1__], 0);\n"
                         "                    check_less_or_equal(function__, \"ar_mat[i_0__][i_1__]\", ar_mat[i_0__][i_1__], 1);\n"
                         "                }\n"
                         "            }\n");
  
  EXPECT_EQ(1, count_matches(expected_2,hpp));


}

TEST(lang, transformed_params_block_var_hpp_get_dims) {
  std::string m1("transformed parameters {\n"
                 "  matrix<lower=0,upper=1>[2,3] ar_mat[4,5];\n"
                 "}\n");
  std::string hpp = model_to_hpp("xformed_parameters_2d_ar_mat", m1);

  std::string expected("        std::vector<size_t> dims__;\n"
                       "        dims__.resize(0);\n"
                       "        dims__.push_back(4);\n"
                       "        dims__.push_back(5);\n"
                       "        dims__.push_back(2);\n"
                       "        dims__.push_back(3);\n");

  EXPECT_EQ(1, count_matches(expected,hpp));
}

TEST(lang, transformed_params_block_var_hpp_write_array) {
  std::string m1("transformed parameters {\n"
                 "  matrix<lower=0,upper=1>[2,3] ar_mat[4,5];\n"
                 "}\n");
  std::string hpp = model_to_hpp("xformed_parameters_2d_ar_mat", m1);

  std::string expected_1("        // declare and define transformed parameters\n"
                         "            current_statement_begin__ = 2;\n"
                         "            validate_non_negative_index(\"ar_mat\", \"2\", 2);\n"
                         "            validate_non_negative_index(\"ar_mat\", \"3\", 3);\n"
                         "            validate_non_negative_index(\"ar_mat\", \"4\", 4);\n"
                         "            validate_non_negative_index(\"ar_mat\", \"5\", 5);\n"
                         "            std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > > ar_mat(4, std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> >(5, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(2, 3)));\n"
                         "            stan::math::initialize(ar_mat, DUMMY_VAR__);\n"
                         "            stan::math::fill(ar_mat, DUMMY_VAR__);\n");
  EXPECT_EQ(1, count_matches(expected_1,hpp));

  std::string expected_2("            // validate transformed parameters\n"
                         "            const char* function__ = \"validate transformed params\";\n"
                         "            (void) function__;  // dummy to suppress unused var warning\n"
                         "\n"
                         "            current_statement_begin__ = 2;\n"
                         "            size_t ar_mat_i_0_max__ = 4;\n"
                         "            size_t ar_mat_i_1_max__ = 5;\n"
                         "            for (size_t i_0__ = 0; i_0__ < ar_mat_i_0_max__; ++i_0__) {\n"
                         "                for (size_t i_1__ = 0; i_1__ < ar_mat_i_1_max__; ++i_1__) {\n"
                         "                    check_greater_or_equal(function__, \"ar_mat[i_0__][i_1__]\", ar_mat[i_0__][i_1__], 0);\n"
                         "                    check_less_or_equal(function__, \"ar_mat[i_0__][i_1__]\", ar_mat[i_0__][i_1__], 1);\n"
                         "                }\n"
                         "            }\n");
  EXPECT_EQ(1, count_matches(expected_2,hpp));

  std::string expected_3("            // write transformed parameters\n"
                         "            if (include_tparams__) {\n"
                         "                size_t ar_mat_j_2_max__ = 3;\n"
                         "                size_t ar_mat_j_1_max__ = 2;\n"
                         "                size_t ar_mat_k_0_max__ = 4;\n"
                         "                size_t ar_mat_k_1_max__ = 5;\n"
                         "                for (size_t j_2__ = 0; j_2__ < ar_mat_j_2_max__; ++j_2__) {\n"
                         "                    for (size_t j_1__ = 0; j_1__ < ar_mat_j_1_max__; ++j_1__) {\n"
                         "                        for (size_t k_1__ = 0; k_1__ < ar_mat_k_1_max__; ++k_1__) {\n"
                         "                            for (size_t k_0__ = 0; k_0__ < ar_mat_k_0_max__; ++k_0__) {\n"
                         "                                vars__.push_back(ar_mat[k_0__][k_1__](j_1__, j_2__));\n"
                         "                            }\n"
                         "                        }\n"
                         "                    }\n"
                         "                }\n"
                         "            }\n");

  EXPECT_EQ(1, count_matches(expected_3,hpp));
}

TEST(lang, transformed_params_block_var_hpp_param_names) {
  std::string m1("transformed parameters {\n"
                 "  matrix<lower=0,upper=1>[2,3] ar_mat[4,5];\n"
                 "}\n");
  std::string hpp = model_to_hpp("xformed_parameters_2d_ar_mat", m1);

  std::string expected("        if (include_tparams__) {\n"
                       "            size_t ar_mat_j_2_max__ = 3;\n"
                       "            size_t ar_mat_j_1_max__ = 2;\n"
                       "            size_t ar_mat_k_0_max__ = 4;\n"
                       "            size_t ar_mat_k_1_max__ = 5;\n"
                       "            for (size_t j_2__ = 0; j_2__ < ar_mat_j_2_max__; ++j_2__) {\n"
                       "                for (size_t j_1__ = 0; j_1__ < ar_mat_j_1_max__; ++j_1__) {\n"
                       "                    for (size_t k_1__ = 0; k_1__ < ar_mat_k_1_max__; ++k_1__) {\n"
                       "                        for (size_t k_0__ = 0; k_0__ < ar_mat_k_0_max__; ++k_0__) {\n"
                       "                            param_name_stream__.str(std::string());\n"
                       "                            param_name_stream__ << \"ar_mat\" << '.' << k_0__ + 1 << '.' << k_1__ + 1 << '.' << j_1__ + 1 << '.' << j_2__ + 1;\n"
                       "                            param_names__.push_back(param_name_stream__.str());\n"
                       "                        }\n"
                       "                    }\n"
                       "                }\n"
                       "            }\n");

  EXPECT_EQ(2, count_matches(expected,hpp));
}



TEST(lang, generated_quantities_block_var_ast) {
  std::string m1("generated quantities {\n"
                 "  matrix<lower=0,upper=1>[2,3] ar_mat[4,5];\n"
                 "}\n");
  stan::lang::program prog = model_to_ast("gqs_2d_ar_mat", m1);
  EXPECT_EQ(1, prog.generated_decl_.first.size());
  stan::lang::block_var_decl bvd = prog.generated_decl_.first[0];
  EXPECT_EQ("ar_mat", bvd.name());
}

TEST(lang, generated_quantities_block_var_hpp_get_dims) {
  std::string m1("generated quantities {\n"
                 "  matrix<lower=0,upper=1>[2,3] ar_mat[4,5];\n"
                 "}\n");
  std::string hpp = model_to_hpp("gqs_2d_ar_mat", m1);

  std::string expected("        std::vector<size_t> dims__;\n"
                       "        dims__.resize(0);\n"
                       "        dims__.push_back(4);\n"
                       "        dims__.push_back(5);\n"
                       "        dims__.push_back(2);\n"
                       "        dims__.push_back(3);\n");

  EXPECT_EQ(1, count_matches(expected,hpp));
}

TEST(lang, generated_quantities_block_var_hpp_write_array) {
  std::string m1("generated quantities {\n"
                 "  matrix<lower=0,upper=1>[2,3] ar_mat[4,5];\n"
                 "}\n");
  std::string hpp = model_to_hpp("gqs_2d_ar_mat", m1);

  std::string expected_1("            if (!include_gqs__) return;\n"
                         "            // declare and define generated quantities\n"
                         "            current_statement_begin__ = 2;\n"
                         "            validate_non_negative_index(\"ar_mat\", \"2\", 2);\n"
                         "            validate_non_negative_index(\"ar_mat\", \"3\", 3);\n"
                         "            validate_non_negative_index(\"ar_mat\", \"4\", 4);\n"
                         "            validate_non_negative_index(\"ar_mat\", \"5\", 5);\n"
                         "            std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > > ar_mat(4, std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> >(5, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(2, 3)));\n"
                         "            stan::math::initialize(ar_mat, DUMMY_VAR__);\n"
                         "            stan::math::fill(ar_mat, DUMMY_VAR__);\n");
  EXPECT_EQ(1, count_matches(expected_1,hpp));

  std::string expected_2("            // validate, write generated quantities\n"
                         "            current_statement_begin__ = 2;\n"
                         "            size_t ar_mat_i_0_max__ = 4;\n"
                         "            size_t ar_mat_i_1_max__ = 5;\n"
                         "            for (size_t i_0__ = 0; i_0__ < ar_mat_i_0_max__; ++i_0__) {\n"
                         "                for (size_t i_1__ = 0; i_1__ < ar_mat_i_1_max__; ++i_1__) {\n"
                         "                    check_greater_or_equal(function__, \"ar_mat[i_0__][i_1__]\", ar_mat[i_0__][i_1__], 0);\n"
                         "                    check_less_or_equal(function__, \"ar_mat[i_0__][i_1__]\", ar_mat[i_0__][i_1__], 1);\n"
                         "                }\n"
                         "            }\n");

  EXPECT_EQ(1, count_matches(expected_2,hpp));

  std::string expected_3("            size_t ar_mat_j_2_max__ = 3;\n"
                         "            size_t ar_mat_j_1_max__ = 2;\n"
                         "            size_t ar_mat_k_0_max__ = 4;\n"
                         "            size_t ar_mat_k_1_max__ = 5;\n"
                         "            for (size_t j_2__ = 0; j_2__ < ar_mat_j_2_max__; ++j_2__) {\n"
                         "                for (size_t j_1__ = 0; j_1__ < ar_mat_j_1_max__; ++j_1__) {\n"
                         "                    for (size_t k_1__ = 0; k_1__ < ar_mat_k_1_max__; ++k_1__) {\n"
                         "                        for (size_t k_0__ = 0; k_0__ < ar_mat_k_0_max__; ++k_0__) {\n"
                         "                            vars__.push_back(ar_mat[k_0__][k_1__](j_1__, j_2__));\n"
                         "                        }\n"
                         "                    }\n"
                         "                }\n"
                         "            }\n");


  EXPECT_EQ(1, count_matches(expected_3,hpp));
}

TEST(lang, generated_quantities_block_var_hpp_param_names) {
  std::string m1("generated quantities {\n"
                 "  matrix<lower=0,upper=1>[2,3] ar_mat[4,5];\n"
                 "}\n");
  std::string hpp = model_to_hpp("gqs_2d_ar_mat", m1);

  std::string expected("        if (!include_gqs__) return;\n"
                       "        size_t ar_mat_j_2_max__ = 3;\n"
                       "        size_t ar_mat_j_1_max__ = 2;\n"
                       "        size_t ar_mat_k_0_max__ = 4;\n"
                       "        size_t ar_mat_k_1_max__ = 5;\n"
                       "        for (size_t j_2__ = 0; j_2__ < ar_mat_j_2_max__; ++j_2__) {\n"
                       "            for (size_t j_1__ = 0; j_1__ < ar_mat_j_1_max__; ++j_1__) {\n"
                       "                for (size_t k_1__ = 0; k_1__ < ar_mat_k_1_max__; ++k_1__) {\n"
                       "                    for (size_t k_0__ = 0; k_0__ < ar_mat_k_0_max__; ++k_0__) {\n"
                       "                        param_name_stream__.str(std::string());\n"
                       "                        param_name_stream__ << \"ar_mat\" << '.' << k_0__ + 1 << '.' << k_1__ + 1 << '.' << j_1__ + 1 << '.' << j_2__ + 1;\n"
                       "                        param_names__.push_back(param_name_stream__.str());\n"
                       "                    }\n"
                       "                }\n"
                       "            }\n"
                       "        }\n");
  EXPECT_EQ(2, count_matches(expected,hpp));
}

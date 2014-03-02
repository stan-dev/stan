#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <istream>
#include <sstream>
#include <exception>
#include <stdexcept>

#include <stan/gm/ast.hpp>
#include <stan/gm/parser.hpp>
#include <stan/gm/generator.hpp>
#include <stan/gm/grammars/program_grammar.hpp>
#include <stan/gm/grammars/whitespace_grammar.hpp>
#include <stan/gm/grammars/expression_grammar.hpp>
#include <stan/gm/grammars/statement_grammar.hpp>
#include <stan/gm/grammars/var_decls_grammar.hpp>


std::string file_name_to_model_name(const std::string& name) {
  std::string name_copy = name;
  size_t last_bk = name_copy.find_last_of('\\');
  if (last_bk != std::string::npos)
    name_copy.erase(0,last_bk + 1);
  size_t last_fwd = name_copy.find_last_of('/');
  if (last_fwd != std::string::npos)
    name_copy.erase(0,last_fwd + 1);
    
  size_t last_dot = name_copy.find_last_of('.');
  if (last_dot != std::string::npos)
    name_copy.erase(last_dot,name_copy.size());

  name_copy += "_model";
  return name_copy;
}

bool is_parsable(const std::string& file_name,
                 std::ostream* msgs = 0) {
  stan::gm::program prog;
  std::ifstream fs(file_name.c_str());
  std::string model_name = file_name_to_model_name(file_name);
  bool parsable = stan::gm::parse(msgs, fs, file_name, model_name, prog);
  return parsable;
}

TEST(gm_parser,eight_schools) {
  EXPECT_TRUE(is_parsable("src/models/misc/eight_schools/eight_schools.stan"));
}

TEST(gm_parser,bugs_1_kidney) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol1/kidney/kidney.stan"));
}
/*TEST(gm_parser,bugs_1_leuk) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol1/leuk/leuk.stan"));
  }*/
/*TEST(gm_parser,bugs_1_leukfr) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol1/leukfr/leukfr.stan"));
}*/
TEST(gm_parser,bugs_1_mice) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol1/mice/mice.stan"));
}
TEST(gm_parser,bugs_1_oxford) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol1/oxford/oxford.stan"));
}
TEST(gm_parser,bugs_1_rats) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol1/rats/rats.stan"));
}
TEST(gm_parser,bugs_1_salm) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol1/salm/salm.stan"));
}
TEST(gm_parser,bugs_1_seeds) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol1/seeds/seeds.stan"));
}
TEST(gm_parser,bugs_1_surgical) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol1/surgical/surgical.stan"));
}

TEST(gm_parser,bugs_2_beetles_cloglog) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol2/beetles/beetles_cloglog.stan"));
}
TEST(gm_parser,bugs_2_beetles_logit) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol2/beetles/beetles_logit.stan"));
}
TEST(gm_parser,bugs_2_beetles_probit) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol2/beetles/beetles_probit.stan"));
}
TEST(gm_parser,bugs_2_birats) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol2/birats/birats.stan"));
}
TEST(gm_parser,bugs_2_dugongs) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol2/dugongs/dugongs.stan"));
}
TEST(gm_parser,bugs_2_eyes) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol2/eyes/eyes.stan"));
}
TEST(gm_parser,bugs_2_ice) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol2/ice/ice.stan"));
}
/*TEST(gm_parser,bugs_2_stagnant) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol2/stagnant/stagnant.stan"));
  }*/

TEST(gm_parser,good_trunc) {
  EXPECT_TRUE(is_parsable("src/test/test-models/reference/gm/good_trunc.stan"));
}

TEST(gm_parser,good_vec_constraints) {
  EXPECT_TRUE(is_parsable("src/test/test-models/reference/gm/good_trunc.stan"));
}

TEST(gm_parser,good_const) {
  EXPECT_TRUE(is_parsable("src/test/test-models/reference/gm/good_const.stan"));
}

TEST(gm_parser,good_matrix_ops) {
  EXPECT_TRUE(is_parsable("src/test/test-models/reference/gm/good_matrix_ops.stan"));
}

TEST(gm_parser,good_funs) {
  EXPECT_TRUE(is_parsable("src/test/test-models/reference/gm/good_funs.stan"));
}

TEST(gm_parser,triangle_lp) {
  EXPECT_TRUE(is_parsable("src/models/basic_distributions/triangle.stan"));
}

TEST(gm_parser,good_vars) {
  EXPECT_TRUE(is_parsable("src/test/test-models/reference/gm/good_vars.stan"));
}

TEST(gm_parser,good_intercept_var) {
  EXPECT_TRUE(is_parsable("src/test/test-models/reference/gm/good_intercept_var.stan"));
}


TEST(gm_parser,good_cov) {
  EXPECT_TRUE(is_parsable("src/test/test-models/reference/gm/good_cov.stan"));
}

TEST(gm_parser,good_local_var_array_size) {
  EXPECT_TRUE(is_parsable("src/test/test-models/reference/gm/good_local_var_array_size.stan"));
}

TEST(gm_parser,parsable_test_bad1) {
  EXPECT_THROW(is_parsable("src/test/test-models/reference/gm/bad1.stan"),
               std::invalid_argument);
}
TEST(gm_parser,parsable_test_bad2) {
  EXPECT_THROW(is_parsable("src/test/test-models/reference/gm/bad2.stan"),
               std::invalid_argument);
}

TEST(gm_parser,parsable_test_bad3) {
  EXPECT_THROW(is_parsable("src/test/test-models/reference/gm/bad3.stan"),
               std::invalid_argument);
}

TEST(gm_parser,parsable_test_bad4) {
  EXPECT_THROW(is_parsable("src/test/test-models/reference/gm/bad4.stan"),
               std::invalid_argument);
}
// TEST(gm_parser,parsable_test_bad5) {
//    EXPECT_THROW(is_parsable("src/test/test-models/reference/gm/bad5.stan"),
//                 std::invalid_argument);
// }

TEST(gm_parser,parsable_test_bad6) {
  EXPECT_THROW(is_parsable("src/test/test-models/reference/gm/bad6.stan"),
               std::invalid_argument);
}

TEST(gm_parser,parsable_test_bad7) {
  EXPECT_THROW(is_parsable("src/test/test-models/reference/gm/bad7.stan"),
               std::invalid_argument);
}
TEST(gm_parser,parsable_test_bad8) {
  EXPECT_THROW(is_parsable("src/test/test-models/reference/gm/bad8.stan"),
               std::invalid_argument);
}
TEST(gm_parser,parsable_test_bad9) {
  EXPECT_THROW(is_parsable("src/test/test-models/reference/gm/bad9.stan"),
               std::invalid_argument);
}
TEST(gm_parser,parsable_test_bad10) {
  EXPECT_THROW(is_parsable("src/test/test-models/reference/gm/bad10.stan"),
               std::invalid_argument);
}
TEST(gm_parser,parsable_test_bad11) {
  EXPECT_THROW(is_parsable("src/test/test-models/reference/gm/bad11.stan"),
               std::invalid_argument);
}
TEST(gm_parser,parsable_test_bad_fun_name) {
  EXPECT_THROW(is_parsable("src/test/test-models/reference/gm/bad_fun_name.stan"),
               std::invalid_argument);
}
TEST(gm_parser,parsable_test_good_fun_name) {
  EXPECT_TRUE(is_parsable("src/test/test-models/reference/gm/good_fun_name.stan"));
}

TEST(gmParser,parsableBadPeriods) {
  EXPECT_THROW(is_parsable("src/test/test-models/reference/gm/bad_periods_data.stan"),
               std::invalid_argument);
  EXPECT_THROW(is_parsable("src/test/test-models/reference/gm/bad_periods_tdata.stan"),
               std::invalid_argument);
  EXPECT_THROW(is_parsable("src/test/test-models/reference/gm/bad_periods_params.stan"),
               std::invalid_argument);
  EXPECT_THROW(is_parsable("src/test/test-models/reference/gm/bad_periods_tparams.stan"),
               std::invalid_argument);
  EXPECT_THROW(is_parsable("src/test/test-models/reference/gm/bad_periods_gqs.stan"),
               std::invalid_argument);
  EXPECT_THROW(is_parsable("src/test/test-models/reference/gm/bad_periods_local.stan"),
               std::invalid_argument);
}
TEST(gmParser,declareVarWithSameNameAsModel) {
  EXPECT_THROW(is_parsable("src/test/test-models/reference/gm/bad_model_name_var.stan"),
               std::invalid_argument);
}
TEST(gm_parser,function_signatures) {
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/good_inf.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures1.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures6.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures7.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_bernoulli.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_beta.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_beta_binomial.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_binomial.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_categorical.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_cauchy.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_chi_square.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_dirichlet.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_double_exponential.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_exponential.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_gamma.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_hypergeometric.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_inv_chi_square.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_inv_gamma.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_logistic.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_lognormal.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_multinomial.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_neg_binomial.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_normal.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_ordered_logistic.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_pareto.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_poisson.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_scaled_inv_chi_square.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_student_t_0.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_student_t_1.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_student_t_2.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_student_t_3.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_uniform.stan"));
  EXPECT_TRUE(is_parsable("src/test/test-models/syntax-only/function_signatures_weibull.stan"));
}

bool is_parsable_syntax(const std::string& model_name,
                        std::ostream* msgs = 0) {
  std::string path("src/test/test-models/syntax-only/");
  path += model_name;
  path += ".stan";
  return is_parsable(path,msgs);
}

void test_parsable(const std::string& model_name) {
  EXPECT_TRUE(is_parsable_syntax(model_name));
}

void test_unparsable(const std::string& model_name) {
  EXPECT_THROW(is_parsable_syntax(model_name), std::invalid_argument);
}

void test_warning(const std::string& model_name, const std::string& warning_msg) {
  std::stringstream msgs;
  EXPECT_TRUE(is_parsable_syntax(model_name,&msgs));
  EXPECT_TRUE(msgs.str().find_first_of(warning_msg) >= 0);
}

/** test that model with specified name in syntax-only path throws
 * an exception containing the second arg as a substring
 *
 * @param model_name Name of model to parse
 * @param msg Substring of error message expected.
 */
void test_throws(const std::string& model_name, const std::string& error_msg) {
  std::stringstream msgs;
  try {
    is_parsable_syntax(model_name,&msgs);
  } catch (const std::invalid_argument& e) {
    EXPECT_TRUE(msgs.str().find_first_of(error_msg) >= 0);
    return;
  }
  EXPECT_TRUE(false);
}

TEST(gmParserStatement2Grammar, addConditionalCondition) {
  test_parsable("conditional_condition_good");
  test_throws("conditional_condition_bad_1",
              "conditions in if-else");
  test_throws("conditional_condition_bad_2",
              "conditions in if-else");
}
TEST(gmParserStatementGrammar, validateIntExpr2) {
  test_parsable("validate_int_expr2_good");
  test_throws("validate_int_expr2_bad1",
              "expression denoting integer required");
  test_throws("validate_int_expr2_bad2",
              "expression denoting integer required");
  test_throws("validate_int_expr2_bad3",
              "expression denoting integer required");
  test_throws("validate_int_expr2_bad4",
              "expression denoting integer required");
}
TEST(gmParserStatementGrammar, validateAllowSample) {
  test_throws("validate_allow_sample_bad1",
              "sampling only allowed in model");
  test_throws("validate_allow_sample_bad2",
              "sampling only allowed in model");
  test_throws("validate_allow_sample_bad3",
              "sampling only allowed in model");
}
TEST(gmParserTermGrammar, multiplicationFun) {
  test_parsable("validate_multiplication");
}
TEST(gmParserTermGrammar, divisionFun) {
  test_warning("validate_division_int_warning", 
               "integer division implicitly rounds");
  test_parsable("validate_division_good");
}
TEST(gmParserTermGrammar, leftDivisionFun) {
  test_parsable("validate_left_division_good");
}
TEST(gmParserTermGrammar, eltMultiplicationFun) {
  test_parsable("validate_elt_multiplication_good");
}
TEST(gmParserTermGrammar, negateExprFun) {
  test_parsable("validate_negate_expr_good");
}
TEST(gmParserTermGrammar, logicalNegateExprFun) {
  test_throws("validate_logical_negate_expr_bad",
              "logical negation operator ! only applies to int or real");
  test_parsable("validate_logical_negate_expr_good");
}

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <istream>
#include <sstream>
#include <exception>
#include <stdexcept>

#include <boost/lexical_cast.hpp>

#include <stan/gm/ast.hpp>
#include <stan/gm/parser.hpp>
#include <stan/gm/generator.hpp>
#include <stan/gm/grammars/program_grammar.hpp>
#include <stan/gm/grammars/whitespace_grammar.hpp>
#include <stan/gm/grammars/expression_grammar.hpp>
#include <stan/gm/grammars/statement_grammar.hpp>
#include <stan/gm/grammars/var_decls_grammar.hpp>


/** extract model name from filepath name
 * @param file_name  Name off model file
 */
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


/** test whether model with specified path name parses successfully
 *
 * @param file_name  Filepath of model file
 * @param msgs Expected error message (default: none)
 */
bool is_parsable(const std::string& file_name,
                 std::ostream* msgs = 0) {
  stan::gm::program prog;
  std::ifstream fs(file_name.c_str());
  std::string model_name = file_name_to_model_name(file_name);
  bool parsable = stan::gm::parse(msgs, fs, file_name, model_name, prog);
  return parsable;
}


/** test whether model with specified name in path syntax-only parses successfully
 *
 * @param model_name Name of model to parse
 * @param folder Path to folder under src/test/test-models (default "syntax-only")
 * @param msgs Warning message
 */
bool is_parsable_folder(const std::string& model_name,
                        const std::string folder = "syntax-only",
                        std::ostream* msgs = 0) {
  std::string path("src/test/test-models/");
  path += folder;
  path += "/";
  path += model_name;
  path += ".stan";
  return is_parsable(path,msgs);
}

/** test that model with specified name in folder "syntax-only"
 *  parses without throwing an exception
 *
 * @param model_name Name of model to parse
 */
void test_parsable(const std::string& model_name) {
  {
    SCOPED_TRACE("parsing: " + model_name);
    EXPECT_TRUE(is_parsable_folder(model_name, "syntax-only"));
  }
}


/** test that model with specified name in folder "reference" throws
 * an exception containing the second arg as a substring
 *
 * @param model_name Name of model to parse
 * @param msg Substring of error message expected.
 */
void test_throws(const std::string& model_name, const std::string& error_msg) {
  std::stringstream msgs;
  try {
    is_parsable_folder(model_name, "reference", &msgs);
  } catch (const std::invalid_argument& e) {
    if (std::string(e.what()).find(error_msg) == std::string::npos
        && msgs.str().find(error_msg) == std::string::npos) {
      FAIL() << std::endl << "*********************************" << std::endl
             << "model name=" << model_name << std::endl
             << "*** EXPECTED: error_msg=" << error_msg << std::endl
             << "*** FOUND: e.what()=" << e.what() << std::endl
             << "*** FOUND: msgs.str()=" << msgs.str() << std::endl
             << "*********************************" << std::endl
             << std::endl;
    }
    return;
  }
  
  FAIL() << "model name=" << model_name 
         << " is parsable and were exepecting msg=" << error_msg
         << std::endl;
}

/** test that model with specified name in syntax-only path parses
 * and returns a warning containing the second arg as a substring
 *
 * @param model_name Name of model to parse
 * @param msg Substring of warning message expected.
 */
void test_warning(const std::string& model_name, const std::string& warning_msg) {
  std::stringstream msgs;
  EXPECT_TRUE(is_parsable_folder(model_name, "syntax-only", &msgs));
  EXPECT_TRUE(msgs.str().find_first_of(warning_msg) != std::string::npos);
}


TEST(gm_parser,eight_schools) {
  EXPECT_TRUE(is_parsable("src/models/misc/eight_schools/eight_schools.stan"));
}

TEST(gm_parser,bugs_1_kidney) {
  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol1/kidney/kidney.stan"));
}


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

// why commented out?
//TEST(gm_parser,bugs_2_stagnant) {
//  EXPECT_TRUE(is_parsable("src/models/bugs_examples/vol2/stagnant/stagnant.stan"));
//  }

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

TEST(gm_parser,parsable_test_bad5) {
  EXPECT_THROW(is_parsable("src/test/test-models/reference/gm/bad5.stan"),
               std::invalid_argument);
}


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

TEST(gm_parser, bernoulli_distribution) {
  test_parsable("function-signatures/distributions/univariate/discrete/poisson/poisson_cdf_log");
  test_parsable("function-signatures/distributions/univariate/discrete/poisson/poisson_cdf");
  test_parsable("function-signatures/distributions/univariate/discrete/poisson/poisson_log_log");
  test_parsable("function-signatures/distributions/univariate/discrete/poisson/poisson_log");
}

TEST(gm_parser, beta_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/univariate/continuous/beta/beta_ccdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/beta/beta_cdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/beta/beta_cdf");
  test_parsable("function-signatures/distributions/univariate/continuous/beta/beta_log");
}

TEST(gm_parser, cauchy_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/univariate/continuous/cauchy/cauchy_ccdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/cauchy/cauchy_cdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/cauchy/cauchy_cdf");
  test_parsable("function-signatures/distributions/univariate/continuous/cauchy/cauchy_log");
}

TEST(gm_parser, chi_square_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/univariate/continuous/chi_square/chi_square_ccdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/chi_square/chi_square_cdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/chi_square/chi_square_cdf");
  test_parsable("function-signatures/distributions/univariate/continuous/chi_square/chi_square_log");
}

TEST(gm_parser, double_exponential_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/univariate/continuous/double_exponential/double_exponential_ccdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/double_exponential/double_exponential_cdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/double_exponential/double_exponential_cdf");
  test_parsable("function-signatures/distributions/univariate/continuous/double_exponential/double_exponential_log");
}

TEST(gm_parser, exp_mod_normal_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/univariate/continuous/exp_mod_normal/exp_mod_normal_ccdf_log_1");
  test_parsable("function-signatures/distributions/univariate/continuous/exp_mod_normal/exp_mod_normal_ccdf_log_2");
  test_parsable("function-signatures/distributions/univariate/continuous/exp_mod_normal/exp_mod_normal_ccdf_log_3");
  test_parsable("function-signatures/distributions/univariate/continuous/exp_mod_normal/exp_mod_normal_ccdf_log_4");
  test_parsable("function-signatures/distributions/univariate/continuous/exp_mod_normal/exp_mod_normal_cdf_log_1");
  test_parsable("function-signatures/distributions/univariate/continuous/exp_mod_normal/exp_mod_normal_cdf_log_2");
  test_parsable("function-signatures/distributions/univariate/continuous/exp_mod_normal/exp_mod_normal_cdf_log_3");
  test_parsable("function-signatures/distributions/univariate/continuous/exp_mod_normal/exp_mod_normal_cdf_log_4");
  test_parsable("function-signatures/distributions/univariate/continuous/exp_mod_normal/exp_mod_normal_cdf_1");
  test_parsable("function-signatures/distributions/univariate/continuous/exp_mod_normal/exp_mod_normal_cdf_2");
  test_parsable("function-signatures/distributions/univariate/continuous/exp_mod_normal/exp_mod_normal_cdf_3");
  test_parsable("function-signatures/distributions/univariate/continuous/exp_mod_normal/exp_mod_normal_cdf_4");
  test_parsable("function-signatures/distributions/univariate/continuous/exp_mod_normal/exp_mod_normal_log_1");
  test_parsable("function-signatures/distributions/univariate/continuous/exp_mod_normal/exp_mod_normal_log_2");
  test_parsable("function-signatures/distributions/univariate/continuous/exp_mod_normal/exp_mod_normal_log_3");
  test_parsable("function-signatures/distributions/univariate/continuous/exp_mod_normal/exp_mod_normal_log_4");
}

TEST(gm_parser, exponential_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/univariate/continuous/exponential/exponential_ccdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/exponential/exponential_cdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/exponential/exponential_cdf");
  test_parsable("function-signatures/distributions/univariate/continuous/exponential/exponential_log");
}

TEST(gm_parser, gamma_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/univariate/continuous/gamma/gamma_ccdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/gamma/gamma_cdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/gamma/gamma_cdf");
  test_parsable("function-signatures/distributions/univariate/continuous/gamma/gamma_log");
}

TEST(gm_parser, gumbel_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/univariate/continuous/gumbel/gumbel_ccdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/gumbel/gumbel_cdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/gumbel/gumbel_cdf");
  test_parsable("function-signatures/distributions/univariate/continuous/gumbel/gumbel_log");
}

TEST(gm_parser, inv_chi_square_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/univariate/continuous/inv_chi_square/inv_chi_square_ccdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/inv_chi_square/inv_chi_square_cdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/inv_chi_square/inv_chi_square_cdf");
  test_parsable("function-signatures/distributions/univariate/continuous/inv_chi_square/inv_chi_square_log");
}

TEST(gm_parser, inv_gamma_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/univariate/continuous/inv_gamma/inv_gamma_ccdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/inv_gamma/inv_gamma_cdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/inv_gamma/inv_gamma_cdf");
  test_parsable("function-signatures/distributions/univariate/continuous/inv_gamma/inv_gamma_log");
}

TEST(gm_parser, logistic_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/univariate/continuous/logistic/logistic_ccdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/logistic/logistic_cdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/logistic/logistic_cdf");
  test_parsable("function-signatures/distributions/univariate/continuous/logistic/logistic_log");
}

TEST(gm_parser, lognormal_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/univariate/continuous/lognormal/lognormal_ccdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/lognormal/lognormal_cdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/lognormal/lognormal_cdf");
  test_parsable("function-signatures/distributions/univariate/continuous/lognormal/lognormal_log");
}

TEST(gm_parser, normal_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/univariate/continuous/normal/normal_ccdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/normal/normal_cdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/normal/normal_cdf");
  test_parsable("function-signatures/distributions/univariate/continuous/normal/normal_log");
}

TEST(gm_parser, pareto_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/univariate/continuous/pareto/pareto_ccdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/pareto/pareto_cdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/pareto/pareto_cdf");
  test_parsable("function-signatures/distributions/univariate/continuous/pareto/pareto_log");
}

TEST(gm_parser, rayleigh_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/univariate/continuous/rayleigh/rayleigh_ccdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/rayleigh/rayleigh_cdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/rayleigh/rayleigh_cdf");
  test_parsable("function-signatures/distributions/univariate/continuous/rayleigh/rayleigh_log");
}

TEST(gm_parser, scaled_inv_chi_square_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/univariate/continuous/scaled_inv_chi_square/scaled_inv_chi_square_ccdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/scaled_inv_chi_square/scaled_inv_chi_square_cdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/scaled_inv_chi_square/scaled_inv_chi_square_cdf");
  test_parsable("function-signatures/distributions/univariate/continuous/scaled_inv_chi_square/scaled_inv_chi_square_log");
}

TEST(gm_parser, skew_normal_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/univariate/continuous/skew_normal/skew_normal_ccdf_log_1");
  test_parsable("function-signatures/distributions/univariate/continuous/skew_normal/skew_normal_ccdf_log_2");
  test_parsable("function-signatures/distributions/univariate/continuous/skew_normal/skew_normal_ccdf_log_3");
  test_parsable("function-signatures/distributions/univariate/continuous/skew_normal/skew_normal_ccdf_log_4");
  test_parsable("function-signatures/distributions/univariate/continuous/skew_normal/skew_normal_cdf_log_1");
  test_parsable("function-signatures/distributions/univariate/continuous/skew_normal/skew_normal_cdf_log_2");
  test_parsable("function-signatures/distributions/univariate/continuous/skew_normal/skew_normal_cdf_log_3");
  test_parsable("function-signatures/distributions/univariate/continuous/skew_normal/skew_normal_cdf_log_4");
  test_parsable("function-signatures/distributions/univariate/continuous/skew_normal/skew_normal_cdf_1");
  test_parsable("function-signatures/distributions/univariate/continuous/skew_normal/skew_normal_cdf_2");
  test_parsable("function-signatures/distributions/univariate/continuous/skew_normal/skew_normal_cdf_3");
  test_parsable("function-signatures/distributions/univariate/continuous/skew_normal/skew_normal_cdf_4");
  test_parsable("function-signatures/distributions/univariate/continuous/skew_normal/skew_normal_log_1");
  test_parsable("function-signatures/distributions/univariate/continuous/skew_normal/skew_normal_log_2");
  test_parsable("function-signatures/distributions/univariate/continuous/skew_normal/skew_normal_log_3");
  test_parsable("function-signatures/distributions/univariate/continuous/skew_normal/skew_normal_log_4");
}

TEST(gm_parser, student_t_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/univariate/continuous/student_t/student_t_ccdf_log_1");
  test_parsable("function-signatures/distributions/univariate/continuous/student_t/student_t_ccdf_log_2");
  test_parsable("function-signatures/distributions/univariate/continuous/student_t/student_t_ccdf_log_3");
  test_parsable("function-signatures/distributions/univariate/continuous/student_t/student_t_ccdf_log_4");
  test_parsable("function-signatures/distributions/univariate/continuous/student_t/student_t_cdf_log_1");
  test_parsable("function-signatures/distributions/univariate/continuous/student_t/student_t_cdf_log_2");
  test_parsable("function-signatures/distributions/univariate/continuous/student_t/student_t_cdf_log_3");
  test_parsable("function-signatures/distributions/univariate/continuous/student_t/student_t_cdf_log_4");
  test_parsable("function-signatures/distributions/univariate/continuous/student_t/student_t_cdf_1");
  test_parsable("function-signatures/distributions/univariate/continuous/student_t/student_t_cdf_2");
  test_parsable("function-signatures/distributions/univariate/continuous/student_t/student_t_cdf_3");
  test_parsable("function-signatures/distributions/univariate/continuous/student_t/student_t_cdf_4");
  test_parsable("function-signatures/distributions/univariate/continuous/student_t/student_t_log_1");
  test_parsable("function-signatures/distributions/univariate/continuous/student_t/student_t_log_2");
  test_parsable("function-signatures/distributions/univariate/continuous/student_t/student_t_log_3");
  test_parsable("function-signatures/distributions/univariate/continuous/student_t/student_t_log_4");
}

TEST(gm_parser, uniform_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/univariate/continuous/uniform/uniform_ccdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/uniform/uniform_cdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/uniform/uniform_cdf");
  test_parsable("function-signatures/distributions/univariate/continuous/uniform/uniform_log");
}

TEST(gm_parser, von_mises_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/univariate/continuous/von_mises/von_mises_log");
}

TEST(gm_parser, weibull_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/univariate/continuous/weibull/weibull_ccdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/weibull/weibull_cdf_log");
  test_parsable("function-signatures/distributions/univariate/continuous/weibull/weibull_cdf");
  test_parsable("function-signatures/distributions/univariate/continuous/weibull/weibull_log");
}

TEST(gm_parser, categorical_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/multivariate/discrete/categorical/categorical_log");
  test_parsable("function-signatures/distributions/multivariate/discrete/categorical/categorical_logit_log");
}

TEST(gm_parser, multinomial_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/multivariate/discrete/multinomial/multinomial_log");
}

TEST(gm_parser, ordered_logistic_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/multivariate/discrete/ordered_logistic/ordered_logistic_log");
}

TEST(gm_parser, dirichlet_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/multivariate/continuous/dirichlet_log");
}

TEST(gm_parser, gaussian_dlm_obs_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/multivariate/continuous/gaussian_dlm_obs_log");
}


TEST(gm_parser, inv_wishart_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/multivariate/continuous/inv_wishart_log");
}

TEST(gm_parser, lkj_corr_cholesky_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/multivariate/continuous/lkj_corr_cholesky_log");
}

TEST(gm_parser, lkj_corr_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/multivariate/continuous/lkj_corr_log");
}

TEST(gm_parser, lkj_cov_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/multivariate/continuous/lkj_cov_log");
}

TEST(gm_parser, multi_gp_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/multivariate/continuous/multi_gp_log");
}

TEST(gm_parser, multi_normal_cholesky_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/multivariate/continuous/multi_normal_cholesky_log");
}

TEST(gm_parser, multi_normal_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/multivariate/continuous/multi_normal_log");
}

TEST(gm_parser, multi_normal_prec_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/multivariate/continuous/multi_normal_prec_log");
}

TEST(gm_parser, multi_student_t_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/multivariate/continuous/multi_student_t_log");
}

TEST(gm_parser, wishart_distribution_function_signatures) {
  test_parsable("function-signatures/distributions/multivariate/continuous/wishart_log");
}

TEST(gm_parser, abs_math_function_signatures) {
  test_parsable("function-signatures/math/functions/abs");
}

TEST(gm_parser, asin_math_function_signatures) {
  test_parsable("function-signatures/math/functions/asin");
}

TEST(gm_parser, asinh_math_function_signatures) {
  test_parsable("function-signatures/math/functions/asinh");
}

TEST(gm_parser, acos_math_function_signatures) {
  test_parsable("function-signatures/math/functions/acos");
}

TEST(gm_parser, acosh_math_function_signatures) {
  test_parsable("function-signatures/math/functions/acosh");
}
 
TEST(gm_parser, atan_math_function_signatures) {
  test_parsable("function-signatures/math/functions/atan");
}

TEST(gm_parser, atan2_math_function_signatures) {
  test_parsable("function-signatures/math/functions/atan2");
}
  
TEST(gm_parser, atanh_math_function_signatures) {
  test_parsable("function-signatures/math/functions/atanh");
}  

TEST(gm_parser, bessel_first_kind_math_function_signatures) {
  test_parsable("function-signatures/math/functions/bessel_first_kind");
}

TEST(gm_parser, bessel_second_kind_math_function_signatures) {
  test_parsable("function-signatures/math/functions/bessel_second_kind");
}

TEST(gm_parser, binary_log_loss_math_function_signatures) {
  test_parsable("function-signatures/math/functions/binary_log_loss");
}

TEST(gm_parser, binomial_coefficient_log_math_function_signatures) {
  test_parsable("function-signatures/math/functions/binomial_coefficient_log");
}

TEST(gm_parser, cbrt_math_function_signatures) {
  test_parsable("function-signatures/math/functions/cbrt");
}

TEST(gm_parser, ceil_math_function_signatures) {
  test_parsable("function-signatures/math/functions/ceil");
}

TEST(gm_parser, constants_math_function_signatures) {
  test_parsable("function-signatures/math/functions/constants");
}

TEST(gm_parser, cos_math_function_signatures) {
  test_parsable("function-signatures/math/functions/cos");
}

TEST(gm_parser, cosh_math_function_signatures) {
  test_parsable("function-signatures/math/functions/cosh");
}

TEST(gm_parser, digamma_math_function_signatures) {
  test_parsable("function-signatures/math/functions/digamma");
}

TEST(gm_parser, erf_math_function_signatures) {
  test_parsable("function-signatures/math/functions/erf"); 
}

TEST(gm_parser, erfc_math_function_signatures) {
  test_parsable("function-signatures/math/functions/erfc"); 
}

TEST(gm_parser, exp_math_function_signatures) {
  test_parsable("function-signatures/math/functions/exp"); 
}

TEST(gm_parser, exp2_math_function_signatures) {
  test_parsable("function-signatures/math/functions/exp2"); 
}

TEST(gm_parser, expm1_math_function_signatures) {
  test_parsable("function-signatures/math/functions/expm1"); 
}

TEST(gm_parser, fabs_math_function_signatures) {
  test_parsable("function-signatures/math/functions/fabs");
}

TEST(gm_parser, falling_factorial_math_function_signatures) {
  test_parsable("function-signatures/math/functions/falling_factorial");
}

TEST(gm_parser, fdim_math_function_signatures) {
  test_parsable("function-signatures/math/functions/fdim");
}
 
TEST(gm_parser, floor_math_function_signatures) {
  test_parsable("function-signatures/math/functions/floor");
}

TEST(gm_parser, fma_math_function_signatures) {
  test_parsable("function-signatures/math/functions/fma");
}

TEST(gm_parser, fmax_math_function_signatures) {
  test_parsable("function-signatures/math/functions/fmax");
}

TEST(gm_parser, fmin_math_function_signatures) {
  test_parsable("function-signatures/math/functions/fmin");
}

TEST(gm_parser, fmod_math_function_signatures) {
  test_parsable("function-signatures/math/functions/fmod");
}

TEST(gm_parser, gamma_p_math_function_signatures) {
  test_parsable("function-signatures/math/functions/gamma_p");
}

TEST(gm_parser, gamma_q_math_function_signatures) {
  test_parsable("function-signatures/math/functions/gamma_q");
}

TEST(gm_parser, hypot_math_function_signatures) {
  test_parsable("function-signatures/math/functions/hypot");
}

TEST(gm_parser, if_else_math_function_signatures) {
  test_parsable("function-signatures/math/functions/if_else");
}

TEST(gm_parser, int_step_math_function_signatures) {
  test_parsable("function-signatures/math/functions/int_step");
}

TEST(gm_parser, inv_math_function_signatures) {
  test_parsable("function-signatures/math/functions/inv");
}

TEST(gm_parser, inv_logit_math_function_signatures) {
  test_parsable("function-signatures/math/functions/inv_logit");
}

TEST(gm_parser, inv_cloglog_math_function_signatures) {
  test_parsable("function-signatures/math/functions/inv_cloglog");
}

TEST(gm_parser, inv_square_math_function_signatures) {
  test_parsable("function-signatures/math/functions/inv_square");
}

TEST(gm_parser, inv_sqrt_math_function_signatures) {
  test_parsable("function-signatures/math/functions/inv_sqrt");
}

TEST(gm_parser, lbeta_math_function_signatures) {
  test_parsable("function-signatures/math/functions/lbeta");
}

TEST(gm_parser, lgamma_math_function_signatures) {
  test_parsable("function-signatures/math/functions/lgamma");
}

TEST(gm_parser, lmgamma_math_function_signatures) {
  test_parsable("function-signatures/math/functions/lmgamma");
}

TEST(gm_parser, log1m_math_function_signatures) {
  test_parsable("function-signatures/math/functions/log1m");
}

TEST(gm_parser, log1m_exp_math_function_signatures) {
  test_parsable("function-signatures/math/functions/log1m_exp");
}

TEST(gm_parser, log1m_inv_logit_math_function_signatures) {
  test_parsable("function-signatures/math/functions/log1m_inv_logit");
}

TEST(gm_parser, log1p_math_function_signatures) {
  test_parsable("function-signatures/math/functions/log1p");
}

TEST(gm_parser, log1p_exp_math_function_signatures) {
  test_parsable("function-signatures/math/functions/log1p_exp");
}

TEST(gm_parser, log_math_function_signatures) {
  test_parsable("function-signatures/math/functions/log");
}

TEST(gm_parser, log10_math_function_signatures) {
  test_parsable("function-signatures/math/functions/log10");
}

TEST(gm_parser, log2_math_function_signatures) {
  test_parsable("function-signatures/math/functions/log2");
}

TEST(gm_parser, log_diff_exp_math_function_signatures) {
  test_parsable("function-signatures/math/functions/log_diff_exp");
}

TEST(gm_parser, log_inv_logit_math_function_signatures) {
  test_parsable("function-signatures/math/functions/log_inv_logit");
}

TEST(gm_parser, logit_math_function_signatures) {
  test_parsable("function-signatures/math/functions/logit"); 
}

TEST(gm_parser, log_falling_factorial_math_function_signatures) {
  test_parsable("function-signatures/math/functions/log_falling_factorial");
}

TEST(gm_parser, log_rising_factorial_math_function_signatures) {
  test_parsable("function-signatures/math/functions/log_rising_factorial");
}

TEST(gm_parser, log_sum_exp_math_function_signatures) {
  test_parsable("function-signatures/math/functions/log_sum_exp");
}

TEST(gm_parser, max_math_function_signatures) {
  test_parsable("function-signatures/math/functions/max");
}

TEST(gm_parser, min_math_function_signatures) {
  test_parsable("function-signatures/math/functions/min");
}

TEST(gm_parser, modified_bessel_first_kind_math_function_signatures) {
  test_parsable("function-signatures/math/functions/modified_bessel_first_kind");
}

TEST(gm_parser, modified_bessel_second_kind_math_function_signatures) {
  test_parsable("function-signatures/math/functions/modified_bessel_second_kind");
}

TEST(gm_parser, multiply_log_math_function_signatures) {
  test_parsable("function-signatures/math/functions/multiply_log");
}

TEST(gm_parser, operators_int_math_function_signatures) {
  test_parsable("function-signatures/math/functions/operators_int");
}

TEST(gm_parser, operators_real_math_function_signatures) {
  test_parsable("function-signatures/math/functions/operators_real");
}

TEST(gm_parser, owens_t_math_function_signatures) {
  test_parsable("function-signatures/math/functions/owens_t");
}

TEST(gm_parser, phi_math_function_signatures) {
  test_parsable("function-signatures/math/functions/phi");
}

TEST(gm_parser, phi_approx_math_function_signatures) {
  test_parsable("function-signatures/math/functions/phi_approx");
}

TEST(gm_parser, pow_math_function_signatures) {
  test_parsable("function-signatures/math/functions/pow");
}

TEST(gm_parser, sin_math_function_signatures) {
  test_parsable("function-signatures/math/functions/sin");
}

TEST(gm_parser, sinh_math_function_signatures) {
  test_parsable("function-signatures/math/functions/sinh");
}

TEST(gm_parser, step_math_function_signatures) {
  test_parsable("function-signatures/math/functions/step");
}

TEST(gm_parser, special_values_math_function_signatures) {
  test_parsable("function-signatures/math/functions/special_values");
}

TEST(gm_parser, sqrt_math_function_signatures) {
  test_parsable("function-signatures/math/functions/sqrt");
}

TEST(gm_parser, square_math_function_signatures) {
  test_parsable("function-signatures/math/functions/square");
}

TEST(gm_parser, rising_factorial_math_function_signatures) {
  test_parsable("function-signatures/math/functions/rising_factorial");
}

TEST(gm_parser, round_math_function_signatures) {
  test_parsable("function-signatures/math/functions/round");
}

TEST(gm_parser, tan_math_function_signatures) {
  test_parsable("function-signatures/math/functions/tan");
}

TEST(gm_parser, tanh_math_function_signatures) {
  test_parsable("function-signatures/math/functions/tanh");
}

TEST(gm_parser, tgamma_math_function_signatures) {
  test_parsable("function-signatures/math/functions/tgamma");
}

TEST(gm_parser, trigamma_math_function_signatures) {
  test_parsable("function-signatures/math/functions/trigamma");
}

TEST(gm_parser, trunc_math_function_signatures) {
  test_parsable("function-signatures/math/functions/trunc");
}

TEST(gm_parser, block_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/block");
}

TEST(gm_parser, broadcast_infix_operators_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/broadcast_infix_operators");
}

TEST(gm_parser, cholesky_decompose_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/cholesky_decompose");
}

TEST(gm_parser, col_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/col");
}

TEST(gm_parser, cols_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/cols");
}

TEST(gm_parser, columns_dot_product_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/columns_dot_product");
}

TEST(gm_parser, columns_dot_self_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/columns_dot_self");
}

TEST(gm_parser, crossprod_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/crossprod");
}

TEST(gm_parser, cumulative_sum_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/cumulative_sum");
}

TEST(gm_parser, determinant_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/determinant");
}

TEST(gm_parser, diag_matrix_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/diag_matrix");
}

TEST(gm_parser, diag_post_multiply_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/diag_post_multiply");
}

TEST(gm_parser, diag_pre_multiply_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/diag_pre_multiply");
}

TEST(gm_parser, diagonal_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/diagonal");
}

TEST(gm_parser, dims_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/dims");
}

TEST(gm_parser, distance_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/distance");
}

TEST(gm_parser, dot_product_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/dot_product");
}

TEST(gm_parser, dot_self_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/dot_self");
}

TEST(gm_parser, eigenvalues_sym_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/eigenvalues_sym");
}

TEST(gm_parser, eigenvectors_sym_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/eigenvectors_sym");
}

TEST(gm_parser, elementwise_products_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/elementwise_products");
}

TEST(gm_parser, exp_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/exp");
}

TEST(gm_parser, head_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/head");
}

TEST(gm_parser, infix_matrix_operators_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/infix_matrix_operators");
}

TEST(gm_parser, inverse_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/inverse");
}

TEST(gm_parser, inverse_spd_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/inverse_spd");
}

TEST(gm_parser, log_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/log");
}

TEST(gm_parser, log_determinant_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/log_determinant");
}

TEST(gm_parser, log_softmax_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/log_softmax");
}

TEST(gm_parser, log_sum_exp_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/log_sum_exp");
}

TEST(gm_parser, division_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/matrix_division");
}

TEST(gm_parser, max_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/max");
}

TEST(gm_parser, mdivide_left_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/mdivide_left");
}

TEST(gm_parser, mdivide_left_tri_low_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/mdivide_left_tri_low");
}

TEST(gm_parser, mdivide_right_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/mdivide_right");
}

TEST(gm_parser, mdivide_right_tri_low_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/mdivide_right_tri_low");
}

TEST(gm_parser, mean_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/mean");
}

TEST(gm_parser, min_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/min");
}

TEST(gm_parser, multiply_lower_tri_self_transpose_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/multiply_lower_tri_self_transpose");
}

TEST(gm_parser, negation_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/negation");
}

TEST(gm_parser, prod_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/prod");
}

TEST(gm_parser, qr_Q_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/qr_Q");
}

TEST(gm_parser, qr_R_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/qr_R");
}

TEST(gm_parser, quad_form_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/quad_form");
}

TEST(gm_parser, quad_form_diag_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/quad_form_diag");
}

TEST(gm_parser, quad_form_sym_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/quad_form_sym");
}

TEST(gm_parser, rank_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/rep_matrix");
}

TEST(gm_parser, rep_matrix_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/rank");
}

TEST(gm_parser, rep_param_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/rep_param"); //mostly rep_array with some other rep_ tests
}

TEST(gm_parser, rep_row_vector_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/rep_row_vector");
}

TEST(gm_parser, rep_vector_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/rep_vector");
}

TEST(gm_parser, row_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/row");
}

TEST(gm_parser, rows_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/rows");
}

TEST(gm_parser, rows_dot_product_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/rows_dot_product");
}

TEST(gm_parser, rows_dot_self_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/rows_dot_self");
}

TEST(gm_parser, singular_values_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/singular_values");
}

TEST(gm_parser, segment_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/segment");
}

TEST(gm_parser, size_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/size");
}

TEST(gm_parser, sd_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/sd");
}

TEST(gm_parser, softmax_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/softmax");
}

TEST(gm_parser, sort_asc_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/sort_asc");
}

TEST(gm_parser, sort_desc_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/sort_desc");
}

TEST(gm_parser, sort_indices_asc_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/sort_indices_asc");
}

TEST(gm_parser, sort_indices_desc_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/sort_indices_desc");
}

TEST(gm_parser, squared_distance_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/squared_distance");
}

TEST(gm_parser, sub_col_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/sub_col");
}

TEST(gm_parser, sub_row_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/sub_row");
}

TEST(gm_parser, sum_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/sum");
}

TEST(gm_parser, tail_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/tail");
}

TEST(gm_parser, tcrossprod_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/tcrossprod");
}

TEST(gm_parser, to_array_1d_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/to_array_1d");
}

TEST(gm_parser, to_array_2d_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/to_array_2d");
}

TEST(gm_parser, to_matrix_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/to_matrix");
}

TEST(gm_parser, to_row_vector_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/to_row_vector");
}

TEST(gm_parser, to_vector_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/to_vector");
}

TEST(gm_parser, trace_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/trace");
}

TEST(gm_parser, trace_gen_quad_form_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/trace_gen_quad_form");
}

TEST(gm_parser, trace_quad_form_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/trace_quad_form");
}

TEST(gm_parser, transpose_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/transpose");
}

TEST(gm_parser, variance_matrix_function_signatures) {
  test_parsable("function-signatures/math/matrix/variance");
}


TEST(gm_parser, operator_and_function_signatures) {
  test_parsable("function-signatures/math/operators/and");
}

TEST(gm_parser, operator_and_or_interaction_function_signatures) {
  test_parsable("function-signatures/math/operators/and_or_interaction");
}

TEST(gm_parser, operator_equal_function_signatures) {
  test_parsable("function-signatures/math/operators/equal");
}

TEST(gm_parser, operator_greater_than_or_equal_function_signatures) {
  test_parsable("function-signatures/math/operators/greater_than_or_equal");
}

TEST(gm_parser, operator_greater_than_function_signatures) {
  test_parsable("function-signatures/math/operators/greater_than");
}

TEST(gm_parser, operator_less_than_or_equal_function_signatures) {
  test_parsable("function-signatures/math/operators/less_than_or_equal");
}

TEST(gm_parser, operator_less_than_function_signatures) {
  test_parsable("function-signatures/math/operators/less_than");
}

TEST(gm_parser, operator_not_equal_function_signatures) {
  test_parsable("function-signatures/math/operators/not_equal");
}

TEST(gm_parser, operator_not_function_signatures) {
  test_parsable("function-signatures/math/operators/not");
}

TEST(gm_parser, operator_or_function_signatures) {
  test_parsable("function-signatures/math/operators/or");
}

TEST(gm_parser, if_else_function_signatures) {
  test_parsable("function-signatures/math/if_else");
}

TEST(gm_parser, while_function_signatures) {
  test_parsable("function-signatures/math/while");
}

TEST(gm_parser, infVariableName) {
  test_parsable("good_inf");
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



TEST(gmParserTermGrammar, infixExponentiation) {
  test_parsable("validate_exponentiation_good");
  test_throws("validate_exponentiation_bad", 
              "type mismatch in assignment; left variable=z;");
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

TEST(gmParserTermGrammar, eltDivisionFun) {
  test_parsable("validate_elt_division_good");
}

TEST(gmParserTermGrammar, negateExprFun) {
  test_parsable("validate_negate_expr_good");
}

TEST(gmParserTermGrammar, logicalNegateExprFun) {
  test_throws("validate_logical_negate_expr_bad",
              "logical negation operator ! only applies to int or real");
  test_parsable("validate_logical_negate_expr_good");
}

TEST(gmParserTermGrammar, addExpressionDimssFun) {
  test_throws("validate_add_expression_dimss_bad",
              "indexes inappropriate");
  test_parsable("validate_add_expression_dimss_good");
}

TEST(gmParserTermGrammar, setFunTypeNamed) {
  test_throws("validate_set_fun_type_named_bad1",
              "random number generators only allowed in generated quantities");
  test_parsable("validate_set_fun_type_named_good");
}

TEST(gmParserVarDeclsGrammarDef, addVar) {
  test_throws("validate_add_var_bad1",
              "duplicate declaration of variable");
  test_throws("validate_add_var_bad2",
              "integer parameters or transformed parameters are not allowed");
  test_parsable("validate_add_var_good");
}

TEST(gmParserVarDeclsGrammarDef, validateIntExpr) {
  test_parsable("validate_validate_int_expr_good");
  for (int i = 1; i <= 13; ++i) {
    std::string model_name("validate_validate_int_expr_bad");
    model_name += boost::lexical_cast<std::string>(i);
    test_throws(model_name,
                "expression denoting integer required");
  }
}

TEST(gmParserVarDeclsGrammarDef, setIntRangeLower) {
  test_parsable("validate_set_int_range_lower_good");
  test_throws("validate_set_int_range_lower_bad1",
              "expression denoting integer required");
  test_throws("validate_set_int_range_lower_bad2",
              "expression denoting integer required");
  test_throws("validate_set_int_range_lower_bad3",
              "expression denoting integer required");
}

TEST(gmParserVarDeclsGrammarDef, setIntRangeUpper) {
  test_parsable("validate_set_int_range_upper_good");
  test_throws("validate_set_int_range_upper_bad1",
              "expression denoting integer required");
  test_throws("validate_set_int_range_upper_bad2",
              "expression denoting integer required");
}

TEST(gmParserVarDeclsGrammarDef, setDoubleRangeLower) {
  test_parsable("validate_set_double_range_lower_good");
  test_throws("validate_set_double_range_lower_bad1",
              "expression denoting real required");
  test_throws("validate_set_double_range_lower_bad2",
              "expression denoting real required");
}

TEST(gmParserVarDeclsGrammarDef, setDoubleRangeUpper) {
  test_parsable("validate_set_double_range_upper_good");
  test_throws("validate_set_double_range_upper_bad1",
              "expression denoting real required");
  test_throws("validate_set_double_range_upper_bad2",
              "expression denoting real required");
}

TEST(gmParserStatementGrammarDef, jacobianAdjustmentWarning) {
  test_parsable("validate_jacobian_warning_good");
  test_warning("validate_jacobian_warning1",
               "you must call increment_log_prob() with the log absolute determinant");
  test_warning("validate_jacobian_warning2",
               "you must call increment_log_prob() with the log absolute determinant");
  test_warning("validate_jacobian_warning3",
               "you must call increment_log_prob() with the log absolute determinant");
  test_warning("validate_jacobian_warning4",
               "you must call increment_log_prob() with the log absolute determinant");
  test_warning("validate_jacobian_warning5",
               "you must call increment_log_prob() with the log absolute determinant");
  test_warning("validate_jacobian_warning6",
               "you must call increment_log_prob() with the log absolute determinant");
}

TEST(gmParserStatementGrammarDef, comparisonsInBoundsTest) {
  test_parsable("validate_bounds_comparison");
  EXPECT_THROW(is_parsable("src/test/test-models/reference/gm/bad_bounds1.stan"),
               std::invalid_argument);
}


TEST(parserFunctions, funsGood0) {
  test_parsable("validate_functions"); // tests proper definitions and use
}

TEST(parserFunctions, funsGood1) {
  test_parsable("functions-good1");
}

TEST(parserFunctions, funsGood2) {
  test_parsable("functions-good2");
}

TEST(parserFunctions, funsGood3) {
  test_parsable("functions-good3");
}

TEST(parserFunctions, funsGood4) {
  test_parsable("functions-good-void");
  test_parsable("functions-good-void"); // test twice to ensure
                                        // symbols are not saved
}
TEST(gmParser, intFun) {
  test_parsable("int_fun");
}

TEST(parserFunctions, funsBad18) {
  test_throws("functions-bad18","variable identifier (name) may not be reserved word");
}

TEST(parserFunctions, funsBad0) {
  test_throws("functions-bad0","Functions cannot contain void argument types");
}

TEST(parserFunctions, funsBad1) {
  test_throws("functions-bad1","Function already declared");
}

TEST(parserFunctions, funsBad2) {
  test_throws("functions-bad2","Function declared, but not defined");
}

TEST(parserFunctions, funsBad3) {
  test_throws("functions-bad3","EXPECTED: \"(\" BUT FOUND");
}

TEST(parserFunctions,funsBad4) {
  test_throws("functions-bad4",
              "Functions used as statements must be declared to have void returns");
}

TEST(parserFunctions,funsBad5) {
  test_throws("functions-bad5",
              "base type mismatch in assignment");
}

TEST(parserFunctions,funsBad6) {
  test_throws("functions-bad6",
              "lp suffixed functions only allowed in");
}

TEST(parserFunctions,funsBad7) {
  test_throws("functions-bad7",
              "lp suffixed functions only allowed in");
}

TEST(parserFunctions,funsBad8) {
  test_throws("functions-bad8",
              "random number generators only allowed in");
}

TEST(parserFunctions,funsBad9) {
  test_throws("functions-bad9",
              "random number generators only allowed in");
}

TEST(parserFunctions,funsBad10) {
  test_throws("functions-bad10",
              "random number generators only allowed in");
}

TEST(parserFunctions,funsBad11) {
  test_throws("functions-bad11",
              "sampling only allowed in model");
}

TEST(parserFunctions,funsBad12) {
  test_throws("functions-bad12",
              "sampling only allowed in model");
}

TEST(parserFunctions,funsBad13) {
  test_throws("functions-bad13",
              "Illegal to assign to function argument variables");
}

TEST(parserFunctions,funsBad14) {
  test_throws("functions-bad14",
              "Function already defined");
}

TEST(parserFunctions,funsBad15) {
  test_throws("functions-bad15",
              "attempt to increment log prob with void expression");
}

TEST(parserFunctions,funsBad16) {
  test_throws("functions-bad16",
              "Function system defined");
}

TEST(parserFunctions,funsBad17) {
  test_throws("functions-bad17",
              "Require real return type for functions ending in _log");
}




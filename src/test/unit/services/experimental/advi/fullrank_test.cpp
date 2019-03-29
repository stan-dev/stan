#include <stan/services/experimental/advi/fullrank.hpp>
#include <gtest/gtest.h>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/services/test_lp.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>

class ServicesExperimentalAdvi : public testing::Test {
public:
  ServicesExperimentalAdvi()
    : model(context, &model_log) {}

  std::stringstream model_log;
  stan::test::unit::instrumented_writer init, parameter, diagnostic;
  stan::test::unit::instrumented_logger logger;
  stan::io::empty_var_context context;
  stan::test::unit::instrumented_interrupt interrupt;
  stan_model model;
};

TEST_F(ServicesExperimentalAdvi, experimental_message) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;
  int grad_samples = 1;
  int elbo_samples = 100;
  int max_iterations = 10000;
  double tol_rel_obj = 0.01;
  double eta = 1.0;
  bool adapt_engaged = true;
  int adapt_iterations = 50;
  int eval_elbo = 100;
  int output_samples = 1000;

  stan::services::experimental::advi
    ::fullrank(model, context,
               seed, chain, init_radius,
               grad_samples, elbo_samples,
               max_iterations, tol_rel_obj,
               eta, adapt_engaged,
               adapt_iterations,
               eval_elbo, output_samples,
               interrupt,
               logger, init, parameter, diagnostic);

  EXPECT_GT(logger.call_count(), 0);
  EXPECT_EQ(logger.call_count(), logger.call_count_info())
    << "all messages go to info";

  EXPECT_EQ(1, logger.find_info("EXPERIMENTAL ALGORITHM"))
    << "Missing experimental algorithm message";
}

TEST_F(ServicesExperimentalAdvi, fullrank) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;
  int grad_samples = 1;
  int elbo_samples = 100;
  int max_iterations = 10000;
  double tol_rel_obj = 0.01;
  double eta = 1.0;
  bool adapt_engaged = true;
  int adapt_iterations = 50;
  int eval_elbo = 100;
  int output_samples = 1000;

  int return_code = stan::services::experimental::advi
    ::fullrank(model, context,
               seed, chain, init_radius,
               grad_samples, elbo_samples,
               max_iterations, tol_rel_obj,
               eta, adapt_engaged,
               adapt_iterations,
               eval_elbo, output_samples,
               interrupt,
               logger, init, parameter, diagnostic);
  EXPECT_EQ(0, return_code);

  std::vector<std::vector<std::string> > parameter_names;
  parameter_names = parameter.vector_string_values();
  std::vector<std::vector<double> > parameter_values;
  parameter_values = parameter.vector_double_values();

  // Expectations of parameter parameter names.
  ASSERT_EQ(8, parameter_names[0].size());
  EXPECT_EQ("lp__", parameter_names[0][0]);
  EXPECT_EQ("log_p__", parameter_names[0][1]);
  EXPECT_EQ("log_g__", parameter_names[0][2]);
  EXPECT_EQ("y.1", parameter_names[0][3]);
  EXPECT_EQ("y.2", parameter_names[0][4]);
  EXPECT_EQ("z.1", parameter_names[0][5]);
  EXPECT_EQ("z.2", parameter_names[0][6]);
  EXPECT_EQ("xgq", parameter_names[0][7]);

  // Expect one name per parameter value.
  EXPECT_EQ(parameter_names[0].size(), parameter_values[0].size());

  ASSERT_EQ(1, init.vector_double_values().size());
  ASSERT_EQ(2, init.vector_double_values().at(0).size());
  std::vector<double> init_values = init.vector_double_values().at(0);
  EXPECT_FLOAT_EQ(0, init_values[0]);
  EXPECT_FLOAT_EQ(0, init_values[1]);

  ASSERT_EQ(output_samples + 1, parameter.vector_double_values().size());
  ASSERT_EQ(eval_elbo, diagnostic.vector_double_values().size());

  EXPECT_EQ(0, interrupt.call_count());
}

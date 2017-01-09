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
  stan::test::unit::instrumented_writer message, init, parameter, diagnostic;
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
               message, init, parameter, diagnostic);

  bool found_experimental_message = false;
  for (size_t n = 0; n < message.string_values().size(); ++n) {
    if (!found_experimental_message
        && message.string_values().at(n).find("EXPERIMENTAL ALGORITHM") != std::string::npos)
      found_experimental_message = true;
  }
  EXPECT_TRUE(found_experimental_message) << "Missing experimental algorithm message";
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
               message, init, parameter, diagnostic);
  EXPECT_EQ(0, return_code);

  ASSERT_EQ(1, init.vector_double_values().size());
  ASSERT_EQ(2, init.vector_double_values().at(0).size());
  std::vector<double> init_values = init.vector_double_values().at(0);
  EXPECT_FLOAT_EQ(0, init_values[0]);
  EXPECT_FLOAT_EQ(0, init_values[1]);

  ASSERT_EQ(output_samples + 1, parameter.vector_double_values().size());
  ASSERT_EQ(eval_elbo, diagnostic.vector_double_values().size());

  EXPECT_EQ(0, interrupt.call_count());
}

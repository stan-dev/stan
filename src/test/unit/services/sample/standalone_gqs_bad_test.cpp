#include <stan/services/error_codes.hpp>
#include <stan/services/sample/standalone_gqs.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/services/test_gq2.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <iostream>

class ServicesStandaloneGQ2 : public testing::Test {
public:
  ServicesStandaloneGQ2()
    : model(context, &model_log) {}

  stan::io::empty_var_context context;
  std::stringstream model_log;
  stan_model model;
  std::stringstream sample_ss, logger_ss;
  stan::test::unit::instrumented_interrupt interrupt;
};

TEST_F(ServicesStandaloneGQ2, no_QoIs) {
  stan::test::unit::instrumented_interrupt interrupt;
  std::stringstream model_log;
  std::stringstream sample_ss, logger_ss;
  stan::callbacks::stream_writer sample_writer(sample_ss, "");
  stan::callbacks::stream_logger logger(logger_ss,
                                        logger_ss,
                                        logger_ss,
                                        logger_ss,
                                        logger_ss);

  int num_params = stan::services::num_constrained_params(model);
  EXPECT_EQ(num_params, 2);
  std::vector<double> draw1;
  draw1.push_back(-2.345);
  draw1.push_back(-6.789);
  std::vector<double> draw2;
  draw2.push_back(-3.123);
  draw2.push_back(-4.123);
  std::vector<std::vector<double> > draws;
  draws.push_back(draw1);
  draws.push_back(draw2);
  const std::vector<std::vector<double> > cdraws(draws);

  int return_code = stan::services::standalone_generate(model,
                                                        cdraws,
                                                        12345,
                                                        interrupt,
                                                        logger,
                                                        sample_writer);
  EXPECT_EQ(return_code, stan::services::error_codes::CONFIG);
  EXPECT_EQ(count_matches("Model doesn't generate any quantities of interest.",
                          logger_ss.str()),1);
}

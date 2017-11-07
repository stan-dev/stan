#include <stan/services/sample/standalone_gqs.hpp>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/services/test_lp.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <iostream>
#include <gtest/gtest.h>

/**
 * Use test model which has GQ, and a constrained param
 */

class ServicesStandaloneGQ : public testing::Test {
public:
  ServicesStandaloneGQ()
    : model(context, &model_log) {}

  std::stringstream model_log;
  stan::io::empty_var_context context;
  stan::test::unit::instrumented_logger logger;
  stan::test::unit::instrumented_writer sample_writer;
  stan_model model;
  stan::test::unit::instrumented_interrupt interrupt;
};

TEST_F(ServicesStandaloneGQ, t1) {
  int num_params = stan::services::num_constrained_params(model);
  EXPECT_EQ(num_params, 2);
}

// TEST_F(ServicesStandaloneGQ, t2) {
//    std::vector<double> draw;
//    draw.push_back(-2.345);
//    draw.push_back(-6.789);
//    std::vector<std::vector<double> > draws;
//    draws.push_back(draw);
//    draws.push_back(draw);
//    const std::vector<std::vector<double> > cdraws(draws);
//    stan::services::standalone_generate(model,
//                                        cdraws,
//                                        12345,
//                                        interrupt,
//                                        logger,
//                                        sample_writer);
//    EXPECT_EQ(1, 1);
// }



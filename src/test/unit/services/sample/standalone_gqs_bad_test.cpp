#include <stan/services/error_codes.hpp>
#include <stan/services/sample/standalone_gqs.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/services/test_gq2.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <test/unit/util.hpp>
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

class ServicesStandaloneGQ2 : public testing::Test {
public:
  ServicesStandaloneGQ2() :
    logger(logger_ss, logger_ss, logger_ss, logger_ss, logger_ss) {}

  void SetUp() {
    stan::io::empty_var_context context;
    model = new stan_model(context);
  }

  void TearDown() {
    delete model;
  }

  stan::test::unit::instrumented_interrupt interrupt;
  std::stringstream logger_ss;
  stan::callbacks::stream_logger logger;
  stan_model *model;
};

TEST_F(ServicesStandaloneGQ2, no_QoIs) {
  std::vector<std::string> param_names;
  std::vector<std::vector<size_t>> param_dimss;
  stan::services::get_model_parameters(*model, param_names, param_dimss);
  EXPECT_EQ(param_names.size(), 1);
  EXPECT_EQ(param_dimss.size(), 1);

  const Eigen::MatrixXd draws(1,1);
  std::stringstream sample_ss;
  stan::callbacks::stream_writer sample_writer(sample_ss, "");
  int return_code = stan::services::standalone_generate(*model,
                                                        draws,
                                                        12345,
                                                        interrupt,
                                                        logger,
                                                        sample_writer);
  EXPECT_EQ(return_code, stan::services::error_codes::CONFIG);
  EXPECT_EQ(count_matches("Model doesn't generate any quantities of interest.",
                          logger_ss.str()),1);
}

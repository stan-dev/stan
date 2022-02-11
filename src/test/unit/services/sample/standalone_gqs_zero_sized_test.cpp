#include <stan/services/error_codes.hpp>
#include <stan/services/sample/standalone_gqs.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <test/test-models/good/services/gq_test_zero_sized.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <test/unit/util.hpp>
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

class ServicesStandaloneGQ2 : public ::testing::Test {
 public:
  ServicesStandaloneGQ2()
      : logger(logger_ss, logger_ss, logger_ss, logger_ss, logger_ss) {}

  void SetUp() {
    stan::io::empty_var_context context;
    model = new stan_model(context);
  }

  void TearDown() { delete model; }

  stan::test::unit::instrumented_interrupt interrupt;
  std::stringstream logger_ss;
  stan::callbacks::stream_logger logger;
  stan_model *model;
};

TEST_F(ServicesStandaloneGQ2, zero_sized_params_get_model_parameters) {
  std::vector<std::string> param_names;
  std::vector<std::vector<size_t>> param_dimss;
  stan::services::get_model_parameters(*model, param_names, param_dimss);

  EXPECT_EQ(param_names.size(), 4);
  EXPECT_EQ(param_dimss.size(), 4);
  EXPECT_EQ(param_dimss[0].size(), 0);
  EXPECT_EQ(param_dimss[1].size(), 1);
  EXPECT_EQ(param_dimss[1][0], 5);
  EXPECT_EQ(param_dimss[2].size(), 1);
  EXPECT_EQ(param_dimss[2][0], 6);
  EXPECT_EQ(param_dimss[3].size(), 2);
  EXPECT_EQ(param_dimss[3][0], 2);
  EXPECT_EQ(param_dimss[3][1], 3);
}

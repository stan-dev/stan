#include <stan/services/error_codes.hpp>
#include <stan/services/sample/standalone_gqs.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <test/test-models/good/services/bug_2390_gq.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <test/unit/util.hpp>
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

class ServicesStandaloneGQ4 : public ::testing::Test {
 public:
  ServicesStandaloneGQ4()
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

TEST_F(ServicesStandaloneGQ4, genDraws_gq_test_vec_len_1) {
  stan::io::stan_csv multidim_csv;
  std::stringstream out;
  std::ifstream csv_stream;
  csv_stream.open(
      "src/test/test-models/good/services/bug_2390_fitted_params.csv");
  multidim_csv = stan::io::stan_csv_reader::parse(csv_stream, &out);
  csv_stream.close();
  EXPECT_EQ("b[1]", multidim_csv.header[7]);

  // model bug_2390_model has 1 param, vector[1] b;
  std::vector<std::string> param_names;
  std::vector<std::vector<size_t>> param_dimss;
  stan::services::get_model_parameters(*model, param_names, param_dimss);

  EXPECT_EQ(param_names.size(), 1);
  EXPECT_EQ(param_dimss.size(), 1);
  int total = 1;
  for (size_t i = 0; i < param_dimss[0].size(); ++i) {
    total *= param_dimss[0][i];
  }
  EXPECT_EQ(total, 1);

  std::stringstream sample_ss;
  stan::callbacks::stream_writer sample_writer(sample_ss, "");
  int return_code = stan::services::standalone_generate(
      *model, multidim_csv.samples.middleCols<1>(7), 12345, interrupt, logger,
      sample_writer);
  EXPECT_EQ(return_code, stan::services::error_codes::OK);
  EXPECT_EQ(count_matches("y_est", sample_ss.str()), 5);
  EXPECT_EQ(count_matches("\n", sample_ss.str()), 1001);
  match_csv_columns(multidim_csv.samples, sample_ss.str(), 1000, 0, 6);
}

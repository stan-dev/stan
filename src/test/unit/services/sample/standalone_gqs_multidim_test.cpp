#include <stan/services/error_codes.hpp>
#include <stan/services/sample/standalone_gqs.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <test/test-models/good/services/gq_test_multidim.hpp>
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

  void TearDown() {
    delete model;
  }

  stan::test::unit::instrumented_interrupt interrupt;
  std::stringstream logger_ss;
  stan::callbacks::stream_logger logger;
  stan_model *model;
};

TEST_F(ServicesStandaloneGQ2, genDraws_gq_test_multidim) {
  stan::io::stan_csv multidim_csv;
  std::stringstream out;
  std::ifstream csv_stream;
  csv_stream.open("src/test/test-models/good/services/gq_test_multidim_fit.csv");
  multidim_csv = stan::io::stan_csv_reader::parse(csv_stream, &out);
  csv_stream.close();
  EXPECT_EQ(12345U, multidim_csv.metadata.seed);
  ASSERT_EQ(247, multidim_csv.header.size());
  EXPECT_EQ("p_ar_mat[1,1,1,1]", multidim_csv.header(7));
  EXPECT_EQ("p_ar_mat[4,5,2,3]", multidim_csv.header(126));
  EXPECT_EQ("gq_ar_mat[1,1,1,1]", multidim_csv.header(127));
  EXPECT_EQ("gq_ar_mat[4,5,2,3]", multidim_csv.header(246));
  ASSERT_EQ(1000, multidim_csv.samples.rows());
  ASSERT_EQ(247, multidim_csv.samples.cols());

  // model gq_test_multidim has 1 param, length 120
  std::vector<std::string> param_names;
  std::vector<std::vector<size_t>> param_dimss;
  stan::services::get_model_parameters(*model, param_names, param_dimss);

  EXPECT_EQ(param_names.size(), 1);
  EXPECT_EQ(param_dimss.size(), 1);
  int total = 1;
  for (size_t i = 0; i < param_dimss[0].size(); ++i) {
    total *= param_dimss[0][i];
  }
  EXPECT_EQ(total, 120);

  std::stringstream sample_ss;
  stan::callbacks::stream_writer sample_writer(sample_ss, "");
  int return_code = stan::services::standalone_generate(*model,
                                    multidim_csv.samples.middleCols<120>(7),
                                    12345,
                                    interrupt,
                                    logger,
                                    sample_writer);
  EXPECT_EQ(return_code, stan::services::error_codes::OK);
  EXPECT_EQ(count_matches("gq_ar_mat",sample_ss.str()),120);
  EXPECT_EQ(count_matches("\n",sample_ss.str()),1001);
  match_csv_columns(multidim_csv.samples, sample_ss.str(), 1000, 120, 127);
}

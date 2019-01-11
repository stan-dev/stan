#include <stan/services/error_codes.hpp>
#include <stan/services/sample/standalone_gqs.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/io/dump.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <test/test-models/good/services/bernoulli.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <boost/tokenizer.hpp>

class ServicesStandaloneGQ : public ::testing::Test {
public:
  ServicesStandaloneGQ() :
    logger(logger_ss, logger_ss, logger_ss, logger_ss, logger_ss) {}

 void SetUp() {
  std::fstream data_stream("src/test/test-models/good/services/bernoulli.data.R",
                           std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();
  model = new stan_model(data_var_context);
 }

  void TearDown() {
    delete(model);
  }

  stan::test::unit::instrumented_interrupt interrupt;
  std::stringstream logger_ss;
  stan::callbacks::stream_logger logger;
  stan_model *model;
};

typedef boost::tokenizer<boost::char_separator<char>> tokenizer;

TEST_F(ServicesStandaloneGQ, genDraws_bernoulli) {
  stan::io::stan_csv bern_csv;
  std::stringstream out;
  std::ifstream csv_stream;
  csv_stream.open("src/test/test-models/good/services/bernoulli_fit.csv");
  bern_csv = stan::io::stan_csv_reader::parse(csv_stream, &out);
  csv_stream.close();
  EXPECT_EQ(12345U, bern_csv.metadata.seed);
  ASSERT_EQ(19, bern_csv.header.size());
  EXPECT_EQ("theta", bern_csv.header(7));
  ASSERT_EQ(1000, bern_csv.samples.rows());
  ASSERT_EQ(19, bern_csv.samples.cols());

  // model bernoulli.stan has 1 param
  std::vector<std::string> param_names;
  std::vector<std::vector<size_t>> param_dimss;
  stan::services::get_model_parameters(*model, param_names, param_dimss);

  EXPECT_EQ(param_names.size(), 1);
  EXPECT_EQ(param_dimss.size(), 1);

  std::stringstream sample_ss;
  stan::callbacks::stream_writer sample_writer(sample_ss, "");
  int return_code = stan::services::standalone_generate(*model,
                                                        bern_csv.samples.middleCols<1>(7),
                                                        12345,
                                                        interrupt,
                                                        logger,
                                                        sample_writer);
  if (return_code != stan::services::error_codes::OK)
    std::cout << "ERROR: " << logger_ss.str() << std::endl;

  EXPECT_EQ(return_code, stan::services::error_codes::OK);
  EXPECT_EQ(count_matches("mu",sample_ss.str()),1);
  EXPECT_EQ(count_matches("y_rep",sample_ss.str()),10);
  EXPECT_EQ(count_matches("\n",sample_ss.str()),1001);

  // compare standalone to sampler QoIs
  std::stringstream sampler_qoi_ss;
  boost::char_separator<char> newline{"\n"};
  boost::char_separator<char> comma{","};
  tokenizer linetok{sample_ss.str(), newline};
  size_t row = 0;
  for (const auto &line : linetok) {
    if (row == 0) {
      ++row;
      continue;
    } 
    tokenizer numtok{line, comma};
    for (const auto &qoi : numtok) {
      sampler_qoi_ss.str(std::string());
      sampler_qoi_ss.clear();
      sampler_qoi_ss << bern_csv.samples(row-1,8);  // 7 diagnostics, 1 param
      EXPECT_EQ(qoi, sampler_qoi_ss.str());
      break;
    }
    ++row;
  }
}      


TEST_F(ServicesStandaloneGQ, genDraws_empty_draws) {
  const Eigen::MatrixXd draws;
  std::stringstream sample_ss;
  stan::callbacks::stream_writer sample_writer(sample_ss, "");
  int return_code = stan::services::standalone_generate(*model,
                                                        draws,
                                                        12345,
                                                        interrupt,
                                                        logger,
                                                        sample_writer);
  EXPECT_EQ(return_code, stan::services::error_codes::DATAERR);
  EXPECT_EQ(count_matches("Empty set of draws",logger_ss.str()),1);
}

TEST_F(ServicesStandaloneGQ, genDraws_bad) {
  Eigen::MatrixXd draws(2,2);
  std::stringstream sample_ss;
  stan::callbacks::stream_writer sample_writer(sample_ss, "");
  int return_code = stan::services::standalone_generate(*model,
                                                        draws,
                                                        12345,
                                                        interrupt,
                                                        logger,
                                                        sample_writer);
  EXPECT_EQ(return_code, stan::services::error_codes::DATAERR);
  EXPECT_EQ(count_matches("Wrong number of parameter values",
                          logger_ss.str()),1);
}

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

typedef bernoulli_model_namespace::bernoulli_model model_class;
typedef boost::ecuyer1988 rng_t;

class ServicesStandaloneGQ : public ::testing::Test {
public:
  ServicesStandaloneGQ() :
    logger(logger_ss, logger_ss, logger_ss, logger_ss, logger_ss) {}

 void SetUp() {
  std::fstream data_stream("src/test/test-models/good/services/bernoulli.data.R",
                           std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();
  model = new model_class(data_var_context);
 }

  void TearDown() {
    delete(model);
  }

  stan::test::unit::instrumented_interrupt interrupt;
  std::stringstream logger_ss;
  stan::callbacks::stream_logger logger;
  model_class *model;
};

TEST_F(ServicesStandaloneGQ, genDraws_bernoulli) {
  // Get param data from fitted model param
  // fitted with bernoulli.data.json, N = 10, y = 2 success, 8 fail
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
  std::vector<std::vector<double> > draws(1000);
  for (int i = 0; i < 1000; ++i) {
    std::vector<double> draw(1);
    draw[0] =  bern_csv.samples(i,7);
    draws[i] = draw;
  }

  // model bernoulli.stan has 2 params
  std::vector<std::string> param_names;
  std::vector<std::vector<size_t>> param_dimss;
  stan::services::get_model_parameters(*model, param_names, param_dimss);

  EXPECT_EQ(param_names.size(), 1);
  EXPECT_EQ(param_dimss.size(), 1);

  std::stringstream sample_ss;
  stan::callbacks::stream_writer sample_writer(sample_ss, "");
  int return_code = stan::services::standalone_generate(*model,
                                                        draws,
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
  //  std::cout << sample_ss.str() << std::endl;
  

  // compare generated sample to original


}

TEST_F(ServicesStandaloneGQ, genDraws_empty_draws) {
  std::vector<std::vector<double> > draws;
  const std::vector<std::vector<double> > cdraws(draws);

  std::stringstream sample_ss;
  stan::callbacks::stream_writer sample_writer(sample_ss, "");
  int return_code = stan::services::standalone_generate(*model,
                                                        cdraws,
                                                        12345,
                                                        interrupt,
                                                        logger,
                                                        sample_writer);

  EXPECT_EQ(return_code, stan::services::error_codes::DATAERR);
  EXPECT_EQ(count_matches("Empty set of draws",logger_ss.str()),1);
}

TEST_F(ServicesStandaloneGQ, genDraws_bad1) {
  std::vector<double> draw1;
  draw1.push_back(0.345);
  std::vector<double> draw2;
  std::vector<std::vector<double> > draws;
  draws.push_back(draw1);
  draws.push_back(draw2);
  const std::vector<std::vector<double> > cdraws(draws);
  std::stringstream sample_ss;
  stan::callbacks::stream_writer sample_writer(sample_ss, "");
  int return_code = stan::services::standalone_generate(*model,
                                                        cdraws,
                                                        12345,
                                                        interrupt,
                                                        logger,
                                                        sample_writer);

  EXPECT_EQ(return_code, stan::services::error_codes::DATAERR);
  EXPECT_EQ(count_matches("Draw 2, wrong number of parameter values.",
                          logger_ss.str()),1);
}


TEST_F(ServicesStandaloneGQ, genDraws_bad2) {
  std::vector<double> draw1;
  draw1.push_back(0.345);
  std::vector<double> draw2;
  draw2.push_back(0.345);
  draw2.push_back(0.345);
  draw2.push_back(0.345);
  std::vector<std::vector<double> > draws;
  draws.push_back(draw1);
  draws.push_back(draw2);
  const std::vector<std::vector<double> > cdraws(draws);
  std::stringstream sample_ss;
  stan::callbacks::stream_writer sample_writer(sample_ss, "");
  int return_code = stan::services::standalone_generate(*model,
                                                        cdraws,
                                                        12345,
                                                        interrupt,
                                                        logger,
                                                        sample_writer);

  EXPECT_EQ(return_code, stan::services::error_codes::DATAERR);
  EXPECT_EQ(count_matches("Draw 2, wrong number of parameter values.",
                          logger_ss.str()),1);
}

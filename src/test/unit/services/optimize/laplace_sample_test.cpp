#include <boost/algorithm/string.hpp>
#include <gtest/gtest.h>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/io/dump.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/optimize/laplace_sample.hpp>
#include <test/test-models/good/services/multi_normal.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <test/unit/util.hpp>
#include <cmath>
#include <iostream>
#include <vector>

class ServicesLaplaceSample : public ::testing::Test {
 public:
  ServicesLaplaceSample() : logger(msgs, msgs, msgs, msgs, msgs) {}

  void SetUp() {
    stan::io::empty_var_context var_context;
    model = new stan_model(var_context);  // typedef from multi_normal.hpp
  }

  void TearDown() { delete model; }

  stan_model* model;
  std::stringstream msgs;
  stan::callbacks::stream_logger logger;
  stan::test::unit::instrumented_interrupt interrupt;
};

TEST_F(ServicesLaplaceSample, values) {
  Eigen::VectorXd theta_hat(2);
  theta_hat << 2, 3;
  int draws = 50000;  // big to enable mean, var & covar test precision
  unsigned int seed = 1234;
  int refresh = 100;
  std::stringstream sample_ss;
  stan::callbacks::stream_writer sample_writer(sample_ss, "");
  int return_code = stan::services::laplace_sample<true>(
      *model, theta_hat, draws, seed, refresh, interrupt, logger,
      sample_writer);
  EXPECT_EQ(stan::services::error_codes::OK, return_code);
  std::string samples_str = sample_ss.str();
  EXPECT_EQ(1, count_matches("log_p__", samples_str));
  EXPECT_EQ(1, count_matches("log_q__", samples_str));
  EXPECT_EQ(2, count_matches("y", samples_str));
  EXPECT_EQ(1, count_matches("y.1", samples_str));
  EXPECT_EQ(1, count_matches("y.2", samples_str));

  std::stringstream out;
  stan::io::stan_csv draws_csv
      = stan::io::stan_csv_reader::parse(sample_ss, &out);

  EXPECT_EQ(4, draws_csv.header.size());
  EXPECT_EQ("log_p__", draws_csv.header[0]);
  EXPECT_EQ("log_q__", draws_csv.header[1]);
  EXPECT_EQ("y[1]", draws_csv.header[2]);
  EXPECT_EQ("y[2]", draws_csv.header[3]);

  Eigen::MatrixXd sample = draws_csv.samples;
  EXPECT_EQ(4, sample.cols());
  EXPECT_EQ(draws, sample.rows());

  Eigen::VectorXd log_p = sample.col(0);
  Eigen::VectorXd log_q = sample.col(1);
  Eigen::VectorXd y1 = sample.col(2);
  Eigen::VectorXd y2 = sample.col(3);

  // because target is normal, laplace approx is exact
  for (int m = 0; m < draws; ++m) {
    EXPECT_FLOAT_EQ(0, log_p(m) - log_q(m));
  }

  // expect mean draws to be location params +/0 MC error
  EXPECT_NEAR(2, stan::math::mean(y1), 0.05);
  EXPECT_NEAR(3, stan::math::mean(y2), 0.05);

  double sum1 = 0;
  for (int m = 0; m < draws; ++m) {
    sum1 += std::pow(y1(m) - 2, 2);
  }
  double var1 = sum1 / draws;
  EXPECT_NEAR(1, var1, 0.05);

  double sum2 = 0;
  for (int m = 0; m < draws; ++m) {
    sum2 += std::pow(y2(m) - 3, 2);
  }
  double var2 = sum2 / draws;
  EXPECT_NEAR(1, var2, 0.05);

  double sum12 = 0;
  for (int m = 0; m < draws; ++m) {
    sum12 += (y1(m) - 2) * (y2(m) - 3);
  }
  double cov = sum12 / draws;
  EXPECT_NEAR(0.8, cov, 0.05);
}

TEST_F(ServicesLaplaceSample, wrongSizeModeError) {
  Eigen::VectorXd theta_hat(3);
  theta_hat << 2, 3, 4;
  int draws = 10;
  unsigned int seed = 1234;
  int refresh = 1;
  std::stringstream sample_ss;
  stan::callbacks::stream_writer sample_writer(sample_ss, "");
  int RC = stan::services::laplace_sample<true>(*model, theta_hat, draws, seed,
                                                refresh, interrupt, logger,
                                                sample_writer);
  EXPECT_EQ(stan::services::error_codes::CONFIG, RC);
}

TEST_F(ServicesLaplaceSample, nonPositiveDrawsError) {
  Eigen::VectorXd theta_hat(2);
  theta_hat << 2, 3;
  int draws = 0;
  unsigned int seed = 1234;
  int refresh = 1;
  std::stringstream sample_ss;
  stan::callbacks::stream_writer sample_writer(sample_ss, "");
  int RC = stan::services::laplace_sample<true>(*model, theta_hat, draws, seed,
                                                refresh, interrupt, logger,
                                                sample_writer);
  EXPECT_EQ(stan::services::error_codes::CONFIG, RC);
}

TEST_F(ServicesLaplaceSample, consoleOutput) {
  Eigen::VectorXd theta_hat(2);
  theta_hat << 2, 3;
  int draws = 10;
  unsigned int seed = 1234;
  int refresh = 1;
  std::stringstream sample_ss;
  stan::callbacks::stream_writer sample_writer(sample_ss, "");
  std::stringstream logger_ss;
  stan::callbacks::stream_logger sample_logger(logger_ss, logger_ss, logger_ss,
                                               logger_ss, logger_ss);
  int return_code = stan::services::laplace_sample<true>(
      *model, theta_hat, draws, seed, refresh, interrupt, sample_logger,
      sample_writer);
  EXPECT_EQ(stan::services::error_codes::OK, return_code);
  std::string console_str = logger_ss.str();
  EXPECT_EQ(1,
            count_matches(
                "Calculating Hessian\nCalculating inverse of Cholesky factor\n",
                console_str));
  EXPECT_EQ(1, count_matches("Generating draws\niteration: 0\niteration: 1",
                             console_str));
}

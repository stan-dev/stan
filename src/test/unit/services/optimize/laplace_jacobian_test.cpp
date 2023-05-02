#include <boost/algorithm/string.hpp>
#include <gtest/gtest.h>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/io/dump.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/optimize/laplace_sample.hpp>
#include <test/test-models/good/optimization/constrain_sigma.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <test/unit/util.hpp>
#include <cmath>
#include <iostream>
#include <vector>

class ServicesLaplaceJacobian : public ::testing::Test {
 public:
  ServicesLaplaceJacobian() : logger(msgs, msgs, msgs, msgs, msgs) {}

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

TEST_F(ServicesLaplaceJacobian, laplace_jacobian_adjust) {
  Eigen::VectorXd theta_hat(1);
  theta_hat << 1.09;  // test for mode sigma at 3.1, take log
  int draws = 5;
  unsigned int seed = 1234;
  int refresh = 1;
  std::stringstream sample1_ss;
  stan::callbacks::stream_writer sample_writer(sample1_ss, "");
  int return_code = stan::services::laplace_sample<true>(
      *model, theta_hat, draws, seed, refresh, interrupt, logger,
      sample_writer);
  EXPECT_EQ(stan::services::error_codes::OK, return_code);
  std::stringstream out;
  stan::io::stan_csv draws1
      = stan::io::stan_csv_reader::parse(sample1_ss, &out);

  std::stringstream sample2_ss;
  stan::callbacks::stream_writer sample_writer2(sample2_ss, "");
  return_code = stan::services::laplace_sample<false>(*model, theta_hat, draws,
                                                      seed, refresh, interrupt,
                                                      logger, sample_writer2);
  EXPECT_EQ(stan::services::error_codes::OK, return_code);
  stan::io::stan_csv draws2
      = stan::io::stan_csv_reader::parse(sample2_ss, &out);

  EXPECT_EQ(3, draws1.header.size());
  EXPECT_EQ("log_p__", draws1.header[0]);
  EXPECT_EQ("log_q__", draws1.header[1]);
  EXPECT_EQ("sigma", draws1.header[2]);

  EXPECT_EQ(3, draws2.header.size());
  EXPECT_EQ("log_p__", draws2.header[0]);
  EXPECT_EQ("log_q__", draws2.header[1]);
  EXPECT_EQ("sigma", draws2.header[2]);

  Eigen::MatrixXd sample1 = draws1.samples;
  Eigen::MatrixXd sample2 = draws2.samples;

  EXPECT_NE(sample1.coeff(0, 0), sample2.coeff(0, 0));
  EXPECT_EQ(sample1.coeff(0, 1), sample2.coeff(0, 1));
  EXPECT_EQ(sample1.coeff(0, 2), sample2.coeff(0, 2));
}

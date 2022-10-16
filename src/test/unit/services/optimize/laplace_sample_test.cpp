// #include <boost/algorithm/string.hpp>
#include <gtest/gtest.h>
#include <stan/io/empty_var_context.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/io/dump.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/optimize/laplace_sample.hpp>
#include <test/test-models/good/services/multi_normal.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <test/unit/util.hpp>

#include <iostream>
#include <vector>

class ServicesLaplaceSample: public ::testing::Test {
 public:
  ServicesLaplaceSample() : logger(msgs, msgs, msgs, msgs, msgs) { }

  void SetUp() {
    stan::io::empty_var_context var_context;
    model = new stan_model(var_context);
  }

  void TearDown() { delete model; }

  stan::test::unit::instrumented_interrupt interrupt;
  std::stringstream msgs;
  stan::callbacks::stream_logger logger;
  stan_model* model;
};

TEST_F(ServicesLaplaceSample, bar) {
  EXPECT_EQ(1, 1);
}

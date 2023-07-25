#include <boost/algorithm/string.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/callbacks/unique_stream_writer.hpp>
#include <stan/io/dump.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/sample/standalone_gqs.hpp>
#include <test/test-models/good/services/bernoulli.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <test/unit/util.hpp>
#include <vector>

auto&& blah = stan::math::init_threadpool_tbb();

static constexpr size_t num_chains = 4;

struct deleter_noop {
  template <typename T>
  constexpr void operator()(T* arg) const {}
};

class ServicesStandaloneGQ : public ::testing::Test {
 public:
  ServicesStandaloneGQ()
      : data_var_context([]() {
          std::fstream data_stream(
              "src/test/test-models/good/services/bernoulli.data.R",
              std::fstream::in);
          stan::io::dump data_context(data_stream);
          data_stream.close();
          return data_context;
        }()),
        interrupt(),
        logger_ss(),
        logger(logger_ss, logger_ss, logger_ss, logger_ss, logger_ss),
        model(data_var_context) {}
  stan::io::dump data_var_context;
  stan::test::unit::instrumented_interrupt interrupt;
  std::stringstream logger_ss;
  stan::callbacks::stream_logger logger;
  stan_model model;
};

TEST_F(ServicesStandaloneGQ, genDraws_bernoulli) {
  stan::io::stan_csv bern_csv;
  std::stringstream out;
  std::ifstream csv_stream;
  csv_stream.open("src/test/test-models/good/services/bernoulli_fit.csv");
  bern_csv = stan::io::stan_csv_reader::parse(csv_stream, &out);
  csv_stream.close();
  EXPECT_EQ(12345U, bern_csv.metadata.seed);
  ASSERT_EQ(19, bern_csv.header.size());
  EXPECT_EQ("theta", bern_csv.header[7]);
  ASSERT_EQ(1000, bern_csv.samples.rows());
  ASSERT_EQ(19, bern_csv.samples.cols());

  std::vector<std::stringstream> sample_ss(num_chains);
  std::vector<
      stan::callbacks::unique_stream_writer<std::stringstream, deleter_noop>>
      sample_writer;
  sample_writer.reserve(num_chains);
  std::vector<Eigen::MatrixXd> draws_vec;
  for (int i = 0; i < num_chains; i++) {
    sample_writer.emplace_back(
        std::unique_ptr<std::stringstream, deleter_noop>(&sample_ss[i]), "");
    draws_vec.push_back(bern_csv.samples.middleCols<1>(7));
  }

  int return_code = stan::services::standalone_generate(
      model, num_chains, draws_vec, 12345, interrupt, logger, sample_writer);
  EXPECT_EQ(return_code, stan::services::error_codes::OK);
  for (int i = 0; i < num_chains; i++) {
    EXPECT_EQ(count_matches("mu", sample_ss[i].str()), 1);
    EXPECT_EQ(count_matches("y_rep", sample_ss[i].str()), 10);
    EXPECT_EQ(count_matches("\n", sample_ss[i].str()), 1001);
    match_csv_columns(bern_csv.samples, sample_ss[i].str(), 1000, 1, 8);
  }
}

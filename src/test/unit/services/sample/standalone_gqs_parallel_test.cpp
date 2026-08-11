#include <boost/algorithm/string.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/callbacks/unique_stream_writer.hpp>
#include <stan/io/json/json_data.hpp>
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
              "src/test/test-models/good/services/bernoulli.data.json",
              std::fstream::in);
          stan::json::json_data data_context(data_stream);
          data_stream.close();
          return data_context;
        }()),
        interrupt(),
        logger_ss(),
        logger(logger_ss, logger_ss, logger_ss, logger_ss, logger_ss),
        model(data_var_context) {}
  stan::json::json_data data_var_context;
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
    EXPECT_EQ(count_matches("\n", sample_ss[i].str()), 1004);
    match_csv_columns(bern_csv.samples, sample_ss[i].str(), 1000, 1, 8);
  }
}

namespace {

Eigen::MatrixXd bernoulli_fit_draws() {
  std::stringstream out;
  std::ifstream csv_stream;
  csv_stream.open("src/test/test-models/good/services/bernoulli_fit.csv");
  stan::io::stan_csv bern_csv
      = stan::io::stan_csv_reader::parse(csv_stream, &out);
  csv_stream.close();
  return bern_csv.samples.middleCols<1>(7);
}

// drop the timing-dependent " Elapsed Time" line, which the test writers
// emit without a comment prefix
std::string data_lines(const std::string& csv) {
  std::stringstream in(csv), out;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line[0] == '#')
      continue;
    if (line.find("Elapsed Time") != std::string::npos)
      continue;
    out << line << "\n";
  }
  return out.str();
}

using test_writer
    = stan::callbacks::unique_stream_writer<std::stringstream, deleter_noop>;

}  // namespace

// The chain id must reach the RNG: two runs of the same draws that differ
// only in chain id must not share a random number stream.
TEST_F(ServicesStandaloneGQ, genDraws_bernoulli_chain_id_rng) {
  Eigen::MatrixXd draws = bernoulli_fit_draws();
  auto gq = [&](unsigned int chain) {
    std::stringstream ss;
    test_writer writer(std::unique_ptr<std::stringstream, deleter_noop>(&ss),
                       "");
    EXPECT_EQ(stan::services::standalone_generate(
                  model, draws, 12345, interrupt, logger, writer, chain),
              stan::services::error_codes::OK);
    return data_lines(ss.str());
  };
  std::string chain_1 = gq(1);
  EXPECT_NE(chain_1, gq(2));
  EXPECT_EQ(chain_1, gq(1));  // reproducible given the same chain id
}

// init_chain_id must offset the per-chain streams: chains started at 3 must
// match chains 3 and 4 of a run started at 1.
TEST_F(ServicesStandaloneGQ, genDraws_bernoulli_init_chain_id_offset) {
  Eigen::MatrixXd draws = bernoulli_fit_draws();
  auto gq = [&](int n_chains, unsigned int init_chain_id) {
    std::vector<std::stringstream> ss(n_chains);
    std::vector<test_writer> writers;
    writers.reserve(n_chains);
    std::vector<Eigen::MatrixXd> draws_vec;
    for (int i = 0; i < n_chains; i++) {
      writers.emplace_back(
          std::unique_ptr<std::stringstream, deleter_noop>(&ss[i]), "");
      draws_vec.push_back(draws);
    }
    EXPECT_EQ(stan::services::standalone_generate(model, n_chains, draws_vec,
                                                  12345, interrupt, logger,
                                                  writers, init_chain_id),
              stan::services::error_codes::OK);
    std::vector<std::string> out;
    for (int i = 0; i < n_chains; i++)
      out.push_back(data_lines(ss[i].str()));
    return out;
  };
  std::vector<std::string> from_1 = gq(4, 1);
  std::vector<std::string> from_3 = gq(2, 3);
  EXPECT_EQ(from_3[0], from_1[2]);
  EXPECT_EQ(from_3[1], from_1[3]);
  EXPECT_NE(from_1[0], from_1[1]);
}

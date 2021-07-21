#include <stan/services/sample/hmc_nuts_dense_e_adapt.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/callbacks/unique_stream_writer.hpp>
#include <test/unit/util.hpp>
#include <src/test/unit/services/util.hpp>
#include <test/test-models/good/optimization/rosenbrock.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <gtest/gtest.h>
#include <iostream>

auto&& blah = stan::math::init_threadpool_tbb();

static constexpr size_t num_chains = 4;
class ServicesSampleHmcNutsDenseEAdaptParMatch : public testing::Test {
 public:
  ServicesSampleHmcNutsDenseEAdaptParMatch()
      : model(std::make_unique<rosenbrock_model_namespace::rosenbrock_model>(
            data_context, 0, &model_log)) {
    for (int i = 0; i < num_chains; ++i) {
      init.push_back(stan::test::unit::instrumented_writer{});
      par_parameters.emplace_back(std::make_unique<std::stringstream>(), "#");
      seq_parameters.emplace_back(std::make_unique<std::stringstream>(), "#");
      diagnostic.push_back(stan::test::unit::instrumented_writer{});
      context.push_back(std::make_shared<stan::io::empty_var_context>());
    }
  }
  stan::io::empty_var_context data_context;
  std::stringstream model_log;
  stan::test::unit::instrumented_logger logger;
  std::vector<stan::test::unit::instrumented_writer> init;
  using str_writer = stan::callbacks::unique_stream_writer<std::stringstream>;
  std::vector<str_writer> par_parameters;
  std::vector<str_writer> seq_parameters;
  std::vector<stan::test::unit::instrumented_writer> diagnostic;
  std::vector<std::shared_ptr<stan::io::empty_var_context>> context;
  std::unique_ptr<rosenbrock_model_namespace::rosenbrock_model> model;
};

/**
 * This test checks that running multiple chains in one call
 * with the same initial id is the same as running multiple calls
 * with incrementing chain ids.
 */
TEST_F(ServicesSampleHmcNutsDenseEAdaptParMatch, single_multi_match) {
  constexpr unsigned int random_seed = 0;
  constexpr unsigned int chain = 0;
  constexpr double init_radius = 0;
  constexpr int num_warmup = 200;
  constexpr int num_samples = 400;
  constexpr int num_thin = 5;
  constexpr bool save_warmup = true;
  constexpr int refresh = 0;
  constexpr double stepsize = 0.1;
  constexpr double stepsize_jitter = 0;
  constexpr int max_depth = 8;
  constexpr double delta = .1;
  constexpr double gamma = .1;
  constexpr double kappa = .1;
  constexpr double t0 = .1;
  constexpr unsigned int init_buffer = 50;
  constexpr unsigned int term_buffer = 50;
  constexpr unsigned int window = 100;
  stan::test::unit::instrumented_interrupt interrupt;
  EXPECT_EQ(interrupt.call_count(), 0);
  int return_code = stan::services::sample::hmc_nuts_dense_e_adapt(
      *model, num_chains, context, random_seed, chain, init_radius, num_warmup,
      num_samples, num_thin, save_warmup, refresh, stepsize, stepsize_jitter,
      max_depth, delta, gamma, kappa, t0, init_buffer, term_buffer, window,
      interrupt, logger, init, par_parameters, diagnostic);

  EXPECT_EQ(0, return_code);

  int num_output_lines = (num_warmup + num_samples) / num_thin;
  EXPECT_EQ((num_warmup + num_samples) * num_chains, interrupt.call_count());
  for (int i = 0; i < num_chains; ++i) {
    stan::test::unit::instrumented_writer seq_init;
    stan::test::unit::instrumented_writer seq_diagnostic;
    return_code = stan::services::sample::hmc_nuts_dense_e_adapt(
        *model, *(context[i]), random_seed, i, init_radius, num_warmup,
        num_samples, num_thin, save_warmup, refresh, stepsize, stepsize_jitter,
        max_depth, delta, gamma, kappa, t0, init_buffer, term_buffer, window,
        interrupt, logger, seq_init, seq_parameters[i], seq_diagnostic);
    EXPECT_EQ(0, return_code);
  }
  std::vector<Eigen::MatrixXd> par_res;
  for (int i = 0; i < num_chains; ++i) {
    auto par_str = par_parameters[i].get_stream().str();
    auto sub_par_str = par_str.substr(par_str.find("Elements") - 1);
    std::istringstream sub_par_stream(sub_par_str);
    Eigen::MatrixXd par_mat
        = stan::test::read_stan_sample_csv(sub_par_stream, 80, 9);
    par_res.push_back(par_mat);
  }
  std::vector<Eigen::MatrixXd> seq_res;
  for (int i = 0; i < num_chains; ++i) {
    auto seq_str = seq_parameters[i].get_stream().str();
    auto sub_seq_str = seq_str.substr(seq_str.find("Elements") - 1);
    std::istringstream sub_seq_stream(sub_seq_str);
    Eigen::MatrixXd seq_mat
        = stan::test::read_stan_sample_csv(sub_seq_stream, 80, 9);
    seq_res.push_back(seq_mat);
  }
  for (int i = 0; i < num_chains; ++i) {
    Eigen::MatrixXd diff_res
        = (par_res[i].array() - seq_res[i].array()).matrix();
    EXPECT_MATRIX_EQ(diff_res, Eigen::MatrixXd::Zero(80, 9));
  }
}

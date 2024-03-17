#include <stan/services/util/run_adaptive_sampler.hpp>
#include <stan/callbacks/table_writer.hpp>
#include <stan/callbacks/csv_writer.hpp>
#include <stan/callbacks/dispatcher.hpp>
#include <stan/callbacks/json_writer.hpp>
#include <stan/callbacks/structured_writer.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/services/util/config_adaptive_sampler.hpp>
#include <test/test-models/good/services/test_lp.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <test/unit/util.hpp>
#include <sstream>
#include <gtest/gtest.h>


// test fixture constructor initializes dispatcher
// writers for: model init params, algo state, warmup draws, sample draws, metric

class ServicesUtilRunAdaptiveSamplerDispatcher : public ::testing::Test {
public:
  ServicesUtilRunAdaptiveSamplerDispatcher() :
      ss_draw_sample(),
      ss_draw_warmup(),
      ss_uparams_sample(),
      ss_uparams_warmup(),
      ss_algo(),
      ss_metric(),
      ss_timing(),
      model(empty_context, 0, &model_ss),
      rng(stan::services::util::create_rng(0, 1)),
      sampler(model, rng),
      num_warmup(3),
      num_samples(3),
      num_thin(1),
      refresh(0),
      save_warmup(true),
      writer_draw_sample(std::unique_ptr<std::stringstream, deleter_noop>(&ss_draw_sample)),
      writer_draw_warmup(std::unique_ptr<std::stringstream, deleter_noop>(&ss_draw_warmup)),
      writer_uparams_sample(std::unique_ptr<std::stringstream, deleter_noop>(&ss_uparams_sample)),
      writer_uparams_warmup(std::unique_ptr<std::stringstream, deleter_noop>(&ss_uparams_warmup)),
      writer_algo(std::unique_ptr<std::stringstream, deleter_noop>(&ss_algo)),
      writer_metric(std::unique_ptr<std::stringstream, deleter_noop>(&ss_metric)),
      writer_timing(std::unique_ptr<std::stringstream, deleter_noop>(&ss_timing))
  {}

  void SetUp() {
    ss_draw_sample.str(std::string());
    ss_draw_sample.clear();
    ss_draw_warmup.str(std::string());
    ss_draw_warmup.clear();
    ss_uparams_sample.str(std::string());
    ss_uparams_sample.clear();
    ss_uparams_warmup.str(std::string());
    ss_uparams_warmup.clear();
    ss_algo.str(std::string());
    ss_algo.clear();
    ss_metric.str(std::string());
    ss_metric.clear();
    ss_timing.str(std::string());
    ss_timing.clear();
  }

  void TearDown() {}

  std::stringstream model_ss;
  stan::io::empty_var_context empty_context;
  stan_model model;
  stan::mcmc::adapt_diag_e_nuts<stan_model, stan::rng_t> sampler;
  stan::rng_t rng;
  stan::test::unit::instrumented_interrupt interrupt;
  stan::test::unit::instrumented_logger logger;
  int num_warmup;
  int num_samples;
  int num_thin;
  int refresh;
  bool save_warmup;

  std::stringstream ss_draw_sample;
  std::stringstream ss_draw_warmup;
  std::stringstream ss_uparams_sample;
  std::stringstream ss_uparams_warmup;
  std::stringstream ss_algo;
  std::stringstream ss_metric;
  std::stringstream ss_timing;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer_draw_sample;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer_draw_warmup;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer_uparams_sample;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer_uparams_warmup;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer_algo;
  stan::callbacks::json_writer<std::stringstream, deleter_noop> writer_metric;
  stan::callbacks::json_writer<std::stringstream, deleter_noop> writer_timing;
};

TEST_F(ServicesUtilRunAdaptiveSamplerDispatcher, run_separate) {
  stan::callbacks::dispatcher dp;
  std::shared_ptr<stan::callbacks::table_writer> writer_draw_sample_ptr
      = std::make_shared<stan::callbacks::csv_writer<
        std::stringstream, deleter_noop>>(std::move(writer_draw_sample));
  std::shared_ptr<stan::callbacks::table_writer> writer_draw_warmup_ptr
      = std::make_shared<stan::callbacks::csv_writer<
        std::stringstream, deleter_noop>>(std::move(writer_draw_warmup));
  std::shared_ptr<stan::callbacks::table_writer> writer_uparams_sample_ptr
      = std::make_shared<stan::callbacks::csv_writer<
        std::stringstream, deleter_noop>>(std::move(writer_uparams_sample));
  std::shared_ptr<stan::callbacks::table_writer> writer_uparams_warmup_ptr
      = std::make_shared<stan::callbacks::csv_writer<
        std::stringstream, deleter_noop>>(std::move(writer_uparams_warmup));
  std::shared_ptr<stan::callbacks::table_writer> writer_algo_ptr
      = std::make_shared<stan::callbacks::csv_writer<
        std::stringstream, deleter_noop>>(std::move(writer_algo));
  std::shared_ptr<stan::callbacks::structured_writer> writer_metric_ptr
      = std::make_shared<stan::callbacks::json_writer<
        std::stringstream, deleter_noop>>(std::move(writer_metric));
  std::shared_ptr<stan::callbacks::structured_writer> writer_timing_ptr
      = std::make_shared<stan::callbacks::json_writer<
        std::stringstream, deleter_noop>>(std::move(writer_timing));

  dp.add_writer(stan::callbacks::table_info_type::DRAW_SAMPLE,
                std::move(writer_draw_sample_ptr));
  dp.add_writer(stan::callbacks::table_info_type::DRAW_WARMUP,
                std::move(writer_draw_warmup_ptr));
  dp.add_writer(stan::callbacks::table_info_type::UPARAMS_SAMPLE,
                std::move(writer_uparams_sample_ptr));
  dp.add_writer(stan::callbacks::table_info_type::UPARAMS_WARMUP,
                std::move(writer_uparams_warmup_ptr));
  dp.add_writer(stan::callbacks::table_info_type::ALGO_STATE,
                std::move(writer_algo_ptr));
  dp.add_writer(stan::callbacks::struct_info_type::INV_METRIC,
                std::move(writer_metric_ptr));
  dp.add_writer(stan::callbacks::struct_info_type::RUN_TIMING,
                std::move(writer_timing_ptr));

  Eigen::VectorXd inv_metric =  Eigen::VectorXd::Ones(model.num_params_r());
  double stepsize = 1;
  double stepsize_jitter = 0;
  int max_depth = 10;
  double delta = 0.90;
  double gamma = 0.05;
  double kappa = 0.75;
  double t0 = 1;
  unsigned int init_buffer = 10;
  unsigned int term_buffer = 2;
  unsigned int window = 25;
  num_warmup = 37;

  stan::services::util::config_adaptive_sampler(sampler, inv_metric, stepsize, stepsize_jitter,
                          max_depth, delta, gamma, kappa, t0, num_warmup,
                          init_buffer, term_buffer, window, logger);

  std::vector<double> init_params = {1.5, -0.5};
  num_samples = 10;
  num_thin = 1;
  refresh = 0;
  save_warmup = true;

  EXPECT_NO_THROW(stan::services::util::run_adaptive_sampler(
      sampler, model, init_params, num_warmup, num_samples, num_thin, refresh,
      save_warmup, rng, interrupt, logger, dp));

  ASSERT_FALSE(ss_metric.str().empty());
// std::cout << "metric" << std::endl;
// std::cout << ss_metric.str() << std::endl;

  ASSERT_FALSE(ss_timing.str().empty());
// std::cout << "timing" << std::endl;
// std::cout << ss_timing.str() << std::endl;

  ASSERT_FALSE(ss_draw_warmup.str().empty());
// std::cout << "warmup" << std::endl;
// std::cout << ss_draw_warmup.str() << std::endl;

  ASSERT_FALSE(ss_draw_sample.str().empty());
// std::cout << "sample" << std::endl;
// std::cout << ss_draw_sample.str() << std::endl;

  ASSERT_FALSE(ss_uparams_warmup.str().empty());
// std::cout << "uparams warmup" << std::endl;
// std::cout << ss_uparams_warmup.str() << std::endl;

  ASSERT_FALSE(ss_uparams_sample.str().empty());
// std::cout << "uparams sample" << std::endl;
// std::cout << ss_uparams_sample.str() << std::endl;

  ASSERT_FALSE(ss_algo.str().empty());
// std::cout << "algo" << std::endl;
// std::cout << ss_algo.str() << std::endl;
}

TEST_F(ServicesUtilRunAdaptiveSamplerDispatcher, run_some) {
  stan::callbacks::dispatcher dp;
  std::shared_ptr<stan::callbacks::table_writer> writer_draw_sample_ptr
      = std::make_shared<stan::callbacks::csv_writer<
        std::stringstream, deleter_noop>>(std::move(writer_draw_sample));
  std::shared_ptr<stan::callbacks::table_writer> writer_uparams_sample_ptr
      = std::make_shared<stan::callbacks::csv_writer<
        std::stringstream, deleter_noop>>(std::move(writer_uparams_sample));
  std::shared_ptr<stan::callbacks::structured_writer> writer_metric_ptr
      = std::make_shared<stan::callbacks::json_writer<
        std::stringstream, deleter_noop>>(std::move(writer_metric));

  dp.add_writer(stan::callbacks::table_info_type::DRAW_SAMPLE,
                     std::move(writer_draw_sample_ptr));
  dp.add_writer(stan::callbacks::table_info_type::UPARAMS_SAMPLE,
                     std::move(writer_uparams_sample_ptr));
  dp.add_writer(stan::callbacks::struct_info_type::INV_METRIC,
                std::move(writer_metric_ptr));

  Eigen::VectorXd inv_metric =  Eigen::VectorXd::Ones(model.num_params_r());
  double stepsize = 1;
  double stepsize_jitter = 0;
  int max_depth = 10;
  double delta = 0.90;
  double gamma = 0.05;
  double kappa = 0.75;
  double t0 = 1;
  unsigned int init_buffer = 10;
  unsigned int term_buffer = 2;
  unsigned int window = 25;
  num_warmup = 37;

  stan::services::util::config_adaptive_sampler(sampler, inv_metric, stepsize, stepsize_jitter,
                          max_depth, delta, gamma, kappa, t0, num_warmup,
                          init_buffer, term_buffer, window, logger);

  std::vector<double> init_params = {1.5, -0.5};
  num_samples = 10;
  num_thin = 1;
  refresh = 0;
  save_warmup = true;
               
  EXPECT_NO_THROW(stan::services::util::run_adaptive_sampler(
      sampler, model, init_params, num_warmup, num_samples, num_thin, refresh,
      save_warmup, rng, interrupt, logger, dp));

  ASSERT_FALSE(ss_draw_sample.str().empty());
  ASSERT_FALSE(ss_uparams_sample.str().empty());
  ASSERT_FALSE(ss_metric.str().empty());
  ASSERT_TRUE(ss_draw_warmup.str().empty());
  ASSERT_TRUE(ss_algo.str().empty());
  ASSERT_TRUE(ss_timing.str().empty());
}

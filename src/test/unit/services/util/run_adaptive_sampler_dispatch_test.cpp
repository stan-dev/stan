#include <stan/services/util/run_adaptive_sampler.hpp>
#include <stan/callbacks/csv_writer.hpp>
#include <stan/callbacks/dispatcher.hpp>
#include <stan/callbacks/json_writer.hpp>
#include <stan/callbacks/structured_writer.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp>
#include <stan/services/util/create_rng.hpp>
#include <test/test-models/good/services/test_lp.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <test/unit/util.hpp>
#include <sstream>
#include <gtest/gtest.h>

class ServicesUtilRunAdaptiveSamplerDispatcher : public ::testing::Test {
public:
  ServicesUtilRunAdaptiveSamplerDispatcher() :
      ss_draw_cnstrn(),
      ss_params_uncnstrn(),
      ss_engine_state(),
      ss_log_prob(),
      ss_metric(),
      model(empty_context, 0, &model_ss),
      rng(stan::services::util::create_rng(0, 1)),
      sampler(model, rng),
      num_warmup(3),
      num_samples(3),
      num_thin(1),
      refresh(0),
      save_warmup(true),
      writer_draw_cnstrn(std::unique_ptr<std::stringstream, deleter_noop>(&ss_draw_cnstrn)),
      writer_params_uncnstrn(std::unique_ptr<std::stringstream, deleter_noop>(&ss_params_uncnstrn)),
      writer_engine_state(std::unique_ptr<std::stringstream, deleter_noop>(&ss_engine_state)),
      writer_log_prob(std::unique_ptr<std::stringstream, deleter_noop>(&ss_log_prob)),
      writer_metric(std::unique_ptr<std::stringstream, deleter_noop>(&ss_metric))
  {
    std::shared_ptr<stan::callbacks::structured_writer> writer_draw_cnstrn_ptr
        = std::make_shared<stan::callbacks::csv_writer<
          std::stringstream, deleter_noop>>(std::move(writer_draw_cnstrn));
    dp.add_writer(stan::callbacks::info_type::DRAW_CONSTRAINED,
                  std::move(writer_draw_cnstrn_ptr));

    std::shared_ptr<stan::callbacks::structured_writer> writer_params_uncnstrn_ptr
        = std::make_shared<stan::callbacks::csv_writer<
          std::stringstream, deleter_noop>>(std::move(writer_params_uncnstrn));
    dp.add_writer(stan::callbacks::info_type::PARAMS_UNCONSTRAINED,
                  std::move(writer_params_uncnstrn_ptr));

    std::shared_ptr<stan::callbacks::structured_writer> writer_engine_state_ptr
        = std::make_shared<stan::callbacks::csv_writer<
          std::stringstream, deleter_noop>>(std::move(writer_engine_state));
    dp.add_writer(stan::callbacks::info_type::ENGINE_STATE,
                  std::move(writer_engine_state_ptr));

    std::shared_ptr<stan::callbacks::structured_writer> writer_log_prob_ptr
        = std::make_shared<stan::callbacks::csv_writer<
          std::stringstream, deleter_noop>>(std::move(writer_log_prob));
    dp.add_writer(stan::callbacks::info_type::LOG_PROB,
                  std::move(writer_log_prob_ptr));

    std::shared_ptr<stan::callbacks::structured_writer> writer_metric_ptr
        = std::make_shared<stan::callbacks::json_writer<
          std::stringstream, deleter_noop>>(std::move(writer_metric));
    dp.add_writer(stan::callbacks::info_type::METRIC,
                  std::move(writer_metric_ptr));
  }

  void SetUp() {
    ss_draw_cnstrn.str(std::string());
    ss_draw_cnstrn.clear();

    ss_params_uncnstrn.str(std::string());
    ss_params_uncnstrn.clear();

    ss_engine_state.str(std::string());
    ss_engine_state.clear();

    ss_log_prob.str(std::string());
    ss_log_prob.clear();

    ss_metric.str(std::string());
    ss_metric.clear();
  }

  void TearDown() {}

  std::stringstream model_ss;
  stan::io::empty_var_context empty_context;
  stan_model model;
  boost::ecuyer1988 rng;
  stan::test::unit::instrumented_interrupt interrupt;
  stan::test::unit::instrumented_logger logger;

  stan::mcmc::adapt_diag_e_nuts<stan_model, boost::ecuyer1988> sampler;
  int num_warmup, num_samples, num_thin, refresh;
  bool save_warmup;

  std::stringstream ss_draw_cnstrn;
  std::stringstream ss_params_uncnstrn;
  std::stringstream ss_engine_state;
  std::stringstream ss_log_prob;
  std::stringstream ss_metric;
  stan::callbacks::dispatcher dp;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer_draw_cnstrn;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer_params_uncnstrn;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer_engine_state;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer_log_prob;
  stan::callbacks::json_writer<std::stringstream, deleter_noop> writer_metric;
};

TEST_F(ServicesUtilRunAdaptiveSamplerDispatcher, run_defaults) {
  std::vector<double> init_params = {1.5, -0.5};

  stan::services::util::run_adaptive_sampler(
      sampler, model, init_params, num_warmup, num_samples, num_thin, refresh,
      save_warmup, rng, interrupt, logger, dp);

  ASSERT_FALSE(ss_draw_cnstrn.str().empty());
  std::cout << ss_draw_cnstrn.str() << std::endl;

  ASSERT_FALSE(ss_metric.str().empty());
  std::cout << ss_metric.str() << std::endl;

  ASSERT_FALSE(ss_params_uncnstrn.str().empty());
  std::cout << "params_uncnstrn" << std::endl;
  std::cout << ss_params_uncnstrn.str() << std::endl;

  ASSERT_FALSE(ss_log_prob.str().empty());
  std::cout << "log_prob" << std::endl;
  std::cout << ss_log_prob.str() << std::endl;

  ASSERT_FALSE(ss_engine_state.str().empty());
  std::cout << "engine_state" << std::endl;
  std::cout << ss_engine_state.str() << std::endl;

}

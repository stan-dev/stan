#include <stan/services/util/initialize.hpp>
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
      ss_metric(),
      ss_draw_cnstrn(),
      model(empty_context, 0, &model_ss),
      rng(stan::services::util::create_rng(0, 1)),
      sampler(model, rng),
      num_warmup(100),
      num_samples(100),
      num_thin(1),
      refresh(0),
      save_warmup(true),
      writer_metric(std::unique_ptr<std::stringstream, deleter_noop>(&ss_metric)),
      writer_draw_cnstrn(std::unique_ptr<std::stringstream, deleter_noop>(&ss_draw_cnstrn)),
      csv_valid_inits(std::unique_ptr<std::stringstream, deleter_noop>(&ss_valid_inits)) {}

  void SetUp() {
    ss_draw_cnstrn.str(std::string());
    ss_draw_cnstrn.clear();
    std::shared_ptr<stan::callbacks::structured_writer> writer_draw_cnstrn_ptr
        = std::make_shared<stan::callbacks::csv_writer<
          std::stringstream, deleter_noop>>(std::move(writer_draw_cnstrn));
    dp.add_writer(stan::callbacks::info_type::DRAW_CONSTRAINED,
                  std::move(writer_draw_cnstrn_ptr));

    ss_metric.str(std::string());
    ss_metric.clear();
    std::shared_ptr<stan::callbacks::structured_writer> writer_metric_ptr
        = std::make_shared<stan::callbacks::json_writer<std::stringstream, deleter_noop>>(std::move(writer_metric));
    dp.add_writer(stan::callbacks::info_type::METRIC,
                  std::move(writer_metric_ptr));

    ss_valid_inits.str(std::string());
    ss_valid_inits.clear();
    std::shared_ptr<stan::callbacks::structured_writer> csv_valid_inits_ptr
        = std::make_shared<stan::callbacks::csv_writer<std::stringstream,
                                                       deleter_noop>>(std::move(csv_valid_inits));
    dp.add_writer(stan::callbacks::info_type::VALID_INIT_PARAMS, std::move(csv_valid_inits_ptr));
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

  std::stringstream ss_valid_inits;
  std::stringstream ss_metric;
  std::stringstream ss_draw_cnstrn;
  stan::callbacks::dispatcher dp;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> csv_valid_inits;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer_draw_cnstrn;
  stan::callbacks::json_writer<std::stringstream, deleter_noop> writer_metric;
};

TEST_F(ServicesUtilRunAdaptiveSamplerDispatcher, run_defaults) {
  std::vector<double> init_params;
  init_params = stan::services::util::initialize(
      model, empty_context, rng, 2, false, logger, dp);

  stan::services::util::run_adaptive_sampler(
      sampler, model, init_params, num_warmup, num_samples, num_thin, refresh,
      save_warmup, rng, interrupt, logger, dp);

  ASSERT_FALSE(ss_draw_cnstrn.str().empty());
  std::cout << ss_draw_cnstrn.str() << std::endl;

  ASSERT_FALSE(ss_valid_inits.str().empty());
  std::cout << ss_valid_inits.str() << std::endl;

  ASSERT_FALSE(ss_metric.str().empty());
  std::cout << ss_metric.str() << std::endl;
}

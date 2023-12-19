#include <stan/services/sample/hmc_nuts_diag_e_adapt.hpp>
#include <stan/callbacks/csv_writer.hpp>
#include <stan/callbacks/dispatcher.hpp>
#include <stan/callbacks/json_writer.hpp>
#include <stan/callbacks/structured_writer.hpp>
#include <stan/io/empty_var_context.hpp>

#include <test/test-models/good/services/test_lp.hpp>
//#include <test/test-models/good/optimization/rosenbrock.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <test/unit/util.hpp>

#include <iostream>
#include <sstream>
#include <gtest/gtest.h>

class ServicesSampleHmcNutsDiagEAdaptDispatch : public testing::Test {
 public:
  ServicesSampleHmcNutsDiagEAdaptDispatch() :
      ss_draw_cnstrn(),
      ss_params_uncnstrn(),
      ss_engine(),
      ss_metric(),
      model(empty_context, 0, &model_ss),
      writer_draw_cnstrn(std::unique_ptr<std::stringstream, deleter_noop>(&ss_draw_cnstrn)),
      writer_params_uncnstrn(std::unique_ptr<std::stringstream, deleter_noop>(&ss_params_uncnstrn)),
      writer_engine(std::unique_ptr<std::stringstream, deleter_noop>(&ss_engine)),
      writer_metric(std::unique_ptr<std::stringstream, deleter_noop>(&ss_metric))
  {
    std::shared_ptr<stan::callbacks::structured_writer> writer_draw_cnstrn_ptr
        = std::make_shared<stan::callbacks::csv_writer<
          std::stringstream, deleter_noop>>(std::move(writer_draw_cnstrn));
    dp.add_writer(stan::callbacks::table_info_type::DRAW_CONSTRAIN,
                  std::move(writer_draw_cnstrn_ptr));

    std::shared_ptr<stan::callbacks::structured_writer> writer_params_uncnstrn_ptr
        = std::make_shared<stan::callbacks::csv_writer<
          std::stringstream, deleter_noop>>(std::move(writer_params_uncnstrn));
    dp.add_writer(stan::callbacks::table_info_type::PARAMS_UNCNSTRN,
                  std::move(writer_params_uncnstrn_ptr));

    std::shared_ptr<stan::callbacks::structured_writer> writer_engine_ptr
        = std::make_shared<stan::callbacks::csv_writer<
          std::stringstream, deleter_noop>>(std::move(writer_engine));
    dp.add_writer(stan::callbacks::table_info_type::DRAW_ENGINE,
                  std::move(writer_engine_ptr));

    std::shared_ptr<stan::callbacks::structured_writer> writer_metric_ptr
        = std::make_shared<stan::callbacks::json_writer<
          std::stringstream, deleter_noop>>(std::move(writer_metric));
    dp.add_writer(stan::callbacks::struct_info_type::INV_METRIC,
                  std::move(writer_metric_ptr));
  }

  void SetUp() {
    ss_draw_cnstrn.str(std::string());
    ss_draw_cnstrn.clear();

    ss_params_uncnstrn.str(std::string());
    ss_params_uncnstrn.clear();

    ss_engine.str(std::string());
    ss_engine.clear();

    ss_metric.str(std::string());
    ss_metric.clear();
  }

  void TearDown() {}

  std::stringstream model_ss;
  stan::io::empty_var_context empty_context;
  stan_model model;

  stan::callbacks::dispatcher dp;
  std::stringstream ss_draw_cnstrn;
  std::stringstream ss_params_uncnstrn;
  std::stringstream ss_metric;
  std::stringstream ss_engine;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer_engine;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer_draw_cnstrn;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer_params_uncnstrn;
  stan::callbacks::json_writer<std::stringstream, deleter_noop> writer_metric;
  //  std::stringstream ss_valid_inits;
};

TEST_F(ServicesSampleHmcNutsDiagEAdaptDispatch, dispatcher_checks) {
  unsigned int random_seed = 0;
  unsigned int chain = 1;
  double init_radius = 2;
  int num_warmup = 1000;
  int num_samples = 10;
  int num_thin = 1;
  bool save_warmup = false;
  int refresh = 0;
  double stepsize = 1;
  double stepsize_jitter = 0;
  int max_depth = 10;
  double delta = .8;
  double gamma = .05;
  double kappa = .75;
  double t0 = 10;
  unsigned int init_buffer = 75;
  unsigned int term_buffer = 50;
  unsigned int window = 100;
  stan::test::unit::instrumented_interrupt interrupt;
  stan::test::unit::instrumented_logger logger;
  EXPECT_EQ(interrupt.call_count(), 0);

  stan::services::sample::hmc_nuts_diag_e_adapt(
      model, empty_context, random_seed, chain, init_radius, num_warmup, num_samples,
      num_thin, save_warmup, refresh, stepsize, stepsize_jitter, max_depth,
      delta, gamma, kappa, t0, init_buffer, term_buffer, window, interrupt,
      logger, dp);

  //  ASSERT_FALSE(ss_valid_inits.str().empty());
  ASSERT_FALSE(ss_metric.str().empty());
  ASSERT_FALSE(ss_draw_cnstrn.str().empty());

  //  std::cout << "validated init param values" << std::endl;
  //  std::cout << ss_valid_inits.str() << std::endl << std::endl;
  std::cout << "metric" << std::endl;
  std::cout << ss_metric.str() << std::endl << std::endl;
  std::cout << "engine" << std::endl;
  std::cout << ss_engine.str() << std::endl << std::endl;
  std::cout << "draws constrained" << std::endl;
  std::cout << ss_draw_cnstrn.str() << std::endl << std::endl;
  //  std::cout << "params unconstrained" << std::endl;
  //  std::cout << ss_params_uncnstrn.str() << std::endl << std::endl;
}

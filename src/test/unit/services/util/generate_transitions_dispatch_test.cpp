#include <stan/services/util/generate_transitions.hpp>
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

class ServicesGenerateTransitionsDispatcher : public testing::Test {
 public: 
  ServicesGenerateTransitionsDispatcher() :
      ss_draw_cnstrn(),
      ss_params_uncnstrn(),
      ss_engine(),
      model(empty_context, 0, &model_ss),
      rng(stan::services::util::create_rng(0, 1)),
      sampler(model, rng),
      writer_draw_cnstrn(std::unique_ptr<std::stringstream, deleter_noop>(&ss_draw_cnstrn)),
      writer_params_uncnstrn(std::unique_ptr<std::stringstream, deleter_noop>(&ss_params_uncnstrn)),
      writer_engine(std::unique_ptr<std::stringstream, deleter_noop>(&ss_engine)) {}

  void SetUp() {
    ss_draw_cnstrn.str(std::string());
    ss_draw_cnstrn.clear();
    std::shared_ptr<stan::callbacks::structured_writer> writer_draw_cnstrn_ptr
        = std::make_shared<stan::callbacks::csv_writer<std::stringstream,
                                                       deleter_noop>>(std::move(writer_draw_cnstrn));
    dp.add_writer(stan::callbacks::table_info_type::DRAW_CONSTRAIN, std::move(writer_draw_cnstrn_ptr));

    ss_params_uncnstrn.str(std::string());
    ss_params_uncnstrn.clear();
    std::shared_ptr<stan::callbacks::structured_writer> writer_params_uncnstrn_ptr
        = std::make_shared<stan::callbacks::csv_writer<std::stringstream,
                                                       deleter_noop>>(std::move(writer_params_uncnstrn));
    dp.add_writer(stan::callbacks::table_info_type::PARAMS_UNCNSTRN, std::move(writer_params_uncnstrn_ptr));

    ss_engine.str(std::string());
    ss_engine.clear();
    std::shared_ptr<stan::callbacks::structured_writer> writer_engine_ptr
        = std::make_shared<stan::callbacks::csv_writer<std::stringstream,
                                                       deleter_noop>>(std::move(writer_engine));
    dp.add_writer(stan::callbacks::table_info_type::DRAW_ENGINE, std::move(writer_engine_ptr));
  }

  void TearDown() {}

  stan::io::empty_var_context empty_context;
  std::stringstream model_ss;
  stan_model model;
  boost::ecuyer1988 rng;
  stan::test::unit::instrumented_interrupt interrupt;
  stan::test::unit::instrumented_logger logger;

  stan::mcmc::adapt_diag_e_nuts<stan_model, boost::ecuyer1988> sampler;

  std::stringstream ss_draw_cnstrn;
  std::stringstream ss_params_uncnstrn;
  std::stringstream ss_engine;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer_draw_cnstrn;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer_params_uncnstrn;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer_engine;
  stan::callbacks::dispatcher dp;
};

TEST_F(ServicesGenerateTransitionsDispatcher, warmup) {
  double init_radius = 0;
  int num_iterations = 10;
  int refresh = 0;
  bool save_warmup = true;
  bool warmup = true;

  stan::test::unit::instrumented_interrupt interrupt;

  std::vector<int> disc_params;
  std::vector<double> init_params = {1.5, -0.5};
  Eigen::Map<Eigen::VectorXd> cont_params(init_params.data(),
                                          init_params.size());

  try {
    sampler.z().q = cont_params;
    sampler.init_stepsize(logger);
  } catch (const std::exception& e) {
    logger.error("Exception initializing step size.");
    logger.error(e.what());
    return;
  }
  stan::mcmc::sample s(cont_params, 0, 0);

  std::vector<std::string> constrained_names;
  model.constrained_param_names(constrained_names, true, true);
  dp.table_header(stan::callbacks::table_info_type::DRAW_CONSTRAIN, constrained_names);

  std::vector<std::string> unconstrained_names;
  model.unconstrained_param_names(unconstrained_names, false, false);
  dp.table_header(stan::callbacks::table_info_type::PARAMS_UNCNSTRN, unconstrained_names);

  std::vector<std::string> engine_names;
  s.get_sample_param_names(engine_names);
  sampler.get_sampler_param_names(engine_names);
  dp.table_header(stan::callbacks::table_info_type::DRAW_ENGINE, engine_names);

  sampler.engage_adaptation();
  stan::services::util::generate_transitions(sampler, num_iterations, 0, 20, 1,
                                             refresh, save_warmup, warmup, dp,
                                             s, model, rng, interrupt, logger);

  ASSERT_FALSE(ss_draw_cnstrn.str().empty());
  std::cout << ss_draw_cnstrn.str() << std::endl;

  ASSERT_FALSE(ss_params_uncnstrn.str().empty());
  std::cout << "params_uncnstrn" << std::endl;
  std::cout << ss_params_uncnstrn.str() << std::endl;

  // during warmup, stepsize changes
  ASSERT_FALSE(ss_engine.str().empty());
  std::cout << "engine" << std::endl;
  std::cout << ss_engine.str() << std::endl;
}

TEST_F(ServicesGenerateTransitionsDispatcher, sample) {
  double init_radius = 0;
  int num_iterations = 10;
  int refresh = 0;
  bool save_warmup = true;
  bool warmup = false;

  stan::test::unit::instrumented_interrupt interrupt;

  std::vector<int> disc_params;
  std::vector<double> init_params = {1.5, -0.5};
  Eigen::Map<Eigen::VectorXd> cont_params(init_params.data(),
                                          init_params.size());
  try {
    sampler.z().q = cont_params;
    sampler.init_stepsize(logger);
  } catch (const std::exception& e) {
    logger.error("Exception initializing step size.");
    logger.error(e.what());
    return;
  }
  stan::mcmc::sample s(cont_params, 0, 0);

  std::vector<std::string> constrained_names;
  model.constrained_param_names(constrained_names, true, true);
  dp.table_header(stan::callbacks::table_info_type::DRAW_CONSTRAIN, constrained_names);

  std::vector<std::string> unconstrained_names;
  model.unconstrained_param_names(unconstrained_names, false, false);
  dp.table_header(stan::callbacks::table_info_type::PARAMS_UNCNSTRN, unconstrained_names);

  std::vector<std::string> engine_names;
  s.get_sample_param_names(engine_names);
  sampler.get_sampler_param_names(engine_names);
  dp.table_header(stan::callbacks::table_info_type::DRAW_ENGINE, engine_names);

  sampler.disengage_adaptation();
  stan::services::util::generate_transitions(sampler, num_iterations, 0, 20, 1,
                                             refresh, save_warmup, warmup, dp,
                                             s, model, rng, interrupt, logger);

  // during sample, stepsize is constant
  ASSERT_FALSE(ss_engine.str().empty());
  std::cout << "engine" << std::endl;
  std::cout << ss_engine.str() << std::endl;
}

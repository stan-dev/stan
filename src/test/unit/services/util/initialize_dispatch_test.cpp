#include <stan/services/util/initialize.hpp>
#include <gtest/gtest.h>
#include <stan/callbacks/csv_writer.hpp>
#include <stan/callbacks/dispatcher.hpp>
#include <stan/callbacks/json_writer.hpp>
#include <stan/callbacks/structured_writer.hpp>
#include <test/unit/util.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <sstream>
#include <test/test-models/good/services/test_lp.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/io/array_var_context.hpp>
#include <stan/services/util/create_rng.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>


auto&& blah = stan::math::init_threadpool_tbb();

static constexpr size_t num_chains = 4;

struct deleter_noop {
  template <typename T>
  constexpr void operator()(T* arg) const {}
};

class ServicesUtilInitializeDispatcher : public testing::Test {
 public:
  ServicesUtilInitializeDispatcher()
      : message_ss(), model_ss(), ss_valid_inits(), 
        model(empty_context, 12345, &model_ss),
        message(message_ss),
        rng(stan::services::util::create_rng(0, 1)),
        writer_inits(std::unique_ptr<std::stringstream, deleter_noop>(&ss_valid_inits)) {}

  void SetUp() {
    ss_valid_inits.str(std::string());
    ss_valid_inits.clear();
    std::shared_ptr<stan::callbacks::structured_writer> writer_inits_ptr
        = std::make_shared<stan::callbacks::csv_writer<std::stringstream,
                                                       deleter_noop>>(std::move(writer_inits));
    dp.add_writer(stan::callbacks::info_type::VALID_INIT_PARAMS, std::move(writer_inits_ptr));
  }

  void TearDown() {}

  stan_model model;
  stan::io::empty_var_context empty_context;
  std::stringstream model_ss;
  std::stringstream message_ss;
  stan::callbacks::stream_writer message;
  stan::test::unit::instrumented_logger logger;
  boost::ecuyer1988 rng;

  std::stringstream ss_valid_inits;
  stan::callbacks::dispatcher dp;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer_inits;
};

TEST_F(ServicesUtilInitializeDispatcher, dispatch_capture) {
  double init_radius = 0;
  bool print_timing = false;
  std::vector<double> params;
  params = stan::services::util::initialize(
      model, empty_context, rng, init_radius, print_timing, logger, dp);
  ASSERT_FALSE(ss_valid_inits.str().empty());
  std::cout << ss_valid_inits.str() << std::endl;
}

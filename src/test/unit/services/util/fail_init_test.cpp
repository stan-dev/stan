#include <stan/services/util/initialize.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/io/array_var_context.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <test/test-models/good/services/test_fail.hpp>
#include <test/unit/util.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <gtest/gtest.h>
#include <sstream>

class ServicesUtilInitialize : public testing::Test {
 public:
  ServicesUtilInitialize()
      : model(empty_context, 12345, &model_ss),
        message(message_ss),
        rng(stan::services::util::create_rng(0, 1)) {}

  stan_model model;
  stan::io::empty_var_context empty_context;
  std::stringstream model_ss;
  std::stringstream message_ss;
  stan::callbacks::stream_writer message;
  stan::test::unit::instrumented_logger logger;
  stan::test::unit::instrumented_writer init;
  stan::rng_t rng;
};

TEST_F(ServicesUtilInitialize, model_throws__full_init) {
  std::vector<std::string> names_r;
  std::vector<double> values_r;
  std::vector<std::vector<size_t> > dim_r;
  names_r.push_back("y");
  values_r.push_back(6.35149);  // 1.5 unconstrained: -10 + 20 * inv.logit(1.5)
  values_r.push_back(-2.449187);  // -0.5 unconstrained
  std::vector<size_t> d;
  d.push_back(2);
  dim_r.push_back(d);
  stan::io::array_var_context init_context(names_r, values_r, dim_r);

  double init_radius = 2;
  bool print_timing = false;
  EXPECT_THROW(
      stan::services::util::initialize(model, init_context, rng, init_radius,
                                       print_timing, logger, init),
      std::domain_error);
  /* Uncomment to print all logs
  auto logs = logger.return_all_logs();
  for (auto&& m : logs) {
    std::cout << m << std::endl;
  }
  */
  EXPECT_EQ(6, logger.call_count());
  EXPECT_EQ(3, logger.call_count_warn());
  EXPECT_EQ(0, logger.find_warn("throwing within log_prob"));
}

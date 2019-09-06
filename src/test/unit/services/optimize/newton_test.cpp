#include <stan/services/optimize/newton.hpp>
#include <gtest/gtest.h>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/optimization/rosenbrock.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <stan/callbacks/stream_writer.hpp>


struct mock_callback : public stan::callbacks::interrupt {
  int n;
  mock_callback() : n(0) { }

  void operator()() {
    n++;
  }
};


class values
  : public stan::callbacks::stream_writer {
public:
  std::vector<std::string> names_;
  std::vector<std::vector<double> > states_;

  values(std::ostream& stream)
    : stan::callbacks::stream_writer(stream) {
  }

  /**
   * Writes a set of names.
   *
   * @param[in] names Names in a std::vector
   */
  void operator()(const std::vector<std::string>& names) {
    names_ = names;
  }

  /**
   * Writes a set of values.
   *
   * @param[in] state Values in a std::vector
   */
  void operator()(const std::vector<double>& state) {
    states_.push_back(state);
  }

};


class ServicesOptimizeNewton : public testing::Test {
public:
  ServicesOptimizeNewton()
    : init(init_ss),
      parameter(parameter_ss),
      model(context, &model_ss) {}

  std::stringstream init_ss, parameter_ss, model_ss;
  stan::test::unit::instrumented_logger logger;
  stan::callbacks::stream_writer init;
  values parameter;
  stan::io::empty_var_context context;
  stan_model model;
};


TEST_F(ServicesOptimizeNewton, rosenbrock) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;

  int num_interations = 1000;
  bool save_iterations = true;
  mock_callback callback;

  int return_code = stan::services::optimize::newton(model, context,
                                                     seed, chain, init_radius,
                                                     num_interations, save_iterations,
                                                     callback,
                                                     logger,
                                                     init,
                                                     parameter);

  EXPECT_EQ(0, return_code);
  EXPECT_EQ(logger.call_count(), logger.call_count_info()) << "all output to info";
  EXPECT_EQ(1, logger.find("Initial log joint probability = -1"));
  EXPECT_EQ(1, logger.find("Iteration  1. Log joint probability ="));

  ASSERT_EQ(3, parameter.names_.size());
  EXPECT_EQ("lp__", parameter.names_[0]);
  EXPECT_EQ("x", parameter.names_[1]);
  EXPECT_EQ("y", parameter.names_[2]);

  EXPECT_GT(parameter.states_.size(), 0);
  EXPECT_FLOAT_EQ(0, parameter.states_.front()[1])
    << "initial value should be (0, 0)";
  EXPECT_FLOAT_EQ(0, parameter.states_.front()[2])
    << "initial value should be (0, 0)";
  EXPECT_NEAR(1, parameter.states_.back()[1], 1e-3)
    << "optimal value should be (1, 1)";
  EXPECT_NEAR(1, parameter.states_.back()[2], 1e-3)
    << "optimal value should be (1, 1)";
  EXPECT_FLOAT_EQ(return_code, 0);
  EXPECT_GT(callback.n, 0);
}

TEST_F(ServicesOptimizeNewton, rosenbrock_no_save_iterations) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;

  int num_interations = 1000;
  bool save_iterations = false;
  mock_callback callback;

  int return_code = stan::services::optimize::newton(model, context,
                                                     seed, chain, init_radius,
                                                     num_interations, save_iterations,
                                                     callback,
                                                     logger,
                                                     init,
                                                     parameter);

  EXPECT_EQ(0, return_code);
  EXPECT_EQ(logger.call_count(), logger.call_count_info()) << "all output to info";
  EXPECT_EQ(1, logger.find("Initial log joint probability = -1"));
  EXPECT_EQ(1, logger.find("Iteration  1. Log joint probability ="));

  EXPECT_EQ("0,0\n", init_ss.str());

  ASSERT_EQ(3, parameter.names_.size());
  EXPECT_EQ("lp__", parameter.names_[0]);
  EXPECT_EQ("x", parameter.names_[1]);
  EXPECT_EQ("y", parameter.names_[2]);

  EXPECT_EQ(1, parameter.states_.size());
  EXPECT_NEAR(1, parameter.states_.back()[1], 1e-3)
    << "optimal value should be (1, 1)";
  EXPECT_NEAR(1, parameter.states_.back()[2], 1e-3)
    << "optimal value should be (1, 1)";
  EXPECT_FLOAT_EQ(return_code, 0);
  EXPECT_GT(callback.n, 0);
}

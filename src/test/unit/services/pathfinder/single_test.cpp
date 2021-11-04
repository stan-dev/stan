#include <stan/services/pathfinder/single.hpp>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/optimization/rosenbrock.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <gtest/gtest.h>

struct mock_callback : public stan::callbacks::interrupt {
  int n;
  mock_callback() : n(0) {}

  void operator()() { n++; }
};

class values : public stan::callbacks::stream_writer {
 public:
  std::vector<std::string> names_;
  std::vector<std::vector<double> > states_;
  Eigen::MatrixXd values_;
  values(std::ostream& stream) : stan::callbacks::stream_writer(stream) {}

  /**
   * Writes a set of names.
   *
   * @param[in] names Names in a std::vector
   */
  void operator()(const std::vector<std::string>& names) { names_ = names; }

  /**
   * Writes a set of values.
   *
   * @param[in] state Values in a std::vector
   */
  void operator()(const std::vector<double>& state) {
    states_.push_back(state);
  }
  void operator()(const Eigen::MatrixXd& vals) { values_ = vals; }
};

class ServicesPathfinderSingle : public testing::Test {
 public:
  ServicesPathfinderSingle()
      : init(init_ss),
        parameter(parameter_ss),
        diagnostics(diagnostic_ss),
        model(context, 0, &model_ss) {}

  std::stringstream init_ss, parameter_ss, diagnostic_ss, model_ss;
  stan::callbacks::stream_writer init;
  stan::test::unit::instrumented_logger logger;
  values parameter;
  values diagnostics;
  stan::io::empty_var_context context;
  stan_model model;
};

TEST_F(ServicesPathfinderSingle, rosenbrock) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;

  bool save_iterations = true;
  int refresh = 0;
  mock_callback callback;

  int return_code = stan::services::optimize::pathfinder_lbfgs_single(
      model, context, seed, chain, init_radius, 5, 0.001, 1e-12, 10000, 1e-8,
      10000000, 1e-8, 2000, save_iterations, refresh, callback, 500, 500, 1,
      logger, init, parameter, diagnostics);
  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, 0, ", ", ", ", "\n", "",
                               "", "");

  std::cout << "Values: \n"
            << parameter.values_.transpose().format(CommaInitFmt) << "\n";
  /*
    EXPECT_EQ(logger.call_count(), logger.call_count_info())
        << "all output to info";
    EXPECT_EQ(1, logger.find("Initial log joint probability = -1"));
    EXPECT_EQ(1, logger.find("Optimization terminated normally: "));
    EXPECT_EQ(1, logger.find("  Convergence detected: relative gradient "
                             "magnitude is below tolerance"));

    EXPECT_EQ("0,0\n", init_ss.str());

    ASSERT_EQ(3, parameter.names_.size());
    EXPECT_EQ("lp__", parameter.names_[0]);
    EXPECT_EQ("x", parameter.names_[1]);
    EXPECT_EQ("y", parameter.names_[2]);

    EXPECT_EQ(23, parameter.states_.size());
    EXPECT_FLOAT_EQ(0, parameter.states_.front()[1])
        << "initial value should be (0, 0)";
    EXPECT_FLOAT_EQ(0, parameter.states_.front()[2])
        << "initial value should be (0, 0)";
    EXPECT_FLOAT_EQ(0.99998301, parameter.states_.back()[1])
        << "optimal value should be (1, 1)";
    EXPECT_FLOAT_EQ(0.99996597, parameter.states_.back()[2])
        << "optimal value should be (1, 1)";
    EXPECT_FLOAT_EQ(return_code, 0);
    EXPECT_EQ(22, callback.n);
  */
}

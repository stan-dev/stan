#include <stan/services/sample/fixed_param.hpp>
#include <gtest/gtest.h>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/optimization/rosenbrock.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/noop_interrupt.hpp>

class memory_writer
  : public stan::callbacks::stream_writer {
public:
  std::vector<std::string> names_;
  std::vector<std::vector<double> > states_;

  memory_writer(std::ostream& stream)
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


class ServicesSamplesFixedParam : public testing::Test {
public:
  ServicesSamplesFixedParam()
    : message(message_ss),
      init(init_ss),
      error(error_ss),
      parameter(parameter_ss),
      diagnostic(diagnostic_ss),
      model(context, &model_ss) {}

  std::stringstream message_ss, init_ss, parameter_ss, model_ss;
  std::stringstream error_ss, diagnostic_ss;
  stan::callbacks::stream_writer message, init, error;
  memory_writer parameter, diagnostic;
  stan::io::empty_var_context context;
  stan_model model;
};


TEST_F(ServicesSamplesFixedParam, rosenbrock) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;

  int refresh = 0;
  stan::callbacks::noop_interrupt callback;
  
  int return_code = stan::services::sample::fixed_param(model, context,
                                                   seed, chain, init_radius,
                                                   20,
                                                   1,
                                                   refresh,
                                                   callback,
                                                   message,
                                                   init,
                                                   error,
                                                   parameter,
                                                   diagnostic);

  EXPECT_EQ("\n Elapsed Time: 0 seconds (Warm-up)\n", 
    message_ss.str().substr(0,36));

  EXPECT_EQ("0,0\n", init_ss.str());

  ASSERT_EQ(4, parameter.names_.size());
  EXPECT_EQ("lp__", parameter.names_[0]);
  EXPECT_EQ("accept_stat__", parameter.names_[1]);
  EXPECT_EQ("x", parameter.names_[2]);
  EXPECT_EQ("y", parameter.names_[3]);

  EXPECT_EQ(20, parameter.states_.size());
  EXPECT_FLOAT_EQ(0, parameter.states_.front()[1])
    << "initial memory_writer should be (0, 0)";
  EXPECT_FLOAT_EQ(0, parameter.states_.front()[2])
    << "initial memory_writer should be (0, 0)";
  EXPECT_FLOAT_EQ(0, parameter.states_.back()[1])
    << "final memory_writer should be (0, 0)";
  EXPECT_FLOAT_EQ(0, parameter.states_.back()[2])
    << "final memory_writer should be (0, 0)";
  EXPECT_FLOAT_EQ(return_code, 0);
}





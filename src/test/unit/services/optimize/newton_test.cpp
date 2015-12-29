#include <gtest/gtest.h>
#include <stan/services/optimize/newton.hpp>
#include <stan/interface_callbacks/writer/stream_writer.hpp>
#include <test/test-models/good/optimization/rosenbrock.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/additive_combine.hpp>

typedef boost::ecuyer1988 rng_t; // (2**50 = 1T samples, 1000 chains)
struct mock_callback {
  int n;
  mock_callback() : n(0) { }
  
  void operator()() {
    n++;
  }
};


class ServicesOptimizeNewton : public testing::Test {
public:
  ServicesOptimizeNewton()
    : dummy_context(data_stream),
      model(dummy_context),
      base_rng(0),
      cont_params(2),
      message_writer(message),
      parameter_writer(parameter) { }

  void SetUp() {
    base_rng = rng_t(0);
    cont_params[0] = -1;
    cont_params[1] = 1;

    message.str("");
    parameter.str("");
    interrupt.n = 0;
  }

  std::stringstream data_stream;
  stan::io::dump dummy_context;
  stan_model model;
  rng_t base_rng;
  Eigen::VectorXd cont_params;
  std::stringstream message;
  std::stringstream parameter;
  stan::interface_callbacks::writer::stream_writer message_writer;
  stan::interface_callbacks::writer::stream_writer parameter_writer;
  mock_callback interrupt;
};

TEST_F(ServicesOptimizeNewton, rosenbrock_save_interations) {
  int return_code;
  
  return_code = stan::services::optimize::newton(model, base_rng, cont_params,
                                                 1000, true,
                                                 interrupt,
                                                 message_writer,
                                                 parameter_writer);
  EXPECT_EQ(0, return_code);
  EXPECT_TRUE(interrupt.n > 1);
  EXPECT_TRUE(message.str().find("Initial log joint probability = -4") != std::string::npos);
  EXPECT_TRUE(message.str().find("Iteration  1. Log joint probability =") != std::string::npos);
  std::string parameter_str = parameter.str();
  EXPECT_TRUE(parameter_str.find("lp__,x,y\n") != std::string::npos);
  EXPECT_TRUE(count(parameter_str.begin(), parameter_str.end(), '\n') > 2);
  EXPECT_FLOAT_EQ(1, cont_params[0]);
  EXPECT_FLOAT_EQ(1, cont_params[1]);
}

TEST_F(ServicesOptimizeNewton, rosenbrock_no_save_interations) {
  int return_code;
  
  return_code = stan::services::optimize::newton(model, base_rng, cont_params,
                                                 1000, false,
                                                 interrupt,
                                                 message_writer,
                                                 parameter_writer);
  EXPECT_EQ(0, return_code);
  EXPECT_TRUE(interrupt.n > 1);
  EXPECT_TRUE(message.str().find("Initial log joint probability = -4") != std::string::npos);
  EXPECT_TRUE(message.str().find("Iteration  1. Log joint probability =") != std::string::npos);

  std::string parameter_str = parameter.str();
  EXPECT_TRUE(parameter_str.find("lp__,x,y\n") != std::string::npos);
  EXPECT_EQ(2, count(parameter_str.begin(), parameter_str.end(), '\n'));
  EXPECT_FLOAT_EQ(1, cont_params[0]);
  EXPECT_FLOAT_EQ(1, cont_params[1]);
}

#include <gtest/gtest.h>
#include <stan/interface_callbacks/writer/stream_writer.hpp>
#include <stan/services/optimize/bfgs.hpp>
#include <test/test-models/good/optimization/rosenbrock.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/additive_combine.hpp>
#include <test/unit/util.hpp>

typedef rosenbrock_model_namespace::rosenbrock_model Model;
typedef boost::ecuyer1988 rng_t; // (2**50 = 1T samples, 1000 chains)

struct mock_callback {
  int n;
  mock_callback() : n(0) { }
  
  void operator()() {
    n++;
  }
};

TEST(ServicesOptimizeBfgs, rosenbrock) {
  Eigen::VectorXd cont_params(2);
  cont_params[0] = -1; cont_params[1] = 1;

  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
  Model model(dummy_context);

  bool save_iterations = true;
  int refresh = 0;
  rng_t base_rng(0);

  mock_callback callback;

  std::stringstream message, parameter;
  stan::interface_callbacks::writer::stream_writer message_writer(message);
  stan::interface_callbacks::writer::stream_writer parameter_writer(parameter);
  int return_code = stan::services::optimize::bfgs(model, base_rng,
                                                   cont_params,
                                                   0.001,
                                                   1e-12,
                                                   10000,
                                                   1e-8,
                                                   10000000,
                                                   1e-8,
                                                   2000,
                                                   save_iterations, refresh,
                                                   callback,
                                                   message_writer, 
                                                   parameter_writer);
  EXPECT_EQ("Initial log joint probability = -4\nOptimization terminated normally: \n  Convergence detected: relative gradient magnitude is below tolerance\n", message.str());
  std::string parameter_str = parameter.str();
  EXPECT_TRUE(parameter_str.find("lp__,x,y\n") != std::string::npos);
  EXPECT_EQ(35, std::count(parameter_str.begin(), parameter_str.end(), '\n'));
  EXPECT_FLOAT_EQ(return_code, 0);
  EXPECT_EQ(33, callback.n);

  EXPECT_FLOAT_EQ(1, cont_params[0]);
  EXPECT_FLOAT_EQ(1, cont_params[1]);
}

TEST(ServicesOptimizeBfgs, rosenbrock_no_save_refresh) {
  Eigen::VectorXd cont_params(2);
  cont_params[0] = -1; cont_params[1] = 1;

  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
  Model model(dummy_context);

  bool save_iterations = false;
  int refresh = 1;
  rng_t base_rng(0);

  mock_callback callback;

  std::stringstream message, parameter;
  stan::interface_callbacks::writer::stream_writer message_writer(message);
  stan::interface_callbacks::writer::stream_writer parameter_writer(parameter);
  int return_code = stan::services::optimize::bfgs(model, base_rng,
                                                   cont_params,
                                                   0.001,
                                                   1e-12,
                                                   10000,
                                                   1e-8,
                                                   10000000,
                                                   1e-8,
                                                   2000,
                                                   save_iterations, refresh,
                                                   callback,
                                                   message_writer, 
                                                   parameter_writer);
  std::string message_str = message.str();
  EXPECT_TRUE(message_str.find("Initial log joint probability = -4\n") != std::string::npos);
  EXPECT_TRUE(std::count(message_str.begin(), message_str.end(), '\n') > 35);
  EXPECT_TRUE(message_str.find("Optimization terminated normally: \n  Convergence detected: relative gradient magnitude is below tolerance\n") != std::string::npos); 
  std::string parameter_str = parameter.str();
  EXPECT_TRUE(parameter_str.find("lp__,x,y\n") != std::string::npos);
  EXPECT_EQ(2, std::count(parameter_str.begin(), parameter_str.end(), '\n'));
  EXPECT_FLOAT_EQ(return_code, 0);
  EXPECT_EQ(33, callback.n);

  EXPECT_FLOAT_EQ(1, cont_params[0]);
  EXPECT_FLOAT_EQ(1, cont_params[1]);
}


TEST(ServicesOptimizeBfgs, rosenbrock_no_save_1_iteration) {
  Eigen::VectorXd cont_params(2);
  cont_params[0] = -1; cont_params[1] = 1;

  static const std::string DATA("");
  std::stringstream data_stream(DATA);
  stan::io::dump dummy_context(data_stream);
  Model model(dummy_context);

  bool save_iterations = false;
  int refresh = 0;
  rng_t base_rng(0);

  mock_callback callback;

  std::stringstream message, parameter;
  stan::interface_callbacks::writer::stream_writer message_writer(message);
  stan::interface_callbacks::writer::stream_writer parameter_writer(parameter);
  int return_code = stan::services::optimize::bfgs(model, base_rng,
                                                   cont_params,
                                                   0.001,
                                                   1e-12,
                                                   10000,
                                                   1e-8,
                                                   10000000,
                                                   1e-8,
                                                   1,
                                                   save_iterations, refresh,
                                                   callback,
                                                   message_writer, 
                                                   parameter_writer);
  EXPECT_TRUE(message.str().find("Initial log joint probability = -4\n") != std::string::npos);
  EXPECT_TRUE(message.str().find("Maximum number of iterations hit") != std::string::npos);
  std::string parameter_str = parameter.str();
  EXPECT_TRUE(parameter_str.find("lp__,x,y\n") != std::string::npos);
  EXPECT_EQ(2, std::count(parameter_str.begin(), parameter_str.end(), '\n'))  << parameter_str << std::endl;
  EXPECT_FLOAT_EQ(return_code, 0);
  EXPECT_EQ(1, callback.n);

  EXPECT_FLOAT_EQ(-0.99599999, cont_params[0]);
  EXPECT_FLOAT_EQ(1, cont_params[1]);
}

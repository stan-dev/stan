#include <stan/services/sample/fixed_param.hpp>
#include <gtest/gtest.h>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/optimization/rosenbrock.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/noop_interrupt.hpp>
#include <iostream>
#include <exception>

namespace stan {
  namespace callbacks {

      class check_interrupt: public interrupt {
      public:
        check_interrupt(int n, bool& called)
          : called_(called), n_(n), counter_(0) {
          called_ = false;
        }
        void operator()() {
          counter_++;
          if (counter_ > n_)
            called_ = true;
        }
      private:
        bool& called_;
        unsigned int n_;
        unsigned int counter_;
      };
  }
}



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


class ServicesSamplesGenerateTransitions : public testing::Test {
public:
  ServicesSamplesGenerateTransitions()
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


TEST_F(ServicesSamplesGenerateTransitions, rosenbrock) {
  unsigned int seed = 0;
  unsigned int chain = 1;
  double init_radius = 0;

  int refresh = 0;
  bool called = false;
  stan::callbacks::check_interrupt callback(10, called);
  EXPECT_EQ(called, false);

  boost::ecuyer1988 rng = stan::services::util::rng(seed, chain);

  std::vector<int> disc_vector;
  std::vector<double> cont_vector
    = stan::services::util::initialize(model, context, rng, init_radius,
                                       false,
                                       message, diagnostic);

  stan::mcmc::fixed_param_sampler sampler;
  stan::services::sample::mcmc_writer
    writer(parameter, diagnostic, message);
  Eigen::VectorXd cont_params(cont_vector.size());
  for (size_t i = 0; i < cont_vector.size(); i++)
    cont_params[i] = cont_vector[i];
  stan::mcmc::sample s(cont_params, 0, 0);

  // Headers
  writer.write_sample_names(s, sampler, model);
  writer.write_diagnostic_names(s, sampler, model);

  stan::services::util::generate_transitions(
    sampler, 20, 0, 20, 1, refresh, true, false, writer,
    s, model, rng, callback, message, error);

  EXPECT_EQ("", message_ss.str());
  EXPECT_EQ("", error_ss.str());
  EXPECT_EQ("lp__", parameter.names_[0]);
  EXPECT_EQ("accept_stat__", parameter.names_[1]);
  EXPECT_EQ("x", parameter.names_[2]);
  EXPECT_EQ("y", parameter.names_[3]);

  for (size_t i=0; i < parameter.states_.size(); i++) {
    for (size_t j=0; j < parameter.names_.size(); j++) {
      EXPECT_FLOAT_EQ(0.0, parameter.states_[i][j]);
    }
  }

  EXPECT_EQ(called, true);
}





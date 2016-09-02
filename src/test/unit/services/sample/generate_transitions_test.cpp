#include <gtest/gtest.h>
#include <stan/services/sample/generate_transitions.hpp>
#include <test/test-models/good/services/test_lp.hpp>
#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/interface_callbacks/writer/stream_writer.hpp>
#include <boost/random/additive_combine.hpp>
#include <sstream>

typedef boost::ecuyer1988 rng_t;
typedef stan::interface_callbacks::writer::stream_writer writer_t;

class mock_sampler : public stan::mcmc::base_mcmc {
public:
  mock_sampler()
    : base_mcmc(), n_transition_called(0) { }

  stan::mcmc::sample transition(stan::mcmc::sample& init_sample,
                                stan::interface_callbacks::writer::base_writer& info_writer,
                                stan::interface_callbacks::writer::base_writer& error_writer) {
    
    n_transition_called++;
    return init_sample;
  }

  int n_transition_called;
};

struct mock_callback {
  int n;
  mock_callback() : n(0) { }
  
  void operator()() {
    n++;
  }
};

class StanServices : public testing::Test {
public:
  StanServices()
    : message_writer(message_output, "# "),
      error_writer(error_output, "# ") { }
  
  void SetUp() {
    model_output.str("");
    sample_output.str("");
    diagnostic_output.str("");
    message_output.str("");
    error_output.str("");

    sampler = new mock_sampler();

    std::fstream empty_data_stream(std::string("").c_str());
    stan::io::dump empty_data_context(empty_data_stream);
    empty_data_stream.close();
    
    model = new stan_model(empty_data_context, &model_output);
    
    writer_t sample_writer(sample_output, "# ");
    writer_t diagnostic_writer(diagnostic_output, "# ");
  
    writer = new stan::services::sample::mcmc_writer<stan_model,
                                                     writer_t,
                                                     writer_t,
                                                     writer_t>
      (sample_writer, diagnostic_writer, message_writer);

    base_rng.seed(123456);

    q = Eigen::VectorXd(0,1);
    log_prob = 0;
    stat = 0;
  }
  void TearDown() {
    delete sampler;
    delete model;
    delete writer;
  }
  
  mock_sampler* sampler;
  stan_model* model;
  stan::services::sample::mcmc_writer<stan_model,
                                      writer_t,
                                      writer_t,
                                      writer_t>* writer;
  rng_t base_rng;

  Eigen::VectorXd q;
  double log_prob;
  double stat;

  std::stringstream model_output;
  std::stringstream sample_output;
  std::stringstream diagnostic_output;
  std::stringstream message_output;
  std::stringstream error_output;
  stan::interface_callbacks::writer::stream_writer message_writer;
  stan::interface_callbacks::writer::stream_writer error_writer;
};


TEST_F(StanServices, generate_transitions) {
  std::string expected_output = "Iteration: 50 / 100 [ 50%]  (Sampling)\nIteration: 53 / 100 [ 53%]  (Sampling)\nIteration: 57 / 100 [ 57%]  (Sampling)\nIteration: 61 / 100 [ 61%]  (Sampling)\nIteration: 65 / 100 [ 65%]  (Sampling)\nIteration: 69 / 100 [ 69%]  (Sampling)\nIteration: 73 / 100 [ 73%]  (Sampling)\nIteration: 77 / 100 [ 77%]  (Sampling)\nIteration: 81 / 100 [ 81%]  (Sampling)\nIteration: 85 / 100 [ 85%]  (Sampling)\nIteration: 89 / 100 [ 89%]  (Sampling)\nIteration: 93 / 100 [ 93%]  (Sampling)\nIteration: 97 / 100 [ 97%]  (Sampling)\nIteration: 100 / 100 [100%]  (Sampling)\n";

  int num_iterations = 51;
  int start = 49;
  int finish = 100;
  int num_thin = 2;
  int refresh = 4;
  bool save = false;
  bool warmup = false;
  stan::mcmc::sample s(q, log_prob, stat);
  std::string prefix = "";
  std::string suffix = "\n";
  std::stringstream ss;
  mock_callback callback;

  stan::services::sample::generate_transitions(sampler,
                                               num_iterations, start, finish,
                                               num_thin, refresh, save, warmup,
                                               *writer, s, *model, base_rng,
                                               prefix, suffix, ss,
                                               callback,
                                               message_writer,
                                               error_writer);
  
  EXPECT_EQ(num_iterations, sampler->n_transition_called);
  EXPECT_EQ(num_iterations, callback.n);

  EXPECT_EQ(expected_output, ss.str());

  EXPECT_EQ("", model_output.str());
  EXPECT_EQ("", sample_output.str());
  EXPECT_EQ("", diagnostic_output.str());
  EXPECT_EQ("", message_output.str());
  EXPECT_EQ("", error_output.str());
}


#include <stan/services/mcmc/sample.hpp>
#include <gtest/gtest.h>
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

TEST_F(StanServices, sample) {
  std::string expected_sample_output = "Iteration: 31 / 80 [ 38%]  (Sampling)\nIteration: 34 / 80 [ 42%]  (Sampling)\nIteration: 38 / 80 [ 47%]  (Sampling)\nIteration: 42 / 80 [ 52%]  (Sampling)\nIteration: 46 / 80 [ 57%]  (Sampling)\nIteration: 50 / 80 [ 62%]  (Sampling)\nIteration: 54 / 80 [ 67%]  (Sampling)\nIteration: 58 / 80 [ 72%]  (Sampling)\nIteration: 62 / 80 [ 77%]  (Sampling)\nIteration: 66 / 80 [ 82%]  (Sampling)\nIteration: 70 / 80 [ 87%]  (Sampling)\nIteration: 74 / 80 [ 92%]  (Sampling)\nIteration: 78 / 80 [ 97%]  (Sampling)\nIteration: 80 / 80 [100%]  (Sampling)\n";
  
  int num_warmup = 30;
  int num_samples = 50;
  int num_thin = 2;
  int refresh = 4;
  bool save = false;
  stan::mcmc::sample s(q, log_prob, stat);
  std::string prefix = "";
  std::string suffix = "\n";
  std::stringstream ss;
  mock_callback callback;

  stan::services::mcmc::sample(sampler,
                               num_warmup, num_samples,
                               num_thin, refresh, save,
                               *writer, s, *model, base_rng,
                               prefix, suffix, ss,
                               callback,
                               message_writer,
                               error_writer);
  

  EXPECT_EQ(num_samples, sampler->n_transition_called);
  EXPECT_EQ(num_samples, callback.n);

  EXPECT_EQ(expected_sample_output, ss.str());

  EXPECT_EQ("", model_output.str());
  EXPECT_EQ("", sample_output.str());
  EXPECT_EQ("", diagnostic_output.str());
  EXPECT_EQ("", message_output.str());
  EXPECT_EQ("", error_output.str());
}


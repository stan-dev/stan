#include <stan/mcmc/base_mcmc.hpp>
#include <stan/services/sample/generate_transitions.hpp>
#include <stan/interface_callbacks/interrupt/base_interrupt.hpp>
#include <stan/interface_callbacks/writer/stringstream.hpp>
#include <gtest/gtest.h>
#include <test/test-models/good/services/test_lp.hpp>
#include <sstream>

typedef boost::ecuyer1988 rng_t;
typedef stan::interface_callbacks::writer::stringstream writer_t;

class mock_sampler : public stan::mcmc::base_mcmc {
public:
  mock_sampler()
  : base_mcmc(), n_transition_called_(0) { }
  
  stan::mcmc::sample transition(stan::mcmc::sample& init_sample) {
    n_transition_called_++;
    return init_sample;
  }
  
  int n_transition_called() {
    return n_transition_called_;
  }
  
private:
  int n_transition_called_;
};

class mock_interrupt: public stan::interface_callbacks::interrupt::base_interrupt {
public:
  mock_interrupt(): n_(0) {}
  
  void operator()() {
    n_++;
  }
  
  void clear() {
    n_ = 0;
  }
  
  int n() {
    return n_;
  }
  
private:
  int n_;
};

class StanServices : public testing::Test {
public:
  void SetUp() {
    
    model_output.str("");
    output.clear();
    diagnostic.clear();
    info.clear();
    
    base_rng.seed(123456);
    
    sampler = new mock_sampler();
    
    std::fstream empty_data_stream(std::string("").c_str(), std::fstream::in);
    stan::io::dump empty_data_context(empty_data_stream);
    empty_data_stream.close();
    
    model = new stan_model(empty_data_context, &model_output);
    
    writer = new stan::services::sample::mcmc_writer
      <stan_model, rng_t, writer_t, writer_t, writer_t>
        (*model, base_rng, output, diagnostic, info);
    
    q = Eigen::VectorXd::Zero(4);
    log_prob = 0;
    stat = 0;
  }
  void TearDown() {
    delete sampler;
    delete model;
    delete writer;
  }
  
  std::stringstream model_output;
  
  writer_t output;
  writer_t diagnostic;
  writer_t info;
  
  rng_t base_rng;
  
  mock_sampler* sampler;
  stan_model* model;
  stan::services::sample::mcmc_writer
    <stan_model, rng_t, writer_t, writer_t, writer_t>* writer;
  
  Eigen::VectorXd q;
  double log_prob;
  double stat;
  
};

TEST_F(StanServices, Warmup) {
  
  std::string expected_output = "";
  std::string expected_diagnostic = "";
  std::string expected_info = "Iteration:  1 / 60 [  1%]  (Warmup)\nIteration:  4 / 60 [  6%]  (Warmup)\nIteration:  8 / 60 [ 13%]  (Warmup)\nIteration: 12 / 60 [ 20%]  (Warmup)\nIteration: 16 / 60 [ 26%]  (Warmup)\nIteration: 20 / 60 [ 33%]  (Warmup)\nIteration: 24 / 60 [ 40%]  (Warmup)\nIteration: 28 / 60 [ 46%]  (Warmup)\n";
  
  int num_warmup = 30;
  int num_samples = 60;
  int num_thin = 2;
  int refresh = 4;
  
  stan::mcmc::sample s(q, log_prob, stat);
  
  mock_interrupt interrupt;
  
  stan::services::sample::generate_transitions(*sampler, s,
                                               num_warmup, 0, num_samples,
                                               num_thin, refresh, false,
                                               true, *writer, interrupt);
  
  EXPECT_EQ(num_warmup, sampler->n_transition_called());
  EXPECT_EQ(num_warmup, interrupt.n());
  
  EXPECT_EQ(expected_info, info.contents());
  
  EXPECT_EQ("", model_output.str());
  EXPECT_EQ(expected_output, output.contents());
  EXPECT_EQ(expected_diagnostic, diagnostic.contents());
}

TEST_F(StanServices, Sample) {
  
  std::string expected_output = "0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n";
  std::string expected_diagnostic = "0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n";
  std::string expected_info = "Iteration: 31 / 90 [ 34%]  (Sampling)\nIteration: 34 / 90 [ 37%]  (Sampling)\nIteration: 38 / 90 [ 42%]  (Sampling)\nIteration: 42 / 90 [ 46%]  (Sampling)\nIteration: 46 / 90 [ 51%]  (Sampling)\nIteration: 50 / 90 [ 55%]  (Sampling)\nIteration: 54 / 90 [ 60%]  (Sampling)\nIteration: 58 / 90 [ 64%]  (Sampling)\nIteration: 62 / 90 [ 68%]  (Sampling)\nIteration: 66 / 90 [ 73%]  (Sampling)\nIteration: 70 / 90 [ 77%]  (Sampling)\nIteration: 74 / 90 [ 82%]  (Sampling)\nIteration: 78 / 90 [ 86%]  (Sampling)\nIteration: 82 / 90 [ 91%]  (Sampling)\nIteration: 86 / 90 [ 95%]  (Sampling)\nIteration: 90 / 90 [100%]  (Sampling)\n";
  
  int num_warmup = 30;
  int num_samples = 60;
  int num_thin = 2;
  int refresh = 4;
  
  stan::mcmc::sample s(q, log_prob, stat);
  
  mock_interrupt interrupt;
  
  stan::services::sample::generate_transitions(*sampler, s,
                                               num_samples, num_warmup, num_warmup + num_samples,
                                               num_thin, refresh, true,
                                               false, *writer, interrupt);
  
  EXPECT_EQ(num_samples, sampler->n_transition_called());
  EXPECT_EQ(num_samples, interrupt.n());
  
  EXPECT_EQ(expected_info, info.contents());
  
  EXPECT_EQ("", model_output.str());
  EXPECT_EQ(expected_output, output.contents());
  EXPECT_EQ(expected_diagnostic, diagnostic.contents());
}


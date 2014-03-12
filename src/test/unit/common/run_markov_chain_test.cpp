#include <gtest/gtest.h>
#include <stan/common/run_markov_chain.hpp>
#include <test/test-models/no-main/common/test_lp.cpp>

typedef boost::ecuyer1988 rng_t;

class mock_sampler : public stan::mcmc::base_mcmc {
public:
  mock_sampler()
    : base_mcmc(&std::cout, &std::cout), n_transition_called(0) { }

  stan::mcmc::sample transition(stan::mcmc::sample& init_sample) {
    n_transition_called++;
    return init_sample;
  }

  int n_transition_called;
};


class StanCommon : public testing::Test {
public:
  void SetUp() {
    sampler = new mock_sampler;

    std::fstream empty_data_stream(std::string("").c_str());
    stan::io::dump empty_data_context(empty_data_stream);
    empty_data_stream.close();
    
    model = new stan_model(empty_data_context, &std::cout);
    
    writer = new stan::io::mcmc_writer<stan_model>(&std::cout, &std::cout, &std::cout);

    base_rng.seed(123456);
  }
  void TearDown() {
    delete sampler;
    delete model;
    delete writer;
  }
  
  mock_sampler* sampler;
  stan_model* model;
  stan::io::mcmc_writer<stan_model>* writer;
  rng_t base_rng;
};


TEST_F(StanCommon, run_markov_chain) {
  std::stringstream out;
  std::stringstream redirect_cout;
  std::streambuf* old_cout_rdbuf = std::cout.rdbuf(redirect_cout.rdbuf());

  std::string expected_std_cout = "Iteration: 50 / 100 [ 50%]  (Sampling)\nIteration: 53 / 100 [ 53%]  (Sampling)\nIteration: 57 / 100 [ 57%]  (Sampling)\nIteration: 61 / 100 [ 61%]  (Sampling)\nIteration: 65 / 100 [ 65%]  (Sampling)\nIteration: 69 / 100 [ 69%]  (Sampling)\nIteration: 73 / 100 [ 73%]  (Sampling)\nIteration: 77 / 100 [ 77%]  (Sampling)\nIteration: 81 / 100 [ 81%]  (Sampling)\nIteration: 85 / 100 [ 85%]  (Sampling)\nIteration: 89 / 100 [ 89%]  (Sampling)\nIteration: 93 / 100 [ 93%]  (Sampling)\nIteration: 97 / 100 [ 97%]  (Sampling)\nIteration: 100 / 100 [100%]  (Sampling)\n";

  int num_iterations = 51;
  int start = 49;
  int finish = 100;
  int num_thin = 2;
  int refresh = 4;
  bool save = false;
  bool warmup = false;
  Eigen::VectorXd q(0, 1);
  double log_prob = 0;
  double stat = 0;
  stan::mcmc::sample s(q, log_prob, stat);
  stan::common::run_markov_chain(sampler,
                                 num_iterations, start, finish,
                                 num_thin, refresh, save, warmup,
                                 *writer, s, *model, base_rng);
  
  EXPECT_EQ(num_iterations, sampler->n_transition_called);

  std::cout.rdbuf(old_cout_rdbuf);
  EXPECT_EQ(expected_std_cout, redirect_cout.str());
}


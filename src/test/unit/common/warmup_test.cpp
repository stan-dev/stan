#include <stan/common/warmup.hpp>
#include <gtest/gtest.h>
#include <test/test-models/no-main/common/test_lp.cpp>
#include <sstream>

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
  stan::io::mcmc_writer<stan_model>* writer;
  rng_t base_rng;

  Eigen::VectorXd q;
  double log_prob;
  double stat;
};


TEST_F(StanCommon, sample) {
  std::stringstream out;
  std::stringstream redirect_cout;
  std::streambuf* old_cout_rdbuf = std::cout.rdbuf(redirect_cout.rdbuf());

  std::string expected_std_cout =  "Iteration:  1 / 100 [  1%]  (Warmup)\nIteration:  4 / 100 [  4%]  (Warmup)\nIteration:  8 / 100 [  8%]  (Warmup)\nIteration: 12 / 100 [ 12%]  (Warmup)\nIteration: 16 / 100 [ 16%]  (Warmup)\nIteration: 20 / 100 [ 20%]  (Warmup)\nIteration: 24 / 100 [ 24%]  (Warmup)\nIteration: 28 / 100 [ 28%]  (Warmup)\nIteration: 32 / 100 [ 32%]  (Warmup)\nIteration: 36 / 100 [ 36%]  (Warmup)\nIteration: 40 / 100 [ 40%]  (Warmup)\nIteration: 44 / 100 [ 44%]  (Warmup)\nIteration: 48 / 100 [ 48%]  (Warmup)\n";
  
  int num_warmup = 50;
  int num_samples = 50;
  int num_thin = 2;
  int refresh = 4;
  bool save = false;
  stan::mcmc::sample s(q, log_prob, stat);
  std::string prefix = "";
  std::string suffix = "\n";
  std::stringstream ss;

  stan::common::warmup(sampler,
                       num_warmup, num_samples,
                       num_thin, refresh, save,
                       *writer, s, *model, base_rng,
                       prefix, suffix, ss);
  
  EXPECT_EQ(num_samples, sampler->n_transition_called);

  std::cout.rdbuf(old_cout_rdbuf);
  EXPECT_EQ(expected_std_cout, ss.str());
  EXPECT_EQ("", redirect_cout.str())
    << "There should be no output to std::cout";
}



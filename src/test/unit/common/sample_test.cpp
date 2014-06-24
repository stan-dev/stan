#include <stan/common/sample.hpp>
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

struct mock_callback {
  int n;
  mock_callback() : n(0) { }
  
  void operator()() {
    n++;
  }
};

class StanCommon : public testing::Test {
public:
  void SetUp() {
    sampler = new mock_sampler;

    std::fstream empty_data_stream(std::string("").c_str());
    stan::io::dump empty_data_context(empty_data_stream);
    empty_data_stream.close();
    
    model = new stan_model(empty_data_context, &std::cout);

    stan::common::recorder::csv sample_recorder(&std::cout, "# ");
    stan::common::recorder::csv diagnostic_recorder(&std::cout, "# ");
    stan::common::recorder::messages message_recorder(&std::cout, "# ");
    
    writer = new stan::io::mcmc_writer<stan_model,
                                       stan::common::recorder::csv,
                                       stan::common::recorder::csv,
                                       stan::common::recorder::messages>
      (sample_recorder, diagnostic_recorder, message_recorder, &std::cout);

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
  stan::io::mcmc_writer<stan_model,
                        stan::common::recorder::csv,
                        stan::common::recorder::csv,
                        stan::common::recorder::messages>* writer;
  rng_t base_rng;

  Eigen::VectorXd q;
  double log_prob;
  double stat;
};


TEST_F(StanCommon, sample) {
  std::stringstream out;
  std::stringstream redirect_cout;
  std::streambuf* old_cout_rdbuf = std::cout.rdbuf(redirect_cout.rdbuf());

  std::string expected_std_cout = "Iteration: 31 / 80 [ 38%]  (Sampling)\nIteration: 34 / 80 [ 42%]  (Sampling)\nIteration: 38 / 80 [ 47%]  (Sampling)\nIteration: 42 / 80 [ 52%]  (Sampling)\nIteration: 46 / 80 [ 57%]  (Sampling)\nIteration: 50 / 80 [ 62%]  (Sampling)\nIteration: 54 / 80 [ 67%]  (Sampling)\nIteration: 58 / 80 [ 72%]  (Sampling)\nIteration: 62 / 80 [ 77%]  (Sampling)\nIteration: 66 / 80 [ 82%]  (Sampling)\nIteration: 70 / 80 [ 87%]  (Sampling)\nIteration: 74 / 80 [ 92%]  (Sampling)\nIteration: 78 / 80 [ 97%]  (Sampling)\nIteration: 80 / 80 [100%]  (Sampling)\n";
  
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

  stan::common::sample(sampler,
                       num_warmup, num_samples,
                       num_thin, refresh, save,
                       *writer, s, *model, base_rng,
                       prefix, suffix, ss,
                       callback);
  
  EXPECT_EQ(num_samples, sampler->n_transition_called);
  EXPECT_EQ(num_samples, callback.n);

  std::cout.rdbuf(old_cout_rdbuf);
  EXPECT_EQ(expected_std_cout, ss.str());
  EXPECT_EQ("", redirect_cout.str())
    << "There should be no output to std::cout";
}


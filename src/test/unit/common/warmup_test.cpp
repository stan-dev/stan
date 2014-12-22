#include <stan/common/warmup.hpp>
#include <gtest/gtest.h>
#include <test/test-models/good/common/test_lp.cpp>
#include <sstream>

typedef boost::ecuyer1988 rng_t;

class mock_sampler : public stan::mcmc::base_mcmc {
public:
  mock_sampler(std::ostream *output, std::ostream *error)
    : base_mcmc(output, error), n_transition_called(0) { }

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
    output.str("");
    error.str("");

    model_output.str("");
    sample_output.str("");
    diagnostic_output.str("");
    message_output.str("");
    writer_output.str("");

    sampler = new mock_sampler(&output, &error);

    std::fstream empty_data_stream(std::string("").c_str());
    stan::io::dump empty_data_context(empty_data_stream);
    empty_data_stream.close();
    
    model = new stan_model(empty_data_context, &model_output);
    
    stan::common::recorder::csv sample_recorder(&sample_output, "# ");
    stan::common::recorder::csv diagnostic_recorder(&diagnostic_output, "# ");
    stan::common::recorder::messages message_recorder(&message_output, "# ");

    writer = new stan::io::mcmc_writer<stan_model,
                                       stan::common::recorder::csv,
                                       stan::common::recorder::csv,
                                       stan::common::recorder::messages>
      (sample_recorder, diagnostic_recorder, message_recorder, &writer_output);

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

  std::stringstream output, error;

  std::stringstream model_output,
    sample_output, diagnostic_output, message_output,
    writer_output;
};


TEST_F(StanCommon, warmup) {
  std::string expected_warmup_output =  "Iteration:  1 / 80 [  1%]  (Warmup)\nIteration:  4 / 80 [  5%]  (Warmup)\nIteration:  8 / 80 [ 10%]  (Warmup)\nIteration: 12 / 80 [ 15%]  (Warmup)\nIteration: 16 / 80 [ 20%]  (Warmup)\nIteration: 20 / 80 [ 25%]  (Warmup)\nIteration: 24 / 80 [ 30%]  (Warmup)\nIteration: 28 / 80 [ 35%]  (Warmup)\n";
  
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

  stan::common::warmup(sampler,
                       num_warmup, num_samples,
                       num_thin, refresh, save,
                       *writer, s, *model, base_rng,
                       prefix, suffix, ss,
                       callback);
  
  EXPECT_EQ(num_warmup, sampler->n_transition_called);
  EXPECT_EQ(num_warmup, callback.n);

  EXPECT_EQ(expected_warmup_output, ss.str());

  EXPECT_EQ("", output.str());
  EXPECT_EQ("", error.str());

  EXPECT_EQ("", model_output.str());
  EXPECT_EQ("", sample_output.str());
  EXPECT_EQ("", diagnostic_output.str());
  EXPECT_EQ("", message_output.str());
  EXPECT_EQ("", writer_output.str());
}



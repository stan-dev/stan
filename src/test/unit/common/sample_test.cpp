#include <stan/common/sample.hpp>
#include <gtest/gtest.h>
#include <sstream>
#include <test/test-models/no-main/common/test_lp.cpp>
#include <test/unit/common/mock_instantiation.hpp>

typedef boost::ecuyer1988 rng_t;


TEST_F(StanCommon, sample) {
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

  stan::common::sample(sampler,
                       num_warmup, num_samples,
                       num_thin, refresh, save,
                       *writer, s, *model, base_rng,
                       prefix, suffix, ss,
                       callback);
  
  EXPECT_EQ(num_samples, sampler->n_transition_called);
  EXPECT_EQ(num_samples, callback.n);

  EXPECT_EQ(expected_sample_output, ss.str());

  EXPECT_EQ("", output.str());
  EXPECT_EQ("", error.str());

  EXPECT_EQ("", model_output.str());
  EXPECT_EQ("", sample_output.str());
  EXPECT_EQ("", diagnostic_output.str());
  EXPECT_EQ("", message_output.str());
  EXPECT_EQ("", writer_output.str());
}


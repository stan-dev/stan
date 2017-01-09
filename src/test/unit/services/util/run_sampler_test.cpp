#include <stan/services/util/run_sampler.hpp>
#include <gtest/gtest.h>
#include <test/test-models/good/services/test_lp.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/services/util/create_rng.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>

class mock_sampler : public stan::mcmc::base_mcmc {
public:
  int n_transition;
  int n_get_sampler_param_names;
  int n_get_sampler_params;
  int n_write_sampler_state;
  int n_get_sampler_diagnostic_names;
  int n_get_sampler_diagnostics;

  mock_sampler() {
    reset();
  }

  void reset() {
    n_transition = 0;
    n_get_sampler_param_names = 0;
    n_get_sampler_params = 0;
    n_write_sampler_state = 0;
    n_get_sampler_diagnostic_names = 0;
    n_get_sampler_diagnostics = 0;
  }
  
  stan::mcmc::sample
  transition(stan::mcmc::sample& init_sample,
             stan::callbacks::writer& info_writer,
             stan::callbacks::writer& error_writer) {
    ++n_transition;
    stan::mcmc::sample result(init_sample);
    return result;
  }

  void get_sampler_param_names(std::vector<std::string>& names) {
    ++n_get_sampler_param_names;
  }

  void get_sampler_params(std::vector<double>& values) {
    ++n_get_sampler_params;
  }

  void write_sampler_state(stan::callbacks::writer& writer) {
    ++n_write_sampler_state;
  }

  void get_sampler_diagnostic_names(std::vector<std::string>& model_names,
                                    std::vector<std::string>& names) {
    ++n_get_sampler_diagnostic_names;
  }

  void get_sampler_diagnostics(std::vector<double>& values) {
    ++n_get_sampler_diagnostics;
  }
};

class ServicesUtil : public testing::Test {
public:
  ServicesUtil()
    : model(context, &model_log),
      rng(stan::services::util::create_rng(0, 1)),
      num_warmup(0),
      num_samples(0),
      num_thin(1),
      refresh(0),
      save_warmup(false) {
    cont_vector.push_back(0);
    cont_vector.push_back(0);
  }

  std::stringstream model_log;
  stan::io::empty_var_context context;
  stan_model model;
  std::vector<double> cont_vector;
  boost::ecuyer1988 rng;
  stan::test::unit::instrumented_interrupt interrupt;
  stan::test::unit::instrumented_writer message_writer, error_writer,
    sample_writer, diagnostic_writer;
  mock_sampler sampler;
  int num_warmup, num_samples, num_thin, refresh;
  bool save_warmup;
};

TEST_F(ServicesUtil, all_zero) {
  stan::services::util::run_sampler(sampler, model,
                                    cont_vector,
                                    num_warmup, num_samples,
                                    num_thin, refresh, save_warmup,
                                    rng,
                                    interrupt,
                                    message_writer, error_writer,
                                    sample_writer, diagnostic_writer);
  EXPECT_EQ(0, interrupt.call_count());

  EXPECT_EQ(3, message_writer.call_count("string"))
    << "Writes the elapsed time";
  EXPECT_EQ(message_writer.call_count("string") + message_writer.call_count("empty"),
            message_writer.call_count())
    << "No other calls to message_writer";
  
  EXPECT_EQ(0, error_writer.call_count());

  EXPECT_EQ(6, sample_writer.call_count());
  EXPECT_EQ(1, sample_writer.call_count("vector_string"))
    << "header line";
  EXPECT_EQ(3, sample_writer.call_count("string"))
    << "elapsed time";
  EXPECT_EQ(2, sample_writer.call_count("empty"))
    << "blank lines";
  
  EXPECT_EQ(6, diagnostic_writer.call_count());
  EXPECT_EQ(1, diagnostic_writer.call_count("vector_string"))
    << "header line";
  EXPECT_EQ(3, diagnostic_writer.call_count("string"))
    << "elapsed time";
  EXPECT_EQ(2, diagnostic_writer.call_count("empty"))
    << "blank lines";
}

TEST_F(ServicesUtil, num_warmup_no_save) {
  num_warmup = 1000;
  stan::services::util::run_sampler(sampler, model,
                                    cont_vector,
                                    num_warmup, num_samples,
                                    num_thin, refresh, save_warmup,
                                    rng,
                                    interrupt,
                                    message_writer, error_writer,
                                    sample_writer, diagnostic_writer);
  EXPECT_EQ(num_warmup, interrupt.call_count());

  EXPECT_EQ(3, message_writer.call_count("string"))
    << "Writes the elapsed time";
  EXPECT_EQ(message_writer.call_count("string") + message_writer.call_count("empty"),
            message_writer.call_count())
    << "No other calls to message_writer";
  
  EXPECT_EQ(0, error_writer.call_count());

  EXPECT_EQ(6, sample_writer.call_count());
  EXPECT_EQ(1, sample_writer.call_count("vector_string"))
    << "header line";
  EXPECT_EQ(3, sample_writer.call_count("string"))
    << "elapsed time";
  EXPECT_EQ(2, sample_writer.call_count("empty"))
    << "blank lines";
  
  EXPECT_EQ(6, diagnostic_writer.call_count());
  EXPECT_EQ(1, diagnostic_writer.call_count("vector_string"))
    << "header line";
  EXPECT_EQ(3, diagnostic_writer.call_count("string"))
    << "elapsed time";
  EXPECT_EQ(2, diagnostic_writer.call_count("empty"))
    << "blank lines";
}

TEST_F(ServicesUtil, num_warmup_save) {
  num_warmup = 1000;
  save_warmup = true;
  stan::services::util::run_sampler(sampler, model,
                                    cont_vector,
                                    num_warmup, num_samples,
                                    num_thin, refresh, save_warmup,
                                    rng,
                                    interrupt,
                                    message_writer, error_writer,
                                    sample_writer, diagnostic_writer);
  EXPECT_EQ(num_warmup, interrupt.call_count());

  EXPECT_EQ(3, message_writer.call_count("string"))
    << "Writes the elapsed time";
  EXPECT_EQ(message_writer.call_count("string") + message_writer.call_count("empty"),
            message_writer.call_count())
    << "No other calls to message_writer";
  
  EXPECT_EQ(0, error_writer.call_count());

  EXPECT_EQ(num_warmup + 6, sample_writer.call_count());
  EXPECT_EQ(1, sample_writer.call_count("vector_string"))
    << "header line";
  EXPECT_EQ(3, sample_writer.call_count("string"))
    << "elapsed time";
  EXPECT_EQ(2, sample_writer.call_count("empty"))
    << "blank lines";
  EXPECT_EQ(num_warmup, sample_writer.call_count("vector_double"))
    << "warmup draws";
  
  EXPECT_EQ(num_warmup + 6, diagnostic_writer.call_count());
  EXPECT_EQ(1, diagnostic_writer.call_count("vector_string"))
    << "header line";
  EXPECT_EQ(3, diagnostic_writer.call_count("string"))
    << "elapsed time";
  EXPECT_EQ(2, diagnostic_writer.call_count("empty"))
    << "blank lines";
  EXPECT_EQ(num_warmup, diagnostic_writer.call_count("vector_double"))
    << "warmup draws";
}


TEST_F(ServicesUtil, num_samples) {
  num_samples = 1000;
  stan::services::util::run_sampler(sampler, model,
                                    cont_vector,
                                    num_warmup, num_samples,
                                    num_thin, refresh, save_warmup,
                                    rng,
                                    interrupt,
                                    message_writer, error_writer,
                                    sample_writer, diagnostic_writer);
  EXPECT_EQ(num_samples, interrupt.call_count());

  EXPECT_EQ(3, message_writer.call_count("string"))
    << "Writes the elapsed time";
  EXPECT_EQ(message_writer.call_count("string") + message_writer.call_count("empty"),
            message_writer.call_count())
    << "No other calls to message_writer";
  
  EXPECT_EQ(0, error_writer.call_count());

  EXPECT_EQ(num_samples + 6, sample_writer.call_count());
  EXPECT_EQ(1, sample_writer.call_count("vector_string"))
    << "header line";
  EXPECT_EQ(3, sample_writer.call_count("string"))
    << "elapsed time";
  EXPECT_EQ(2, sample_writer.call_count("empty"))
    << "blank lines";
  EXPECT_EQ(num_samples, sample_writer.call_count("vector_double"))
    << "num_samples draws";
  
  EXPECT_EQ(num_samples + 6, diagnostic_writer.call_count());
  EXPECT_EQ(1, diagnostic_writer.call_count("vector_string"))
    << "header line";
  EXPECT_EQ(3, diagnostic_writer.call_count("string"))
    << "elapsed time";
  EXPECT_EQ(2, diagnostic_writer.call_count("empty"))
    << "blank lines";
  EXPECT_EQ(num_samples, sample_writer.call_count("vector_double"))
    << "num_samples draws";
}

TEST_F(ServicesUtil, num_warmup_save_num_samples_num_thin) {
  num_warmup = 500;
  save_warmup = true;
  num_samples = 500;
  num_thin = 10;
  stan::services::util::run_sampler(sampler, model,
                                    cont_vector,
                                    num_warmup, num_samples,
                                    num_thin, refresh, save_warmup,
                                    rng,
                                    interrupt,
                                    message_writer, error_writer,
                                    sample_writer, diagnostic_writer);
  EXPECT_EQ(num_warmup + num_samples, interrupt.call_count());

  EXPECT_EQ(3, message_writer.call_count("string"))
    << "Writes the elapsed time";
  EXPECT_EQ(message_writer.call_count("string") + message_writer.call_count("empty"),
            message_writer.call_count())
    << "No other calls to message_writer";
  
  EXPECT_EQ(0, error_writer.call_count());

  EXPECT_EQ((num_warmup + num_samples) / num_thin + 6,
            sample_writer.call_count());
  EXPECT_EQ(1, sample_writer.call_count("vector_string"))
    << "header line";
  EXPECT_EQ(3, sample_writer.call_count("string"))
    << "elapsed time";
  EXPECT_EQ(2, sample_writer.call_count("empty"))
    << "blank lines";
  EXPECT_EQ((num_warmup + num_samples) / num_thin,
            sample_writer.call_count("vector_double"))
    << "thinned warmup and draws";
  
  EXPECT_EQ((num_warmup + num_samples) / num_thin + 6,
            diagnostic_writer.call_count());
  EXPECT_EQ(1, diagnostic_writer.call_count("vector_string"))
    << "header line";
  EXPECT_EQ(3, diagnostic_writer.call_count("string"))
    << "elapsed time";
  EXPECT_EQ(2, diagnostic_writer.call_count("empty"))
    << "blank lines";
  EXPECT_EQ((num_warmup + num_samples) / num_thin,
            diagnostic_writer.call_count("vector_double"))
    << "thinned warmup and draws";
}


TEST_F(ServicesUtil, num_warmup_num_samples_refresh) {
  num_warmup = 500;
  num_samples = 500;
  refresh = 10;

  stan::services::util::run_sampler(sampler, model,
                                    cont_vector,
                                    num_warmup, num_samples,
                                    num_thin, refresh, save_warmup,
                                    rng,
                                    interrupt,
                                    message_writer, error_writer,
                                    sample_writer, diagnostic_writer);
  EXPECT_EQ(num_warmup + num_samples, interrupt.call_count());

  EXPECT_EQ((num_warmup + num_samples) / refresh + 2 + 3, message_writer.call_count("string"))
    << "Writes 1 to start warmup, 1 to start post-warmup, and "
    << "(num_warmup + num_samples) / refresh, then the elapsed time";
  EXPECT_EQ(message_writer.call_count("string") + message_writer.call_count("empty"),
            message_writer.call_count())
    << "No other calls to message_writer";
  
  EXPECT_EQ(0, error_writer.call_count());

  EXPECT_EQ(num_samples + 6,
            sample_writer.call_count());
  EXPECT_EQ(1, sample_writer.call_count("vector_string"))
    << "header line";
  EXPECT_EQ(3, sample_writer.call_count("string"))
    << "elapsed time";
  EXPECT_EQ(2, sample_writer.call_count("empty"))
    << "blank lines";
  EXPECT_EQ(num_samples,
            sample_writer.call_count("vector_double"))
    << "draws";
  
  EXPECT_EQ(num_samples + 6,
            diagnostic_writer.call_count());
  EXPECT_EQ(1, diagnostic_writer.call_count("vector_string"))
    << "header line";
  EXPECT_EQ(3, diagnostic_writer.call_count("string"))
    << "elapsed time";
  EXPECT_EQ(2, diagnostic_writer.call_count("empty"))
    << "blank lines";
  EXPECT_EQ(num_samples,
            diagnostic_writer.call_count("vector_double"))
    << "draws";
}

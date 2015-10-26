#include <stan/mcmc/base_mcmc.hpp>
#include <stan/services/sample/run_sampler.hpp>
#include <stan/services/sample/mcmc_writer.hpp>
#include <stan/interface_callbacks/interrupt/base_interrupt.hpp>
#include <stan/interface_callbacks/writer/stream_writer.hpp>
#include <gtest/gtest.h>
#include <test/test-models/good/services/test_lp.hpp>
#include <boost/random/additive_combine.hpp>
#include <sstream>

typedef boost::ecuyer1988 rng_t;
typedef stan::interface_callbacks::writer::stream_writer writer_t;

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

  StanServices()
    : output(output_ss),
      diagnostic(diagnostic_ss),
      info(info_ss),
      err(err_ss) {}
  
  void SetUp() {

    model_output.str("");
    output_ss.str("");
    diagnostic_ss.str("");
    info_ss.str("");
    err_ss.str("");

    base_rng.seed(123456);

    sampler = new mock_sampler();

    std::fstream empty_data_stream(std::string("").c_str(), std::fstream::in);
    stan::io::dump empty_data_context(empty_data_stream);
    empty_data_stream.close();

    model = new stan_model(empty_data_context, &model_output);

    writer = new stan::services::sample::mcmc_writer
      <stan_model, rng_t, writer_t, writer_t, writer_t, writer_t>
      (*model, base_rng, output, diagnostic, info, err);

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
  std::stringstream output_ss;
  std::stringstream diagnostic_ss;
  std::stringstream info_ss;
  std::stringstream err_ss;

  writer_t output;
  writer_t diagnostic;
  writer_t info;
  writer_t err;

  rng_t base_rng;

  mock_sampler* sampler;
  stan_model* model;
  stan::services::sample::mcmc_writer
    <stan_model, rng_t, writer_t, writer_t, writer_t, writer_t>* writer;

  Eigen::VectorXd q;
  double log_prob;
  double stat;

};

TEST_F(StanServices, sample) {

  std::string expected_output = "lp__,accept_stat__,y.1,y.2,z.1,z.2,xgq\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n0,0,0,0,1,1,2713\n\n# ";
  std::string expected_diagnostic = "lp__,accept_stat__\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n0,0\n\n# ";
  std::string expected_info = "Iteration:  1 / 30 [  3%]  (Warmup)\nIteration:  4 / 30 [ 13%]  (Warmup)\nIteration:  8 / 30 [ 26%]  (Warmup)\nIteration: 12 / 30 [ 40%]  (Warmup)\nIteration: 16 / 30 [ 53%]  (Warmup)\nIteration: 20 / 30 [ 66%]  (Warmup)\nIteration: 24 / 30 [ 80%]  (Warmup)\nIteration: 28 / 30 [ 93%]  (Warmup)\nIteration: 30 / 30 [100%]  (Warmup)\nIteration: 31 / 80 [ 38%]  (Sampling)\nIteration: 34 / 80 [ 42%]  (Sampling)\nIteration: 38 / 80 [ 47%]  (Sampling)\nIteration: 42 / 80 [ 52%]  (Sampling)\nIteration: 46 / 80 [ 57%]  (Sampling)\nIteration: 50 / 80 [ 62%]  (Sampling)\nIteration: 54 / 80 [ 67%]  (Sampling)\nIteration: 58 / 80 [ 72%]  (Sampling)\nIteration: 62 / 80 [ 77%]  (Sampling)\nIteration: 66 / 80 [ 82%]  (Sampling)\nIteration: 70 / 80 [ 87%]  (Sampling)\nIteration: 74 / 80 [ 92%]  (Sampling)\nIteration: 78 / 80 [ 97%]  (Sampling)\nIteration: 80 / 80 [100%]  (Sampling)\n\n";
  std::string expected_err = "";

  int num_warmup = 30;
  int num_samples = 50;
  int num_thin = 2;
  int refresh = 4;
  bool save_warmup = false;
  stan::mcmc::sample s(q, log_prob, stat);

  mock_interrupt interrupt;

  stan::services::sample::run_sampler(*sampler, s,
                                      num_warmup, num_samples,
                                      num_thin, refresh, save_warmup,
                                      *writer, interrupt);

  EXPECT_EQ(num_warmup + num_samples, sampler->n_transition_called());
  EXPECT_EQ(num_warmup + num_samples, interrupt.n());

  // Strip away elapsed time which is too variable
  EXPECT_EQ("", model_output.str());
  EXPECT_EQ(expected_output,
            output_ss.str().substr(0, output_ss.str().find("Elapsed")));
  EXPECT_EQ(expected_diagnostic,
            diagnostic_ss.str().substr(0, diagnostic_ss.str().find("Elapsed")));
  EXPECT_EQ(expected_info,
            info_ss.str().substr(0, info_ss.str().find("Elapsed")));
  EXPECT_EQ(expected_err, err_ss.str());
}

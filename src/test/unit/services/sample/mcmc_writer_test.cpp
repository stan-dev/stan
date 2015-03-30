#include <stan/services/sample/mcmc_writer.hpp>
#include <stan/interface_callbacks/writer/stringstream.hpp>
#include <test/test-models/good/io_example.hpp>

#include <vector>
#include <boost/random/additive_combine.hpp>

#include <stan/mcmc/sample.hpp>
#include <stan/mcmc/hmc/nuts/diag_e_nuts.hpp>

#include <sstream>
#include <string>

#include <gtest/gtest.h>

typedef boost::ecuyer1988 rng_t;
typedef stan::interface_callbacks::writer::stringstream writer_t;

TEST(StanIoMcmcWriter, write_names) {
  
  // Model
  std::fstream data_stream("", std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  std::stringstream model_output;
  stan_model model(data_var_context, &model_output);
  
  // Sample
  Eigen::VectorXd real(2);
  real(0) = 1.43;
  real(1) = 2.71;
  
  double log_prob = 3.14;
  double accept_stat = 0.84;
  
  stan::mcmc::sample sample(real, log_prob, accept_stat);
  
  // Sampler
  rng_t base_rng(0);
  
  writer_t sampler_writer;
  stan::mcmc::adapt_diag_e_nuts<stan_model, rng_t, writer_t>
    sampler(model, base_rng, sampler_writer);
  sampler.seed(real);
  
  // Writer
  writer_t output;
  writer_t diagnostic;
  writer_t info;
  
  stan::services::sample::mcmc_writer
    <stan_model, rng_t, writer_t, writer_t, writer_t>
      writer(model, base_rng, output, diagnostic, info);

  writer.write_names(sample, sampler);
  
  std::string expected_output =
    "lp__,accept_stat__,stepsize__,treedepth__,n_leapfrog__,n_divergent__,"
    "mu1,mu2\n";
  std::string expected_diagnostic =
    "lp__,accept_stat__,stepsize__,treedepth__,n_leapfrog__,n_divergent__,"
    "mu1,mu2,p_mu1,p_mu2,g_mu1,g_mu2\n";
  std::string expected_info = "";

  EXPECT_EQ("", model_output.str());
  
  EXPECT_EQ(expected_output, output.contents());
  EXPECT_EQ(expected_diagnostic, diagnostic.contents());
  EXPECT_EQ(expected_info, info.contents());

}

TEST(StanIoMcmcWriter, write_state) {
  
  // Model
  std::fstream data_stream("", std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();
  
  std::stringstream model_output;
  stan_model model(data_var_context, &model_output);
  
  // Sample
  Eigen::VectorXd real(2);
  real(0) = 1.43;
  real(1) = 2.71;
  
  double log_prob = 3.14;
  double accept_stat = 0.84;
  
  stan::mcmc::sample sample(real, log_prob, accept_stat);
  
  // Sampler
  rng_t base_rng(0);
  
  writer_t sampler_writer;
  stan::mcmc::adapt_diag_e_nuts<stan_model, rng_t, writer_t>
    sampler(model, base_rng, sampler_writer);
  sampler.seed(real);
  sampler.z().p.setZero();
  sampler.z().g.setZero();
  
  // Writer
  writer_t output;
  writer_t diagnostic;
  writer_t info;
  
  stan::services::sample::mcmc_writer
    <stan_model, rng_t, writer_t, writer_t, writer_t>
      writer(model, base_rng, output, diagnostic, info);
  
  writer.write_state(sample, sampler);
  
  std::string expected_output =
    "3.14,0.84,0.1,0,0,0,1.43,2.71\n";
  std::string expected_diagnostic =
    "3.14,0.84,0.1,0,0,0,1.43,2.71,0,0,0,0\n";
  std::string expected_info = "";
  
  EXPECT_EQ("", model_output.str());
  
  EXPECT_EQ(expected_output, output.contents());
  EXPECT_EQ(expected_diagnostic, diagnostic.contents());
  EXPECT_EQ(expected_info, info.contents());
  
}

TEST(StanIoMcmcWriter, write_adapt_finish) {
  
  // Model
  std::fstream data_stream("", std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();
  
  std::stringstream model_output;
  stan_model model(data_var_context, &model_output);
  
  // Sample
  Eigen::VectorXd real(2);
  real(0) = 1.43;
  real(1) = 2.71;
  
  double log_prob = 3.14;
  double accept_stat = 0.84;
  
  stan::mcmc::sample sample(real, log_prob, accept_stat);
  
  // Sampler
  rng_t base_rng(0);
  
  writer_t sampler_writer;
  stan::mcmc::adapt_diag_e_nuts<stan_model, rng_t, writer_t>
    sampler(model, base_rng, sampler_writer);
  sampler.seed(real);
  
  // Writer
  writer_t output;
  writer_t diagnostic;
  writer_t info;
  
  stan::services::sample::mcmc_writer
    <stan_model, rng_t, writer_t, writer_t, writer_t>
      writer(model, base_rng, output, diagnostic, info);
  
  writer.write_adapt_finish(sampler);
  
  std::string expected_output =
    "# Adaptation terminated\nStep size = 0.1\n"
    "# Diagonal elements of inverse mass matrix:\nM_inv = 1,1\n";
  std::string expected_diagnostic =
    "# Adaptation terminated\nStep size = 0.1\n"
    "# Diagonal elements of inverse mass matrix:\nM_inv = 1,1\n";
  std::string expected_info = "";
  
  EXPECT_EQ("", model_output.str());
  
  EXPECT_EQ(expected_output, output.contents());
  EXPECT_EQ(expected_diagnostic, diagnostic.contents());
  EXPECT_EQ(expected_info, info.contents());
  
}

TEST(StanIoMcmcWriter, write_timing) {
  
  // Model
  std::fstream data_stream("", std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();
  
  std::stringstream model_output;
  stan_model model(data_var_context, &model_output);
  
  // Sample
  Eigen::VectorXd real(2);
  real(0) = 1.43;
  real(1) = 2.71;
  
  double log_prob = 3.14;
  double accept_stat = 0.84;
  
  stan::mcmc::sample sample(real, log_prob, accept_stat);
  
  // Sampler
  rng_t base_rng(0);
  
  writer_t sampler_writer;
  stan::mcmc::adapt_diag_e_nuts<stan_model, rng_t, writer_t>
    sampler(model, base_rng, sampler_writer);
  sampler.seed(real);
  
  // Writer
  writer_t output;
  writer_t diagnostic;
  writer_t info;
  
  stan::services::sample::mcmc_writer
    <stan_model, rng_t, writer_t, writer_t, writer_t>
      writer(model, base_rng, output, diagnostic, info);
  
  writer.write_timing(10, 10);
  
  std::string expected_output =
    "\n# Elapsed Time (seconds):\n# Warmup = 10\n# Sampling = 10\n# Total = 20\n\n";
  std::string expected_diagnostic =
    "\n# Elapsed Time (seconds):\n# Warmup = 10\n# Sampling = 10\n# Total = 20\n\n";
  std::string expected_info =
    "\nElapsed Time (seconds):\nWarmup = 10\nSampling = 10\nTotal = 20\n\n";
  
  EXPECT_EQ("", model_output.str());
  
  EXPECT_EQ(expected_output, output.contents());
  EXPECT_EQ(expected_diagnostic, diagnostic.contents());
  EXPECT_EQ(expected_info, info.contents());
  
}

TEST(StanIoMcmcWriter, write_message) {
  
  // Model
  std::fstream data_stream("", std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();
  
  std::stringstream model_output;
  stan_model model(data_var_context, &model_output);
  
  // Sample
  Eigen::VectorXd real(2);
  real(0) = 1.43;
  real(1) = 2.71;
  
  double log_prob = 3.14;
  double accept_stat = 0.84;
  
  stan::mcmc::sample sample(real, log_prob, accept_stat);
  
  // Sampler
  rng_t base_rng(0);
  
  writer_t sampler_writer;
  stan::mcmc::adapt_diag_e_nuts<stan_model, rng_t, writer_t>
    sampler(model, base_rng, sampler_writer);
  sampler.seed(real);
  
  // Writer
  writer_t output;
  writer_t diagnostic;
  writer_t info;
  
  stan::services::sample::mcmc_writer
    <stan_model, rng_t, writer_t, writer_t, writer_t>
      writer(model, base_rng, output, diagnostic, info);
  
  writer.write_message("Important message");
  
  std::string expected_output = "";
  std::string expected_diagnostic = "";
  std::string expected_info = "Important message\n";
  
  EXPECT_EQ("", model_output.str());
  
  EXPECT_EQ(expected_output, output.contents());
  EXPECT_EQ(expected_diagnostic, diagnostic.contents());
  EXPECT_EQ(expected_info, info.contents());
  
}

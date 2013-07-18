#include <stan/io/mcmc_writer.hpp>
#include <test/io/test_model/example.cpp>

#include <vector>
#include <boost/random/additive_combine.hpp>

#include <stan/mcmc/sample.hpp>
#include <stan/mcmc/hmc/nuts/diag_e_nuts.hpp>

#include <sstream>
#include <string>

#include <gtest/gtest.h>


TEST(StanIoMcmcWriter, print_sample_names) {
  
  // Model
  std::fstream data_stream("", std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();
  
  example_model_namespace::example_model model(data_var_context, &std::cout);
  
  // Sample
  std::vector<double> real;
  real.push_back(1.43);
  real.push_back(2.71);
  
  std::vector<int> discrete;
  
  double log_prob = 3.14;
  double accept_stat = 0.84;
  
  stan::mcmc::sample sample(real, discrete, log_prob, accept_stat);
  
  // Sampler
  typedef boost::ecuyer1988 rng_t;
  rng_t base_rng(0);
  
  stan::mcmc::adapt_diag_e_nuts<example_model_namespace::example_model, rng_t> sampler(model, base_rng, 0);
  sampler.seed(real, discrete);
  
  // Writer
  std::stringstream sample_stream;
  std::stringstream diagnostic_stream;
  
  stan::io::mcmc_writer<example_model_namespace::example_model> writer(&sample_stream, &diagnostic_stream);
  
  writer.print_sample_names(sample, sampler, model);
  
  std::string line;
  std::getline(sample_stream, line);
  
  EXPECT_EQ("lp__,accept_stat__,stepsize__,treedepth__,mu1,mu2", line);
  
}

TEST(StanIoMcmcWriter, print_sample_params) {
  
  // Model
  std::fstream data_stream("", std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();
  
  example_model_namespace::example_model model(data_var_context, &std::cout);
  
  // Sample
  std::vector<double> real;
  real.push_back(1.43);
  real.push_back(2.71);
  
  std::vector<int> discrete;
  
  double log_prob = 3.14;
  double accept_stat = 0.84;
  
  stan::mcmc::sample sample(real, discrete, log_prob, accept_stat);
  
  // Sampler
  typedef boost::ecuyer1988 rng_t;
  rng_t base_rng(0);
  
  stan::mcmc::adapt_diag_e_nuts<example_model_namespace::example_model, rng_t> sampler(model, base_rng, 0);
  sampler.seed(real, discrete);
  
  // Writer
  std::stringstream sample_stream;
  std::stringstream diagnostic_stream;
  
  stan::io::mcmc_writer<example_model_namespace::example_model> writer(&sample_stream, &diagnostic_stream);
  
  
  writer.print_sample_params(base_rng, sample, sampler, model);
  
  std::string line;
  std::getline(sample_stream, line);
  
  std::stringstream expected_stream;
  expected_stream << log_prob << ",";
  expected_stream << accept_stat << ",";
  expected_stream << sampler.get_current_stepsize() << ",";
  expected_stream << 0 << ",";
  expected_stream << real.at(0) << ",";
  expected_stream << real.at(1);
  
  std::string expected_line;
  std::getline(expected_stream, expected_line);
  
  EXPECT_EQ(expected_line, line);
  
}

TEST(StanIoMcmcWriter, print_adapt_finish) {
  
  // Model
  std::fstream data_stream("", std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();
  
  example_model_namespace::example_model model(data_var_context, &std::cout);
  
  // Sample
  std::vector<double> real;
  real.push_back(1.43);
  real.push_back(2.71);
  
  std::vector<int> discrete;
  
  double log_prob = 3.14;
  double accept_stat = 0.84;
  
  stan::mcmc::sample sample(real, discrete, log_prob, accept_stat);
  
  // Sampler
  typedef boost::ecuyer1988 rng_t;
  rng_t base_rng(0);
  
  stan::mcmc::adapt_diag_e_nuts<example_model_namespace::example_model, rng_t> sampler(model, base_rng, 0);
  sampler.seed(real, discrete);
  
  // Writer
  std::stringstream sample_stream;
  std::stringstream diagnostic_stream;
  
  stan::io::mcmc_writer<example_model_namespace::example_model> writer(&sample_stream, &diagnostic_stream);
  
  writer.print_adapt_finish(sampler);
  
  std::stringstream expected_stream;
  expected_stream << "# Adaptation terminated" << std::endl;
  expected_stream << "# Step size = " << sampler.get_current_stepsize() << std::endl;
  expected_stream << "# Diagonal elements of inverse mass matrix:" << std::endl;
  expected_stream << "# " << sampler.z().mInv(0) << ", " << sampler.z().mInv(1) << std::endl;
  
  std::string line;
  std::string expected_line;
  
  // Line 1
  std::getline(expected_stream, expected_line);
  
  std::getline(sample_stream, line);
  EXPECT_EQ(expected_line, line);
  
  std::getline(diagnostic_stream, line);
  EXPECT_EQ(expected_line, line);
  
  // Line 2
  std::getline(expected_stream, expected_line);
  
  std::getline(sample_stream, line);
  EXPECT_EQ(expected_line, line);
  
  std::getline(diagnostic_stream, line);
  EXPECT_EQ(expected_line, line);
  
  // Line 3
  std::getline(expected_stream, expected_line);
  
  std::getline(sample_stream, line);
  EXPECT_EQ(expected_line, line);
  
  std::getline(diagnostic_stream, line);
  EXPECT_EQ(expected_line, line);
  
  // Line 4
  std::getline(expected_stream, expected_line);
  
  std::getline(sample_stream, line);
  EXPECT_EQ(expected_line, line);
  
  std::getline(diagnostic_stream, line);
  EXPECT_EQ(expected_line, line);
  
}

TEST(StanIoMcmcWriter, print_diagnostic_names) {
  
  // Model
  std::fstream data_stream("", std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();
  
  example_model_namespace::example_model model(data_var_context, &std::cout);
  
  // Sample
  std::vector<double> real;
  real.push_back(1.43);
  real.push_back(2.71);
  
  std::vector<int> discrete;
  
  double log_prob = 3.14;
  double accept_stat = 0.84;
  
  stan::mcmc::sample sample(real, discrete, log_prob, accept_stat);
  
  // Sampler
  typedef boost::ecuyer1988 rng_t;
  rng_t base_rng(0);
  
  stan::mcmc::adapt_diag_e_nuts<example_model_namespace::example_model, rng_t> sampler(model, base_rng, 0);
  sampler.seed(real, discrete);
  
  // Writer
  std::stringstream sample_stream;
  std::stringstream diagnostic_stream;
  
  stan::io::mcmc_writer<example_model_namespace::example_model> writer(&sample_stream, &diagnostic_stream);
  
  writer.print_diagnostic_names(sample, sampler, model);
  
  std::string line;
  std::getline(diagnostic_stream, line);
  
  // FIXME: make this work, too
  EXPECT_EQ("lp__,accept_stat__,stepsize__,treedepth__,mu1,mu2,p_mu1,p_mu2,g_mu1,g_mu2", line);
  
}

TEST(StanIoMcmcWriter, print_diagnostic_params) {
  
  // Model
  std::fstream data_stream("", std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();
  
  example_model_namespace::example_model model(data_var_context, &std::cout);
  
  // Sample
  std::vector<double> real;
  real.push_back(1.43);
  real.push_back(2.71);
  
  std::vector<int> discrete;
  
  double log_prob = 3.14;
  double accept_stat = 0.84;
  
  stan::mcmc::sample sample(real, discrete, log_prob, accept_stat);
  
  // Sampler
  typedef boost::ecuyer1988 rng_t;
  rng_t base_rng(0);
  
  stan::mcmc::adapt_diag_e_nuts<example_model_namespace::example_model, rng_t> sampler(model, base_rng, 0);
  sampler.seed(real, discrete);
  sampler.z().p(0) = 0;
  sampler.z().p(1) = 0;
  sampler.z().g(0) = 0;
  sampler.z().g(1) = 0;
  
  // Writer
  std::stringstream sample_stream;
  std::stringstream diagnostic_stream;
  
  stan::io::mcmc_writer<example_model_namespace::example_model> writer(&sample_stream, &diagnostic_stream);
  
  writer.print_diagnostic_params(sample, sampler);
  
  std::string line;
  std::getline(diagnostic_stream, line);
  
  std::stringstream expected_stream;
  expected_stream << log_prob << ",";
  expected_stream << accept_stat << ",";
  expected_stream << sampler.get_current_stepsize() << ",";
  expected_stream << 0 << ",";
  expected_stream << real.at(0) << ",";
  expected_stream << real.at(1) << ",";
  expected_stream << 0 << ",";
  expected_stream << 0 << ",";
  expected_stream << 0 << ",";
  expected_stream << 0;
  
  std::string expected_line;
  std::getline(expected_stream, expected_line);
  
  EXPECT_EQ(expected_line, line);
  
}

TEST(StanIoMcmcWriter, print_timing) {
  
  // Model
  std::fstream data_stream("", std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();
  
  example_model_namespace::example_model model(data_var_context, &std::cout);
  
  // Sample
  std::vector<double> real;
  real.push_back(1.43);
  real.push_back(2.71);
  
  std::vector<int> discrete;
  
  double log_prob = 3.14;
  double accept_stat = 0.84;
  
  stan::mcmc::sample sample(real, discrete, log_prob, accept_stat);
  
  // Sampler
  typedef boost::ecuyer1988 rng_t;
  rng_t base_rng(0);
  
  stan::mcmc::adapt_diag_e_nuts<example_model_namespace::example_model, rng_t> sampler(model, base_rng, 0);
  sampler.seed(real, discrete);
  
  // Writer
  std::stringstream sample_stream;
  std::stringstream diagnostic_stream;
  
  stan::io::mcmc_writer<example_model_namespace::example_model> writer(&sample_stream, &diagnostic_stream);
  
  double warm = 0.193933;
  double sampling = 0.483830;

  writer.print_timing(warm, sampling, &sample_stream);

  std::stringstream expected_stream;
  expected_stream << std::endl;
  expected_stream << "# Elapsed Time: " << warm << " seconds (Warm-up)" << std::endl;
  expected_stream << "#               " << sampling << " seconds (Sampling)" << std::endl;
  expected_stream << "#               " << warm + sampling << " seconds (Total)" << std::endl;
  expected_stream << std::endl;
  
  std::string line;
  std::string expected_line;

  // Line 1
  std::getline(expected_stream, expected_line);
  
  std::getline(sample_stream, line);
  EXPECT_EQ(expected_line, line);
  
  std::getline(diagnostic_stream, line);
  EXPECT_EQ(expected_line, line);
  
  // Line 2
  std::getline(expected_stream, expected_line);
  
  std::getline(sample_stream, line);
  EXPECT_EQ(expected_line, line);
  
  std::getline(diagnostic_stream, line);
  EXPECT_EQ(expected_line, line);
  
  // Line 3
  std::getline(expected_stream, expected_line);
  
  std::getline(sample_stream, line);
  EXPECT_EQ(expected_line, line);
  
  std::getline(diagnostic_stream, line);
  EXPECT_EQ(expected_line, line);
  
  // Line 4
  std::getline(expected_stream, expected_line);
  
  std::getline(sample_stream, line);
  EXPECT_EQ(expected_line, line);
  
  std::getline(diagnostic_stream, line);
  EXPECT_EQ(expected_line, line);
  
  // Line 5
  std::getline(expected_stream, expected_line);
  
  std::getline(sample_stream, line);
  EXPECT_EQ(expected_line, line);
  
  std::getline(diagnostic_stream, line);
  EXPECT_EQ(expected_line, line);
  
}

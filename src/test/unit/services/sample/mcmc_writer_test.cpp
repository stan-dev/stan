#include <stan/services/sample/mcmc_writer.hpp>
#include <stan/interface_callbacks/writer/stream_writer.hpp>
#include <stan/interface_callbacks/writer/noop_writer.hpp>
#include <test/test-models/good/io_example.hpp>

#include <vector>
#include <boost/random/additive_combine.hpp>

#include <stan/mcmc/sample.hpp>
#include <stan/mcmc/hmc/nuts/diag_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp>

#include <sstream>
#include <string>

#include <gtest/gtest.h>

typedef stan::interface_callbacks::writer::stream_writer writer_t;

TEST(StanIoMcmcWriter, write_sample_names) {

  // Model
  std::fstream data_stream("", std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  std::stringstream output;
  io_example_model_namespace::io_example_model model(data_var_context, &output);

  // Sample
  Eigen::VectorXd real(2);
  real(0) = 1.43;
  real(1) = 2.71;

  double log_prob = 3.14;
  double accept_stat = 0.84;

  stan::mcmc::sample sample(real, log_prob, accept_stat);

  // Sampler
  typedef boost::ecuyer1988 rng_t;
  rng_t base_rng(0);


  stan::interface_callbacks::writer::noop_writer sampler_writer;

  stan::mcmc::adapt_diag_e_nuts<io_example_model_namespace::io_example_model, rng_t>
    sampler(model, base_rng);
  sampler.seed(real);

  // Writer
  std::stringstream sample_stream;
  std::stringstream diagnostic_stream;
  std::stringstream message_stream;

  writer_t sample_writer(sample_stream, "# ");
  writer_t diagnostic_writer(diagnostic_stream, "# ");
  writer_t message_writer(message_stream, "# ");

  stan::services::sample::mcmc_writer<io_example_model_namespace::io_example_model,
                                      writer_t,
                                      writer_t,
                                      writer_t>
    writer(sample_writer, diagnostic_writer, message_writer);

  writer.write_sample_names(sample, &sampler, model);

  std::string line;
  std::getline(sample_stream, line);

  EXPECT_EQ("lp__,accept_stat__,stepsize__,treedepth__,n_leapfrog__,divergent__,energy__,mu1,mu2", line);
  EXPECT_EQ("", message_stream.str());
  EXPECT_EQ("", output.str());
}

TEST(StanIoMcmcWriter, write_sample_params) {

  // Model
  std::fstream data_stream("", std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  std::stringstream output;
  io_example_model_namespace::io_example_model model(data_var_context, &output);

  // Sample
  Eigen::VectorXd real(2);
  real(0) = 1.43;
  real(1) = 2.71;

  double log_prob = 3.14;
  double accept_stat = 0.84;

  stan::mcmc::sample sample(real, log_prob, accept_stat);

  // Sampler
  typedef boost::ecuyer1988 rng_t;
  rng_t base_rng(0);

  stan::interface_callbacks::writer::noop_writer sampler_writer;

  stan::mcmc::adapt_diag_e_nuts<io_example_model_namespace::io_example_model, rng_t>
    sampler(model, base_rng);
  sampler.seed(real);

  sampler.z().p(0) = 0;
  sampler.z().p(1) = 0;

  // Writer
  std::stringstream sample_stream;
  std::stringstream diagnostic_stream;
  std::stringstream message_stream;

  writer_t sample_writer(sample_stream, "# ");
  writer_t diagnostic_writer(diagnostic_stream, "# ");
  writer_t message_writer(message_stream, "# ");

  stan::services::sample::mcmc_writer<io_example_model_namespace::io_example_model,
                                      writer_t,
                                      writer_t,
                                      writer_t>
    writer(sample_writer, diagnostic_writer, message_writer);

  writer.write_sample_params<rng_t>(base_rng, sample, sampler, model);

  std::string line;
  std::getline(sample_stream, line);

  std::stringstream expected_stream;
  expected_stream << log_prob << ",";
  expected_stream << accept_stat << ",";
  expected_stream << sampler.get_current_stepsize() << ",";
  expected_stream << 0 << ",";
  expected_stream << 0 << ",";
  expected_stream << 0 << ",";
  expected_stream << 0 << ",";
  expected_stream << real(0) << ",";
  expected_stream << real(1);

  std::string expected_line;
  std::getline(expected_stream, expected_line);

  EXPECT_EQ(expected_line, line);
  EXPECT_EQ("", message_stream.str());
  EXPECT_EQ("", output.str());
}

TEST(StanIoMcmcWriter, write_adapt_finish) {

  // Model
  std::fstream data_stream("", std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  std::stringstream output;
  io_example_model_namespace::io_example_model model(data_var_context, &output);

  // Sample
  Eigen::VectorXd real(2);
  real(0) = 1.43;
  real(1) = 2.71;

  double log_prob = 3.14;
  double accept_stat = 0.84;

  stan::mcmc::sample sample(real, log_prob, accept_stat);

  // Sampler
  typedef boost::ecuyer1988 rng_t;
  rng_t base_rng(0);

  std::stringstream sample_stream;
  std::stringstream diagnostic_stream;
  std::stringstream message_stream;
  stan::interface_callbacks::writer::stream_writer sampler_writer(sample_stream, "# ");

  stan::mcmc::adapt_diag_e_nuts<io_example_model_namespace::io_example_model, rng_t>
    sampler(model, base_rng);
  sampler.seed(real);

  // Writer

  writer_t sample_writer(sample_stream, "# ");
  writer_t diagnostic_writer(diagnostic_stream, "# ");
  writer_t message_writer(message_stream, "# ");

  stan::services::sample::mcmc_writer<io_example_model_namespace::io_example_model,
                                      writer_t,
                                      writer_t,
                                      writer_t>
    writer(sample_writer, diagnostic_writer, message_writer);

  writer.write_adapt_finish(&sampler);

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
  EXPECT_EQ("", line);

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

  EXPECT_EQ("", message_stream.str());
  EXPECT_EQ("", output.str());
}

TEST(StanIoMcmcWriter, write_diagnostic_names) {

  // Model
  std::fstream data_stream("", std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  std::stringstream output;
  io_example_model_namespace::io_example_model model(data_var_context, &output);

  // Sample
  Eigen::VectorXd real(2);
  real(0) = 1.43;
  real(1) = 2.71;

  double log_prob = 3.14;
  double accept_stat = 0.84;

  stan::mcmc::sample sample(real, log_prob, accept_stat);

  // Sampler
  typedef boost::ecuyer1988 rng_t;
  rng_t base_rng(0);

  stan::interface_callbacks::writer::noop_writer sampler_writer;

  stan::mcmc::adapt_diag_e_nuts<io_example_model_namespace::io_example_model, rng_t>
    sampler(model, base_rng);
  sampler.seed(real);

  // Writer
  std::stringstream sample_stream;
  std::stringstream diagnostic_stream;
  std::stringstream message_stream;

  writer_t sample_writer(sample_stream, "# ");
  writer_t diagnostic_writer(diagnostic_stream, "# ");
  writer_t message_writer(message_stream, "# ");

  stan::services::sample::mcmc_writer<io_example_model_namespace::io_example_model,
                                      writer_t,
                                      writer_t,
                                      writer_t>
    writer(sample_writer, diagnostic_writer, message_writer);

  writer.write_diagnostic_names(sample, &sampler, model);

  std::string line;
  std::getline(diagnostic_stream, line);

  // FIXME: make this work, too
  EXPECT_EQ("lp__,accept_stat__,stepsize__,treedepth__,n_leapfrog__,divergent__,energy__,mu1,mu2,p_mu1,p_mu2,g_mu1,g_mu2", line);

  EXPECT_EQ("", message_stream.str());
  EXPECT_EQ("", output.str());
}

TEST(StanIoMcmcWriter, write_diagnostic_params) {

  // Model
  std::fstream data_stream("", std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  std::stringstream output;
  io_example_model_namespace::io_example_model model(data_var_context, &output);

  // Sample
  Eigen::VectorXd real(2);
  real(0) = 1.43;
  real(1) = 2.71;

  double log_prob = 3.14;
  double accept_stat = 0.84;

  stan::mcmc::sample sample(real, log_prob, accept_stat);

  // Sampler
  typedef boost::ecuyer1988 rng_t;
  rng_t base_rng(0);

  stan::interface_callbacks::writer::noop_writer sampler_writer;

  stan::mcmc::adapt_diag_e_nuts<io_example_model_namespace::io_example_model, rng_t>
    sampler(model, base_rng);
  sampler.seed(real);
  sampler.z().p(0) = 0;
  sampler.z().p(1) = 0;
  sampler.z().g(0) = 0;
  sampler.z().g(1) = 0;

  // Writer
  std::stringstream sample_stream;
  std::stringstream diagnostic_stream;
  std::stringstream message_stream;

  writer_t sample_writer(sample_stream, "# ");
  writer_t diagnostic_writer(diagnostic_stream, "# ");
  writer_t message_writer(message_stream, "# ");

  stan::services::sample::mcmc_writer<io_example_model_namespace::io_example_model,
                                      writer_t,
                                      writer_t,
                                      writer_t>
    writer(sample_writer, diagnostic_writer, message_writer);

  writer.write_diagnostic_params(sample, &sampler);

  std::string line;
  std::getline(diagnostic_stream, line);

  std::stringstream expected_stream;
  expected_stream << log_prob << ",";
  expected_stream << accept_stat << ",";
  expected_stream << sampler.get_current_stepsize() << ",";
  expected_stream << 0 << ",";
  expected_stream << 0 << ",";
  expected_stream << 0 << ",";
  expected_stream << 0 << ",";
  expected_stream << real(0) << ",";
  expected_stream << real(1) << ",";
  expected_stream << 0 << ",";
  expected_stream << 0 << ",";
  expected_stream << 0 << ",";
  expected_stream << 0;

  std::string expected_line;
  std::getline(expected_stream, expected_line);

  EXPECT_EQ(expected_line, line);
  EXPECT_EQ("", output.str());
}

TEST(StanIoMcmcWriter, write_timing) {

  // Model
  std::fstream data_stream("", std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  std::stringstream output;
  io_example_model_namespace::io_example_model model(data_var_context, &output);

  // Sample
  Eigen::VectorXd real(2);
  real(0) = 1.43;
  real(1) = 2.71;

  double log_prob = 3.14;
  double accept_stat = 0.84;

  stan::mcmc::sample sample(real, log_prob, accept_stat);

  // Sampler
  typedef boost::ecuyer1988 rng_t;
  rng_t base_rng(0);

  stan::interface_callbacks::writer::noop_writer sampler_writer;

  stan::mcmc::adapt_diag_e_nuts<io_example_model_namespace::io_example_model, rng_t>
    sampler(model, base_rng);
  sampler.seed(real);

  // Writer
  std::stringstream sample_stream;
  std::stringstream diagnostic_stream;
  std::stringstream message_stream;

  writer_t sample_writer(sample_stream, "# ");
  writer_t diagnostic_writer(diagnostic_stream, "# ");
  writer_t message_writer(message_stream, "# ");

  stan::services::sample::mcmc_writer<io_example_model_namespace::io_example_model,
                                      writer_t,
                                      writer_t,
                                      writer_t>
    writer(sample_writer, diagnostic_writer, message_writer);

  double warm = 0.193933;
  double sampling = 0.483830;

  writer.write_timing(warm, sampling, sample_writer);

  std::stringstream expected_stream;
  expected_stream << "# " << std::endl;
  expected_stream << "#  Elapsed Time: " << warm << " seconds (Warm-up)" << std::endl;
  expected_stream << "#                " << sampling << " seconds (Sampling)" << std::endl;
  expected_stream << "#                " << warm + sampling << " seconds (Total)" << std::endl;
  expected_stream << "# " << std::endl;

  std::string line;
  std::string expected_line;

  // Line 1
  std::getline(expected_stream, expected_line);

  std::getline(sample_stream, line);
  EXPECT_EQ(expected_line, line);

  std::getline(diagnostic_stream, line);
  EXPECT_EQ("", line);

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

  EXPECT_EQ("", message_stream.str());
  EXPECT_EQ("", output.str());
}

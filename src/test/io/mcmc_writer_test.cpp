#include <stan/io/mcmc_writer.hpp>
#include <test/io/test_model/example.cpp>

#include <vector>
#include <boost/random/additive_combine.hpp>

#include <stan/mcmc/sample.hpp>
#include <stan/mcmc/hmc/nuts/diag_e_nuts.hpp>

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
  
  stan::mcmc::sample sample(real, discrete, 3.14, 1.41);
  
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
  
  std::cout << sample_stream << std::endl;
  std::cout << diagnostic_stream << std::endl;
  
  EXPECT_EQ(1, 1);
  
  /*
   EXPECT_EQ(1, metadata.stan_version_major);
   EXPECT_EQ(3, metadata.stan_version_minor);
   EXPECT_EQ(0, metadata.stan_version_patch);
   
   EXPECT_EQ("blocker_model", metadata.model);
   EXPECT_EQ("src/models/bugs_examples/vol1/blocker/blocker.data.R", metadata.data);
   EXPECT_EQ("src/models/bugs_examples/vol1/blocker/blocker.init.R", metadata.init);
   EXPECT_FALSE(metadata.append_samples);
   EXPECT_FALSE(metadata.save_warmup);
   EXPECT_EQ(4085885484U, metadata.seed);
   EXPECT_FALSE(metadata.random_seed);
   EXPECT_EQ(1U, metadata.chain_id);
   EXPECT_EQ(4000U, metadata.iter);
   EXPECT_EQ(2000U, metadata.warmup);
   EXPECT_EQ(2U, metadata.thin);
   EXPECT_FALSE(metadata.equal_step_sizes);
   EXPECT_EQ(-1, metadata.leapfrog_steps);
   EXPECT_EQ(10, metadata.max_treedepth);
   EXPECT_FLOAT_EQ(-1, metadata.epsilon);
   EXPECT_FLOAT_EQ(0, metadata.epsilon_pm);
   EXPECT_FLOAT_EQ(0.5, metadata.delta);
   EXPECT_FLOAT_EQ(0.05, metadata.gamma);
   EXPECT_EQ("NUTS with a diagonal Euclidean metric", metadata.algorithm);
   */
}

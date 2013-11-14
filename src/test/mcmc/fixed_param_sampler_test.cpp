#include <stan/mcmc/chains.hpp>
#include <stan/mcmc/fixed_param_sampler.hpp>
#include <stan/io/stan_csv_reader.hpp>

#include <gtest/gtest.h>
#include <test/models/utility.hpp>


TEST(McmcFixedParamSampler, check_persistency) {
  
  std::vector<std::string> model_path;
  model_path.push_back("src");
  model_path.push_back("test");
  model_path.push_back("mcmc");
  model_path.push_back("models");
  model_path.push_back("proper");
  
  std::string command = convert_model_path(model_path);
  command += " sample algorithm=fixed_param output file=" + convert_model_path(model_path) + ".csv";
  run_command_output command_output;
  
  try {
    command_output = run_command(command);
  } catch(...) {
    ADD_FAILURE() << "Failed running command: " << command;
  }
  
  std::ifstream output_stream;
  output_stream.open( (convert_model_path(model_path) + ".csv").data() );
  
  stan::io::stan_csv parsed_output = stan::io::stan_csv_reader::parse(output_stream);
  stan::mcmc::chains<> chains(parsed_output);
  
  for (int i = 0; i < chains.num_params(); ++i) {
    EXPECT_EQ(chains.samples(0, i).tail(1)(0), chains.samples(0, i).head(1)(0));
  }
  
}


TEST(McmcFixedParamSampler, check_empty) {
  
  std::vector<std::string> model_path;
  model_path.push_back("src");
  model_path.push_back("test");
  model_path.push_back("mcmc");
  model_path.push_back("models");
  model_path.push_back("empty");
  
  std::string command = convert_model_path(model_path);
  command += " sample algorithm=fixed_param output file=" + convert_model_path(model_path) + ".csv";
  run_command_output command_output;
  
  bool success = TRUE;
  
  try {
    command_output = run_command(command);
  } catch(...) {
    success = FALSE;
  }

  EXPECT_EQ(success, TRUE);
  
}
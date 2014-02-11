#include <stan/mcmc/chains.hpp>
#include <stan/mcmc/fixed_param_sampler.hpp>
#include <stan/io/stan_csv_reader.hpp>

#include <gtest/gtest.h>
#include <test/CmdStan/models/utility.hpp>


void test_constant(const Eigen::VectorXd& samples) {
  for (int i = 1; i < samples.size(); i++)
    EXPECT_EQ(samples(0), samples(i));
}

TEST(McmcPersistentSampler, check_persistency) {
  
  std::vector<std::string> model_path;
  model_path.push_back("src");
  model_path.push_back("test");
  model_path.push_back("test-models");
  model_path.push_back("compiled");
  model_path.push_back("CmdStan");
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
    test_constant(chains.samples(0, i));
  }
  
}


TEST(McmcFixedParamSampler, check_empty) {
  
  std::vector<std::string> model_path;
  model_path.push_back("src");
  model_path.push_back("test");
  model_path.push_back("test-models");
  model_path.push_back("compiled");
  model_path.push_back("CmdStan");
  model_path.push_back("empty");
  
  std::string command = convert_model_path(model_path);
  command += " sample algorithm=fixed_param output file=" + convert_model_path(model_path) + ".csv";
  run_command_output command_output;
  
  bool success = true;
  
  try {
    command_output = run_command(command);
  } catch(...) {
    success = false;
  }

  EXPECT_EQ(success, true);
  
}

TEST(McmcFixedParamSampler, check_empty_but_algorithm_not_fixed_param) {
  
  std::vector<std::string> model_path;
  model_path.push_back("src");
  model_path.push_back("test");
  model_path.push_back("test-models");
  model_path.push_back("compiled");
  model_path.push_back("CmdStan");
  model_path.push_back("empty");
  
  std::string command = convert_model_path(model_path);
  command += " sample output file=" + convert_model_path(model_path) + ".csv";
  run_command_output command_output;
  
  bool success = true;
  
  try {
    command_output = run_command(command);
  } catch(...) {
    success = false;
  }

  EXPECT_EQ(success, true);
  EXPECT_NE(0, command_output.err_code);
  char const * errmsg =
    "Must use algorithm=fixed_param for model that has no parameters";
  EXPECT_NE(std::string::npos, command_output.output.find(errmsg));  
}

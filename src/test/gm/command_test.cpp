#include <gtest/gtest.h>
#include <string>
#include <test/models/utility.hpp>

TEST(StanGmCommand, zero_init_value_fail) {
  std::vector<std::string> model_path;
  model_path.push_back("src");
  model_path.push_back("test");
  model_path.push_back("gm");
  model_path.push_back("model_specs");
  model_path.push_back("compiled");
  model_path.push_back("value_fail");

  std::string command = convert_model_path(model_path) + " --init=0";
  std::string command_output;
  long time;
  
  try {
    command_output = run_command(command, time);
  } catch(...) {
    ADD_FAILURE() << "Failed running command: " << command;
  }
  EXPECT_EQ("Rejecting inititialization at zero because of vanishing density.\n",
            command_output);
}

TEST(StanGmCommand, zero_init_domain_fail) {
  std::vector<std::string> model_path;
  model_path.push_back("src");
  model_path.push_back("test");
  model_path.push_back("gm");
  model_path.push_back("model_specs");
  model_path.push_back("compiled");
  model_path.push_back("domain_fail");

  std::string command = convert_model_path(model_path) + " --init=0";
  std::string command_output;
  long time;

  try {
    command_output = run_command(command, time);
  } catch(...) {
    ADD_FAILURE() << "Failed running command: " << command;
  }
  
  EXPECT_EQ("Rejecting inititialization at zero because of grad_log_prob failure.\n",
            command_output);
}

#include <stan/gm/error_codes.hpp>
#include <gtest/gtest.h>
#include <string>
#include <test/models/utility.hpp>

TEST(StanGmCommand, countMatches) {
  EXPECT_EQ(-1, count_matches("", ""));
  EXPECT_EQ(-1, count_matches("", "abc"));

  EXPECT_EQ(0, count_matches("abc", ""));
  EXPECT_EQ(0, count_matches("abc", "ab"));
  EXPECT_EQ(0, count_matches("abc", "dab"));
  EXPECT_EQ(0, count_matches("abc", "abde"));

  EXPECT_EQ(0, count_matches("aa","a"));
  EXPECT_EQ(1, count_matches("aa","aa"));
  EXPECT_EQ(1, count_matches("aa","aaa"));
  EXPECT_EQ(2, count_matches("aa","aaaa"));
}

void test_sample_prints(const std::string& base_cmd) {
  std::string cmd(base_cmd);
  cmd += " num_samples=100 num_warmup=100";
  std::string cmd_output = run_command(cmd).output;
  // transformed data
  EXPECT_EQ(1, count_matches("x=", cmd_output)); 
  // transformed parameters
  EXPECT_TRUE(count_matches("z=", cmd_output) >= 200); 
  // model
  EXPECT_TRUE(count_matches("y=", cmd_output) >= 200);
  // generated quantities [only on saved iterations, should be num samples]
  EXPECT_TRUE(count_matches("w=", cmd_output) == 100);
}

void test_optimize_prints(const std::string& base_cmd) {
  std::string cmd(base_cmd);
  // cmd += " num_samples=100 num_warmup=100";
  std::string cmd_output = run_command(cmd).output;
  // transformed data
  EXPECT_EQ(1, count_matches("x=", cmd_output)); 
  // transformed parameters
  EXPECT_TRUE(count_matches("z=", cmd_output) >= 1); 
  // model
  EXPECT_TRUE(count_matches("y=", cmd_output) >= 1);
  // generated quantities [only on saved iterations, should be num samples]
  EXPECT_TRUE(count_matches("w=", cmd_output) == 1);
}

TEST(StanGmCommand, printReallyPrints) {
  // files auto-cleaned by clean-all due to location
  std::string cmd
    = "make CC=clang++ O=0 src/test/gm/model_specs/printer";
  run_command(cmd);

  // SAMPLING
  // static HMC
  // + adapt
  test_sample_prints("src/test/gm/model_specs/printer sample algorithm=hmc engine=static metric=unit_e adapt engaged=0");
  test_sample_prints("src/test/gm/model_specs/printer sample algorithm=hmc engine=static metric=diag_e adapt engaged=0");
  test_sample_prints("src/test/gm/model_specs/printer sample algorithm=hmc engine=static metric=dense_e adapt engaged=0");

  // - adapt
  test_sample_prints("src/test/gm/model_specs/printer sample algorithm=hmc engine=static metric=unit_e adapt engaged=1");
  test_sample_prints("src/test/gm/model_specs/printer sample algorithm=hmc engine=static metric=diag_e adapt engaged=1");
  test_sample_prints("src/test/gm/model_specs/printer sample algorithm=hmc engine=static metric=dense_e adapt engaged=1");

  // NUTS
  // + adapt
  test_sample_prints("src/test/gm/model_specs/printer sample algorithm=hmc engine=nuts metric=unit_e adapt engaged=0");
  test_sample_prints("src/test/gm/model_specs/printer sample algorithm=hmc engine=nuts metric=diag_e adapt engaged=0");
  test_sample_prints("src/test/gm/model_specs/printer sample algorithm=hmc engine=nuts metric=dense_e adapt engaged=0");

  // - adapt
  test_sample_prints("src/test/gm/model_specs/printer sample algorithm=hmc engine=nuts metric=unit_e adapt engaged=1");
  test_sample_prints("src/test/gm/model_specs/printer sample algorithm=hmc engine=nuts metric=diag_e adapt engaged=1");
  test_sample_prints("src/test/gm/model_specs/printer sample algorithm=hmc engine=nuts metric=dense_e adapt engaged=1");

  // OPTIMIZATION
  test_optimize_prints("src/test/gm/model_specs/printer optimize algorithm=newton");
  test_optimize_prints("src/test/gm/model_specs/printer optimize algorithm=nesterov");
  test_optimize_prints("src/test/gm/model_specs/printer optimize algorithm=bfgs");
}


TEST(StanGmCommand, zero_init_value_fail) {
  std::vector<std::string> model_path;
  model_path.push_back("src");
  model_path.push_back("test");
  model_path.push_back("gm");
  model_path.push_back("model_specs");
  model_path.push_back("compiled");
  model_path.push_back("value_fail");

  std::string command = convert_model_path(model_path) + " sample init=0 output file=test/gm/samples.csv";
  run_command_output out = run_command(command);
  EXPECT_EQ(int(stan::gm::error_codes::OK), out.err_code);
  EXPECT_EQ("Rejecting inititialization at zero because of vanishing density.\n", 
            out.output)
    << "Failed running: " << out.command;
}

TEST(StanGmCommand, zero_init_domain_fail) {
  std::vector<std::string> model_path;
  model_path.push_back("src");
  model_path.push_back("test");
  model_path.push_back("gm");
  model_path.push_back("model_specs");
  model_path.push_back("compiled");
  model_path.push_back("domain_fail");
  
  std::string command = convert_model_path(model_path) + " sample init=0 output file=test/gm/samples.csv";
  
  run_command_output out = run_command(command);
  EXPECT_EQ(int(stan::gm::error_codes::OK), out.err_code);
  EXPECT_EQ("Rejecting inititialization at zero because of log_prob_grad failure.\n",
            out.output)
    << "Failed running: " << out.command;
}

TEST(StanGmCommand, user_init_value_fail) {
  std::vector<std::string> model_path;
  model_path.push_back("src");
  model_path.push_back("test");
  model_path.push_back("gm");
  model_path.push_back("model_specs");
  model_path.push_back("compiled");
  model_path.push_back("value_fail");
  
  std::vector<std::string> init_path;
  init_path.push_back("src");
  init_path.push_back("test");
  init_path.push_back("gm");
  init_path.push_back("model_specs");
  init_path.push_back("compiled");
  init_path.push_back("value_fail.init.R");
  
  std::string command = convert_model_path(model_path)
    + " sample init=" + convert_model_path(init_path)
    + " output file=test/gm/samples.csv";

  run_command_output out = run_command(command);
  EXPECT_EQ(int(stan::gm::error_codes::OK), out.err_code);
  EXPECT_EQ("Rejecting user-specified inititialization because of vanishing density.\n",
            out.output)
    << "Failed running: " << out.command;
}

TEST(StanGmCommand, user_init_domain_fail) {
  std::vector<std::string> model_path;
  model_path.push_back("src");
  model_path.push_back("test");
  model_path.push_back("gm");
  model_path.push_back("model_specs");
  model_path.push_back("compiled");
  model_path.push_back("domain_fail");
  
  std::vector<std::string> init_path;
  init_path.push_back("src");
  init_path.push_back("test");
  init_path.push_back("gm");
  init_path.push_back("model_specs");
  init_path.push_back("compiled");
  init_path.push_back("domain_fail.init.R");
  
  std::string command = convert_model_path(model_path)
    + " sample init=" + convert_model_path(init_path)
    + " output file=test/gm/samples.csv";
  
  run_command_output out = run_command(command);
  EXPECT_EQ(int(stan::gm::error_codes::OK), out.err_code);
  EXPECT_EQ("Rejecting user-specified inititialization because of log_prob_grad failure.\n",
            out.output)
    << "Failed running: " << out.command;
}

TEST(StanGmCommand, CheckCommand_default) {
  std::vector<std::string> model_path;
  model_path.push_back("src");
  model_path.push_back("test");
  model_path.push_back("gm");
  model_path.push_back("model_specs");
  model_path.push_back("compiled");
  model_path.push_back("domain_fail"); // can use any model here
   
  std::string command = convert_model_path(model_path);
  run_command_output out = run_command(command);
  EXPECT_EQ(int(stan::gm::error_codes::USAGE), out.err_code);
}

TEST(StanGmCommand, CheckCommand_help) {
  std::vector<std::string> model_path;
  model_path.push_back("src");
  model_path.push_back("test");
  model_path.push_back("gm");
  model_path.push_back("model_specs");
  model_path.push_back("compiled");
  model_path.push_back("domain_fail"); // can use any model here
  
   std::string command = convert_model_path(model_path) + " help";

  run_command_output out = run_command(command);
  EXPECT_EQ(int(stan::gm::error_codes::OK), out.err_code);
}

TEST(StanGmCommand, CheckCommand_unrecognized_argument) {
  std::vector<std::string> model_path;
  model_path.push_back("src");
  model_path.push_back("test");
  model_path.push_back("gm");
  model_path.push_back("model_specs");
  model_path.push_back("compiled");
  model_path.push_back("domain_fail"); // can use any model here
  
  std::string command = convert_model_path(model_path) + " foo";

  run_command_output out = run_command(command);
  EXPECT_EQ(int(stan::gm::error_codes::USAGE), out.err_code);
}

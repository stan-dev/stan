#include <gtest/gtest.h>
#include <test/CmdStan/models/utility.hpp>
#include <stan/mcmc/chains.hpp>

class CmdStan : public testing::Test {
 public:
  void SetUp() {
    std::vector<std::string> model_path;
    model_path.push_back("src");
    model_path.push_back("test");
    model_path.push_back("test-models");
    model_path.push_back("compiled");
    model_path.push_back("CmdStan");
    model_path.push_back("optimization_output");
    
    output_file = "test/CmdStan/output.csv";

    base_command = convert_model_path(model_path)
      + " output file=" + output_file;

    y11 = "y[1,1]";
    y12 = "y[1,2]";
    y21 = "y[2,1]";
    y22 = "y[2,2]";
  }

  stan::mcmc::chains<> parse_output_file() {
    std::ifstream output_stream;
    output_stream.open(output_file.data());
  
    stan::io::stan_csv parsed_output = stan::io::stan_csv_reader::parse(output_stream);
    stan::mcmc::chains<> chains(parsed_output);
    output_stream.close();
    return chains;
  }

  std::string base_command;
  std::string output_file;
  std::string y11, y12, y21, y22;
};


TEST_F(CmdStan, optimize_default) {
  run_command_output out = run_command(base_command + " optimize");

  ASSERT_EQ(0, out.err_code);

  stan::mcmc::chains<> chains = parse_output_file();
  ASSERT_EQ(1, chains.num_chains());
  ASSERT_EQ(1, chains.num_samples());

  EXPECT_FLOAT_EQ(1, chains.samples(y11)[0]);
  EXPECT_FLOAT_EQ(100, chains.samples(y21)[0]);
  EXPECT_FLOAT_EQ(10000, chains.samples(y12)[0]);
  EXPECT_FLOAT_EQ(1000000, chains.samples(y22)[0]);
}

TEST_F(CmdStan, optimize_nesterov) {
  run_command_output out = run_command(base_command + " optimize algorithm=nesterov");

  ASSERT_EQ(0, out.err_code);

  stan::mcmc::chains<> chains = parse_output_file();
  ASSERT_EQ(1, chains.num_chains());
  ASSERT_EQ(1, chains.num_samples());

  EXPECT_NEAR(1, chains.samples(y11)[0], 0.5);
  EXPECT_NEAR(100, chains.samples(y21)[0], 5);
  EXPECT_NEAR(10000, chains.samples(y12)[0], 50);
  EXPECT_NEAR(1000000, chains.samples(y22)[0], 500);
}

TEST_F(CmdStan, optimize_bfgs) {
  run_command_output out = run_command(base_command + " optimize algorithm=bfgs");

  ASSERT_EQ(0, out.err_code);

  stan::mcmc::chains<> chains = parse_output_file();
  ASSERT_EQ(1, chains.num_chains());
  ASSERT_EQ(1, chains.num_samples());

  EXPECT_FLOAT_EQ(1, chains.samples(y11)[0]);
  EXPECT_FLOAT_EQ(100, chains.samples(y21)[0]);
  EXPECT_FLOAT_EQ(10000, chains.samples(y12)[0]);
  EXPECT_FLOAT_EQ(1000000, chains.samples(y22)[0]);
}

TEST_F(CmdStan, optimize_newton) {
  run_command_output out = run_command(base_command + " optimize algorithm=newton");

  ASSERT_EQ(0, out.err_code);

  stan::mcmc::chains<> chains = parse_output_file();
  ASSERT_EQ(1, chains.num_chains());
  ASSERT_EQ(1, chains.num_samples());

  EXPECT_FLOAT_EQ(1, chains.samples(y11)[0]);
  EXPECT_FLOAT_EQ(100, chains.samples(y21)[0]);
  EXPECT_FLOAT_EQ(10000, chains.samples(y12)[0]);
  EXPECT_FLOAT_EQ(1000000, chains.samples(y22)[0]);
}

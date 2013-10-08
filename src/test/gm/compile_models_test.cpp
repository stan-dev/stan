#include <gtest/gtest.h>
#include <stan/gm/error_codes.hpp>
#include <test/models/utility.hpp>
#include <stan/mcmc/chains.hpp>

TEST(gm,compile_models) {
  SUCCEED() 
    << "Model compilation done through makefile dependencies." << std::endl
    << "Should have compiled: src/test/gm/model_specs/compiled/*.stan";
}

TEST(gm,issue91_segfault_printing_uninitialized) {
  std::vector<std::string> model_path;
  model_path.push_back("src");
  model_path.push_back("test");
  model_path.push_back("gm");
  model_path.push_back("model_specs");
  model_path.push_back("compiled");
  model_path.push_back("issue91");

  std::string command 
    = convert_model_path(model_path)
    + " sample num_warmup=0 num_samples=0"
    + " output file=" + convert_model_path(model_path) + ".csv";
  
  run_command_output out = run_command(command);
  EXPECT_EQ(int(stan::gm::error_codes::OK), out.err_code);
  EXPECT_FALSE(out.hasError);
}

TEST(gm,issue109_csv_header_consistent_with_samples) {
  std::vector<std::string> model_path;
  model_path.push_back("src");
  model_path.push_back("test");
  model_path.push_back("gm");
  model_path.push_back("model_specs");
  model_path.push_back("compiled");
  model_path.push_back("issue109");

  std::string path = convert_model_path(model_path);
  std::string samples = path + ".csv";

  std::string command
    = path
    + " sample num_warmup=0 num_samples=1"
    + " output file=" + samples;

  run_command_output out = run_command(command);
  EXPECT_EQ(int(stan::gm::error_codes::OK), out.err_code);
  EXPECT_FALSE(out.hasError);

  std::ifstream ifstream;
  ifstream.open(samples.c_str());
  stan::mcmc::chains<> chains(stan::io::stan_csv_reader::parse(ifstream));
  ifstream.close();
  
  EXPECT_EQ(1, chains.num_samples());
  EXPECT_FLOAT_EQ(1, chains.samples("z[1,1]")(0));
  EXPECT_FLOAT_EQ(2, chains.samples("z[1,2]")(0));
  EXPECT_FLOAT_EQ(3, chains.samples("z[2,1]")(0));
  EXPECT_FLOAT_EQ(4, chains.samples("z[2,2]")(0));
  EXPECT_FLOAT_EQ(1, chains.samples("z_mat[1,1]")(0));
  EXPECT_FLOAT_EQ(2, chains.samples("z_mat[1,2]")(0));
  EXPECT_FLOAT_EQ(3, chains.samples("z_mat[2,1]")(0));
  EXPECT_FLOAT_EQ(4, chains.samples("z_mat[2,2]")(0));
}

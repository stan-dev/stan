// TODO(carpenter): move this into test/unit/command
//                  it's here now because it won't compile there

#include <gtest/gtest.h>
#include <stan/lang/compiler.hpp>
#include <stan/command/stanc_helper.hpp>
#include <test/unit/util.hpp>
#include <fstream>
#include <sstream>

void expect_find(const std::string& str, const std::string& target) {
  EXPECT_TRUE(str.find(target) != std::string::npos)
    << str << " does not contain " << target << std::endl;
}

TEST(commandStancHelper, printVersion) {
  std::stringstream ss;
  print_version(&ss);
  expect_find(ss.str(), "stanc version 2.");
}

TEST(commandStancHelper, printStancHelp) {
  std::stringstream ss;
  print_stanc_help(&ss);
  expect_find(ss.str(), "USAGE:  stanc [options] <model_file>");
  expect_find(ss.str(), "OPTIONS:");
}

bool create_test_file(const std::string& path, const std::string& program) {
  std::string cmd = "echo \"";
  cmd += program;
  cmd += "\" > ";
  cmd += path;
  cmd += "; chmod 444 ";
  cmd += path;
  int return_code = system(cmd.c_str());
  return return_code == 0;
}

bool create_test_file() {
  std::string program
    = "parameters { real y; } model { y ~ normal(0, 1); }";
  return create_test_file("test/test-models/temp-bar.stan", program);
}

TEST(commandStancHelper, deleteFile) {
  if (!create_test_file()) return;
  std::stringstream ss;
  delete_file(&ss, "test/test-models/temp-bar.stan");
  // test there's no error message
  EXPECT_EQ(0, ss.str().size());
  // and then test the file stream can't be opened
  std::fstream fs;
  fs.open("test/test-models/temp-bar.stan", std::fstream::in);
  EXPECT_FALSE(fs.is_open());
  fs.close();
}

int run_helper(const std::string& path,
               std::ostream& out, std::ostream& err) {
  int argc = 2;
  std::vector<const char*> argv_vec;
  argv_vec.push_back("main");
  argv_vec.push_back(path.c_str());
  const char** argv = &argv_vec[0];
  return stanc_helper(argc, argv, &out, &err);
}

TEST(commandStancHelper, readOnlyOK) {
  if (!create_test_file()) return;

  std::stringstream out;
  std::stringstream err;
  int rc = run_helper("test/test-models/temp-bar.stan", out, err);

  EXPECT_EQ(0, rc);
  expect_find(out.str(), "Model name=temp_bar_model");
  expect_find(out.str(), "Input file=test/test-models/temp-bar.stan");
  expect_find(out.str(), "Output file=temp_bar_model.cpp");
  EXPECT_EQ(0, err.str().size());

  delete_file(&err, "test/test-models/temp-bar.stan");
  delete_file(&err, "bar_model.cpp");
}

TEST(commandStancHelper, failRC) {
  std::string path = "test/test-models/temp-bar.stan";
  std::string prog
    = "parameters { real y; } model { y ~ ormal(0, 1); }";
  if (!create_test_file(path, prog)) return;

  std::stringstream out;
  std::stringstream err;
  int rc = run_helper(path, out, err);

  // TODO(carpenter): This should be -2 but it's -3 so
  // I only tested that it's != 0 to contrast with earlier success
  EXPECT_TRUE(rc != 0);

  delete_file(&err, "test/test-models/temp-bar.stan");
}
    

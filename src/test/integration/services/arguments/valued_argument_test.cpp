#include <stan/services/arguments/valued_argument.hpp>
#include <gtest/gtest.h>

class test_arg_impl : public stan::services::valued_argument {
  std::string print_value() {
    return "";
  }
  std::string print_valid() {
    return "";
  }
  bool is_default() {
    return true;
  }
};


class StanServicesArgumentsValuedArgument : public testing::Test {
public:
  void SetUp () {
    arg = new test_arg_impl;
  }
  void TearDown() {
    delete(arg);
  }
  
  stan::services::argument *arg;
};


TEST_F(StanServicesArgumentsValuedArgument,Constructor) {
  // test fixture would have created the argument.
}

TEST_F(StanServicesArgumentsValuedArgument,name) {
  EXPECT_EQ("", arg->name());
}

TEST_F(StanServicesArgumentsValuedArgument,description) {
  EXPECT_EQ("", arg->description());
}

TEST_F(StanServicesArgumentsValuedArgument,print) {
  // FIXME: write test
}

TEST_F(StanServicesArgumentsValuedArgument,print_help) {
  // FIXME: write test
}

TEST_F(StanServicesArgumentsValuedArgument,parse_args) {
  bool return_value;
  std::vector<std::string> args;
  bool help_flag;
  
  return_value = false;
  args.clear();
  help_flag = false;
  return_value = arg->parse_args(args,0,0,help_flag);
  
  EXPECT_TRUE(return_value);
  EXPECT_FALSE(help_flag);
  EXPECT_EQ(0U, args.size());

  
  return_value = false;
  args.clear();
  args.push_back("help");
  help_flag = false;
  return_value = arg->parse_args(args,0,0,help_flag);
  
  EXPECT_TRUE(return_value);
  EXPECT_FALSE(help_flag);
  EXPECT_EQ(1U, args.size());
}

TEST_F(StanServicesArgumentsValuedArgument,parse_args_unexpected) {
  bool return_value;
  std::vector<std::string> args;
  bool help_flag;

  return_value = false;
  args.clear();
  args.push_back("foo=bar");
  help_flag = false;
  return_value = arg->parse_args(args,0,0,help_flag);
  
  EXPECT_TRUE(return_value);
  EXPECT_FALSE(help_flag);
  EXPECT_EQ(1U, args.size());
}

TEST_F(StanServicesArgumentsValuedArgument,arg) {
  EXPECT_EQ(0, arg->arg(""));
  EXPECT_EQ(0, arg->arg("foo"));
}

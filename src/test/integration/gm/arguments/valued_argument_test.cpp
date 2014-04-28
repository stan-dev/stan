#include <stan/gm/arguments/valued_argument.hpp>
#include <gtest/gtest.h>

class test_arg_impl : public stan::gm::valued_argument {
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


class StanGmArgumentsValuedArgument : public testing::Test {
public:
  void SetUp () {
    arg = new test_arg_impl;
  }
  void TearDown() {
    delete(arg);
  }
  
  stan::gm::argument *arg;
};


TEST_F(StanGmArgumentsValuedArgument,Constructor) {
  // test fixture would have created the argument.
}

TEST_F(StanGmArgumentsValuedArgument,name) {
  EXPECT_EQ("", arg->name());
}

TEST_F(StanGmArgumentsValuedArgument,description) {
  EXPECT_EQ("", arg->description());
}

TEST_F(StanGmArgumentsValuedArgument,print) {
  // FIXME: write test
}

TEST_F(StanGmArgumentsValuedArgument,print_help) {
  // FIXME: write test
}

TEST_F(StanGmArgumentsValuedArgument,parse_args) {
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

TEST_F(StanGmArgumentsValuedArgument,parse_args_unexpected) {
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

TEST_F(StanGmArgumentsValuedArgument,arg) {
  EXPECT_EQ(0, arg->arg(""));
  EXPECT_EQ(0, arg->arg("foo"));
}

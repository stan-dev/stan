#include <stan/gm/arguments/argument.hpp>
#include <gtest/gtest.h>

class test_arg_impl : public stan::gm::argument {
  void print(std::ostream* s, int depth, const std::string prefix) {}
  void print_help(std::ostream* s, int depth, bool recurse) {}
};

class StanGmArgumentsArgument : public testing::Test {
public:
  void SetUp () {
    arg = new test_arg_impl;
  }
  void TearDown() {
    delete(arg);
  }
  
  stan::gm::argument *arg;
};


TEST_F(StanGmArgumentsArgument,Constructor) {
  // test fixture would have created the argument.
}

TEST_F(StanGmArgumentsArgument,name) {
  EXPECT_EQ("", arg->name());
}

TEST_F(StanGmArgumentsArgument,description) {
  EXPECT_EQ("", arg->description());
}

TEST_F(StanGmArgumentsArgument,split_arg) {
  std::string arg_string;
  std::string name;
  std::string value;

  arg_string = "";
  arg->split_arg(arg_string, name, value);
  EXPECT_EQ("", name);
  EXPECT_EQ("", value);
  
  arg_string = "foo=bar";
  arg->split_arg(arg_string, name, value);
  EXPECT_EQ("foo", name);
  EXPECT_EQ("bar", value);


  arg_string = " foo=bar ";
  arg->split_arg(arg_string, name, value);
  EXPECT_EQ(" foo", name);
  EXPECT_EQ("bar ", value);

  arg_string = "\nfoo=\rbar\t";
  arg->split_arg(arg_string, name, value);
  EXPECT_EQ("\nfoo", name);
  EXPECT_EQ("\rbar\t", value);
}


#include <stan/services/arguments/argument.hpp>
#include <gtest/gtest.h>
#include <stan/interface_callbacks/writer/base_writer.hpp>

class test_arg_impl : public stan::services::argument {
  void print(stan::interface_callbacks::writer::base_writer& w,
             int depth, const std::string& prefix) {}
  void print_help(stan::interface_callbacks::writer::base_writer& w,
                  int depth, bool recurse) {}
};

class StanServicesArgumentsArgument : public testing::Test {
public:
  void SetUp () {
    arg = new test_arg_impl;
  }
  void TearDown() {
    delete(arg);
  }
  
  stan::services::argument *arg;
};


TEST_F(StanServicesArgumentsArgument,Constructor) {
  // test fixture would have created the argument.
}

TEST_F(StanServicesArgumentsArgument,name) {
  EXPECT_EQ("", arg->name());
}

TEST_F(StanServicesArgumentsArgument,description) {
  EXPECT_EQ("", arg->description());
}

TEST_F(StanServicesArgumentsArgument,split_arg) {
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


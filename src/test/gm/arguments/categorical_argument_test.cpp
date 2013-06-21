#include <stan/gm/arguments/categorical_argument.hpp>
#include <gtest/gtest.h>

class StanGmArgumentsCategoricalArgument : public testing::Test {
public:
  void SetUp () {
    arg = new stan::gm::categorical_argument;
  }
  void TearDown() {
    delete(arg);
  }
  
  stan::gm::argument *arg;
};


TEST_F(StanGmArgumentsCategoricalArgument,Constructor) {
  // test fixture would have created the argument.
}

TEST_F(StanGmArgumentsCategoricalArgument,name) {
  EXPECT_EQ("", arg->name());
}

TEST_F(StanGmArgumentsCategoricalArgument,description) {
  EXPECT_EQ("", arg->description());
}

TEST_F(StanGmArgumentsCategoricalArgument,print) {
  // FIXME: write test
}

TEST_F(StanGmArgumentsCategoricalArgument,print_help) {
  // FIXME: write test
}

TEST_F(StanGmArgumentsCategoricalArgument,parse_args) {
  bool return_value;
  std::vector<std::string> args;
  bool help_flag;
  
  return_value = false;
  args.clear();
  help_flag = false;
  return_value = arg->parse_args(args,0,0,help_flag);
  
  EXPECT_TRUE(return_value);
  EXPECT_FALSE(help_flag);
  EXPECT_EQ(0, args.size());

  
  return_value = false;
  args.clear();
  args.push_back("help");
  help_flag = false;
  return_value = arg->parse_args(args,0,0,help_flag);
  
  EXPECT_TRUE(return_value);
  EXPECT_TRUE(help_flag);
  EXPECT_EQ(0, args.size());


  return_value = false;
  args.clear();
  args.push_back("help-all");
  help_flag = false;
  return_value = arg->parse_args(args,0,0,help_flag);
  
  EXPECT_TRUE(return_value);
  EXPECT_TRUE(help_flag);
  EXPECT_EQ(0, args.size());
}

TEST_F(StanGmArgumentsCategoricalArgument,parse_args_unexpected) {
  bool return_value;
  std::vector<std::string> args;
  bool help_flag;

  return_value = false;
  args.clear();
  args.push_back("foo=bar");
  help_flag = false;
  return_value = arg->parse_args(args,0,0,help_flag);
  
  EXPECT_FALSE(return_value);
  EXPECT_FALSE(help_flag);
  EXPECT_EQ(1, args.size());
  

  return_value = false;
  args.clear();
  args.push_back("help");
  args.push_back("foo=bar");
  help_flag = false;
  return_value = arg->parse_args(args,0,0,help_flag);
  
  EXPECT_FALSE(return_value);
  EXPECT_FALSE(help_flag);
  EXPECT_EQ(2, args.size());


  return_value = false;
  args.clear();
  args.push_back("foo=bar");
  args.push_back("help");
  help_flag = false;
  return_value = arg->parse_args(args,0,0,help_flag);
  
  EXPECT_TRUE(return_value);
  EXPECT_TRUE(help_flag);
  EXPECT_EQ(0, args.size());


  return_value = false;
  args.clear();
  args.push_back("foo");
  help_flag = false;
  return_value = arg->parse_args(args,0,0,help_flag);
  
  EXPECT_FALSE(return_value);
  EXPECT_FALSE(help_flag);
  EXPECT_EQ(1, args.size());
}

TEST_F(StanGmArgumentsCategoricalArgument,arg) {
  EXPECT_EQ(0, arg->arg(""));
  EXPECT_EQ(0, arg->arg("foo"));
}

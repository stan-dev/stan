#include <gtest/gtest.h>
#include <stan/gm/arguments/singleton_argument.hpp>


template <typename T>
T argument_value() {
  return 0;
}

template <typename T>
std::string argument_string() {
  return boost::lexical_cast<std::string>(argument_value<T>());
}

template<>
double argument_value<double>() {
  return 1.234;
}

template<>
int argument_value<int>() {
  return 567;
}

template<>
bool argument_value<bool>() {
  return true;
}

template<>
std::string argument_value<std::string>() {
  return "value";
}



template <typename T>
class StanGmArgumentsSingleton : public ::testing::Test {
public:
  StanGmArgumentsSingleton() 
    : arg(new stan::gm::singleton_argument<T>("argument")) { }
  
  virtual ~StanGmArgumentsSingleton() {
    delete(arg);
  }
  
  stan::gm::argument *arg;
};

TYPED_TEST_CASE_P(StanGmArgumentsSingleton);

TYPED_TEST_P(StanGmArgumentsSingleton, constructor) {
  // test fixture would have created the argument
}

TYPED_TEST_P(StanGmArgumentsSingleton, name) {
  EXPECT_EQ("argument", this->arg->name());
}

TYPED_TEST_P(StanGmArgumentsSingleton, description) {
  EXPECT_EQ("", this->arg->description());
}

TYPED_TEST_P(StanGmArgumentsSingleton, print) {
  // FIXME: write test
}

TYPED_TEST_P(StanGmArgumentsSingleton, print_help) {
  // FIXME: write test
}

TYPED_TEST_P(StanGmArgumentsSingleton, parse_args) {
  bool return_value;
  std::vector<std::string> args;
  bool help_flag;
  
  return_value = false;
  args.clear();
  help_flag = false;
  return_value = this->arg->parse_args(args,0,0,help_flag);
  
  EXPECT_TRUE(return_value);
  EXPECT_FALSE(help_flag);
  EXPECT_EQ(0U, args.size());

  return_value = false;
  args.clear();
  args.push_back("help");
  help_flag = false;
  return_value = this->arg->parse_args(args,0,0,help_flag);
  
  EXPECT_TRUE(return_value);
  EXPECT_TRUE(help_flag);
  EXPECT_EQ(0U, args.size());

  return_value = false;
  args.clear();
  args.push_back("help-all");
  help_flag = false;
  return_value = this->arg->parse_args(args,0,0,help_flag);
  
  EXPECT_TRUE(return_value);
  EXPECT_TRUE(help_flag);
  EXPECT_EQ(0U, args.size());


  return_value = false;
  args.clear();
  args.push_back("argument=" + argument_string<TypeParam>());
  help_flag = false;
  return_value = this->arg->parse_args(args,0,0,help_flag);
  
  EXPECT_TRUE(return_value);
  EXPECT_FALSE(help_flag);
  EXPECT_EQ(0U, args.size());
  EXPECT_EQ(argument_value<TypeParam>(), 
            static_cast<stan::gm::singleton_argument<TypeParam>*>(this->arg)->value());
}

TYPED_TEST_P(StanGmArgumentsSingleton, parse_args_unexpected) {
  bool return_value;
  std::vector<std::string> args;
  bool help_flag;
  
  return_value = false;
  args.clear();
  args.push_back("foo=bar");
  help_flag = false;
  return_value = this->arg->parse_args(args,0,0,help_flag);
  
  EXPECT_TRUE(return_value);
  EXPECT_FALSE(help_flag);
  EXPECT_EQ(1U, args.size());
}

TYPED_TEST_P(StanGmArgumentsSingleton, argument_lookup) {
  // EXPECT_EQ(0, this->arg->arg(""));
  // EXPECT_EQ(0, this->arg->arg("foo"));
}

REGISTER_TYPED_TEST_CASE_P(StanGmArgumentsSingleton,
                           constructor,
                           name,
                           description,
                           print,
                           print_help,
                           parse_args,
                           parse_args_unexpected,
                           argument_lookup);

INSTANTIATE_TYPED_TEST_CASE_P(real, StanGmArgumentsSingleton, double);
INSTANTIATE_TYPED_TEST_CASE_P(int, StanGmArgumentsSingleton, int);
INSTANTIATE_TYPED_TEST_CASE_P(bool, StanGmArgumentsSingleton, bool);
INSTANTIATE_TYPED_TEST_CASE_P(string, StanGmArgumentsSingleton, std::string);


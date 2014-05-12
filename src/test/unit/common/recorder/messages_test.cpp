#include <gtest/gtest.h>
#include <stan/common/recorder/messages.hpp>
#include <sstream>
#include <vector>

class StanCommonRecorder: public ::testing::Test {
public:
  StanCommonRecorder()
    : ss(), recorder(&ss, "prefix") {}
  void SetUp() { }
  
  void TearDown() { }
  
  std::stringstream ss;
  stan::common::recorder::messages recorder;
};

TEST_F(StanCommonRecorder, messages_vector_double) {
  std::vector<double> x;
  for (int i = 1; i < 10; i++) 
    x.push_back(i);
  EXPECT_NO_THROW(recorder(x));

  x.clear();
  for (int i = 1; i < 10; i++) 
    x.push_back(10*i);
  
  EXPECT_NO_THROW(recorder(x));
  
  x.clear();
  x.push_back(1);
  EXPECT_NO_THROW(recorder(x));

  EXPECT_EQ("", ss.str());
}

TEST_F(StanCommonRecorder, messages_vector_string) {
  std::vector<std::string> y;

  y.push_back("abc");
  y.push_back("def");
  EXPECT_NO_THROW(recorder(y));
  
  y.clear();
  y.push_back("ghi");
  EXPECT_NO_THROW(recorder(y));

  EXPECT_EQ("", ss.str());
}

TEST_F(StanCommonRecorder, messages_string) {
  std::string x;

  x = "abcd";
  EXPECT_NO_THROW(recorder(x));
  
  EXPECT_EQ("prefix" + x + "\n", ss.str());
}

TEST_F(StanCommonRecorder, messages_noargs) {
  EXPECT_NO_THROW(recorder());
  EXPECT_NO_THROW(recorder());
  
  EXPECT_EQ("\n\n", ss.str());
}

TEST_F(StanCommonRecorder, messages_is_recording) {
  EXPECT_TRUE(recorder.is_recording());
  
  EXPECT_EQ("", ss.str());
}


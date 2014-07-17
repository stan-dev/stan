#include <gtest/gtest.h>
#include <stan/common/recorder/no_op.hpp>
#include <sstream>
#include <vector>

class StanCommonRecorder : public ::testing::Test {
public:
  void SetUp() { }
  
  void TearDown() { }
  
  stan::common::recorder::no_op recorder;
};

TEST_F(StanCommonRecorder, no_op_vector_double) {
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
}

TEST_F(StanCommonRecorder, no_op_vector_string) {
  std::vector<std::string> y;

  y.push_back("abc");
  y.push_back("def");
  EXPECT_NO_THROW(recorder(y));
  
  y.clear();
  y.push_back("ghi");
  EXPECT_NO_THROW(recorder(y));
}

TEST_F(StanCommonRecorder, no_op_string) {
  std::string x;

  x = "abcd";
  EXPECT_NO_THROW(recorder(x));
}

TEST_F(StanCommonRecorder, no_op_noargs) {
  EXPECT_NO_THROW(recorder());
  EXPECT_NO_THROW(recorder());
}

TEST_F(StanCommonRecorder, no_op_is_recording) {
  EXPECT_FALSE(recorder.is_recording());
}

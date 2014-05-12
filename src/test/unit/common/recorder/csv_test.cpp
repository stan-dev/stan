#include <gtest/gtest.h>
#include <stan/common/recorder/csv.hpp>
#include <sstream>
#include <vector>

class StanCommonRecorder : public ::testing::Test {
public:
  StanCommonRecorder() :
    prefix("prefix"),
    recorder(&ss, prefix) { }
  
  void SetUp() {
    ss.str("");
  }
  
  void TearDown() {
  }
  
  const std::string prefix;
  std::stringstream ss;
  stan::common::recorder::csv recorder;
};



TEST_F(StanCommonRecorder, csv_vector_double) {
  ASSERT_EQ("", ss.str());
  std::vector<double> x;
  std::string expected;
  

  expected = "1,2,3,4,5,6,7,8,9\n";
  for (int i = 1; i < 10; i++) 
    x.push_back(i);
  recorder(x);
  EXPECT_EQ(expected, ss.str());

  
  expected += "10,20,30,40,50,60,70,80,90\n";
  x.clear();
  for (int i = 1; i < 10; i++) 
    x.push_back(10*i);
  
  recorder(x);
  EXPECT_EQ(expected, ss.str());


  ss.str("");
  expected = "1\n";
  x.clear();
  x.push_back(1);
  recorder(x);
  EXPECT_EQ(expected, ss.str());
}

TEST_F(StanCommonRecorder, csv_vector_string) {
  ASSERT_EQ("", ss.str());
  
  std::vector<std::string> y;
  std::string expected;
  
  expected = "abc,def\n";
  y.push_back("abc");
  y.push_back("def");
  recorder(y);
  EXPECT_EQ(expected, ss.str());

  expected += "ghi\n";
  y.clear();
  y.push_back("ghi");
  recorder(y);
  EXPECT_EQ(expected, ss.str());
}

TEST_F(StanCommonRecorder, csv_string) {
  ASSERT_EQ("", ss.str());
  std::string x;
  std::string expected;

  x = "abcd";
  expected = prefix + x + "\n";
  recorder(x);
  EXPECT_EQ(expected, ss.str());
}

TEST_F(StanCommonRecorder, csv_noargs) {
  ASSERT_EQ("", ss.str());
  
  recorder();
  EXPECT_EQ("\n", ss.str());

  recorder();
  EXPECT_EQ("\n\n", ss.str());
}

TEST_F(StanCommonRecorder, csv_is_recording) {
  EXPECT_TRUE(recorder.is_recording());
  
  stan::common::recorder::csv null_recorder(0, "");
  EXPECT_FALSE(null_recorder.is_recording());
}

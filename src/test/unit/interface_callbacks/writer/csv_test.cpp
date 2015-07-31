#include <gtest/gtest.h>
#include <stan/interface_callbacks/writer/csv.hpp>
#include <sstream>
#include <vector>

class StanInterfaceWriter : public ::testing::Test {
public:
  StanInterfaceWriter() :
    prefix("prefix"),
    writer(&ss, prefix) { }
  
  void SetUp() {
    ss.str("");
  }
  
  void TearDown() {
  }
  
  const std::string prefix;
  std::stringstream ss;
  stan::interface_callbacks::writer::csv writer;
};



TEST_F(StanInterfaceWriter, csv_vector_double) {
  ASSERT_EQ("", ss.str());
  std::vector<double> x;
  std::string expected;
  

  expected = "1,2,3,4,5,6,7,8,9\n";
  for (int i = 1; i < 10; i++) 
    x.push_back(i);
  writer(x);
  EXPECT_EQ(expected, ss.str());

  
  expected += "10,20,30,40,50,60,70,80,90\n";
  x.clear();
  for (int i = 1; i < 10; i++) 
    x.push_back(10*i);
  
  writer(x);
  EXPECT_EQ(expected, ss.str());


  ss.str("");
  expected = "1\n";
  x.clear();
  x.push_back(1);
  writer(x);
  EXPECT_EQ(expected, ss.str());
}

TEST_F(StanInterfaceWriter, csv_vector_string) {
  ASSERT_EQ("", ss.str());
  
  std::vector<std::string> y;
  std::string expected;
  
  expected = "abc,def\n";
  y.push_back("abc");
  y.push_back("def");
  writer(y);
  EXPECT_EQ(expected, ss.str());

  expected += "ghi\n";
  y.clear();
  y.push_back("ghi");
  writer(y);
  EXPECT_EQ(expected, ss.str());
}

TEST_F(StanInterfaceWriter, csv_string) {
  ASSERT_EQ("", ss.str());
  std::string x;
  std::string expected;

  x = "abcd";
  expected = prefix + x + "\n";
  writer(x);
  EXPECT_EQ(expected, ss.str());
}

TEST_F(StanInterfaceWriter, csv_noargs) {
  ASSERT_EQ("", ss.str());
  
  writer();
  EXPECT_EQ("\n", ss.str());

  writer();
  EXPECT_EQ("\n\n", ss.str());
}

TEST_F(StanInterfaceWriter, csv_is_writing) {
  EXPECT_TRUE(writer.is_writing());
  
  stan::interface_callbacks::writer::csv null_writer(0, "");
  EXPECT_FALSE(null_writer.is_writing());
}

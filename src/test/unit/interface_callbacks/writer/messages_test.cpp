#include <gtest/gtest.h>
#include <stan/interface_callbacks/writer/messages.hpp>
#include <sstream>
#include <vector>

class StanInterfaceWriter: public ::testing::Test {
public:
  StanInterfaceWriter()
    : ss(), writer(&ss, "prefix") {}
  void SetUp() { }
  
  void TearDown() { }
  
  std::stringstream ss;
  stan::interface_callbacks::writer::messages writer;
};

TEST_F(StanInterfaceWriter, messages_vector_double) {
  std::vector<double> x;
  for (int i = 1; i < 10; i++) 
    x.push_back(i);
  EXPECT_NO_THROW(writer(x));

  x.clear();
  for (int i = 1; i < 10; i++) 
    x.push_back(10*i);
  
  EXPECT_NO_THROW(writer(x));
  
  x.clear();
  x.push_back(1);
  EXPECT_NO_THROW(writer(x));

  EXPECT_EQ("", ss.str());
}

TEST_F(StanInterfaceWriter, messages_vector_string) {
  std::vector<std::string> y;

  y.push_back("abc");
  y.push_back("def");
  EXPECT_NO_THROW(writer(y));
  
  y.clear();
  y.push_back("ghi");
  EXPECT_NO_THROW(writer(y));

  EXPECT_EQ("", ss.str());
}

TEST_F(StanInterfaceWriter, messages_string) {
  std::string x;

  x = "abcd";
  EXPECT_NO_THROW(writer(x));
  
  EXPECT_EQ("prefix" + x + "\n", ss.str());
}

TEST_F(StanInterfaceWriter, messages_noargs) {
  EXPECT_NO_THROW(writer());
  EXPECT_NO_THROW(writer());
  
  EXPECT_EQ("\n\n", ss.str());
}

TEST_F(StanInterfaceWriter, messages_is_writing) {
  EXPECT_TRUE(writer.is_writing());
  
  EXPECT_EQ("", ss.str());
}


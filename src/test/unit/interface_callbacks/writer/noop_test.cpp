#include <gtest/gtest.h>
#include <stan/interface_callbacks/writer/noop.hpp>
#include <sstream>
#include <vector>

class StanInterfaceWriter : public ::testing::Test {
public:
  void SetUp() { }
  
  void TearDown() { }
  
  stan::interface_callbacks::writer::noop writer;
};

TEST_F(StanInterfaceWriter, noop_vector_double) {
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
}

TEST_F(StanInterfaceWriter, noop_vector_string) {
  std::vector<std::string> y;

  y.push_back("abc");
  y.push_back("def");
  EXPECT_NO_THROW(writer(y));
  
  y.clear();
  y.push_back("ghi");
  EXPECT_NO_THROW(writer(y));
}

TEST_F(StanInterfaceWriter, noop_string) {
  std::string x;

  x = "abcd";
  EXPECT_NO_THROW(writer(x));
}

TEST_F(StanInterfaceWriter, noop_noargs) {
  EXPECT_NO_THROW(writer());
  EXPECT_NO_THROW(writer());
}

TEST_F(StanInterfaceWriter, noop_is_writing) {
  EXPECT_FALSE(writer.is_writing());
}

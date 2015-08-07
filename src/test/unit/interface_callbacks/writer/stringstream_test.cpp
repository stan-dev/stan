#include <gtest/gtest.h>
#include <stan/interface_callbacks/writer/stringstream.hpp>
#include <sstream>
#include <vector>

class WriterStringstream : public ::testing::Test {
public:
  void SetUp() {
    writer = new stan::interface_callbacks::writer::stringstream(ss);
  }
  
  void TearDown() {
    delete(writer);
    ss.str(std::string());
    ss.clear();
  }
  
  stan::interface_callbacks::writer::stringstream* writer;
  std::stringstream ss;
};

TEST_F(WriterStringstream, key_value) {
  writer->operator()("key", 0.0);
  EXPECT_EQ("key = 0\n", ss.str());
}

TEST_F(WriterStringstream, key_value2) {
  writer->operator()("key", "value");
  EXPECT_EQ("key = value\n", ss.str());
}

TEST_F(WriterStringstream, key_values) {
  double values[10];
  for (int n = 0; n < 10; n++)
    values[n] = n;
  writer->operator()("key", values, 10);
  EXPECT_EQ("key = 0,1,2,3,4,5,6,7,8,9\n", ss.str());
}

TEST_F(WriterStringstream, key_values2) {
  double values[10];
  for (int n = 0; n < 10; n++)
    values[n] = n;
  writer->operator()("key", values, 2, 5);
  EXPECT_EQ("key = [0,1,2,3,4\n5,6,7,8,9]\n", ss.str());
}

TEST_F(WriterStringstream, names) {
  std::vector<std::string> names;
  writer->operator()(names);
  EXPECT_EQ("", ss.str());

  names.push_back("a");
  names.push_back("b");
  names.push_back("c");
  writer->operator()(names);
  EXPECT_EQ("a,b,c\n", ss.str());
}

TEST_F(WriterStringstream, state) {
  std::vector<double> state(10);
  for (int n = 0; n < 10; n++)
    state[n] = n;
  writer->operator()(state);
  EXPECT_EQ("0,1,2,3,4,5,6,7,8,9\n", ss.str());
}

TEST_F(WriterStringstream, no_arg) {
  writer->operator()();
  EXPECT_EQ("\n", ss.str());
}

TEST_F(WriterStringstream, message) {
  std::string message;
  message = "foo bar baz";
  writer->operator()(message);
  EXPECT_EQ("foo bar baz\n", ss.str());
}

TEST_F(WriterStringstream, message2) {
  writer->operator()("foo bar baz");
  EXPECT_EQ("foo bar baz\n", ss.str());
}

TEST_F(WriterStringstream, is_writing) {
  EXPECT_TRUE(writer->is_writing());
}

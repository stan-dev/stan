#include <gtest/gtest.h>
#include <boost/lexical_cast.hpp>
#include <stan/callbacks/json_writer.hpp>

struct deleter_noop {
  template <typename T>
  constexpr void operator()(T* arg) const {}
};
class StanInterfaceCallbacksJsonWriter : public ::testing::Test {
 public:
  StanInterfaceCallbacksJsonWriter()
      : ss(), writer(std::unique_ptr<std::stringstream, deleter_noop>(&ss)) {}

  void SetUp() {
    ss.str(std::string());
    ss.clear();
  }

  void TearDown() { writer.reset(); }

  std::stringstream ss;
  stan::callbacks::json_writer<std::stringstream, deleter_noop> writer;
};

TEST_F(StanInterfaceCallbacksJsonWriter, begin_record_end) {
  EXPECT_NO_THROW(writer.begin_record());
  EXPECT_NO_THROW(writer.end_record());
  EXPECT_EQ("{}", ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, write_double_vector) {
  std::string key("key");
  const int N = 5;
  std::vector<double> x;
  for (int n = 0; n < N; ++n)
    x.push_back(n);

  EXPECT_NO_THROW(writer.write(key, x));
  EXPECT_EQ("\"key\" : [ 0, 1, 2, 3, 4 ]", ss.str());

  EXPECT_NO_THROW(writer.write(key, x));
  EXPECT_EQ(
      "\"key\" : [ 0, 1, 2, 3, 4 ]"
      ", \"key\" : [ 0, 1, 2, 3, 4 ]",
      ss.str());

  EXPECT_NO_THROW(writer.write(key, x));
  EXPECT_EQ(
      "\"key\" : [ 0, 1, 2, 3, 4 ]"
      ", \"key\" : [ 0, 1, 2, 3, 4 ]"
      ", \"key\" : [ 0, 1, 2, 3, 4 ]",
      ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, single_member) {
  std::string key("key");
  std::string value("value");
  EXPECT_NO_THROW(writer.begin_record());
  EXPECT_NO_THROW(writer.write(key, value));
  EXPECT_NO_THROW(writer.end_record());
  EXPECT_EQ("{\"key\" : \"value\"}", ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, more_members) {
  std::string key("key");
  std::string value("value");
  EXPECT_NO_THROW(writer.begin_record());
  EXPECT_NO_THROW(writer.write(key, value));
  EXPECT_NO_THROW(writer.write(key, value));

  EXPECT_EQ(
      "{\"key\" : \"value\""
      ", \"key\" : \"value\"",
      ss.str());

  EXPECT_NO_THROW(writer.write(key, value));
  EXPECT_NO_THROW(writer.end_record());
  EXPECT_EQ(
      "{\"key\" : \"value\""
      ", \"key\" : \"value\""
      ", \"key\" : \"value\"}",
      ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, write_double_vector_precision2) {
  ss << std::setprecision(2);
  std::string key("key");
  const int N = 5;
  std::vector<double> x{1.23456789, 2.3456789, 3.45678910, 4.567890123};
  EXPECT_NO_THROW(writer.write(key, x));
  EXPECT_EQ("\"key\" : [ 1.2, 2.3, 3.5, 4.6 ]", ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, write_string_special_characters) {
  ss << std::setprecision(2);
  std::string key("key");
  std::string x(
      "the\\quick\"brown/\bfox\fjumped\nover\rthe\tlazy\vdog\atwotimes");
  EXPECT_NO_THROW(writer.write(key, x));
  EXPECT_EQ(
      "\"key\" : "
      "\"the\\\\quick\\\"brown\\/"
      "\\bfox\\fjumped\\nover\\rthe\\tlazy\\vdog\\atwotimes\"",
      ss.str());
}
TEST_F(StanInterfaceCallbacksJsonWriter, write_double_vector_precision3) {
  const int N = 5;
  std::vector<double> x{1.23456789, 2.3456789, 3.45678910, 4.567890123};
  ss.precision(3);
  EXPECT_NO_THROW(writer.write("key", x));
  EXPECT_EQ("\"key\" : [ 1.23, 2.35, 3.46, 4.57 ]", ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, write_string_vector) {
  const int N = 5;
  std::vector<std::string> x;
  for (int n = 0; n < N; ++n)
    x.push_back(boost::lexical_cast<std::string>(n));

  EXPECT_NO_THROW(writer.write("key", x));
  EXPECT_EQ("\"key\" : [ 0, 1, 2, 3, 4 ]", ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, write_null) {
  EXPECT_NO_THROW(writer.write("message"));
  EXPECT_EQ("\"message\" : \"null\" ", ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, write_string) {
  EXPECT_NO_THROW(writer.write("key", "value"));
  EXPECT_EQ("\"key\" : \"value\"", ss.str());
}

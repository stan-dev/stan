#include <gtest/gtest.h>
#include <boost/lexical_cast.hpp>
#include <stan/callbacks/json_writer.hpp>

class StanInterfaceCallbacksJsonWriter : public ::testing::Test {
 public:
  StanInterfaceCallbacksJsonWriter()
      : writer(std::make_unique<std::stringstream>(std::stringstream{})) {}

  void SetUp() {
    static_cast<std::stringstream&>(writer.get_stream()).str(std::string());
    static_cast<std::stringstream&>(writer.get_stream()).clear();
  }
  void TearDown() {
    writer.reset();
  }

  stan::callbacks::json_writer<std::ostream> writer;
};

TEST_F(StanInterfaceCallbacksJsonWriter, begin_end) {
  EXPECT_NO_THROW(writer.begin());
  EXPECT_NO_THROW(writer.end());
  EXPECT_EQ("{\n}",
            static_cast<std::stringstream&>(writer.get_stream()).str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, keyed_double_vector) {
  std::string key("key");
  const int N = 5;
  std::vector<double> x;
  for (int n = 0; n < N; ++n)
    x.push_back(n);

  EXPECT_NO_THROW(writer.keyed_values(key, x));
  EXPECT_EQ("\"key\" : [ 0, 1, 2, 3, 4 ]",
            static_cast<std::stringstream&>(writer.get_stream()).str());

  EXPECT_NO_THROW(writer.keyed_values(key, x));
  EXPECT_EQ("\"key\" : [ 0, 1, 2, 3, 4 ]"
            ", \"key\" : [ 0, 1, 2, 3, 4 ]",
            static_cast<std::stringstream&>(writer.get_stream()).str());

  EXPECT_NO_THROW(writer.keyed_values(key, x));
  EXPECT_EQ("\"key\" : [ 0, 1, 2, 3, 4 ]"
            ", \"key\" : [ 0, 1, 2, 3, 4 ]"
            ", \"key\" : [ 0, 1, 2, 3, 4 ]",
            static_cast<std::stringstream&>(writer.get_stream()).str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, single_member) {
  std::string key("key");
  std::string value("value");
  EXPECT_NO_THROW(writer.begin());
  EXPECT_NO_THROW(writer.keyed_string(key, value));
  EXPECT_NO_THROW(writer.end());
  EXPECT_EQ("{\n\"key\" : \"value\"}",
            static_cast<std::stringstream&>(writer.get_stream()).str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, more_members) {
  std::string key("key");
  std::string value("value");
  EXPECT_NO_THROW(writer.begin());
  EXPECT_NO_THROW(writer.keyed_string(key, value));
  EXPECT_NO_THROW(writer.keyed_string(key, value));

  EXPECT_EQ("{\n\"key\" : \"value\""
            ", \"key\" : \"value\"",
            static_cast<std::stringstream&>(writer.get_stream()).str());

  EXPECT_NO_THROW(writer.keyed_string(key, value));
  EXPECT_NO_THROW(writer.end());
  EXPECT_EQ("{\n\"key\" : \"value\""
            ", \"key\" : \"value\""
            ", \"key\" : \"value\"}",
            static_cast<std::stringstream&>(writer.get_stream()).str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, keyed_double_vector_precision2) {
  std::string key("key");
  const int N = 5;
  std::vector<double> x{1.23456789, 2.3456789, 3.45678910, 4.567890123};
  writer.get_stream().precision(2);
  EXPECT_NO_THROW(writer.keyed_values(key, x));
  EXPECT_EQ("\"key\" : [ 1.2, 2.3, 3.5, 4.6 ]",
            static_cast<std::stringstream&>(writer.get_stream()).str());
}

// TEST_F(StanInterfaceCallbacksJsonWriter, keyed_double_vector_precision3) {
//   const int N = 5;
//   std::vector<double> x{1.23456789, 2.3456789, 3.45678910, 4.567890123};
//   writer.get_stream().precision(3);
//   EXPECT_NO_THROW(writer(x));
//   EXPECT_EQ("1.23,2.35,3.46,4.57\n",
//             static_cast<std::stringstream&>(writer.get_stream()).str());
// }

// TEST_F(StanInterfaceCallbacksJsonWriter, keyed_string_vector) {
//   const int N = 5;
//   std::vector<std::string> x;
//   for (int n = 0; n < N; ++n)
//     x.push_back(boost::lexical_cast<std::string>(n));

//   EXPECT_NO_THROW(writer(x));
//   EXPECT_EQ("0,1,2,3,4\n",
//             static_cast<std::stringstream&>(writer.get_stream()).str());
// }

// TEST_F(StanInterfaceCallbacksJsonWriter, keyed_null) {
//   EXPECT_NO_THROW(writer());
//   EXPECT_EQ("\n", static_cast<std::stringstream&>(writer.get_stream()).str());
// }

// TEST_F(StanInterfaceCallbacksJsonWriter, keyed_string) {
//   EXPECT_NO_THROW(writer("message"));
//   EXPECT_EQ("message\n",
//             static_cast<std::stringstream&>(writer.get_stream()).str());
// }

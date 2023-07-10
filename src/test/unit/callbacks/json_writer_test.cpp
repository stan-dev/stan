#include <stan/callbacks/json_writer.hpp>
#include <rapidjson/document.h>
#include <gtest/gtest.h>
#include <string>

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

  void TearDown() {}

  std::stringstream ss;
  stan::callbacks::json_writer<std::stringstream, deleter_noop> writer;
};

TEST_F(StanInterfaceCallbacksJsonWriter, begin_end_record) {
  writer.begin_record();
  writer.end_record();
  EXPECT_EQ("{}\n", ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, begin_end_record_nested) {
  std::string key("key");
  std::string value("value");
  writer.begin_record();
  writer.begin_record("1");
  writer.write(key, value);
  writer.end_record();  // 1
  writer.begin_record("2");
  writer.write(key, value);
  writer.begin_record("2.1");
  writer.write(key, value);
  writer.write(key, value);
  writer.end_record();  // 2.1
  writer.begin_record("2.2");
  writer.write(key, value);
  writer.write(key, value);
  writer.end_record();  // 2.2
  writer.end_record();  // 2
  writer.end_record();
  EXPECT_EQ(
      "{\"1\" : {\"key\" : \"value\"},\n"
      "\"2\" : {\"key\" : \"value\","
      " \"2.1\" : {\"key\" : \"value\", \"key\" : \"value\"},\n"
      "\"2.2\" : {\"key\" : \"value\", \"key\" : \"value\"}}}\n",
      ss.str());
  rapidjson::Document document;
  ASSERT_FALSE(document.Parse<0>(ss.str().c_str()).HasParseError());
}

TEST_F(StanInterfaceCallbacksJsonWriter, write_double_vector) {
  std::string key("key");
  const int N = 5;
  std::vector<double> x;
  for (int n = 0; n < N; ++n)
    x.push_back(n);

  writer.write(key, x);
  EXPECT_EQ("\"key\" : [ 0, 1, 2, 3, 4 ]", ss.str());

  writer.write(key, x);
  EXPECT_EQ(
      "\"key\" : [ 0, 1, 2, 3, 4 ]"
      ", \"key\" : [ 0, 1, 2, 3, 4 ]",
      ss.str());

  writer.write(key, x);
  EXPECT_EQ(
      "\"key\" : [ 0, 1, 2, 3, 4 ]"
      ", \"key\" : [ 0, 1, 2, 3, 4 ]"
      ", \"key\" : [ 0, 1, 2, 3, 4 ]",
      ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, single_member) {
  std::string key("key");
  std::string value("value");
  writer.begin_record();
  writer.write(key, value);
  writer.end_record();
  EXPECT_EQ("{\"key\" : \"value\"}\n", ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, more_members) {
  std::string key("key");
  std::string value("value");
  writer.begin_record();
  writer.write(key, value);
  writer.write(key, value);

  EXPECT_EQ("{\"key\" : \"value\", \"key\" : \"value\"", ss.str());

  writer.write(key, value);
  writer.end_record();
  EXPECT_EQ("{\"key\" : \"value\", \"key\" : \"value\", \"key\" : \"value\"}\n",
            ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, write_double_vector_precision2) {
  ss << std::setprecision(2);
  std::string key("key");
  const int N = 5;
  std::vector<double> x{1.23456789, 2.3456789, 3.45678910, 4.567890123};
  writer.write(key, x);
  EXPECT_EQ("\"key\" : [ 1.2, 2.3, 3.5, 4.6 ]", ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, write_double_vector_nan_inf) {
  std::string key("key");
  std::vector<double> x;
  x.push_back(std::numeric_limits<double>::quiet_NaN());
  x.push_back(std::numeric_limits<double>::infinity());
  x.push_back(-std::numeric_limits<double>::infinity());
  writer.write(key, x);
  EXPECT_EQ("\"key\" : [ NaN, Inf, -Inf ]", ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, write_string_special_characters) {
  ss << std::setprecision(2);
  std::string key("key");
  std::string x(
      "the\\quick\"brown/\bfox\fjumped\nover\rthe\tlazy\vdog\atwotimes");
  writer.write(key, x);
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
  writer.write("key", x);
  EXPECT_EQ("\"key\" : [ 1.23, 2.35, 3.46, 4.57 ]", ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, write_string_vector) {
  const int N = 5;
  std::vector<std::string> x;
  for (int n = 0; n < N; ++n)
    x.push_back(std::to_string(n));

  writer.write("key", x);
  EXPECT_EQ("\"key\" : [ 0, 1, 2, 3, 4 ]", ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, write_null) {
  writer.write("message");
  EXPECT_EQ("\"message\" : null", ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, write_string) {
  writer.write("key", "value");
  EXPECT_EQ("\"key\" : \"value\"", ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, write_int_vector) {
  std::string key("key");
  const int N = 5;
  std::vector<int> x;
  for (int n = 0; n < N; ++n)
    x.push_back(n);

  writer.write(key, x);
  EXPECT_EQ("\"key\" : [ 0, 1, 2, 3, 4 ]", ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, write_empty_vector) {
  std::string key("key");
  std::vector<double> x;
  writer.write(key, x);
  EXPECT_EQ("\"key\" : [  ]", ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, write_complex) {
  std::string key("key");
  std::complex<double> x(1.110, 2.110);
  writer.write(key, x);
  EXPECT_EQ("\"key\" : [1.11, 2.11]", ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, write_complex_inf) {
  std::string key("key");
  std::complex<double> x(std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity());
  writer.write(key, x);
  EXPECT_EQ("\"key\" : [Inf, -Inf]", ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, write_complex_vector) {
  std::string key("key");
  const int N = 3;
  std::vector<std::complex<double>> x;
  for (int n = 0; n < N; ++n)
    x.push_back(std::complex<double>(1.110, 2.110));

  writer.write(key, x);
  EXPECT_EQ("\"key\" : [ [1.11, 2.11], [1.11, 2.11], [1.11, 2.11] ]", ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, write_eigen_vector) {
  std::string key("key");
  Eigen::VectorXd x{{1.0, 2.0, 3.0, 4.0}};
  writer.write(key, x);
  EXPECT_EQ("\"key\" : [ 1, 2, 3, 4 ]", ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, write_empty_eigen_vector) {
  std::string key("key");
  Eigen::VectorXd x;
  writer.write(key, x);
  EXPECT_EQ("\"key\" : [  ]", ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, write_eigen_rowvector) {
  std::string key("key");
  Eigen::RowVectorXd x{{1.0, 2.0, 3.0, 4.0}};
  writer.write(key, x);
  EXPECT_EQ("\"key\" : [ 1, 2, 3, 4 ]", ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, write_eigen_matrix) {
  std::string key("key");
  Eigen::MatrixXd x{{1.0, 2.0}, {3.0, 4.0}};
  writer.write(key, x);
  EXPECT_EQ("\"key\" : [ [ 1, 2 ], [ 3, 4 ] ]", ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, write_empty_eigen_matrix) {
  std::string key("key");
  Eigen::MatrixXd x;
  writer.write(key, x);
  EXPECT_EQ("\"key\" : [  ]", ss.str());
}

TEST_F(StanInterfaceCallbacksJsonWriter, no_op_writer) {
  std::string key("key");
  std::string value("value");
  stan::callbacks::json_writer<std::stringstream, deleter_noop> writer;
  EXPECT_NO_THROW(writer.write(key, value));
}

TEST_F(StanInterfaceCallbacksJsonWriter, no_op_writer2) {
  std::string key("key");
  std::string value("value");
  stan::callbacks::json_writer<std::stringstream, deleter_noop> writer(nullptr);
  EXPECT_NO_THROW(writer.write(key, value));
}

TEST_F(StanInterfaceCallbacksJsonWriter, move_ctor) {
  std::string key("key");
  std::string value("value");
  std::vector<stan::callbacks::json_writer<std::stringstream, deleter_noop>>
      jwriters;
  jwriters.reserve(3);
  for (int i = 0; i < 3; i++) {
    std::stringstream* raw_ptr = new std::stringstream();
    std::unique_ptr<std::stringstream, deleter_noop> oss(raw_ptr,
                                                         deleter_noop());
    stan::callbacks::json_writer<std::stringstream, deleter_noop> writer(
        std::move(oss));
    EXPECT_NO_THROW(jwriters.emplace_back(std::move(writer)));
    EXPECT_NO_THROW(writer.write(key, value));
  }
}

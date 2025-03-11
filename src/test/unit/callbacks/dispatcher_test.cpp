#include <stan/callbacks/dispatcher.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/json_writer.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <Eigen/Dense>

using stan::callbacks::deleter_noop;
using stan::callbacks::dispatcher;
using stan::callbacks::info_type;

class DispatcherTest : public ::testing::Test {
 public:
  DispatcherTest() : ss_sample(), ss_config(), ss_metric(), dispatcher() {
    // Create shared writers
    writer_sample = std::make_shared<stan::callbacks::stream_writer>(ss_sample);
    writer_config = std::make_shared<stan::callbacks::stream_writer>(ss_config);
    writer_metric = std::make_shared<
        stan::callbacks::json_writer<std::stringstream, deleter_noop>>(
        std::unique_ptr<std::stringstream, deleter_noop>(&ss_metric));
  }

  void SetUp() {
    ss_sample.str(std::string());
    ss_sample.clear();
    ss_config.str(std::string());
    ss_config.clear();
    ss_metric.str(std::string());
    ss_metric.clear();

    // Add managed resources
    dispatcher.add_managed_resource(writer_sample);
    dispatcher.add_managed_resource(writer_config);
    dispatcher.add_managed_resource(writer_metric);

    // Register channels
    dispatcher.register_channel(
        info_type::CONFIG,
        std::unique_ptr<stan::callbacks::channel>(
            new stan::callbacks::writer_channel(writer_config.get())));

    dispatcher.register_channel(
        info_type::SAMPLE,
        std::unique_ptr<stan::callbacks::channel>(
            new stan::callbacks::writer_channel(writer_sample.get())));

    dispatcher.register_channel(
        info_type::METRIC, std::unique_ptr<stan::callbacks::channel>(
                               new stan::callbacks::structured_writer_channel(
                                   writer_metric.get())));
  }

  void TearDown() {}

  std::stringstream ss_sample;
  std::stringstream ss_config;
  std::stringstream ss_metric;

  std::shared_ptr<stan::callbacks::stream_writer> writer_sample;
  std::shared_ptr<stan::callbacks::stream_writer> writer_config;
  std::shared_ptr<stan::callbacks::json_writer<std::stringstream, deleter_noop>>
      writer_metric;

  stan::callbacks::dispatcher dispatcher;
};

// Test basic string dispatch to plain writer
TEST_F(DispatcherTest, StringDispatch) {
  dispatcher.dispatch(info_type::CONFIG, std::string("Message1"));
  EXPECT_EQ(ss_config.str(), "Message1\n");
}

// Test multiple string dispatches
TEST_F(DispatcherTest, MultipleStringDispatch) {
  dispatcher.dispatch(info_type::CONFIG, std::string("Message1"));
  dispatcher.dispatch(info_type::CONFIG, std::string("Message2"));
  dispatcher.dispatch(info_type::CONFIG, std::string("Message3"));
  EXPECT_EQ(ss_config.str(), "Message1\nMessage2\nMessage3\n");
}

// Test empty call dispatch
TEST_F(DispatcherTest, EmptyDispatch) {
  dispatcher.dispatch(info_type::CONFIG);
  // Empty dispatch should produce just a newline in stream_writer
  EXPECT_EQ(ss_config.str(), "\n");
}

// Test vector of doubles dispatch
TEST_F(DispatcherTest, VectorDoubleDispatch) {
  std::vector<double> values = {1.1, 2.2, 3.3};
  dispatcher.dispatch(info_type::SAMPLE, values);
  std::string output = ss_sample.str();
  EXPECT_NE(output.find("1.1"), std::string::npos);
  EXPECT_NE(output.find("2.2"), std::string::npos);
  EXPECT_NE(output.find("3.3"), std::string::npos);
}

// Test vector of strings dispatch
TEST_F(DispatcherTest, VectorStringDispatch) {
  std::vector<std::string> names = {"alpha", "beta", "gamma"};
  dispatcher.dispatch(info_type::SAMPLE, names);
  std::string output = ss_sample.str();
  EXPECT_NE(output.find("alpha"), std::string::npos);
  EXPECT_NE(output.find("beta"), std::string::npos);
  EXPECT_NE(output.find("gamma"), std::string::npos);
}

// Test Eigen matrix dispatch
TEST_F(DispatcherTest, EigenMatrixDispatch) {
  Eigen::MatrixXd matrix(2, 2);
  matrix << 1.0, 2.0, 3.0, 4.0;
  dispatcher.dispatch(info_type::SAMPLE, matrix);
  std::string output = ss_sample.str();
  EXPECT_NE(output.find("1"), std::string::npos);
  EXPECT_NE(output.find("2"), std::string::npos);
  EXPECT_NE(output.find("3"), std::string::npos);
  EXPECT_NE(output.find("4"), std::string::npos);
}

// Test Eigen vector dispatch
TEST_F(DispatcherTest, EigenVectorDispatch) {
  Eigen::VectorXd vector(3);
  vector << 1.0, 2.0, 3.0;
  dispatcher.dispatch(info_type::SAMPLE, vector);
  std::string output = ss_sample.str();
  EXPECT_NE(output.find("1"), std::string::npos);
  EXPECT_NE(output.find("2"), std::string::npos);
  EXPECT_NE(output.find("3"), std::string::npos);
}

// Test Eigen row vector dispatch
TEST_F(DispatcherTest, EigenRowVectorDispatch) {
  Eigen::RowVectorXd vector(3);
  vector << 1.0, 2.0, 3.0;
  dispatcher.dispatch(info_type::SAMPLE, vector);
  std::string output = ss_sample.str();
  EXPECT_NE(output.find("1"), std::string::npos);
  EXPECT_NE(output.find("2"), std::string::npos);
  EXPECT_NE(output.find("3"), std::string::npos);
}

// Test structured writer begin/end record
TEST_F(DispatcherTest, StructuredBeginEndRecord) {
  dispatcher.begin_record(info_type::METRIC);
  dispatcher.end_record(info_type::METRIC);
  std::string output = ss_metric.str();
  // JSON output should contain opening and closing braces
  EXPECT_NE(output.find("{"), std::string::npos);
  EXPECT_NE(output.find("}"), std::string::npos);
}

TEST_F(DispatcherTest, MetricStructuredKeyValueRecord) {
  // For METRIC (structured writer), open a record, dispatch key/value pairs,
  // then close the record.
  dispatcher.begin_record(info_type::METRIC);
  dispatcher.dispatch(info_type::METRIC, "metric_type", std::string("diag"));
  dispatcher.dispatch(info_type::METRIC, "stepsize", 0.6789);
  std::vector<double> inv_metric = {0.1, 0.2, 0.3};
  dispatcher.dispatch(info_type::METRIC, "inv_metric", inv_metric);
  dispatcher.end_record(info_type::METRIC);
  // Expected output:
  // Begin record marker, followed by key/value pairs each formatted as
  // "key:value;" and then end record marker.
  std::string output = ss_metric.str();
  EXPECT_NE(output.find("metric_type"), std::string::npos);
  EXPECT_NE(output.find("diag"), std::string::npos);
}

// Test structured writer with multiple key-value types
TEST_F(DispatcherTest, StructuredMultipleValueTypes) {
  dispatcher.begin_record(info_type::METRIC);
  dispatcher.dispatch(info_type::METRIC, "string_key",
                      std::string("string_value"));
  dispatcher.dispatch(info_type::METRIC, "int_key", 42);
  dispatcher.dispatch(info_type::METRIC, "double_key", 3.14159);
  dispatcher.dispatch(info_type::METRIC, "bool_key", true);
  dispatcher.end_record(info_type::METRIC);

  std::string output = ss_metric.str();
  EXPECT_NE(output.find("string_key"), std::string::npos);
  EXPECT_NE(output.find("string_value"), std::string::npos);
  EXPECT_NE(output.find("int_key"), std::string::npos);
  EXPECT_NE(output.find("42"), std::string::npos);
  EXPECT_NE(output.find("double_key"), std::string::npos);
  EXPECT_NE(output.find("3.14159"), std::string::npos);
  EXPECT_NE(output.find("bool_key"), std::string::npos);
  EXPECT_NE(output.find("true"), std::string::npos);
}

// Test structured writer with Eigen values
TEST_F(DispatcherTest, StructuredEigenValues) {
  dispatcher.begin_record(info_type::METRIC);

  Eigen::MatrixXd matrix(2, 2);
  matrix << 1.0, 2.0, 3.0, 4.0;
  dispatcher.dispatch(info_type::METRIC, "matrix", matrix);

  Eigen::VectorXd vector(3);
  vector << 5.0, 6.0, 7.0;
  dispatcher.dispatch(info_type::METRIC, "vector", vector);

  dispatcher.end_record(info_type::METRIC);

  std::string output = ss_metric.str();
  EXPECT_NE(output.find("matrix"), std::string::npos);
  EXPECT_NE(output.find("1"), std::string::npos);
  EXPECT_NE(output.find("4"), std::string::npos);
  EXPECT_NE(output.find("vector"), std::string::npos);
  EXPECT_NE(output.find("5"), std::string::npos);
  EXPECT_NE(output.find("7"), std::string::npos);
}

// Test unregistered channel
TEST_F(DispatcherTest, UnregisteredChannel) {
  // Dispatch to unregistered channel should silently do nothing
  dispatcher.dispatch(info_type::ALGORITHM_STATE, std::string("Message"));
  dispatcher.dispatch(info_type::ALGORITHM_STATE,
                      std::vector<double>{1.0, 2.0});
  dispatcher.begin_record(info_type::ALGORITHM_STATE);
  dispatcher.dispatch(info_type::ALGORITHM_STATE, "key", "value");
  dispatcher.end_record(info_type::ALGORITHM_STATE);

  // No exceptions should be thrown
}

// Test named record
TEST_F(DispatcherTest, NamedRecord) {
  dispatcher.begin_record(info_type::METRIC, "record_name");
  dispatcher.dispatch(info_type::METRIC, "key", "value");
  dispatcher.end_record(info_type::METRIC);

  std::string output = ss_metric.str();
  EXPECT_NE(output.find("record_name"), std::string::npos);
  EXPECT_NE(output.find("key"), std::string::npos);
  EXPECT_NE(output.find("value"), std::string::npos);
}

// Test that begin_record and end_record on a plain writer channel are
// silently ignored
TEST_F(DispatcherTest, RecordOperationsOnPlainWriter) {
  dispatcher.begin_record(info_type::CONFIG);
  dispatcher.end_record(info_type::CONFIG);

  // Should not generate any output
  EXPECT_EQ(ss_config.str(), "");
}

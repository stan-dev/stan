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

// For this test we assume that InfoType has at least these values:
using stan::callbacks::dispatcher;
using stan::callbacks::InfoType;

struct deleter_noop {
  template <typename T>
  constexpr void operator()(T* arg) const {}
};

class DispatcherTest : public ::testing::Test {
 public:
  DispatcherTest()
      : ss_sample(),
        ss_config(),
        ss_metric(),
        writer_sample(ss_sample),
        writer_config(ss_config),
        writer_metric(
            std::unique_ptr<std::stringstream, deleter_noop>(&ss_metric)),
        dispatcher() {}

  void SetUp() {
    ss_sample.str(std::string());
    ss_sample.clear();
    ss_config.str(std::string());
    ss_config.clear();
    ss_metric.str(std::string());
    ss_metric.clear();

    dispatcher.register_channel(
        InfoType::CONFIG,
        std::unique_ptr<stan::callbacks::Channel>(
            new stan::callbacks::WriterChannel(&writer_config)));

    dispatcher.register_channel(
        InfoType::SAMPLE,
        std::unique_ptr<stan::callbacks::Channel>(
            new stan::callbacks::WriterChannel(&writer_sample)));

    dispatcher.register_channel(
        InfoType::METRIC,
        std::unique_ptr<stan::callbacks::Channel>(
            new stan::callbacks::StructuredWriterChannel(&writer_metric)));
  }

  void TearDown() {}

  std::stringstream ss_sample;
  std::stringstream ss_config;
  std::stringstream ss_metric;

  stan::callbacks::stream_writer writer_sample;
  stan::callbacks::stream_writer writer_config;
  stan::callbacks::json_writer<std::stringstream, deleter_noop> writer_metric;
  stan::callbacks::dispatcher dispatcher;
};

// Test basic string dispatch to plain writer
TEST_F(DispatcherTest, StringDispatch) {
  dispatcher.dispatch(InfoType::CONFIG, std::string("Message1"));
  EXPECT_EQ(ss_config.str(), "Message1\n");
}

// Test multiple string dispatches
TEST_F(DispatcherTest, MultipleStringDispatch) {
  dispatcher.dispatch(InfoType::CONFIG, std::string("Message1"));
  dispatcher.dispatch(InfoType::CONFIG, std::string("Message2"));
  dispatcher.dispatch(InfoType::CONFIG, std::string("Message3"));
  EXPECT_EQ(ss_config.str(), "Message1\nMessage2\nMessage3\n");
}

// Test empty call dispatch
TEST_F(DispatcherTest, EmptyDispatch) {
  dispatcher.dispatch(InfoType::CONFIG);
  // Empty dispatch should produce just a newline in stream_writer
  EXPECT_EQ(ss_config.str(), "\n");
}

// Test vector of doubles dispatch
TEST_F(DispatcherTest, VectorDoubleDispatch) {
  std::vector<double> values = {1.1, 2.2, 3.3};
  dispatcher.dispatch(InfoType::SAMPLE, values);
  std::string output = ss_sample.str();
  EXPECT_NE(output.find("1.1"), std::string::npos);
  EXPECT_NE(output.find("2.2"), std::string::npos);
  EXPECT_NE(output.find("3.3"), std::string::npos);
}

// Test vector of strings dispatch
TEST_F(DispatcherTest, VectorStringDispatch) {
  std::vector<std::string> names = {"alpha", "beta", "gamma"};
  dispatcher.dispatch(InfoType::SAMPLE, names);
  std::string output = ss_sample.str();
  EXPECT_NE(output.find("alpha"), std::string::npos);
  EXPECT_NE(output.find("beta"), std::string::npos);
  EXPECT_NE(output.find("gamma"), std::string::npos);
}

// Test Eigen matrix dispatch
TEST_F(DispatcherTest, EigenMatrixDispatch) {
  Eigen::MatrixXd matrix(2, 2);
  matrix << 1.0, 2.0, 3.0, 4.0;
  dispatcher.dispatch(InfoType::SAMPLE, matrix);
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
  dispatcher.dispatch(InfoType::SAMPLE, vector);
  std::string output = ss_sample.str();
  EXPECT_NE(output.find("1"), std::string::npos);
  EXPECT_NE(output.find("2"), std::string::npos);
  EXPECT_NE(output.find("3"), std::string::npos);
}

// Test Eigen row vector dispatch
TEST_F(DispatcherTest, EigenRowVectorDispatch) {
  Eigen::RowVectorXd vector(3);
  vector << 1.0, 2.0, 3.0;
  dispatcher.dispatch(InfoType::SAMPLE, vector);
  std::string output = ss_sample.str();
  EXPECT_NE(output.find("1"), std::string::npos);
  EXPECT_NE(output.find("2"), std::string::npos);
  EXPECT_NE(output.find("3"), std::string::npos);
}

// Test structured writer begin/end record
TEST_F(DispatcherTest, StructuredBeginEndRecord) {
  dispatcher.begin_record(InfoType::METRIC);
  dispatcher.end_record(InfoType::METRIC);
  std::string output = ss_metric.str();
  // JSON output should contain opening and closing braces
  EXPECT_NE(output.find("{"), std::string::npos);
  EXPECT_NE(output.find("}"), std::string::npos);
=======
  TEST_F(DispatcherTest, MetricStructuredKeyValueRecord) {
    // For METRIC (structured writer), open a record, dispatch key/value pairs,
    // then close the record.
    dispatcher.begin_record(InfoType::METRIC);
    dispatcher.dispatch(InfoType::METRIC, "metric_type", std::string("diag"));
    dispatcher.dispatch(InfoType::METRIC, "stepsize", 0.6789);
    // For the inv_metric, assume the caller converts the vector to a
    // comma-separated string.
    std::vector<double> inv_metric = {0.1, 0.2, 0.3};
    std::string inv_metric_str;
    for (size_t i = 0; i < inv_metric.size(); ++i) {
      inv_metric_str += std::to_string(inv_metric[i]);
      if (i != inv_metric.size() - 1)
        inv_metric_str += ",";
    }
    dispatcher.dispatch(InfoType::METRIC, "inv_metric", inv_metric_str);
    dispatcher.end_record(InfoType::METRIC);
    // Expected output:
    // Begin record marker, followed by key/value pairs each formatted as
    // "key:value;" and then end record marker.
    std::cout << ss_metric.str() << std::endl;
>>>>>>> 89d756b23a601c560cb63a09930f5fcbce011efc
  }

  // Test structured writer key-value pairs with string value
  TEST_F(DispatcherTest, StructuredKeyStringValue) {
    dispatcher.begin_record(InfoType::METRIC);
    dispatcher.dispatch(InfoType::METRIC, "key1", std::string("value1"));
    dispatcher.end_record(InfoType::METRIC);

    std::string output = ss_metric.str();
    EXPECT_NE(output.find("key1"), std::string::npos);
    EXPECT_NE(output.find("value1"), std::string::npos);
  }

<<<<<<< HEAD
  // Test structured writer with multiple key-value types
  TEST_F(DispatcherTest, StructuredMultipleValueTypes) {
    dispatcher.begin_record(InfoType::METRIC);
    dispatcher.dispatch(InfoType::METRIC, "string_key",
                        std::string("string_value"));
    dispatcher.dispatch(InfoType::METRIC, "int_key", 42);
    dispatcher.dispatch(InfoType::METRIC, "double_key", 3.14159);
    dispatcher.dispatch(InfoType::METRIC, "bool_key", true);
    dispatcher.end_record(InfoType::METRIC);

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

  // Test structured writer with vector values
  TEST_F(DispatcherTest, StructuredVectorValues) {
    dispatcher.begin_record(InfoType::METRIC);

    std::vector<double> doubles = {1.1, 2.2, 3.3};
    dispatcher.dispatch(InfoType::METRIC, "doubles", doubles);

    std::vector<std::string> strings = {"one", "two", "three"};
    dispatcher.dispatch(InfoType::METRIC, "strings", strings);

    dispatcher.end_record(InfoType::METRIC);

    std::string output = ss_metric.str();
    EXPECT_NE(output.find("doubles"), std::string::npos);
    EXPECT_NE(output.find("1.1"), std::string::npos);
    EXPECT_NE(output.find("strings"), std::string::npos);
    EXPECT_NE(output.find("one"), std::string::npos);
  }

  // Test structured writer with Eigen values
  TEST_F(DispatcherTest, StructuredEigenValues) {
    dispatcher.begin_record(InfoType::METRIC);

    Eigen::MatrixXd matrix(2, 2);
    matrix << 1.0, 2.0, 3.0, 4.0;
    dispatcher.dispatch(InfoType::METRIC, "matrix", matrix);

    Eigen::VectorXd vector(3);
    vector << 5.0, 6.0, 7.0;
    dispatcher.dispatch(InfoType::METRIC, "vector", vector);

    dispatcher.end_record(InfoType::METRIC);

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
    dispatcher.dispatch(InfoType::ALGORITHM_STATE, std::string("Message"));
    dispatcher.dispatch(InfoType::ALGORITHM_STATE,
                        std::vector<double>{1.0, 2.0});
    dispatcher.begin_record(InfoType::ALGORITHM_STATE);
    dispatcher.dispatch(InfoType::ALGORITHM_STATE, "key", "value");
    dispatcher.end_record(InfoType::ALGORITHM_STATE);

    // No exceptions should be thrown
  }

  // Test named record
  TEST_F(DispatcherTest, NamedRecord) {
    dispatcher.begin_record(InfoType::METRIC, "record_name");
    dispatcher.dispatch(InfoType::METRIC, "key", "value");
    dispatcher.end_record(InfoType::METRIC);

    std::string output = ss_metric.str();
    EXPECT_NE(output.find("record_name"), std::string::npos);
    EXPECT_NE(output.find("key"), std::string::npos);
    EXPECT_NE(output.find("value"), std::string::npos);
  }

  // Test that begin_record and end_record on a plain writer channel are
  // silently ignored
  TEST_F(DispatcherTest, RecordOperationsOnPlainWriter) {
    dispatcher.begin_record(InfoType::CONFIG);
    dispatcher.end_record(InfoType::CONFIG);

    // Should not generate any output
    EXPECT_EQ(ss_config.str(), "");
  }

  // Test complex sampler metric output pattern
  TEST_F(DispatcherTest, ComplexSamplerMetricPattern) {
    // This test simulates a more complex real-world usage pattern

    // Start a record for a sampling iteration
    dispatcher.begin_record(InfoType::METRIC);

    // Add various diagnostic info
    dispatcher.dispatch(InfoType::METRIC, "iter", 10);
    dispatcher.dispatch(InfoType::METRIC, "lp", -105.2);
    dispatcher.dispatch(InfoType::METRIC, "accept_stat", 0.8);

    // Add a nested object for adaptation
    dispatcher.begin_record(InfoType::METRIC, "adaptation");
    dispatcher.dispatch(InfoType::METRIC, "step_size", 0.85);

    // Add an inverse metric matrix
    Eigen::MatrixXd inv_metric(2, 2);
    inv_metric << 1.2, 0.1, 0.1, 0.9;
    dispatcher.dispatch(InfoType::METRIC, "inv_metric", inv_metric);

    // End adaptation object
    dispatcher.end_record(InfoType::METRIC);

    // End the main record
    dispatcher.end_record(InfoType::METRIC);

    // Verify key entries exist in the output
    std::string output = ss_metric.str();
    EXPECT_NE(output.find("iter"), std::string::npos);
    EXPECT_NE(output.find("10"), std::string::npos);
    EXPECT_NE(output.find("lp"), std::string::npos);
    EXPECT_NE(output.find("-105.2"), std::string::npos);
    EXPECT_NE(output.find("adaptation"), std::string::npos);
    EXPECT_NE(output.find("step_size"), std::string::npos);
    EXPECT_NE(output.find("inv_metric"), std::string::npos);
  }

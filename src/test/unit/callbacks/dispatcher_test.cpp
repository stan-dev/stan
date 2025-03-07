#include <stan/callbacks/dispatcher.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/json_writer.hpp>
#include <stan/callbacks/in_memory_writer.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <Eigen/Dense>

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
        writer_sample_in_memory(5,3),
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

    dispatcher.register_channel(
        InfoType::SAMPLE_RAW,
        std::unique_ptr<stan::callbacks::Channel>(
            new stan::callbacks::WriterChannel(&writer_sample_in_memory)));
  }

  void TearDown() {}

  std::stringstream ss_sample;
  std::stringstream ss_config;
  std::stringstream ss_metric;

  stan::callbacks::stream_writer writer_sample;
  stan::callbacks::stream_writer writer_config;
  stan::callbacks::json_writer<std::stringstream, deleter_noop> writer_metric;
  stan::callbacks::in_memory_writer writer_sample_in_memory;
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
}

TEST_F(DispatcherTest, MetricStructuredKeyValueRecord) {
  // For METRIC (structured writer), open a record, dispatch key/value pairs,
  // then close the record.
  dispatcher.begin_record(InfoType::METRIC);
  dispatcher.dispatch(InfoType::METRIC, "metric_type", std::string("diag"));
  dispatcher.dispatch(InfoType::METRIC, "stepsize", 0.6789);
  std::vector<double> inv_metric = {0.1, 0.2, 0.3};
  dispatcher.dispatch(InfoType::METRIC, "inv_metric", inv_metric);
  dispatcher.end_record(InfoType::METRIC);
  // Expected output:
  // Begin record marker, followed by key/value pairs each formatted as
  // "key:value;" and then end record marker.
  std::string output = ss_metric.str();
  EXPECT_NE(output.find("metric_type"), std::string::npos);
  EXPECT_NE(output.find("diag"), std::string::npos);
}

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
  dispatcher.dispatch(InfoType::ALGORITHM_STATE, std::vector<double>{1.0, 2.0});
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

// Test in_memory_writer integration with dispatcher
TEST_F(DispatcherTest, InMemoryWriterBasic) {
  // Write rows of data to the in_memory_writer via the dispatcher
  std::vector<double> row1 = {1.1, 2.2, 3.3};
  std::vector<double> row2 = {4.4, 5.5, 6.6};
  std::vector<double> row3 = {7.7, 8.8, 9.9};

  dispatcher.dispatch(InfoType::SAMPLE_RAW, row1);
  dispatcher.dispatch(InfoType::SAMPLE_RAW, row2);
  dispatcher.dispatch(InfoType::SAMPLE_RAW, row3);

  // Check that the data was stored correctly
  const Eigen::MatrixXd& data = writer_sample_in_memory.get_eigen_state_values();

  EXPECT_EQ(data.rows(), 5);  // As initialized
  EXPECT_EQ(data.cols(), 3);  // As initialized

  // Check the stored values (first 3 rows should have our data)
  EXPECT_DOUBLE_EQ(data(0, 0), 1.1);
  EXPECT_DOUBLE_EQ(data(0, 1), 2.2);
  EXPECT_DOUBLE_EQ(data(0, 2), 3.3);

  EXPECT_DOUBLE_EQ(data(1, 0), 4.4);
  EXPECT_DOUBLE_EQ(data(1, 1), 5.5);
  EXPECT_DOUBLE_EQ(data(1, 2), 6.6);

  EXPECT_DOUBLE_EQ(data(2, 0), 7.7);
  EXPECT_DOUBLE_EQ(data(2, 1), 8.8);
  EXPECT_DOUBLE_EQ(data(2, 2), 9.9);
}

// Test in_memory_writer with row overflow
TEST_F(DispatcherTest, InMemoryWriterRowOverflow) {
  // Try to write more rows than allocated
  std::vector<double> row = {1.0, 2.0, 3.0};

  // Should be able to write 5 rows (as initialized)
  for (int i = 0; i < 5; i++) {
    dispatcher.dispatch(InfoType::SAMPLE_RAW, row);
  }

  // Sixth row should throw exception
  EXPECT_THROW(dispatcher.dispatch(InfoType::SAMPLE_RAW, row),
               std::runtime_error);
}

// Test in_memory_writer with column mismatch
TEST_F(DispatcherTest, InMemoryWriterColumnMismatch) {
  // Try to write a row with wrong number of columns
  std::vector<double> wrong_size_row
      = {1.0, 2.0};  // Only 2 columns when we need 3

  EXPECT_THROW(dispatcher.dispatch(InfoType::SAMPLE_RAW, wrong_size_row),
               std::runtime_error);

  std::vector<double> also_wrong_size
      = {1.0, 2.0, 3.0, 4.0};  // 4 columns when we need 3

  EXPECT_THROW(dispatcher.dispatch(InfoType::SAMPLE_RAW, also_wrong_size),
               std::runtime_error);
}

// Test in_memory_writer reset
TEST_F(DispatcherTest, InMemoryWriterReset) {
  // Write some data
  std::vector<double> row = {1.1, 2.2, 3.3};
  dispatcher.dispatch(InfoType::SAMPLE_RAW, row);

  // Verify data is written
  const Eigen::MatrixXd& data1 = writer_sample_in_memory.get_eigen_state_values();
  EXPECT_DOUBLE_EQ(data1(0, 0), 1.1);

  // Reset the writer
  writer_sample_in_memory.reset();

  // Verify data is cleared
  const Eigen::MatrixXd& data2 = writer_sample_in_memory.get_eigen_state_values();
  EXPECT_DOUBLE_EQ(data2(0, 0), 0.0);

  // Write new data
  std::vector<double> new_row = {4.4, 5.5, 6.6};
  dispatcher.dispatch(InfoType::SAMPLE_RAW, new_row);

  // Verify new data is written
  const Eigen::MatrixXd& data3 = writer_sample_in_memory.get_eigen_state_values();
  EXPECT_DOUBLE_EQ(data3(0, 0), 4.4);
}

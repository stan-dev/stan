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

TEST_F(DispatcherTest, ConfigPlainMultipleMessages) {
  // Dispatch several string messages to a plain writer (CONFIG).
  dispatcher.dispatch(InfoType::CONFIG, std::string("Config1"));
  dispatcher.dispatch(InfoType::CONFIG, std::string("Config2"));
  dispatcher.dispatch(InfoType::CONFIG, std::string("Config3"));
  EXPECT_EQ(ss_config.str(), "Config1\nConfig2\nConfig3\n");
}

TEST_F(DispatcherTest, SamplePlainVector) {
  // Dispatch a vector of doubles to a plain writer (SAMPLE).
  std::vector<double> sample = {1.1, 2.2, 3.3};
  dispatcher.dispatch(InfoType::SAMPLE, sample);
  std::string output(ss_sample.str());
  EXPECT_EQ(count_matches("1.1", output), 1);
  EXPECT_EQ(count_matches("2.2", output), 1);
  EXPECT_EQ(count_matches("2.2", output), 1);
}

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
}

// TEST_F(DispatcherTest, NonStringDispatchToPlain) {
//   // Dispatch a non-string (e.g., an int) to a plain writer (CONFIG).
//   dispatcher.dispatch(InfoType::CONFIG, 999);
//   // Expect conversion via std::to_string, so output should be "999".
//   EXPECT_EQ(plainWriter.output, "999");
// }

// TEST_F(DispatcherTest, NonStringDispatchToStructuredIgnored) {
//   // Dispatch a non-string (e.g., an int) to a structured writer (METRIC)
//   should be ignored. dispatcher.dispatch(InfoType::METRIC, 123);
//   EXPECT_EQ(structuredWriter.output, "");
// }

// TEST_F(DispatcherTest, UnregisteredInfoType) {
//   // Dispatch to an unregistered InfoType (e.g., ALGORITHM_STATE) produces no
//   output. dispatcher.dispatch(InfoType::ALGORITHM_STATE,
//   std::string("NoOutput")); EXPECT_EQ(plainWriter.output, "");
//   EXPECT_EQ(structuredWriter.output, "");
// }

#include <stan/callbacks/output_controller.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/matrix_writer.hpp>
#include <stan/callbacks/structured_writer.hpp>
#include <stan/callbacks/json_writer.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <fstream>

// Mock writer for testing
class mock_writer : public stan::callbacks::writer {
 public:
  std::vector<std::vector<double>> data;
  
  void operator()(const std::vector<double>& x) override {
    data.push_back(x);
  }
};

TEST(output_controller, configure_and_write_streaming) {
  stan::callbacks::output_controller controller;
  
  // Configure output for samples
  controller.configure_output("samples", 
      {stan::callbacks::OutputFormat::MATRIX, "memory", 100, 3});
  
  // Write sample data
  std::vector<double> sample = {1.0, 2.0, 3.0};
  controller.write("samples", sample);
  
  // Get writer and verify data
  auto writer = controller.get_writer("samples");
  auto* matrix_writer = dynamic_cast<stan::callbacks::matrix_writer*>(writer.get());
  EXPECT_NE(matrix_writer, nullptr);
  EXPECT_EQ(matrix_writer->rows(), 100);
  EXPECT_EQ(matrix_writer->cols(), 3);
  EXPECT_EQ(matrix_writer->current_row(), 1);
}

TEST(output_controller, write_metadata) {
  stan::callbacks::output_controller controller;
  
  // Configure output for metadata
  controller.configure_output("model_info", 
      {stan::callbacks::OutputFormat::JSON, "model.json"});
  
  // Get JSON writer - note we don't need to specify template parameters
  auto* json_writer = controller.get_json_writer("model_info");
  EXPECT_NE(json_writer, nullptr);
  
  // Write metadata as a properly structured JSON record
  json_writer->begin_record();
  json_writer->write("model_name", "bernoulli");
  json_writer->write("version", "2.29.0");
  json_writer->end_record();
}

TEST(output_controller, get_json_writer) {
  stan::callbacks::output_controller controller;
  
  // Configure JSON writer for metric
  controller.configure_output("metric", 
      {stan::callbacks::OutputFormat::JSON, "metric.json"});
  
  // Get JSON writer directly
  auto* metric_writer = controller.get_json_writer<std::ofstream>("metric");
  EXPECT_NE(metric_writer, nullptr);
  
  // Test using JSON writer interface directly
  metric_writer->write("stepsize", "0.1");
  metric_writer->write("metric_type", "dense");
}

TEST(output_controller, get_json_writer_wrong_type) {
  stan::callbacks::output_controller controller;
  
  // Configure non-JSON writer
  controller.configure_output("samples", 
      {stan::callbacks::OutputFormat::MATRIX, "memory", 100, 3});
  
  // Attempt to get JSON writer
  EXPECT_THROW(controller.get_json_writer<std::ofstream>("samples"), std::runtime_error);
}

TEST(output_controller, multiple_formats) {
  stan::callbacks::output_controller controller;
  
  // Configure different formats for different information types
  controller.configure_output("samples", 
      {stan::callbacks::OutputFormat::MATRIX, "memory", 100, 3});
  controller.configure_output("diagnostics", 
      {stan::callbacks::OutputFormat::CSV, "diagnostics.csv"});
  controller.configure_output("metric", 
      {stan::callbacks::OutputFormat::JSON, "metric.json"});
  
  // Write streaming data
  std::vector<double> sample = {1.0, 2.0, 3.0};
  std::vector<double> diagnostic = {4.0, 5.0, 6.0};
  controller.write("samples", sample);
  controller.write("diagnostics", diagnostic);
  
  // Write metric using JSON writer interface
  auto* metric_writer = controller.get_json_writer<std::ofstream>("metric");
  metric_writer->write("stepsize", "0.1");
  
  // Verify writers
  auto sample_writer = controller.get_writer("samples");
  auto diagnostic_writer = controller.get_writer("diagnostics");
  
  EXPECT_NE(dynamic_cast<stan::callbacks::matrix_writer*>(sample_writer.get()), nullptr);
  EXPECT_NE(dynamic_cast<stan::callbacks::stream_writer*>(diagnostic_writer.get()), nullptr);
  EXPECT_NE(controller.get_json_writer<std::ofstream>("metric"), nullptr);
}

TEST(output_controller, unconfigured_output) {
  stan::callbacks::output_controller controller;
  
  // Attempt to write without configuration
  std::vector<double> data = {1.0, 2.0, 3.0};
  std::vector<std::string> metadata = {"model_name", "bernoulli"};
  
  EXPECT_THROW(controller.write("samples", data), std::runtime_error);
  EXPECT_THROW(controller.get_json_writer<std::ofstream>("metric"), std::runtime_error);
}

TEST(output_controller, invalid_format) {
  stan::callbacks::output_controller controller;
  
  // Configure with invalid format - should throw during configuration
  EXPECT_THROW(controller.configure_output("samples", 
      {static_cast<stan::callbacks::OutputFormat>(999), "invalid"}),
      std::runtime_error);
  
  // Verify we can't write after invalid configuration
  std::vector<double> data = {1.0, 2.0, 3.0};
  EXPECT_THROW(controller.write("samples", data), std::runtime_error);
} 

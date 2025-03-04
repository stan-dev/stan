#include <stan/io/output_controller.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/matrix_writer.hpp>
#include <stan/callbacks/structured_writer.hpp>
#include <stan/callbacks/json_writer.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

// Mock writer for testing
class mock_writer : public stan::callbacks::writer {
 public:
  std::vector<std::vector<double>> data;
  std::vector<std::vector<std::string>> metadata;
  
  void operator()(const std::vector<double>& x) override {
    data.push_back(x);
  }
  
  void operator()(const std::vector<std::string>& x) override {
    metadata.push_back(x);
  }
};

TEST(output_controller, configure_and_write_streaming) {
  stan::io::output_controller controller;
  
  // Configure output for samples
  controller.configure_output("samples", 
      {stan::io::OutputFormat::MATRIX, "memory", 100, 3});
  
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
  stan::io::output_controller controller;
  
  // Configure output for metadata
  controller.configure_output("model_info", 
      {stan::io::OutputFormat::JSON, "model.json"});
  
  // Write metadata
  std::vector<std::string> metadata = {"model_name", "bernoulli", "version", "2.29.0"};
  controller.write_metadata("model_info", metadata);
  
  // Get writer and verify it's a JSON writer
  auto writer = controller.get_writer("model_info");
  auto* json_writer = dynamic_cast<stan::callbacks::json_writer*>(writer.get());
  EXPECT_NE(json_writer, nullptr);
}

TEST(output_controller, get_json_writer) {
  stan::io::output_controller controller;
  
  // Configure JSON writer for metric
  controller.configure_output("metric", 
      {stan::io::OutputFormat::JSON, "metric.json"});
  
  // Get JSON writer directly
  auto* metric_writer = controller.get_json_writer("metric");
  EXPECT_NE(metric_writer, nullptr);
  
  // Test using JSON writer interface directly
  std::vector<std::string> metric_data = {"stepsize", "0.1", "metric_type", "dense"};
  metric_writer->operator()(metric_data);
}

TEST(output_controller, get_json_writer_wrong_type) {
  stan::io::output_controller controller;
  
  // Configure non-JSON writer
  controller.configure_output("samples", 
      {stan::io::OutputFormat::MATRIX, "memory", 100, 3});
  
  // Attempt to get JSON writer
  EXPECT_THROW(controller.get_json_writer("samples"), std::runtime_error);
}

TEST(output_controller, get_arrow_writer) {
  stan::io::output_controller controller;
  
  // Configure Arrow writer for samples
  controller.configure_output("samples", 
      {stan::io::OutputFormat::ARROW, "samples.arrow"});
  
  // Get Arrow writer directly
  auto* arrow_writer = controller.get_arrow_writer<std::ofstream>("samples");
  EXPECT_NE(arrow_writer, nullptr);
  
  // Test using Arrow writer interface directly
  std::vector<double> sample = {1.0, 2.0, 3.0};
  arrow_writer->operator()(sample);
}

TEST(output_controller, get_arrow_writer_wrong_type) {
  stan::io::output_controller controller;
  
  // Configure non-Arrow writer
  controller.configure_output("samples", 
      {stan::io::OutputFormat::MATRIX, "memory", 100, 3});
  
  // Attempt to get Arrow writer
  EXPECT_THROW(controller.get_arrow_writer<std::ofstream>("samples"), std::runtime_error);
}

TEST(output_controller, multiple_formats_with_arrow) {
  stan::io::output_controller controller;
  
  // Configure different formats for different information types
  controller.configure_output("samples", 
      {stan::io::OutputFormat::ARROW, "samples.arrow"});
  controller.configure_output("diagnostics", 
      {stan::io::OutputFormat::CSV, "diagnostics.csv"});
  controller.configure_output("metric", 
      {stan::io::OutputFormat::JSON, "metric.json"});
  
  // Write streaming data
  std::vector<double> sample = {1.0, 2.0, 3.0};
  std::vector<double> diagnostic = {4.0, 5.0, 6.0};
  controller.write("samples", sample);
  controller.write("diagnostics", diagnostic);
  
  // Write metric using JSON writer interface
  auto* metric_writer = controller.get_json_writer<std::ofstream>("metric");
  std::vector<std::string> metric_data = {"stepsize", "0.1"};
  metric_writer->operator()(metric_data);
  
  // Verify writers
  auto sample_writer = controller.get_writer("samples");
  auto diagnostic_writer = controller.get_writer("diagnostics");
  
  EXPECT_NE(dynamic_cast<stan::callbacks::arrow_writer<std::ofstream>*>(sample_writer.get()), nullptr);
  EXPECT_NE(dynamic_cast<stan::callbacks::stream_writer*>(diagnostic_writer.get()), nullptr);
  EXPECT_NE(controller.get_json_writer<std::ofstream>("metric"), nullptr);
}

TEST(output_controller, unconfigured_output) {
  stan::io::output_controller controller;
  
  // Attempt to write without configuration
  std::vector<double> data = {1.0, 2.0, 3.0};
  std::vector<std::string> metadata = {"model_name", "bernoulli"};
  
  EXPECT_THROW(controller.write("samples", data), std::runtime_error);
  EXPECT_THROW(controller.write_metadata("model_info", metadata), std::runtime_error);
  EXPECT_THROW(controller.get_json_writer("metric"), std::runtime_error);
}

TEST(output_controller, invalid_format) {
  stan::io::output_controller controller;
  
  // Configure with invalid format
  controller.configure_output("samples", 
      {static_cast<stan::io::OutputFormat>(999), "invalid"});
  
  std::vector<double> data = {1.0, 2.0, 3.0};
  EXPECT_THROW(controller.write("samples", data), std::runtime_error);
} 
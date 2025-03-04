#include <stan/io/output_controller.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/matrix_writer.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

// Mock writer for testing
class mock_writer : public stan::callbacks::writer {
 public:
  std::vector<std::vector<double>> data;
  
  void operator()(const std::vector<double>& x) override {
    data.push_back(x);
  }
  
  void operator()(const std::vector<std::string>& x) override {
    // Not used in tests
  }
};

TEST(output_controller, configure_and_write) {
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

TEST(output_controller, multiple_formats) {
  stan::io::output_controller controller;
  
  // Configure different formats for different information types
  controller.configure_output("samples", 
      {stan::io::OutputFormat::MATRIX, "memory", 100, 3});
  controller.configure_output("diagnostics", 
      {stan::io::OutputFormat::CSV, "diagnostics.csv"});
  
  // Write data
  std::vector<double> sample = {1.0, 2.0, 3.0};
  std::vector<double> diagnostic = {4.0, 5.0, 6.0};
  
  controller.write("samples", sample);
  controller.write("diagnostics", diagnostic);
  
  // Verify writers
  auto sample_writer = controller.get_writer("samples");
  auto diagnostic_writer = controller.get_writer("diagnostics");
  
  EXPECT_NE(dynamic_cast<stan::callbacks::matrix_writer*>(sample_writer.get()), nullptr);
  EXPECT_NE(dynamic_cast<stan::callbacks::stream_writer*>(diagnostic_writer.get()), nullptr);
}

TEST(output_controller, unconfigured_output) {
  stan::io::output_controller controller;
  
  // Attempt to write without configuration
  std::vector<double> data = {1.0, 2.0, 3.0};
  EXPECT_THROW(controller.write("samples", data), std::runtime_error);
}

TEST(output_controller, invalid_format) {
  stan::io::output_controller controller;
  
  // Configure with invalid format
  controller.configure_output("samples", 
      {static_cast<stan::io::OutputFormat>(999), "invalid"});
  
  std::vector<double> data = {1.0, 2.0, 3.0};
  EXPECT_THROW(controller.write("samples", data), std::runtime_error);
} 
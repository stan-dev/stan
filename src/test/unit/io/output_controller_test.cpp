#include <stan/io/output_controller.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/test/unit/util.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

// Mock writer for testing
class mock_writer : public stan::callbacks::writer {
 public:
  std::vector<std::vector<double>> samples;
  std::vector<std::vector<double>> diagnostics;
  
  void operator()(const std::vector<double>& x) override {
    samples.push_back(x);
  }
  
  void operator()(const std::vector<std::string>& x) override {
    // Not used in tests
  }
};

TEST(output_controller, add_and_write_sample) {
  stan::io::output_controller controller;
  
  // Create two mock writers
  auto writer1 = std::make_unique<mock_writer>();
  auto writer2 = std::make_unique<mock_writer>();
  
  // Store raw pointers for testing
  auto* writer1_ptr = writer1.get();
  auto* writer2_ptr = writer2.get();
  
  // Add writers to controller
  controller.add_sample_writer(std::move(writer1));
  controller.add_sample_writer(std::move(writer2));
  
  // Write sample data
  std::vector<double> sample = {1.0, 2.0, 3.0};
  controller.write_sample(sample);
  
  // Check both writers received the data
  EXPECT_EQ(writer1_ptr->samples.size(), 1);
  EXPECT_EQ(writer2_ptr->samples.size(), 1);
  EXPECT_EQ(writer1_ptr->samples[0], sample);
  EXPECT_EQ(writer2_ptr->samples[0], sample);
}

TEST(output_controller, add_and_write_diagnostic) {
  stan::io::output_controller controller;
  
  // Create two mock writers
  auto writer1 = std::make_unique<mock_writer>();
  auto writer2 = std::make_unique<mock_writer>();
  
  // Store raw pointers for testing
  auto* writer1_ptr = writer1.get();
  auto* writer2_ptr = writer2.get();
  
  // Add writers to controller
  controller.add_diagnostic_writer(std::move(writer1));
  controller.add_diagnostic_writer(std::move(writer2));
  
  // Write diagnostic data
  std::vector<double> diagnostic = {4.0, 5.0, 6.0};
  controller.write_diagnostic(diagnostic);
  
  // Check both writers received the data
  EXPECT_EQ(writer1_ptr->diagnostics.size(), 1);
  EXPECT_EQ(writer2_ptr->diagnostics.size(), 1);
  EXPECT_EQ(writer1_ptr->diagnostics[0], diagnostic);
  EXPECT_EQ(writer2_ptr->diagnostics[0], diagnostic);
}

TEST(output_controller, empty_writers) {
  stan::io::output_controller controller;
  
  // Writing to empty writers should not crash
  std::vector<double> data = {1.0, 2.0, 3.0};
  controller.write_sample(data);
  controller.write_diagnostic(data);
  
  // Test passes if no crash occurs
}

TEST(output_controller, multiple_writes) {
  stan::io::output_controller controller;
  
  auto writer = std::make_unique<mock_writer>();
  auto* writer_ptr = writer.get();
  
  controller.add_sample_writer(std::move(writer));
  
  // Write multiple samples
  std::vector<double> sample1 = {1.0, 2.0, 3.0};
  std::vector<double> sample2 = {4.0, 5.0, 6.0};
  
  controller.write_sample(sample1);
  controller.write_sample(sample2);
  
  // Check all samples were written
  EXPECT_EQ(writer_ptr->samples.size(), 2);
  EXPECT_EQ(writer_ptr->samples[0], sample1);
  EXPECT_EQ(writer_ptr->samples[1], sample2);
} 
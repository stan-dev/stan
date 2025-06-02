#include <stan/run/output_writers.hpp>
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <thread>
#include <chrono>

class OutputWritersTest : public ::testing::Test {
protected:
  void SetUp() override {
    test_dir = std::filesystem::temp_directory_path() / "stan_run_test_output";
    std::filesystem::create_directories(test_dir);
  }
  
  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(test_dir, ec);
  }
  
  std::filesystem::path test_dir;
};

bool is_whitespace(char c) { return c == ' ' || c == '\n'; }

std::string output_sans_whitespace(std::stringstream& ss) {
  auto out = ss.str();
  out.erase(std::remove_if(out.begin(), out.end(), is_whitespace), out.end());
  return out;
}

TEST_F(OutputWritersTest, GenerateTimestampFormat) {
  std::string timestamp = stan::run::generate_timestamp();
  
  // Should match format YYYYMMDD_HHMMSS
  std::regex timestamp_regex(R"(\d{8}_\d{6})");
  EXPECT_TRUE(std::regex_match(timestamp, timestamp_regex));
  
  // Length should be exactly 15 characters
  EXPECT_EQ(timestamp.length(), 15);
}

TEST_F(OutputWritersTest, GenerateTimestampUnique) {
  std::string timestamp1 = stan::run::generate_timestamp();
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  std::string timestamp2 = stan::run::generate_timestamp();
  
  // Timestamps should be different (assuming system clock resolution)
  EXPECT_NE(timestamp1, timestamp2);
}

TEST_F(OutputWritersTest, GenerateFilename) {
  std::string filename = stan::run::generate_filename(
    "test_model", "20250522_143000", 2, "sample", ".csv");
  
  EXPECT_EQ(filename, "test_model_20250522_143000_chain2_sample.csv");
}

TEST_F(OutputWritersTest, GenerateFilenameEdgeCases) {
  // Test with empty model name
  std::string filename1 = stan::run::generate_filename(
    "", "20250522_143000", 1, "sample", ".csv");
  EXPECT_EQ(filename1, "_20250522_143000_chain1_sample.csv");
  
  // Test with chain 0 (should still work)
  std::string filename2 = stan::run::generate_filename(
    "model", "20250522_143000", 0, "sample", ".csv");
  EXPECT_EQ(filename2, "model_20250522_143000_chain0_sample.csv");
  
  // Test with no extension
  std::string filename3 = stan::run::generate_filename(
    "model", "20250522_143000", 1, "sample", "");
  EXPECT_EQ(filename3, "model_20250522_143000_chain1_sample");
}

TEST_F(OutputWritersTest, EnsureOutputDirectoryCreatesDirectory) {
  std::filesystem::path new_dir = test_dir / "new_subdir";
  
  EXPECT_FALSE(std::filesystem::exists(new_dir));
  
  stan::run::ensure_output_directory(new_dir.string());
  
  EXPECT_TRUE(std::filesystem::exists(new_dir));
  EXPECT_TRUE(std::filesystem::is_directory(new_dir));
}

TEST_F(OutputWritersTest, EnsureOutputDirectoryExistingDirectory) {
  // Should not throw when directory already exists
  EXPECT_NO_THROW(stan::run::ensure_output_directory(test_dir.string()));
}

TEST_F(OutputWritersTest, EnsureOutputDirectoryEmptyPath) {
  // Should handle empty path gracefully
  EXPECT_NO_THROW(stan::run::ensure_output_directory(""));
}

TEST_F(OutputWritersTest, EnsureOutputDirectoryNestedPath) {
  std::filesystem::path nested_dir = test_dir / "level1" / "level2" / "level3";
  
  EXPECT_FALSE(std::filesystem::exists(nested_dir));
  
  stan::run::ensure_output_directory(nested_dir.string());
  
  EXPECT_TRUE(std::filesystem::exists(nested_dir));
  EXPECT_TRUE(std::filesystem::is_directory(nested_dir));
}

TEST_F(OutputWritersTest, CreateFilePath) {
  std::string filepath1 = stan::run::create_file_path(test_dir.string(), "test.csv");
  std::filesystem::path expected1 = test_dir / "test.csv";
  EXPECT_EQ(filepath1, expected1.string());
  
  // Test with empty directory
  std::string filepath2 = stan::run::create_file_path("", "test.csv");
  EXPECT_EQ(filepath2, "test.csv");
}

TEST_F(OutputWritersTest, CreateCSVWriter) {
  std::string filename = "test_sample.csv";
  std::string filepath = (test_dir / filename).string();
  
  auto writer = stan::run::create_writer<stan::run::csv_writer>(
    test_dir.string(), "test_model", "20250522_143000", 1, 
    "sample", ".csv", "# ");
  
  EXPECT_TRUE(writer != nullptr);
  
  // Write some test data
  std::vector<std::string> headers = {"param1", "param2"};
  std::vector<double> values = {1.5, 2.5};
  
  writer->operator()(headers);
  writer->operator()(values);
  
  // Flush and close writer
  writer.reset();
  
  // Verify file was created and contains expected content
  std::string expected_filepath = stan::run::create_file_path(
    test_dir.string(),
    stan::run::generate_filename("test_model", "20250522_143000", 1, "sample", ".csv")
  );
  
  EXPECT_TRUE(std::filesystem::exists(expected_filepath));
  
  std::ifstream file(expected_filepath);
  std::string line;
  
  // Check header line
  std::getline(file, line);
  EXPECT_EQ(line, "param1,param2");
  
  // Check data line
  std::getline(file, line);
  EXPECT_EQ(line, "1.5,2.5");
}

TEST_F(OutputWritersTest, CreateJSONWriter) {
  auto writer = stan::run::create_writer<stan::run::json_writer>(
    test_dir.string(), "test_model", "20250522_143000", 1, 
    "metric", ".json");
  
  EXPECT_TRUE(writer != nullptr);
  
  // Write some test JSON data
  writer->begin_record();
  writer->begin_record("name");
  writer->end_record();
  writer->write("dummy");
  writer->end_record();
  
  // Verify file was created
  std::string expected_filepath = stan::run::create_file_path(
    test_dir.string(),
    stan::run::generate_filename("test_model", "20250522_143000", 1, "metric", ".json")
  );
  
  EXPECT_TRUE(std::filesystem::exists(expected_filepath));
}

TEST_F(OutputWritersTest, CreateWriterInvalidDirectory) {
  // Try to create writer in non-writable directory (simulate permission error)
  std::string invalid_dir = "/root/non_writable_dir_that_should_not_exist";
  
  EXPECT_THROW(
    stan::run::create_writer<stan::run::csv_writer>(
      invalid_dir, "test_model", "20250522_143000", 1, 
      "sample", ".csv", "# "),
    std::runtime_error
  );
}

TEST_F(OutputWritersTest, TraitsDetection) {
  // Test type traits for writer detection
  EXPECT_TRUE(stan::run::traits::is_stream_writer<stan::run::csv_writer>::value);
  EXPECT_FALSE(stan::run::traits::is_json_writer<stan::run::csv_writer>::value);
  
  EXPECT_FALSE(stan::run::traits::is_stream_writer<stan::run::json_writer>::value);
  EXPECT_TRUE(stan::run::traits::is_json_writer<stan::run::json_writer>::value);
  
  // Test with non-writer types
  EXPECT_FALSE(stan::run::traits::is_stream_writer<int>::value);
  EXPECT_FALSE(stan::run::traits::is_json_writer<std::string>::value);
}

#include <stan/callbacks/in_memory_writer.hpp>
#include <stan/callbacks/dispatcher.hpp>
#include <gtest/gtest.h>
#include <sstream>
#include <vector>
#include <memory>
#include <string>
#include <Eigen/Dense>

using stan::callbacks::in_memory_writer;
using stan::callbacks::dispatcher;
using stan::callbacks::InfoType;
using stan::callbacks::WriterChannel;

// Helper function to split a comma-separated line into tokens
std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> tokens;
  std::stringstream ss(line);
  std::string token;
  
  while (std::getline(ss, token, ',')) {
    // Trim whitespace
    token.erase(0, token.find_first_not_of(" \t"));
    token.erase(token.find_last_not_of(" \t") + 1);
    tokens.push_back(token);
  }
  
  return tokens;
}

// Helper function to convert string tokens to doubles
std::vector<double> convert_to_doubles(const std::vector<std::string>& tokens) {
  std::vector<double> values;
  for (const auto& token : tokens) {
    values.push_back(std::stod(token));
  }
  return values;
}

class InMemoryWriterTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Parse the header
    header_tokens = split_csv_line("theta, mu,y_rep.1,y_rep.2,y_rep.3,y_rep.4,y_rep.5,y_rep.6,y_rep.7,y_rep.8,y_rep.9,y_rep.10");
    num_cols = header_tokens.size();
    
    // Parse the sample rows
    std::vector<std::string> sample_lines = {
      "0.309252,0.309252,0,0,1,1,0,0,0,1,0,0",
      "0.572524,0.572524,0,0,0,1,0,1,1,0,1,0",
      "0.0795978,0.0795978,0,0,0,0,0,0,0,0,0,0"
    };
    
    for (const auto& line : sample_lines) {
      auto tokens = split_csv_line(line);
      auto values = convert_to_doubles(tokens);
      sample_rows.push_back(values);
    }
    
    num_rows = sample_rows.size();
  }
  
  std::vector<std::string> header_tokens;
  std::vector<std::vector<double>> sample_rows;
  size_t num_rows;
  size_t num_cols;
};

// Test basic writing and retrieval of samples
TEST_F(InMemoryWriterTest, BasicWriteAndRetrieve) {
  // Create writer with exact capacity
  in_memory_writer writer(num_rows, num_cols);
  
  // Write header
  writer(header_tokens);
  
  // Write sample rows
  for (const auto& row : sample_rows) {
    writer(row);
  }
  
  // Verify names were stored correctly
  ASSERT_EQ(writer.get_names().size(), num_cols);
  for (size_t i = 0; i < num_cols; ++i) {
    EXPECT_EQ(writer.get_names()[i], header_tokens[i]);
  }
  
  // Verify data was stored correctly
  const Eigen::MatrixXd& data = writer.get_eigen_state_values();
  ASSERT_EQ(data.rows(), num_rows);
  ASSERT_EQ(data.cols(), num_cols);
  
  for (size_t i = 0; i < num_rows; ++i) {
    for (size_t j = 0; j < num_cols; ++j) {
      EXPECT_DOUBLE_EQ(data(i, j), sample_rows[i][j]);
    }
  }
  
  // Verify row count
  EXPECT_EQ(writer.get_row_count(), num_rows);
}

// Test writing more data than allocated
TEST_F(InMemoryWriterTest, WriteOverflow) {
  // Create writer with less capacity than needed
  in_memory_writer writer(num_rows - 1, num_cols);
  
  // Write header
  writer(header_tokens);
  
  // Write sample rows until overflow
  for (size_t i = 0; i < num_rows - 1; ++i) {
    writer(sample_rows[i]);
  }
  
  // The next write should throw an exception
  EXPECT_THROW(writer(sample_rows.back()), std::runtime_error);
}

// Test writing rows with incorrect column count
TEST_F(InMemoryWriterTest, ColumnMismatch) {
  // Create writer
  in_memory_writer writer(num_rows, num_cols);
  
  // Write header
  writer(header_tokens);
  
  // Create an invalid row with too few columns
  std::vector<double> invalid_row(num_cols - 1, 0.5);
  
  // Writing should throw an exception
  EXPECT_THROW(writer(invalid_row), std::runtime_error);
  
  // Create an invalid row with too many columns
  invalid_row.resize(num_cols + 1, 0.5);
  
  // Writing should throw an exception
  EXPECT_THROW(writer(invalid_row), std::runtime_error);
}

// Test reset functionality
TEST_F(InMemoryWriterTest, Reset) {
  // Create writer
  in_memory_writer writer(num_rows, num_cols);
  
  // Write header and first row
  writer(header_tokens);
  writer(sample_rows[0]);
  
  // Verify row was written
  EXPECT_EQ(writer.get_row_count(), 1);
  EXPECT_DOUBLE_EQ(writer.get_eigen_state_values()(0, 0), sample_rows[0][0]);
  
  // Reset the writer
  writer.reset();
  
  // Verify row count is reset
  EXPECT_EQ(writer.get_row_count(), 0);
  
  // Verify data is cleared
  EXPECT_DOUBLE_EQ(writer.get_eigen_state_values()(0, 0), 0.0);
  
  // Verify names are preserved
  ASSERT_EQ(writer.get_names().size(), num_cols);
  EXPECT_EQ(writer.get_names()[0], header_tokens[0]);
  
  // Write a different row
  writer(sample_rows[1]);
  
  // Verify new data is written
  EXPECT_EQ(writer.get_row_count(), 1);
  EXPECT_DOUBLE_EQ(writer.get_eigen_state_values()(0, 0), sample_rows[1][0]);
}

// Test clear functionality
TEST_F(InMemoryWriterTest, Clear) {
  // Create writer
  in_memory_writer writer(num_rows, num_cols);
  
  // Write header and first row
  writer(header_tokens);
  writer(sample_rows[0]);
  
  // Clear the writer
  writer.clear();
  
  // Verify row count is reset
  EXPECT_EQ(writer.get_row_count(), 0);
  
  // Verify data is cleared
  EXPECT_DOUBLE_EQ(writer.get_eigen_state_values()(0, 0), 0.0);
  
  // Verify names are also cleared
  EXPECT_EQ(writer.get_names().size(), 0);
}

// Test with different size allocations
TEST_F(InMemoryWriterTest, DifferentSizeAllocations) {
  // Create writer with more capacity than needed
  in_memory_writer writer(num_rows * 2, num_cols);
  
  // Write header
  writer(header_tokens);
  
  // Write sample rows
  for (const auto& row : sample_rows) {
    writer(row);
  }
  
  // Verify data was stored correctly
  const Eigen::MatrixXd& data = writer.get_eigen_state_values();
  ASSERT_EQ(data.rows(), num_rows * 2);  // Total allocation
  ASSERT_EQ(data.cols(), num_cols);
  
  // Verify only written rows have data
  for (size_t i = 0; i < num_rows; ++i) {
    for (size_t j = 0; j < num_cols; ++j) {
      EXPECT_DOUBLE_EQ(data(i, j), sample_rows[i][j]);
    }
  }
  
  // Remaining rows should be zero
  for (size_t i = num_rows; i < num_rows * 2; ++i) {
    for (size_t j = 0; j < num_cols; ++j) {
      EXPECT_DOUBLE_EQ(data(i, j), 0.0);
    }
  }
}

// Test integration with dispatcher
TEST_F(InMemoryWriterTest, DispatcherIntegration) {
  // Create writer and dispatcher
  in_memory_writer writer(num_rows, num_cols);
  dispatcher disp;
  
  // Register channel
  disp.register_channel(
      InfoType::SAMPLE_RAW,
      std::unique_ptr<stan::callbacks::Channel>(
          new WriterChannel(&writer))
  );
  
  // Send header and data through dispatcher
  disp.dispatch(InfoType::SAMPLE_RAW, header_tokens);
  for (const auto& row : sample_rows) {
    disp.dispatch(InfoType::SAMPLE_RAW, row);
  }
  
  // Verify data was received correctly
  const Eigen::MatrixXd& data = writer.get_eigen_state_values();
  for (size_t i = 0; i < num_rows; ++i) {
    for (size_t j = 0; j < num_cols; ++j) {
      EXPECT_DOUBLE_EQ(data(i, j), sample_rows[i][j]);
    }
  }
}

// Test handling of string messages
TEST_F(InMemoryWriterTest, StringMessages) {
  in_memory_writer writer(num_rows, num_cols);
  
  // Write header
  writer(header_tokens);
  
  // Write a string message (should be ignored)
  writer("This is a message that should be ignored");
  
  // Write sample row
  writer(sample_rows[0]);
  
  // Verify row count is correct (only the data row should be counted)
  EXPECT_EQ(writer.get_row_count(), 1);
  
  // Verify data was stored correctly
  const Eigen::MatrixXd& data = writer.get_eigen_state_values();
  for (size_t j = 0; j < num_cols; ++j) {
    EXPECT_DOUBLE_EQ(data(0, j), sample_rows[0][j]);
  }
}

// Test handling of empty calls
TEST_F(InMemoryWriterTest, EmptyCalls) {
  in_memory_writer writer(num_rows, num_cols);
  
  // Write header
  writer(header_tokens);
  
  // Write an empty call (should be ignored)
  writer();
  
  // Write sample row
  writer(sample_rows[0]);
  
  // Verify row count is correct (only the data row should be counted)
  EXPECT_EQ(writer.get_row_count(), 1);
  
  // Verify data was stored correctly
  const Eigen::MatrixXd& data = writer.get_eigen_state_values();
  for (size_t j = 0; j < num_cols; ++j) {
    EXPECT_DOUBLE_EQ(data(0, j), sample_rows[0][j]);
  }
}

// Test handling of Eigen input types
TEST_F(InMemoryWriterTest, EigenInputs) {
  in_memory_writer writer(num_rows + 3, num_cols);
  
  // Write header
  writer(header_tokens);
  
  // Write sample row as vector<double>
  writer(sample_rows[0]);
  
  // Write sample row as Eigen::VectorXd
  Eigen::VectorXd vec(num_cols);
  for (size_t j = 0; j < num_cols; ++j) {
    vec(j) = sample_rows[1][j];
  }
  writer(vec);
  
  // Write sample row as Eigen::RowVectorXd
  Eigen::RowVectorXd row_vec(num_cols);
  for (size_t j = 0; j < num_cols; ++j) {
    row_vec(j) = sample_rows[2][j];
  }
  writer(row_vec);
  
  // Create a matrix with all three rows
  Eigen::MatrixXd matrix(3, num_cols);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < num_cols; ++j) {
      matrix(i, j) = sample_rows[i][j];
    }
  }
  
  // Write all rows at once as a matrix
  writer(matrix);
  
  // Verify row count (should be 1 + 1 + 1 + 3 = 6)
  EXPECT_EQ(writer.get_row_count(), 6);
  
  // Verify data was stored correctly
  const Eigen::MatrixXd& data = writer.get_eigen_state_values();
  
  // First three rows should match the original inputs
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < num_cols; ++j) {
      EXPECT_DOUBLE_EQ(data(i, j), sample_rows[i][j]);
    }
  }
  
  // Next three rows should match the matrix input
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < num_cols; ++j) {
      EXPECT_DOUBLE_EQ(data(i + 3, j), sample_rows[i][j]);
    }
  }
}

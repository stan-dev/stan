#ifndef STAN_CALLBACKS_IN_MEMORY_WRITER_HPP
#define STAN_CALLBACKS_IN_MEMORY_WRITER_HPP

#include <vector>
#include <stdexcept>
#include <cstddef>
#include <Eigen/Dense>
#include <stan/callbacks/writer.hpp>

namespace stan {
namespace callbacks {

/**
 * A general in-memory writer that stores draws from Stan's write_array
 * callback into a contiguous Eigen matrix. The matrix is allocated with
 * column-major storage (Eigen's default), meaning that the element at
 * row i and column j is stored at index (i + j * num_rows) in memory.
 *
 * The writer accepts draws row-by-row and writes them into the matrix.
 * If more rows are written than initially allocated, or if the input row
 * does not have the expected number of columns, an exception is thrown.
 */
class in_memory_writer : public stan::callbacks::writer {
 public:
  /**
   * Construct an in-memory writer.
   *
   * @param num_rows the total number of rows (draws) expected
   * @param num_cols the number of columns (parameters) per draw
   *
   * The underlying Eigen matrix is allocated to have dimensions
   * num_rows x num_cols in column-major order.
   */
  in_memory_writer(std::size_t num_rows, std::size_t num_cols)
      : num_rows_(num_rows),
        num_cols_(num_cols),
        data_(Eigen::MatrixXd::Zero(num_rows, num_cols)),  // column-major
        current_row_(0),
        names_() {}

  virtual ~in_memory_writer() {}

  /**
   * Writes a set of names.
   *
   * @param[in] names Names in a std::vector
   */
  void operator()(const std::vector<std::string>& names) override {
    names_ = names;
  }

  /**
   * Writes a single row (draw) to the in-memory matrix.
   *
   * The input vector must have exactly num_cols elements.
   * The row is written into the matrix at the current row index.
   * If the matrix is full, an exception is thrown.
   *
   * @param row A vector containing one draw from the posterior.
   */
  void operator()(const std::vector<double>& row) override {
    if (row.size() != num_cols_) {
      throw std::runtime_error("Row size does not match the number of columns");
    }
    if (current_row_ >= num_rows_) {
      throw std::runtime_error("Attempted to write more rows than allocated");
    }
    // Because Eigen::MatrixXd is column-major by default, simply assigning
    // row by row will place the data in memory in the order:
    // index = current_row_ + (column_index * num_rows_).
    for (std::size_t j = 0; j < num_cols_; ++j) {
      data_(current_row_, j) = row[j];
    }
    ++current_row_;
  }

  /**
   * Default implementation for empty call.
   */
  void operator()() override {}

  /**
   * Default implementation for string message.
   */
  void operator()(const std::string& message) override {}

  /**
   * Handles Eigen matrix input by converting to rows and inserting.
   * This method treats each row of the input matrix as a separate draw.
   */
  void operator()(const Eigen::Matrix<double, -1, -1>& matrix) override {
    for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
      std::vector<double> row(matrix.cols());
      for (Eigen::Index j = 0; j < matrix.cols(); ++j) {
        row[j] = matrix(i, j);
      }
      (*this)(row);  // Use the vector<double> operator to insert the row
    }
  }

  /**
   * Handles Eigen vector input by inserting as a single row.
   */
  void operator()(const Eigen::Matrix<double, -1, 1>& vector) override {
    std::vector<double> row(vector.size());
    for (Eigen::Index i = 0; i < vector.size(); ++i) {
      row[i] = vector(i);
    }
    (*this)(row);  // Use the vector<double> operator to insert the row
  }

  /**
   * Handles Eigen row vector input by inserting as a single row.
   */
  void operator()(const Eigen::Matrix<double, 1, -1>& vector) override {
    std::vector<double> row(vector.size());
    for (Eigen::Index i = 0; i < vector.size(); ++i) {
      row[i] = vector(i);
    }
    (*this)(row);  // Use the vector<double> operator to insert the row
  }

  /**
   * Always returns true as the in-memory writer is always valid.
   */
  bool is_valid() const noexcept override { return true; }

  /**
   * Returns a const reference to the in-memory Eigen matrix containing all
   * draws.
   *
   * The matrix is stored in column-major order.
   *
   * @return const reference to the Eigen::MatrixXd holding the draws.
   */
  const Eigen::MatrixXd& get_eigen_state_values() const { return data_; }

  /**
   * Returns a const reference to the column names.
   *
   * @return const reference to the vector of column names.
   */
  const std::vector<std::string>& get_names() const { return names_; }

  /**
   * Returns the number of rows that have been written so far.
   *
   * @return The current row count.
   */
  std::size_t get_row_count() const { return current_row_; }

  /**
   * Resets the writer to its initial state.
   *
   * Clears the stored data (sets the matrix to zero) and resets the current row
   * index. Column names are retained.
   */
  void reset() {
    current_row_ = 0;
    data_.setZero();
  }

  /**
   * Fully resets the writer, including clearing column names.
   */
  void clear() {
    current_row_ = 0;
    data_.setZero();
    names_.clear();
  }

 private:
  std::size_t num_rows_;  // Total number of draws (rows) expected.
  std::size_t num_cols_;  // Number of parameters (columns) per draw.
  Eigen::MatrixXd data_;  // Internal storage; Eigen matrices are column-major.
  std::size_t current_row_;         // Next row index to be written.
  std::vector<std::string> names_;  // Column names
};

}  // namespace callbacks
}  // namespace stan

#endif  // STAN_CALLBACKS_IN_MEMORY_WRITER_HPP

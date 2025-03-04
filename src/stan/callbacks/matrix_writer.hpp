#ifndef STAN_CALLBACKS_MATRIX_WRITER_HPP
#define STAN_CALLBACKS_MATRIX_WRITER_HPP

#include <stan/callbacks/writer.hpp>
#include <vector>
#include <memory>

namespace stan {
namespace callbacks {

class matrix_writer : public writer {
 public:
  matrix_writer(size_t rows, size_t cols)
      : rows_(rows), cols_(cols), data_(rows * cols) {}

  void operator()(const std::vector<double>& x) override {
    if (current_row_ >= rows_) {
      throw std::runtime_error("Matrix writer: too many rows");
    }
    if (x.size() != cols_) {
      throw std::runtime_error("Matrix writer: incorrect number of columns");
    }

    // Store in column-major order
    for (size_t j = 0; j < cols_; ++j) {
      data_[j * rows_ + current_row_] = x[j];
    }
    current_row_++;
  }

  void operator()(const std::vector<std::string>& x) override {
    throw std::runtime_error("Matrix writer does not support string data");
  }

  double* data() { return data_.data(); }
  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }
  size_t current_row() const { return current_row_; }

 private:
  size_t rows_;
  size_t cols_;
  size_t current_row_{0};
  std::vector<double> data_;
};

}  // namespace callbacks
}  // namespace stan

#endif


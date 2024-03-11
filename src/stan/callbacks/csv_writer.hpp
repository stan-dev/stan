#ifndef STAN_CALLBACKS_CSV_WRITER_HPP
#define STAN_CALLBACKS_CSV_WRITER_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/callbacks/process_string.hpp>
#include <stan/callbacks/table_writer.hpp>
#include <ostream>
#include <string>
#include <vector>
#include <memory>
#include <cstring>
#include <iostream>

namespace stan {
namespace callbacks {

/**
 * The `csv_writer` callback is used output a single CSV file,
 * consisting of a header row, data rows, and comment rows.
 * Data rows contain one or more values, either real or integer.
 * The header and all data rows must contain the same number of elements.
 * Comments are prefixed by "#" and consist of a either a label
 * or a label, value pair
 *
 * @tparam Stream A type with with a valid `operator<<(std::string)`
 * @tparam Deleter A class with a valid `operator()` method for deleting the
 * output stream
 */
template <typename Stream, typename Deleter = std::default_delete<Stream>>
class csv_writer final : public table_writer  {
 private:
  // Output stream
  std::unique_ptr<Stream, Deleter> output_{nullptr};

  // Internal state:  header row, data row, items per row
  size_t header_size_ = 0;
  size_t row_size_ = 0;
  bool wrote_header = false;

  /**
   * Writes a single value.  Corrects capitalization for inf and nans.
   *
   * @param[in] v value
   */
  void write_value(double v) {
    if (unlikely(std::isinf(v))) {
      if (v > 0) {
        *output_ << "Inf";
      } else {
        *output_ << "-Inf";
      }
    } else if (unlikely(std::isnan(v))) {
      *output_ << "NaN";
    } else {
      *output_ << v;
    }
  }

  /**
   * Writes a single complex value.
   *
   * @param[in] v value
   */
  void write_complex_value(std::complex<double> v) {
    write_value(v.real());
    *output_ << ", ";
    write_value(v.imag());
  }

  /**
   * Writes the set of comma separated values in an Eigen (row) vector.
   *
   * @param[in] v Values in an `Eigen::Vector`
   */
  template <typename Derived>
  void write_eigen_vector(const Eigen::DenseBase<Derived>& v) {
    *output_ << "# ";
    if (v.size() > 0) {
      size_t last = v.size() - 1;
      for (Eigen::Index i = 0; i < last; ++i) {
        write_value(v[i]);
        *output_ << ", ";
      }
      write_value(v[last]);
    }
    *output_ << std::endl;
  }

 public:
  /**
   * Constructs a no-op csv writer.
   */
  csv_writer() : output_(nullptr) {}

  /**
   * Constructs a csv writer with an output stream.
   *
   * @param[in, out] output unique pointer to a type inheriting from
   * `std::ostream`
   */
  explicit csv_writer(std::unique_ptr<Stream, Deleter>&& output)
      : output_(std::move(output)) {}

  /** copy constructor */
  csv_writer(csv_writer& other) = delete;

  /** move constructor */
  csv_writer(csv_writer&& other) noexcept
      : output_(std::move(other.output_)) {}

  virtual ~csv_writer() {}

  /**
   * Setup header row
   */
  void begin_header() {
    if (output_ == nullptr) {
      return;
    }
    if (header_size_ != 0) {
      throw std::domain_error("Error, multiple heaader rows");
    } else if (row_size_ != 0) {
      throw std::domain_error("Error, data before header");
    }
  }

  /**
   * Outputs newline
   */
  void end_header() {
    if (output_ == nullptr ) {
      return;
    }
    if (header_size_ > 0) {
      *output_ << std::endl;
    }
  }

  /**
   * Writes a vector of column names for a data table.
   * @param values vector of strings to write.
   */
  void write_header(const std::vector<std::string>& values) {
    if (output_ == nullptr) {
      return;
    }
    if (header_size_ > 0) {
      *output_ << ", ";
    }
    if (values.size() > 0) {
      auto last = values.end();
      --last;
      for (auto it = values.begin(); it != last; ++it) {
        *output_ << process_string(*it) << ", ";
      }
      *output_ << process_string(values.back());
      header_size_ += values.size();
    }
  }

  /**
   * Don't check state; just set up next row.
   */
  void begin_row() {
    if (output_ == nullptr) {
      return;
    }
    if (row_size_ > 0) {
      *output_ << std::endl;
      row_size_ = 0;
    }
  }

  /**
   * Output newline as needed, reset state
   */
  void end_row() {
    if (output_ == nullptr) {
      return;
    }
    if (row_size_ > 0) {
      *output_ << std::endl;
      row_size_ = 0;
    }
  }

  /**
   * Check state, pad output row as needed, output newline, reset state
   */
  void pad_row() {
    if (output_ == nullptr) {
      return;
    }
    if (row_size_ == 0) {  // avoid spurious newlines
      return;
    }
    if (row_size_ == header_size_) {
      *output_ << std::endl;
    } else if (row_size_ < header_size_) {
      size_t missing = header_size_ - row_size_;
      for (auto i = 0; i < missing; ++i) {
        *output_ << ", 0";
      }
      *output_ << std::endl;
    }
    row_size_ = 0;
  }      

  /// write_flat - like write, but no key
  void write_flat(const std::vector<double>& values) {
    if (output_ == nullptr) {
      return;
    }
    if (row_size_ > 0) {
      *output_ << ", ";
    }
    auto last = values.end();
    --last;
    for (auto it = values.begin(); it != last; ++it) {
      write_value(*it);
      *output_ << ", ";
    }
    write_value(values.back());
    row_size_ += values.size();
  }

};

}  // namespace callbacks
}  // namespace stan
#endif

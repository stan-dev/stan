#ifndef STAN_CALLBACKS_CSV_WRITER_HPP
#define STAN_CALLBACKS_CSV_WRITER_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/callbacks/structured_writer.hpp>
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
class csv_writer final : public structured_writer {
 private:
  // Output stream
  std::unique_ptr<Stream, Deleter> output_{nullptr};

  // Internal state:  header row, data row, items per row
  size_t header_size_ = 0;
  size_t row_size_ = 0;

  template <typename T>
  void write_int_like(const std::string& key, T value) {
    if (output_ == nullptr) {
      return;
    }
    if (row_size_ > 0) {
      *output_ << std::endl;
      row_size_ = 0;
    }
    *output_ << process_string(key) << " = " << value << std::endl;
  }

  /**
   * Writes key as initial element of comment line followed by " = "
   *
   * @param[in] key value
   */
  void write_key(const std::string& key) {
    if (row_size_ > 0) {
      *output_ << std::endl;
      row_size_ = 0;
    }
    *output_ << "# " << process_string(key) << " = ";
  }

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
    if (output_ == nullptr) {
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
   * Output newline, as needed.
   */
  void end_row() {
    if (output_ == nullptr) {
      return;
    }
    if (row_size_ == 0) {  // avoid spurious newlines
      return;
    }
    *output_ << std::endl;
    row_size_ = 0;
  }

  /**
   * Check state, pad output row as needed, output newline, reset state
   */
  void end_row_padded() {
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
    row_size_ == 0;
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

  /**
   * No-op implementation.
   */
  void begin_record() {
      return;
  }

  /**
   * Writes "" key\n".
   * @param[in] key The name of the record.
   */
  void begin_record(const std::string& key) {
    if (output_ == nullptr) {
      return;
    }
    if (row_size_ > 0) {
      *output_ << std::endl;
      row_size_ = 0;
    }
    *output_ << "# " << process_string(key) << std::endl;
  }

  /**
   * No-op implementation.
   */
  void end_record() {
      return;
  }

  /**
   * Write a key-value pair to the output stream with a value of null as the
   * value.
   * @param key Name of the value pair
   */
  void write(const std::string& key) {
    if (output_ == nullptr) {
      return;
    }
    if (row_size_ > 0) {
      *output_ << std::endl;
      row_size_ = 0;
    }
    *output_ << "# " << process_string(key) << std::endl;
  }

  /**
   * Write a key-value pair where the value is a string.
   * @param key Name of the value pair
   * @param value string to write.
   */
  void write(const std::string& key, const std::string& value) {
    if (output_ == nullptr) {
      return;
    }
    write_key(key);
    *output_ << process_string(value) << std::endl;
  }

  /**
   * Write a key-value pair where the value is a const char*.
   * @param key Name of the value pair
   * @param value pointer to chars to write.
   */
  void write(const std::string& key, const char* value) {
    if (output_ == nullptr) {
      return;
    }
    write_key(key);
    *output_ << process_string(value) << std::endl;
  }

  /**
   * Write a key-value pair where the value is a bool.
   * @param key Name of the value pair
   * @param value bool to write.
   */
  void write(const std::string& key, bool value) {
    if (output_ == nullptr) {
      return;
    }
    write_key(key);
    *output_ << (value ? "true" : "false") << std::endl;
  }

  /**
   * Write a key-value pair where the value is an int.
   * @param key Name of the value pair
   * @param value int to write.
   */
  void write(const std::string& key, int value) { write_int_like(key, value); }

  /**
   * Write a key-value pair where the value is an `std::size_t`.
   * @param key Name of the value pair
   * @param value `std::size_t` to write.
   */
  void write(const std::string& key, std::size_t value) {
    write_int_like(key, value);
  }

  /**
   * Write a key-value pair where the value is an `long long int`.
   * @param key Name of the value pair
   * @param value `long long int` to write.
   */
  void write(const std::string& key,
             long long int value  // NOLINT(runtime/int)
  ) {
    write_int_like(key, value);
  }

  /**
   * Write a key-value pair where the value is an `unsigned int`.
   * @param key Name of the value pair
   * @param value `unsigned int` to write.
   */
  void write(const std::string& key, unsigned int value) {
    write_int_like(key, value);
  }

  /**
   * Write a key-value pair where the value is a double.
   * @param key Name of the value pair
   * @param value double to write.
   */
  void write(const std::string& key, double value) {
    if (output_ == nullptr) {
      return;
    }
    write_key(key);
    *output_ << value << std::endl;
  }

  /**
   * Write a key-value pair where the value is a complex value.
   * @param key Name of the value pair
   * @param value complex value to write.
   */
  void write(const std::string& key, const std::complex<double>& value) {
    if (output_ == nullptr) {
      return;
    }
    write_key(key);
    write_complex_value(value);
    *output_ << std::endl;
  }

  /**
   * Write a key-value pair where the value is a vector to be made a list.
   * @param key Name of the value pair
   * @param values vector to write.
   */
  void write(const std::string& key, const std::vector<std::string>& values) {
    if (output_ == nullptr) {
      return;
    }
    write_key(key);
    if (values.size() > 0) {
      auto last = values.end();
      --last;
      for (auto it = values.begin(); it != last; ++it) {
        *output_ << process_string(*it) << ", ";
      }
    }
    *output_ << values.back() << std::endl;
  }

  /**
   * Write a key-value pair where the value is a vector to be made a list.
   * @param key Name of the value pair
   * @param values vector to write.
   */
  void write(const std::string& key, const std::vector<double>& values) {
    if (output_ == nullptr) {
      return;
    }
    write_key(key);
    if (values.size() > 0) {
      auto last = values.end();
      --last;
      for (auto it = values.begin(); it != last; ++it) {
        write_value(*it);
        *output_ << ", ";
      }
      write_value(values.back());
    }
    *output_ << std::endl;
  }

  /**
   * Write a key-value pair where the value is a vector to be made a list.
   * @param key Name of the value pair
   * @param values vector to write.
   */
  void write(const std::string& key, const std::vector<int>& values) {
    if (output_ == nullptr) {
      return;
    }
    write_key(key);
    if (values.size() > 0) {
      auto last = values.end();
      --last;
      for (auto it = values.begin(); it != last; ++it) {
        *output_ << *it << ", ";
      }
    }
    *output_ << values.back() << std::endl;
  }

  /**
   * Write a key-value pair where the value is a vector to be made a list.
   * @param key Name of the value pair
   * @param values vector to write.
   */
  void write(const std::string& key,
             const std::vector<std::complex<double>>& values) {
    if (output_ == nullptr) {
      return;
    }
    write_key(key);
    if (values.size() > 0) {
      size_t last = values.size() - 1;
      for (size_t i = 0; i < last; ++i) {
        write_complex_value(values[i]);
        *output_ << ", ";
      }
      write_complex_value(values[last]);
    }
    *output_ << std::endl;
  }

  /**
   * Write a key-value pair where the value is an Eigen Vector.
   * @param key Name of the value pair
   * @param vec Eigen Vector to write.
   */
  void write(const std::string& key, const Eigen::VectorXd& vec) {
    if (output_ == nullptr) {
      return;
    }
    write_key(key);
    write_eigen_vector(vec);
  }

  /**
   * Write a key-value pair where the value is an Eigen Vector.
   * @param key Name of the value pair
   * @param vec Eigen Vector to write.
   */
  void write(const std::string& key, const Eigen::RowVectorXd& vec) {
    if (output_ == nullptr) {
      return;
    }
    write_key(key);
    write_eigen_vector(vec);
  }

  /**
   * Write a key-value pair where the value is an Eigen Matrix.
   * @param key Name of the value pair
   * @param mat Eigen Matrix to write.
   */
  void write(const std::string& key, const Eigen::MatrixXd& mat) {
    if (output_ == nullptr) {
      return;
    }
    write_key(key);
    if (mat.rows() > 0) {
      for (Eigen::Index i = 0; i < mat.rows(); ++i) {
        write_eigen_vector(mat.row(i));
      }
    } else {
    *output_ << std::endl;
    }      
  }

};

}  // namespace callbacks
}  // namespace stan
#endif

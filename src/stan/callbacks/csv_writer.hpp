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
 * The `csv_writer` callback is used output a single CSV table,
 * which consists of a header row, followed by zero or more data rows.
 * Data rows contain one or more values, either real or integer.
 * The header and all data rows must contain the same number of elements.
 * The writer doesn't try to validate the object's internal structure
 * or object completeness, only syntactic correctness.
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

  // Whether or not header row has been written
  bool has_header_ = false;

  size_t num_cols_ = 0;


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

 public:
  /**
   * Constructs a no-op csv writer.
   *
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
   * Writes a vector of column names for a data table.
   * @param values vector of strings to write.
   */
  void table_header(const std::vector<std::string>& values) {
    if (!has_header_) {
      if (values.size() > 0) {
        auto last = values.end();
        --last;
        for (auto it = values.begin(); it != last; ++it) {
          *output_ << process_string(*it) << ", ";
        }
        *output_ << process_string(values.back()) << std::endl;
        has_header_ = true;
        num_cols_ = values.size();
      }
    } // else illegal
  }

  void table_row(const std::vector<double>& values) {
    std::stringstream msg;
    if (has_header_) {
      if (values.size() == num_cols_ && values.size() > 0) {
        auto last = values.end();
        --last;
        for (auto it = values.begin(); it != last; ++it) {
          write_value(*it);
          *output_ << ", ";
        }
        write_value(values.back());
        *output_ << std::endl;
      }
    }
  }


  // /**
  //  * Writes a row of a data table.
  //  * @param values vector of values.
  //  */
  // void table_row(const std::vector<double>& values, bool pad = false) {
  //   std::stringstream msg;
  //   std::cout << "table_row, num_values: " << values.size() << std::endl;
  //   if (has_header_) {
  //     if (values.size() == num_cols_ && values.size() > 0) {
  //       auto last = values.end();
  //       --last;
  //       for (auto it = values.begin(); it != last; ++it) {
  //         write_value(*it);
  //         *output_ << ", ";
  //       }
  //       write_value(values.back());
  //       *output_ << std::endl;
  //     } else if (pad && values.size() > 0) {
  //       size_t missing = num_cols_ - values.size();
  //       auto last = values.end();
  //       for (auto it = values.begin(); it != last; ++it) {
  //         write_value(*it);
  //         *output_ << ", ";
  //       }
  //       for (auto i = 0; i < missing; ++i) {
  //         *output_ << "0, ";
  //       }
  //       *output_ << "0" << std::endl;
  //     } else {
  //       msg << "Output error, expected " << num_cols_
  //           << " values, got " << values.size() ;
  //       throw std::domain_error(msg.str());
  //     }
  //   } else {
  //     msg << "Output error, missing header.";
  //     throw std::domain_error(msg.str());
  //   }
  // }


};

}  // namespace callbacks
}  // namespace stan
#endif

#ifndef STAN_IO_JSON_JSON_DATA_HANDLER_HPP
#define STAN_IO_JSON_JSON_DATA_HANDLER_HPP

#include <stan/io/json/json_error.hpp>
#include <stan/io/json/json_handler.hpp>
#include <stan/io/json/rapidjson_parser.hpp>
#include <stan/io/var_context.hpp>
#include <cctype>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace stan {

namespace json {

typedef std::map<std::string,
                 std::pair<std::vector<double>, std::vector<size_t>>>
    vars_map_r;

typedef std::map<std::string, std::pair<std::vector<int>, std::vector<size_t>>>
    vars_map_i;

/**
 * A <code>json_data_handler</code> is an implementation of a
 * <code>json_handler</code> that restricts the allowed JSON text
 * to a set of Stan variable declarations in JSON format.
 * Each Stan variable consists of a JSON key : value pair.
 * The key is a string and the value is either a single numeric
 * scalar value or a JSON array of numeric values.
 *
 * <p>The <code>json_data_handler</code> checks that the top-level
 * JSON object contains a set of name-value pairs
 * where the values can be either numeric scalar objects or
 * or numeric arrays of any dimensionality. Arrays must be rectangular.
 * The strings \"Inf\" and \"Infinity\" are mapped to positive infinity,
 * the strings \"-Inf\" and \"-Infinity\" are mapped to negative infinity,
 * and the string \"NaN\" is mapped to not-a-number.
 * Bare versions of Infinity, -Infinity, and NaN are also allowed.
 */
class json_data_handler : public stan::json::json_handler {
 private:
  vars_map_r &vars_r_;
  vars_map_i &vars_i_;
  std::string key_;
  std::vector<double> values_r_;
  std::vector<int> values_i_;
  std::vector<size_t> dims_;
  std::vector<size_t> dims_verify_;
  std::vector<bool> dims_unknown_;
  size_t dim_idx_;
  size_t dim_last_;
  bool is_int_;

  void reset() {
    key_.clear();
    values_r_.clear();
    values_i_.clear();
    dims_.clear();
    dims_verify_.clear();
    dims_unknown_.clear();
    dim_idx_ = 0;
    dim_last_ = 0;
    is_int_ = true;
  }

  bool is_init() {
    return (key_.size() == 0 && values_r_.size() == 0 && values_i_.size() == 0
            && dims_.size() == 0 && dims_verify_.size() == 0
            && dims_unknown_.size() == 0 && dim_idx_ == 0 && dim_last_ == 0
            && is_int_);
  }

 public:
  /**
   * Construct a json_data_handler object.
   *
   * <b>Warning:</b> This method does not close the input stream.
   *
   * @param vars_r name-value map for real-valued variables
   * @param vars_i name-value map for int-valued variables
   */
  json_data_handler(vars_map_r &vars_r, vars_map_i &vars_i)
      : json_handler(),
        vars_r_(vars_r),
        vars_i_(vars_i),
        key_(),
        values_r_(),
        values_i_(),
        dims_(),
        dims_verify_(),
        dims_unknown_(),
        dim_idx_(0),
        dim_last_(0),
        is_int_(true) {}

  void start_text() {
    vars_i_.clear();
    vars_r_.clear();
    reset();
  }

  void end_text() { reset(); }

  void start_array() {
    if (0 == key_.size()) {
      throw json_error("expecting JSON object, found array");
    }
    if (dim_idx_ > 0 && dim_last_ == dim_idx_) {
      std::stringstream errorMsg;
      errorMsg << "variable: " << key_ << ", error: non-scalar array value";
      throw json_error(errorMsg.str());
    }
    incr_dim_size();
    dim_idx_++;
    if (dims_.size() < dim_idx_) {
      dims_.push_back(0);
      dims_unknown_.push_back(true);
      dims_verify_.push_back(0);
    } else {
      dims_verify_[dim_idx_ - 1] = 0;
    }
  }

  void end_array() {
    if (dims_unknown_[dim_idx_ - 1] == true) {
      dims_unknown_[dim_idx_ - 1] = false;
    } else if (dims_verify_[dim_idx_ - 1] != dims_[dim_idx_ - 1]) {
      std::stringstream errorMsg;
      errorMsg << "variable: " << key_ << ", error: non-rectangular array";
      throw json_error(errorMsg.str());
    }
    if (0 == dim_last_
        && ((is_int_ && values_i_.size() > 0) || (values_r_.size() > 0)))
      dim_last_ = dim_idx_;
    dim_idx_--;
  }

  void start_object() {
    if (!is_init()) {
      std::stringstream errorMsg;
      errorMsg << "variable: " << key_ << ", error: nested objects not allowed";
      throw json_error(errorMsg.str());
    }
  }

  void end_object() {
    save_current_key_value_pair();
    reset();
  }

  void promote_to_double() {
    if (is_int_) {
      for (std::vector<int>::iterator it = values_i_.begin();
           it != values_i_.end(); ++it)
        values_r_.push_back(*it);
      is_int_ = false;
    }
  }

  void null() {
    std::stringstream errorMsg;
    errorMsg << "variable: " << key_ << ", error: null values not allowed";
    throw json_error(errorMsg.str());
  }

  void boolean(bool p) {
    std::stringstream errorMsg;
    errorMsg << "variable: " << key_ << ", error: boolean values not allowed";
    throw json_error(errorMsg.str());
  }

  void string(const std::string &s) {
    double tmp;
    if (0 == s.compare("-Inf")) {
      tmp = -std::numeric_limits<double>::infinity();
    } else if (0 == s.compare("-Infinity")) {
      tmp = -std::numeric_limits<double>::infinity();
    } else if (0 == s.compare("Inf")) {
      tmp = std::numeric_limits<double>::infinity();
    } else if (0 == s.compare("Infinity")) {
      tmp = std::numeric_limits<double>::infinity();
    } else if (0 == s.compare("NaN")) {
      tmp = std::numeric_limits<double>::quiet_NaN();
    } else {
      std::stringstream errorMsg;
      errorMsg << "variable: " << key_ << ", error: string values not allowed";
      throw json_error(errorMsg.str());
    }
    promote_to_double();
    values_r_.push_back(tmp);
    incr_dim_size();
  }

  void key(const std::string &key) {
    save_current_key_value_pair();
    reset();
    key_ = key;
  }

  void number_double(double x) {
    set_last_dim();
    promote_to_double();
    values_r_.push_back(x);
    incr_dim_size();
  }

  void number_int(int n) {
    set_last_dim();
    if (is_int_) {
      values_i_.push_back(n);
    } else {
      values_r_.push_back(n);
    }
    incr_dim_size();
  }

  void number_unsigned_int(unsigned n) {
    set_last_dim();
    // if integer overflow, promote numeric data to double
    if (n > (unsigned)std::numeric_limits<int>::max())
      promote_to_double();
    if (is_int_) {
      values_i_.push_back(static_cast<int>(n));
    } else {
      values_r_.push_back(n);
    }
    incr_dim_size();
  }

  void number_int64(int64_t n) {
    // the number doesn't fit in int (otherwise number_int() would be called)
    number_double(n);
  }

  void number_unsigned_int64(uint64_t n) {
    // the number doesn't fit in int (otherwise number_unsigned_int() would be
    // called)
    number_double(n);
  }

  void save_current_key_value_pair() {
    if (0 == key_.size())
      return;

    // redefinition or variables not allowed
    if (vars_r_.find(key_) != vars_r_.end()
        || vars_i_.find(key_) != vars_i_.end()) {
      std::stringstream errorMsg;
      errorMsg << "attempt to redefine variable: " << key_;
      throw json_error(errorMsg.str());
    }

    // transpose order of array values to column-major
    if (is_int_) {
      std::pair<std::vector<int>, std::vector<size_t>> pair;
      if (dims_.size() > 1) {
        std::vector<int> cm_values_i(values_i_.size());
        to_column_major(cm_values_i, values_i_, dims_);
        pair = make_pair(cm_values_i, dims_);

      } else {
        pair = make_pair(values_i_, dims_);
      }
      vars_i_[key_] = pair;
    } else {
      std::pair<std::vector<double>, std::vector<size_t>> pair;
      if (dims_.size() > 1) {
        std::vector<double> cm_values_r(values_r_.size());
        to_column_major(cm_values_r, values_r_, dims_);
        pair = make_pair(cm_values_r, dims_);
      } else {
        pair = make_pair(values_r_, dims_);
      }
      vars_r_[key_] = pair;
    }
  }

  void incr_dim_size() {
    if (dim_idx_ > 0) {
      if (dims_unknown_[dim_idx_ - 1])
        dims_[dim_idx_ - 1]++;
      else
        dims_verify_[dim_idx_ - 1]++;
    }
  }

  template <typename T>
  void to_column_major(std::vector<T> &cm_vals, const std::vector<T> &rm_vals,
                       const std::vector<size_t> &dims) {
    for (size_t i = 0; i < rm_vals.size(); i++) {
      size_t idx = convert_offset_rtl_2_ltr(i, dims);
      cm_vals[idx] = rm_vals[i];
    }
  }

  void set_last_dim() {
    if (dim_last_ > 0 && dim_idx_ < dim_last_) {
      std::stringstream errorMsg;
      errorMsg << "variable: " << key_ << ", error: non-rectangular array";
      throw json_error(errorMsg.str());
    }
    dim_last_ = dim_idx_;
  }

  // convert row-major offset to column-major offset
  size_t convert_offset_rtl_2_ltr(size_t rtl_offset,
                                  const std::vector<size_t> &dims) {
    size_t rtl_dsize = 1;
    for (size_t i = 1; i < dims.size(); i++)
      rtl_dsize *= dims[i];

    // array index should be valid, but check just in case
    if (rtl_offset >= rtl_dsize * dims[0]) {
      std::stringstream errorMsg;
      errorMsg << "variable: " << key_ << ", unexpected error";
      throw json_error(errorMsg.str());
    }

    // calculate offset by working left-to-right to get array indices
    // for row-major offset left-most dimensions are divided out
    // for column-major offset successive dimensions are multiplied in
    size_t rem = rtl_offset;
    size_t ltr_offset = 0;
    size_t ltr_dsize = 1;
    for (size_t i = 0; i < dims.size() - 1; i++) {
      size_t idx = rem / rtl_dsize;
      ltr_offset += idx * ltr_dsize;
      rem = rem - idx * rtl_dsize;
      rtl_dsize = rtl_dsize / dims[i + 1];
      ltr_dsize *= dims[i];
    }
    ltr_offset += rem * ltr_dsize;  // for loop stops 1 early

    return ltr_offset;
  }
};

}  // namespace json

}  // namespace stan

#endif

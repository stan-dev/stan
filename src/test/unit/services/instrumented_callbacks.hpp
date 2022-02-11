#ifndef TEST__UNIT__INSTRUMENTED_CALLBACKS_HPP
#define TEST__UNIT__INSTRUMENTED_CALLBACKS_HPP

#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <map>
#include <string>
#include <iostream>
#include <exception>
#include <atomic>
#include <mutex>
namespace stan {
namespace test {
namespace unit {

/**
 * instrumented_interrupt counts the number of times it is
 * called and makes the count accessible via a method.
 */
class instrumented_interrupt : public stan::callbacks::interrupt {
 public:
  instrumented_interrupt() : counter_(0) {}

  void operator()() { counter_++; }

  unsigned int call_count() { return counter_; }

 private:
  std::atomic<unsigned int> counter_;
};

/**
 * instrumented_writer counts the number of times it is called through
 * each route and makes the count available via methods.
 * Stores all arguments passed and makes them available via
 * methods.
 */
class instrumented_writer : public stan::callbacks::writer {
 public:
  instrumented_writer() {}

  void operator()(const std::string& key, double value) {
    counter_["string_double"]++;
    string_double.push_back(std::make_pair(key, value));
  }

  void operator()(const std::string& key, int value) {
    counter_["string_int"]++;
    string_int.push_back(std::make_pair(key, value));
  }

  void operator()(const std::string& key, const std::string& value) {
    counter_["string_string"]++;
    string_string.push_back(std::make_pair(key, value));
  }

  void operator()(const std::string& key, const double* values, int n_values) {
    counter_["string_pdouble_int"]++;
    if (n_values == 0)
      return;
    std::vector<double> value(n_values);
    for (int i = 0; i < n_values; ++i)
      value[i] = values[i];
    string_pdouble_int.push_back(std::make_pair(key, value));
  }

  void operator()(const std::string& key, const double* values, int n_rows,
                  int n_cols) {
    counter_["string_pdouble_int_int"]++;
    if (n_rows == 0 || n_cols == 0)
      return;
    Eigen::MatrixXd value(n_rows, n_cols);
    for (int i = 0; i < n_rows; ++i)
      for (int j = 0; j < n_cols; ++j)
        value(i, j) = values[i * n_cols + j];
    string_pdouble_int_int.push_back(std::make_pair(key, value));
  }

  void operator()(const std::vector<std::string>& names) {
    counter_["vector_string"]++;
    vector_string.push_back(names);
  }

  void operator()(const std::vector<double>& state) {
    counter_["vector_double"]++;
    vector_double.push_back(state);
  }

  void operator()() { counter_["empty"]++; }

  void operator()(const std::string& message) {
    counter_["string"]++;
    string.push_back(message);
  }

  unsigned int call_count() {
    unsigned int n = 0;
    for (auto& it : counter_)
      n += it.second;
    return n;
  }

  unsigned int call_count(std::string s) { return counter_[s]; }

  std::vector<std::pair<std::string, double>> string_double_values() {
    return string_double;
  };

  std::vector<std::pair<std::string, int>> string_int_values() {
    return string_int;
  };

  std::vector<std::pair<std::string, std::string>> string_string_values() {
    return string_string;
  };

  std::vector<std::pair<std::string, std::vector<double>>>
  string_pdouble_int_values() {
    return string_pdouble_int;
  };

  std::vector<std::pair<std::string, Eigen::MatrixXd>>
  string_pdouble_int_int_values() {
    return string_pdouble_int_int;
  };

  std::vector<std::vector<std::string>> vector_string_values() {
    return vector_string;
  };

  std::vector<std::vector<double>> vector_double_values() {
    return vector_double;
  };

  std::vector<std::string> string_values() { return string; };

 private:
  std::map<std::string, std::atomic<int>> counter_;
  std::vector<std::pair<std::string, double>> string_double;
  std::vector<std::pair<std::string, int>> string_int;
  std::vector<std::pair<std::string, std::string>> string_string;
  std::vector<std::pair<std::string, std::vector<double>>> string_pdouble_int;
  std::vector<std::pair<std::string, Eigen::MatrixXd>> string_pdouble_int_int;
  std::vector<std::vector<std::string>> vector_string;
  std::vector<std::vector<double>> vector_double;
  std::vector<std::string> string;
};

/**
 * instrumented_logger counts the number of times it is called through
 * each route and makes the count available via methods.
 * Stores all arguments passed and makes them available via
 * methods.
 */
class instrumented_logger : public stan::callbacks::logger {
 public:
  std::mutex logger_guard;
  instrumented_logger() {}

  void debug(const std::string& message) {
    std::lock_guard<std::mutex> guard(logger_guard);
    debug_.push_back(message);
  }

  void debug(const std::stringstream& message) {
    std::lock_guard<std::mutex> guard(logger_guard);
    debug_.push_back(message.str());
  }

  void info(const std::string& message) {
    std::lock_guard<std::mutex> guard(logger_guard);
    info_.push_back(message);
  }

  void info(const std::stringstream& message) {
    std::lock_guard<std::mutex> guard(logger_guard);
    info_.push_back(message.str());
  }

  void warn(const std::string& message) {
    std::lock_guard<std::mutex> guard(logger_guard);
    warn_.push_back(message);
  }

  void warn(const std::stringstream& message) {
    std::lock_guard<std::mutex> guard(logger_guard);
    warn_.push_back(message.str());
  }

  void error(const std::string& message) {
    std::lock_guard<std::mutex> guard(logger_guard);
    error_.push_back(message);
  }

  void error(const std::stringstream& message) {
    std::lock_guard<std::mutex> guard(logger_guard);
    error_.push_back(message.str());
  }

  void fatal(const std::string& message) { fatal_.push_back(message); }

  void fatal(const std::stringstream& message) {
    std::lock_guard<std::mutex> guard(logger_guard);
    fatal_.push_back(message.str());
  }

  unsigned int call_count() {
    return debug_.size() + info_.size() + warn_.size() + error_.size()
           + fatal_.size();
  }

  unsigned int call_count_debug() { return debug_.size(); }

  unsigned int call_count_info() { return info_.size(); }

  unsigned int call_count_warn() { return warn_.size(); }

  unsigned int call_count_error() { return error_.size(); }

  unsigned int call_count_fatal() { return fatal_.size(); }

  unsigned int find_debug(const std::string& msg) { return find_(debug_, msg); }

  unsigned int find_info(const std::string& msg) { return find_(info_, msg); }

  unsigned int find_warn(const std::string& msg) { return find_(warn_, msg); }

  unsigned int find_error(const std::string& msg) { return find_(error_, msg); }

  unsigned int find_fatal(const std::string& msg) { return find_(fatal_, msg); }

  unsigned int find(const std::string& msg) {
    return find_debug(msg) + find_info(msg) + find_warn(msg) + find_error(msg)
           + find_fatal(msg);
  }

  void print_info(std::ostream& o) {
    for (size_t n = 0; n < info_.size(); ++n)
      o << info_[n] << std::endl;
  }

 private:
  unsigned int find_(const std::vector<std::string>& vec,
                     const std::string& msg) {
    unsigned int count = 0;
    for (size_t n = 0; n < vec.size(); ++n)
      if (vec[n].find(msg) != std::string::npos)
        count++;
    return count;
  }

  std::vector<std::string> debug_;
  std::vector<std::string> info_;
  std::vector<std::string> warn_;
  std::vector<std::string> error_;
  std::vector<std::string> fatal_;
};

}  // namespace unit
}  // namespace test
}  // namespace stan

#endif

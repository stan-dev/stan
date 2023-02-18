#ifndef TEST_UNIT_IO_JSON_UTIL_HPP
#define TEST_UNIT_IO_JSON_UTIL_HPP

#include <stan/io/json/json_handler.hpp>
#include <stan/io/json/rapidjson_parser.hpp>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

/**
 * Reports events, 1 per line.
 */
class recording_handler : public stan::json::json_handler {
 public:
  std::stringstream os_;
  recording_handler() : json_handler(), os_() {}
  void start_text() { os_ << "S:text" << std::endl; }
  void end_text() { os_ << "E:text" << std::endl; }
  void start_array() { os_ << "S:arr" << std::endl; }
  void end_array() { os_ << "E:arr" << std::endl; }
  void start_object() { os_ << "S:obj" << std::endl; }
  void end_object() { os_ << "E:obj" << std::endl; }
  void null() { os_ << "NULL:null" << std::endl; }
  void boolean(bool p) { os_ << "BOOL:" << p << std::endl; }
  void string(const std::string &s) {
    os_ << "STR:\"" << s << "\"" << std::endl;
  }
  void key(const std::string &key) {
    os_ << "KEY:\"" << key << "\"" << std::endl;
  }
  void number_double(double x) { os_ << "D(REAL):" << x << std::endl; }
  void number_int(int n) { os_ << "I(INT):" << n << std::endl; }
  void number_unsigned_int(unsigned n) { os_ << "U(INT):" << n << std::endl; }
  void number_int64(int64_t n) { os_ << "I64(INT):" << n << std::endl; }
  void number_unsigned_int64(uint64_t n) {
    os_ << "U64(INT):" << n << std::endl;
  }
};

/**
 * Gets the path separator for the OS.
 *
 * @return '\' for Windows, '/' otherwise.
 */
char get_slash() {
#if defined(WIN32) || defined(_WIN32) \
    || defined(__WIN32) && !defined(__CYGWIN__)
  static char path_separator = '\\';
#else
  static char path_separator = '/';
#endif
  return path_separator;
}

/**
 * Returns the path as a string with the appropriate path separator.
 *
 * @param file_path vector of strings representing path to the file
 * @return the string representation of the path
 */
std::string paths_to_fname(const std::vector<std::string> &path) {
  std::string pathstr;
  if (path.size() > 0) {
    pathstr.append(path[0]);
    for (size_t i = 1; i < path.size(); i++) {
      pathstr.append(1, get_slash());
      pathstr.append(path[i]);
    }
  }
  return pathstr;
}

/**
 * Echo contents of filename to stdout
 *
 * @param filename path to the file
 */
void show_file(const std::string &fileName) {
  std::ifstream ifs(fileName.c_str(),
                    std::ios::in | std::ios::binary | std::ios::ate);
  std::ifstream::pos_type fileSize = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  std::vector<char> bytes(fileSize);
  ifs.read(bytes.data(), fileSize);
  std::string tmp(bytes.data(), fileSize);
  std::cout << tmp << std::endl << std::flush;
}

/**
 * Echo parse of filename to stdout
 *
 * @param filename path to the file
 */
void show_parse(const std::string &fileName) {
  recording_handler handler;
  std::ifstream ifs(fileName.c_str(),
                    std::ios::in | std::ios::binary | std::ios::ate);
  ifs.seekg(0, std::ios::beg);
  stan::json::rapidjson_parse(ifs, handler);
  std::cout << handler.os_.str() << std::endl << std::flush;
}

void test_int_var(stan::json::json_data &jdata, const std::string &name,
                  const std::vector<int> &expected_vals,
                  const std::vector<size_t> &expected_dims) {
  EXPECT_EQ(true, jdata.contains_i(name));
  std::vector<size_t> dims = jdata.dims_i(name);
  EXPECT_EQ(expected_dims.size(), dims.size());
  for (size_t i = 0; i < dims.size(); i++)
    EXPECT_EQ(expected_dims[i], dims[i]);
  std::vector<int> vals = jdata.vals_i(name);
  EXPECT_EQ(expected_vals.size(), vals.size());
  for (size_t i = 0; i < vals.size(); i++)
    EXPECT_EQ(expected_vals[i], vals[i]);
}

void test_empty_int_arr(stan::json::json_data &jdata, const std::string &name,
                        const std::vector<int> &expected_vals) {
  EXPECT_EQ(true, jdata.contains_i(name));
  std::vector<size_t> dims = jdata.dims_i(name);
  EXPECT_EQ(1, dims.size());
  std::vector<int> vals = jdata.vals_i(name);
  EXPECT_EQ(expected_vals.size(), vals.size());
  for (size_t i = 0; i < vals.size(); i++)
    EXPECT_EQ(expected_vals[i], vals[i]);
}

void test_real_var(stan::json::json_data &jdata, const std::string &name,
                   const std::vector<double> &expected_vals,
                   const std::vector<size_t> &expected_dims) {
  EXPECT_EQ(true, jdata.contains_r(name));
  std::vector<size_t> dims = jdata.dims_r(name);
  EXPECT_EQ(expected_dims.size(), dims.size());
  for (size_t i = 0; i < dims.size(); i++)
    EXPECT_EQ(expected_dims[i], dims[i]);
  std::vector<double> vals = jdata.vals_r(name);
  EXPECT_EQ(expected_vals.size(), vals.size());
  for (size_t i = 0; i < vals.size(); i++)
    EXPECT_EQ(expected_vals[i], vals[i]);
}

void test_complex_var(stan::json::json_data &jdata, const std::string &name,
                      const std::vector<std::complex<double>> &expected_vals,
                      const std::vector<size_t> &expected_dims) {
  EXPECT_EQ(true, (jdata.contains_r(name) || jdata.contains_i(name)));
  std::vector<size_t> dims = jdata.dims_r(name);
  dims.pop_back();
  EXPECT_EQ(expected_dims.size(), dims.size());
  for (size_t i = 0; i < dims.size(); i++)
    EXPECT_EQ(expected_dims[i], dims[i]);
  std::vector<std::complex<double>> vals = jdata.vals_c(name);
  EXPECT_EQ(expected_vals.size(), vals.size());
  for (size_t i = 0; i < vals.size(); i++)
    EXPECT_EQ(expected_vals[i], vals[i]);
}

void test_exception(const std::string &input,
                    const std::string &exception_text) {
  try {
    std::stringstream s(input);
    stan::json::json_data jdata(s);
  } catch (const std::exception &e) {
    EXPECT_EQ(e.what(), exception_text);
    return;
  }
  FAIL();  // didn't throw an exception as expected.
}

#endif

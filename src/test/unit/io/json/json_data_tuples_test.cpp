#include <stan/io/json/json_data.hpp>
#include <stan/io/json/json_data_handler.hpp>
#include <stan/io/json/json_error.hpp>
#include <stan/io/json/json_handler.hpp>
#include <stan/io/json/rapidjson_parser.hpp>

#include <test/unit/io/json/util.hpp>

#include <boost/limits.hpp>
#include <boost/math/concepts/real_concept.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>
#include <complex>
#include <iostream>
#include <fstream>


// (int, real) x;
TEST(ioJson, jsonData_tuple_int_real) {
  std::vector<std::string> json_path;
  json_path = {"src", "test", "test-data", "tuple_int_real.json"};
  std::string filename = paths_to_fname(json_path);
  std::ifstream in(filename);
  stan::json::json_data jdata(in);
  std::vector<int> expected_vals_i;
  std::vector<double> expected_vals_r;
  std::vector<size_t> expected_dims;
  expected_vals_i.push_back(1);
  test_int_var(jdata, "x.1", expected_vals_i, expected_dims);
  expected_vals_r.push_back(6.28);
  test_real_var(jdata, "x.2", expected_vals_r, expected_dims);
  expected_vals_r.clear();
  expected_vals_r.push_back(0.0000123);
  test_real_var(jdata, "y", expected_vals_r, expected_dims);
}

// (real, (int, real)) x;
TEST(ioJson, jsonData_tuple_nested) {
  std::vector<std::string> json_path;
  json_path = {"src", "test", "test-data", "tuple_nested.json"};
  std::string filename = paths_to_fname(json_path);
  std::ifstream in(filename);
  stan::json::json_data jdata(in);
  std::vector<int> expected_vals_i;
  std::vector<double> expected_vals_r;
  std::vector<size_t> expected_dims;
  expected_vals_r.push_back(3.214);
  test_real_var(jdata, "x.1", expected_vals_r, expected_dims);
  expected_vals_i.push_back(1);
  test_int_var(jdata, "x.2.1", expected_vals_i, expected_dims);
  expected_vals_r.clear();
  expected_vals_r.push_back(6.28);
  test_real_var(jdata, "x.2.2", expected_vals_r, expected_dims);
  expected_vals_r.clear();
  expected_vals_r.push_back(0.0000123);
  test_real_var(jdata, "y", expected_vals_r, expected_dims);
}

// array[2] (array[3] real, real) x;
TEST(ioJson, jsonData_array_tuple_simple) {
  std::vector<std::string> json_path;
  json_path = {"src", "test", "test-data", "d1_array_tuple_1d_real.json"};
  std::string filename = paths_to_fname(json_path);
  std::ifstream in(filename);
  stan::json::json_data jdata(in);

  std::vector<double> expected_vals_x1 = {11, 21, 12, 22.2, 13, 23};
  std::vector<size_t> expected_dims_x1 = {2, 3};
  test_real_var(jdata, "x.1", expected_vals_x1, expected_dims_x1);

  std::vector<double> expected_vals_x2 = {1, 2.2};
  std::vector<size_t> expected_dims_x2 = {2};
  test_real_var(jdata, "x.2", expected_vals_x2, expected_dims_x2);

  std::vector<double> expected_vals_y = { 3.214};
  std::vector<size_t> expected_dims_y;
  test_real_var(jdata, "y", expected_vals_y, expected_dims_y);
}


// array[3] (array[2] int, (array[2] real, real)) x;
TEST(ioJson, jsonData_array_tuple_1d_tuple_1d_real) {
  std::vector<std::string> json_path;
  json_path = {"src", "test", "test-data", "d1_array_tuple_1d_tuple.json"};
  std::string filename = paths_to_fname(json_path);
  std::ifstream in(filename);
  stan::json::json_data jdata(in);

  std::vector<int> expected_vals_x1 = { 11, 21, 31, 12, 22, 32 };
  std::vector<size_t> expected_dims_x1 = {3, 2};
  test_int_var(jdata, "x.1", expected_vals_x1, expected_dims_x1);

  std::vector<double> expected_vals_x2_1 = {1.1, 3.3, 5.5, 2.2, 4.4, 6.6};
  std::vector<size_t> expected_dims_x2_1  = {3, 2};
  test_real_var(jdata, "x.2.1", expected_vals_x2_1, expected_dims_x2_1);

  std::vector<double> expected_vals_x2_2 = {6.66, 7.77, 8.88};
  std::vector<size_t> expected_dims_x2_2  = {3};
  test_real_var(jdata, "x.2.2", expected_vals_x2_2, expected_dims_x2_2);

  std::vector<double> expected_vals_y = { 3.214};
  std::vector<size_t> expected_dims_y;
  test_real_var(jdata, "y", expected_vals_y, expected_dims_y);
}

// equivalent decls:
// array[2, 2] (array[3] real, real) x;
// (array[2, 2, 3] real, array[2, 2] real) x;
TEST(ioJson, jsonData_array_tuple_array) {
  std::vector<std::string> json_path;
  json_path = {"src", "test", "test-data", "d2_array_tuple_1d_real.json"};
  std::string filename1 = paths_to_fname(json_path);
  std::ifstream in1(filename1);
  stan::json::json_data jdata1(in1);
  json_path = {"src", "test", "test-data", "tuple_array_3d_2d.json"};
  std::string filename2 = paths_to_fname(json_path);
  std::ifstream in2(filename2);
  stan::json::json_data jdata2(in2);

  std::vector<double> expected_vals_x1 = { 11.1, 31, 21, 41, 12, 32, 22.2, 42, 13, 33.3, 23, 43 };
  std::vector<size_t> expected_dims_x1 = { 2, 2, 3 };
  test_real_var(jdata1, "x.1", expected_vals_x1, expected_dims_x1);
  test_real_var(jdata2, "x.1", expected_vals_x1, expected_dims_x1);

  std::vector<double> expected_vals_x2 = { 1, 3.3, 2.2, 4 };
  std::vector<size_t> expected_dims_x2 = { 2, 2 };
  test_real_var(jdata1, "x.2", expected_vals_x2, expected_dims_x2);
  test_real_var(jdata2, "x.2", expected_vals_x2, expected_dims_x2);

  std::vector<double> expected_vals_y = { 3.214};
  std::vector<size_t> expected_dims_y;
  test_real_var(jdata1, "y", expected_vals_y, expected_dims_y);
  test_real_var(jdata2, "y", expected_vals_y, expected_dims_y);
}


// sanity check - non-tuple vars OK
TEST(ioJson, jsonData_no_tuples) {
  std::vector<std::string> json_path;
  json_path = {"src", "test", "test-data", "arrays.json"};
  std::string filename = paths_to_fname(json_path);
  std::ifstream in(filename);
  stan::json::json_data jdata(in);

  std::vector<double> expected_vals_x = { 11.1, 31, 21, 41, 12, 32, 22.2, 42, 13, 33.3, 23, 43 };
  std::vector<size_t> expected_dims_x = { 2, 2, 3 };
  test_real_var(jdata, "x", expected_vals_x, expected_dims_x);

  std::vector<double> expected_vals_y = { 1, 3.3, 2.2, 4 };
  std::vector<size_t> expected_dims_y = { 2, 2 };
  test_real_var(jdata, "y", expected_vals_y, expected_dims_y);

  std::vector<double> expected_vals_z = { 3.214};
  std::vector<size_t> expected_dims_z;
  test_real_var(jdata, "z", expected_vals_z, expected_dims_z);
}

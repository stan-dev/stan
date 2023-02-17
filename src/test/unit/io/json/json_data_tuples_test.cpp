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
  show_file(filename);
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
  show_file(filename);
  std::ifstream in(filename);
  stan::json::json_data jdata(in);
  EXPECT_EQ(true, jdata.contains_i("x.1"));
  EXPECT_EQ(true, jdata.contains_r("x.2.1"));
  EXPECT_EQ(true, jdata.contains_r("x.2.2"));
  EXPECT_EQ(true, jdata.contains_r("y"));
}

// equivalent decls:
// array[2, 2] (array[3] real, real) x;
// (array[2, 2, 3] real, array[2, 2] real) x;
TEST(ioJson, jsonData_array_tuple_array) {
  std::vector<std::string> json_path;
  json_path = {"src", "test", "test-data", "d2_array_tuple_1d_real.json"};
  std::string filename = paths_to_fname(json_path);
  std::cout << filename << std::endl << std::flush;
  show_file(filename);
  std::ifstream in(filename);
  stan::json::json_data jdata1(in);
  EXPECT_EQ(true, jdata1.contains_r("y"));
  std::vector<double> y_vals = jdata1.vals_r("y");
  std::cout << "y: " << y_vals[0] << std::endl;
  EXPECT_EQ(true, jdata1.contains_r("x.1"));
  EXPECT_EQ(true, jdata1.contains_r("x.2"));
  std::vector<double> x1_vals = jdata1.vals_r("x.1");
  std::cout << "x.1: ";
  for (size_t i = 0; i < x1_vals.size(); i++)
    std::cout << x1_vals[i] << ", ";
  std::cout << std::endl;
  std::vector<double> x2_vals = jdata1.vals_r("x.2");
  std::cout << "x.2: ";
  for (size_t i = 0; i < x2_vals.size(); i++)
    std::cout << x2_vals[i] << ", ";
  std::cout << std::endl << std::flush;
}

// sanity check - non-tuple vars OK
TEST(ioJson, jsonData_no_tuples) {
  std::vector<std::string> json_path;
  json_path = {"src", "test", "test-data", "arrays.json"};
  std::string filename = paths_to_fname(json_path);
  show_file(filename);
  std::ifstream in(filename);
  stan::json::json_data jdata(in);
  EXPECT_EQ(true, jdata.contains_r("x"));
  std::vector<double> x_vals = jdata.vals_r("x");
  EXPECT_EQ(true, jdata.contains_r("y"));
  std::vector<double> y_vals = jdata.vals_r("y");
  EXPECT_EQ(true, jdata.contains_r("z"));
  std::vector<double> z_vals = jdata.vals_r("z");
  EXPECT_EQ(1, z_vals.size());
}

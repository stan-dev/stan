#include <stan/io/json/json_data.hpp>
#include <stan/io/json/json_data_handler.hpp>
#include <stan/io/json/json_error.hpp>
#include <stan/io/json/json_handler.hpp>
#include <stan/io/json/rapidjson_parser.hpp>

#include <test/unit/util.hpp>
#include <test/unit/io/json/util.hpp>

#include <gtest/gtest.h>

// tuple(int, real) x - also real y;
TEST(ioJson, jsonData_tuple_int_real) {
  std::vector<std::string> json_path;
  json_path
      = {"src", "test", "unit", "io", "test_json_files", "tuple_int_real.json"};
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

// tuple(real, (int, real)) x;
TEST(ioJson, jsonData_tuple_nested) {
  std::vector<std::string> json_path;
  json_path
      = {"src", "test", "unit", "io", "test_json_files", "tuple_nested.json"};
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

TEST(ioJson, jsonData_array_tuple_arrays) {
  std::vector<std::string> json_path;
  json_path = {"src",
               "test",
               "unit",
               "io",
               "test_json_files",
               "d1_array_tuple_1d_real.json"};
  std::string filename = paths_to_fname(json_path);
  std::ifstream in(filename);
  stan::json::json_data jdata(in);

  std::vector<double> expected_vals_x1 = {11, 21, 12, 22.2, 13, 23};
  std::vector<size_t> expected_dims_x1 = {2, 3};
  test_real_var(jdata, "x.1", expected_vals_x1, expected_dims_x1);

  std::vector<double> expected_vals_x2 = {5, 5.5, 7, 7.7, 6, 6.6, 8, 8.8};
  std::vector<size_t> expected_dims_x2 = {2, 2, 2};
  test_real_var(jdata, "x.2", expected_vals_x2, expected_dims_x2);

  std::vector<double> expected_vals_x3 = {1, 2.2};
  std::vector<size_t> expected_dims_x3 = {2};
  test_real_var(jdata, "x.3", expected_vals_x3, expected_dims_x3);

  std::vector<double> expected_vals_y = {3.214};
  std::vector<size_t> expected_dims_y;
  test_real_var(jdata, "y", expected_vals_y, expected_dims_y);

  std::vector<double> expected_vals_z4 = {0.1, 0.4, 0.2, 0.5, 0.3, 0.6};
  std::vector<size_t> expected_dims_z4 = {2, 3};
  test_real_var(jdata, "z.4", expected_vals_z4, expected_dims_z4);
}

TEST(ioJson, jsonData_array_tuple_1d_tuple_1d_real) {
  std::vector<std::string> json_path;
  json_path = {"src",
               "test",
               "unit",
               "io",
               "test_json_files",
               "d1_array_tuple_1d_tuple.json"};
  std::string filename = paths_to_fname(json_path);
  std::ifstream in(filename);
  stan::json::json_data jdata(in);

  std::vector<int> expected_vals_x1 = {11, 21, 31, 12, 22, 32};
  std::vector<size_t> expected_dims_x1 = {3, 2};
  test_int_var(jdata, "x.1", expected_vals_x1, expected_dims_x1);

  std::vector<double> expected_vals_x2_1 = {1.1, 3.3, 5.5, 2.2, 4.4, 6.6};
  std::vector<size_t> expected_dims_x2_1 = {3, 2};
  test_real_var(jdata, "x.2.1", expected_vals_x2_1, expected_dims_x2_1);

  std::vector<double> expected_vals_x2_2 = {6.66, 7.77, 8.88};
  std::vector<size_t> expected_dims_x2_2 = {3};
  test_real_var(jdata, "x.2.2", expected_vals_x2_2, expected_dims_x2_2);

  std::vector<double> expected_vals_y = {3.214};
  std::vector<size_t> expected_dims_y;
  test_real_var(jdata, "y", expected_vals_y, expected_dims_y);
}

// different var decl and different JSON, but equivalent vars:
// array[2, 2] (array[3] real, real) x;
// (array[2, 2, 3] real, array[2, 2] real) x;
TEST(ioJson, jsonData_array_tuple_array) {
  std::vector<std::string> json_path;
  json_path = {"src",
               "test",
               "unit",
               "io",
               "test_json_files",
               "d2_array_tuple_1d_real.json"};
  std::string filename1 = paths_to_fname(json_path);
  std::ifstream in1(filename1);
  stan::json::json_data jdata1(in1);
  json_path = {
      "src", "test", "unit", "io", "test_json_files", "tuple_array_3d_2d.json"};
  std::string filename2 = paths_to_fname(json_path);
  std::ifstream in2(filename2);
  stan::json::json_data jdata2(in2);

  std::vector<double> expected_vals_x1
      = {11.1, 31, 21, 41, 12, 32, 22.2, 42, 13, 33.3, 23, 43};
  std::vector<size_t> expected_dims_x1 = {2, 2, 3};
  test_real_var(jdata1, "x.1", expected_vals_x1, expected_dims_x1);
  test_real_var(jdata2, "x.1", expected_vals_x1, expected_dims_x1);

  std::vector<double> expected_vals_x2 = {1, 3.3, 2.2, 4};
  std::vector<size_t> expected_dims_x2 = {2, 2};
  test_real_var(jdata1, "x.2", expected_vals_x2, expected_dims_x2);
  test_real_var(jdata2, "x.2", expected_vals_x2, expected_dims_x2);

  std::vector<double> expected_vals_y = {3.214};
  std::vector<size_t> expected_dims_y;
  test_real_var(jdata1, "y", expected_vals_y, expected_dims_y);
  test_real_var(jdata2, "y", expected_vals_y, expected_dims_y);
}

TEST(ioJson, jsonData_array_tuple_multi) {
  std::vector<std::string> json_path;
  json_path = {
      "src", "test", "unit", "io", "test_json_files", "array_tuple_multi.json"};
  std::string filename = paths_to_fname(json_path);
  std::ifstream in(filename);
  stan::json::json_data jdata(in);

  std::vector<int> expected_vals_x11 = {111, 444, 222, 555, 333, 666};
  std::vector<size_t> expected_dims_x11 = {2, 3};
  test_int_var(jdata, "x.1.1", expected_vals_x11, expected_dims_x11);

  std::vector<double> expected_vals_x12
      = {1, 91.1, 3, 93.3, 5, 95.5, 2, 92.2, 4, 94.4, 6, 96.6};
  std::vector<size_t> expected_dims_x12 = {2, 3, 2};
  test_real_var(jdata, "x.1.2", expected_vals_x12, expected_dims_x12);

  std::vector<int> expected_vals_x2 = {37, 47};
  std::vector<size_t> expected_dims_x2 = {2};
  test_int_var(jdata, "x.2", expected_vals_x2, expected_dims_x2);
}

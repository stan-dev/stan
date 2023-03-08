#include <stan/io/json/json_data.hpp>
#include <stan/io/json/json_data_handler.hpp>
#include <stan/io/json/json_error.hpp>
#include <stan/io/json/json_handler.hpp>
#include <stan/io/json/rapidjson_parser.hpp>

#include <test/unit/util.hpp>
#include <test/unit/io/json/util.hpp>

#include <boost/limits.hpp>
#include <boost/math/concepts/real_concept.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

#include <complex>

TEST(ioJson, jsonData_scalar_int) {
  std::string txt = "{ \"foo\" : 1 }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<int> expected_vals;
  expected_vals.push_back(1);
  std::vector<size_t> expected_dims;
  test_int_var(jdata, "foo", expected_vals, expected_dims);
}

TEST(ioJson, jsonData_scalar_real) {
  std::string txt = "{ \"foo\" : 1.1 }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals;
  expected_vals.push_back(1.1);
  std::vector<size_t> expected_dims;
  test_real_var(jdata, "foo", expected_vals, expected_dims);
}

TEST(ioJson, jsonData_scalar_complex) {
  std::string txt = "{ \"foo\" : [1.1, 2.2] }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<std::complex<double>> expected_vals;
  expected_vals.push_back(std::complex<double>(1.1, 2.2));
  std::vector<size_t> expected_dims;
  test_complex_var(jdata, "foo", expected_vals, expected_dims);
}

TEST(ioJson, jsonData_mult_vars) {
  std::string txt = "{ \"foo\" : 1, \"bar\" : 0.1 }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<int> expected_vals_i;
  expected_vals_i.push_back(1);
  std::vector<size_t> expected_dims;
  test_int_var(jdata, "foo", expected_vals_i, expected_dims);
  std::vector<double> expected_vals_r;
  expected_vals_r.push_back(0.1);
  test_real_var(jdata, "bar", expected_vals_r, expected_dims);
}

TEST(ioJson, jsonData_mult_vars2) {
  std::string txt = "{ \"foo\" : \"-Inf\", \"bar\" : 0.1 }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals_r;
  expected_vals_r.push_back(-std::numeric_limits<double>::infinity());
  std::vector<size_t> expected_dims;
  test_real_var(jdata, "foo", expected_vals_r, expected_dims);
  expected_vals_r.clear();
  expected_vals_r.push_back(0.1);
  test_real_var(jdata, "bar", expected_vals_r, expected_dims);
}

TEST(ioJson, jsonData_mult_vars3) {
  std::string txt
      = "{ \"foo\" : \"-Inf\", "
        "                  \"bar\" : 0.1 ,"
        "                  \"baz\" : [ \"-Inf\", 0.1 , 1 ] }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals_r;
  expected_vals_r.push_back(-std::numeric_limits<double>::infinity());
  std::vector<size_t> expected_dims;
  test_real_var(jdata, "foo", expected_vals_r, expected_dims);
  expected_vals_r.clear();
  expected_vals_r.push_back(0.1);
  test_real_var(jdata, "bar", expected_vals_r, expected_dims);
  expected_vals_r.clear();
  expected_vals_r.push_back(-std::numeric_limits<double>::infinity());
  expected_vals_r.push_back(0.1);
  expected_vals_r.push_back(1);
  expected_dims.push_back(3);
  test_real_var(jdata, "baz", expected_vals_r, expected_dims);
}

TEST(ioJson, jsonData_real_array_1D) {
  std::string txt = "{ \"foo\" : [ 1.1, 2.2 ] }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals;
  expected_vals.push_back(1.1);
  expected_vals.push_back(2.2);
  std::vector<size_t> expected_dims;
  expected_dims.push_back(2);
  test_real_var(jdata, "foo", expected_vals, expected_dims);
}

TEST(ioJson, jsonData_complex_array_1D) {
  std::string txt = "{ \"foo\" : [ [1.1, 2.2], [3, 4] ] }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<std::complex<double>> expected_vals;
  expected_vals.push_back(std::complex<double>(1.1, 2.2));
  expected_vals.push_back(std::complex<double>(3, 4));
  std::vector<size_t> expected_dims;
  expected_dims.push_back(2);
  test_complex_var(jdata, "foo", expected_vals, expected_dims);
}

TEST(ioJson, jsonData_array_1D_inf) {
  std::string txt = "{ \"foo\" : [ 1.1, \"Inf\" ] }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals;
  expected_vals.push_back(1.1);
  expected_vals.push_back(std::numeric_limits<double>::infinity());
  std::vector<size_t> expected_dims;
  expected_dims.push_back(2);
  test_real_var(jdata, "foo", expected_vals, expected_dims);
}

TEST(ioJson, jsonData_array_1D_inf2) {
  std::string txt = "{ \"foo\" : [ 1, \"Inf\" ] }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals;
  expected_vals.push_back(1);
  expected_vals.push_back(std::numeric_limits<double>::infinity());
  std::vector<size_t> expected_dims;
  expected_dims.push_back(2);
  test_real_var(jdata, "foo", expected_vals, expected_dims);
}

TEST(ioJson, jsonData_array_1D_neg_inf) {
  std::string txt = "{ \"foo\" : [ 1.1, \"-Inf\" ] }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals;
  expected_vals.push_back(1.1);
  expected_vals.push_back(-std::numeric_limits<double>::infinity());
  std::vector<size_t> expected_dims;
  expected_dims.push_back(2);
  test_real_var(jdata, "foo", expected_vals, expected_dims);
}

TEST(ioJson, jsonData_real_array_2D) {
  std::string txt
      = "{ \"foo\" : [ [ 1.1, 1.2 ], [ 2.1, 2.2 ], [ 3.1, 3.2] ]  }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals;
  expected_vals.push_back(1.1);
  expected_vals.push_back(2.1);
  expected_vals.push_back(3.1);
  expected_vals.push_back(1.2);
  expected_vals.push_back(2.2);
  expected_vals.push_back(3.2);
  std::vector<size_t> expected_dims;
  expected_dims.push_back(3);
  expected_dims.push_back(2);
  test_real_var(jdata, "foo", expected_vals, expected_dims);
}

TEST(ioJson, jsonData_complex_array_2D) {
  std::string txt = "{ \"foo\" : [ [ [1, 2], [3, 4] ], [ [5, 6], [7, 8] ] ] }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<std::complex<double>> expected_vals;
  expected_vals.push_back(std::complex<double>(1, 2));
  expected_vals.push_back(std::complex<double>(5, 6));
  expected_vals.push_back(std::complex<double>(3, 4));
  expected_vals.push_back(std::complex<double>(7, 8));
  std::vector<size_t> expected_dims;
  expected_dims.push_back(2);
  expected_dims.push_back(2);
  test_complex_var(jdata, "foo", expected_vals, expected_dims);
}

TEST(ioJson, jsonData_real_array_3D) {
  std::string txt
      = "{ \"foo\" : [ [ [ 11.1, 11.2, 11.3, 11.4 ], [ 12.1, 12.2, 12.3, 12.4 "
        "], "
        "[ 13.1, 13.2, 13.3, 13.4] ],"
        "                            [ [ 21.1, 21.2, 21.3, 21.4 ], [ 22.1, "
        "22.2, "
        "22.3, 22.4 ], [ 23.1, 23.2, 23.3, 23.4] ] ] }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals;
  expected_vals.push_back(11.1);
  expected_vals.push_back(21.1);
  expected_vals.push_back(12.1);
  expected_vals.push_back(22.1);
  expected_vals.push_back(13.1);
  expected_vals.push_back(23.1);
  expected_vals.push_back(11.2);
  expected_vals.push_back(21.2);
  expected_vals.push_back(12.2);
  expected_vals.push_back(22.2);
  expected_vals.push_back(13.2);
  expected_vals.push_back(23.2);
  expected_vals.push_back(11.3);
  expected_vals.push_back(21.3);
  expected_vals.push_back(12.3);
  expected_vals.push_back(22.3);
  expected_vals.push_back(13.3);
  expected_vals.push_back(23.3);
  expected_vals.push_back(11.4);
  expected_vals.push_back(21.4);
  expected_vals.push_back(12.4);
  expected_vals.push_back(22.4);
  expected_vals.push_back(13.4);
  expected_vals.push_back(23.4);
  std::vector<size_t> expected_dims;
  expected_dims.push_back(2);  // two rows
  expected_dims.push_back(3);  // three cols
  expected_dims.push_back(4);  // four shelves
  test_real_var(jdata, "foo", expected_vals, expected_dims);
}

TEST(ioJson, jsonData_complex_array_3D) {
  std::string txt
      = "{ \"foo\" : [ [ [ [11.1, 11.2], [11.3, 11.4] ], [ [12.1,"
        " 12.2], [12.3, 12.4] ], "
        "[ [13.1, 13.2], [13.3, 13.4]] ],"
        " [ [ [21.1, 21.2], [21.3, 21.4] ], [ [22.1, 22.2], "
        "[22.3, 22.4] ], [ [23.1, 23.2], [23.3, 23.4]] ] ] }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<std::complex<double>> expected_vals;
  expected_vals.push_back(std::complex<double>(11.1, 11.2));
  expected_vals.push_back(std::complex<double>(21.1, 21.2));
  expected_vals.push_back(std::complex<double>(12.1, 12.2));
  expected_vals.push_back(std::complex<double>(22.1, 22.2));
  expected_vals.push_back(std::complex<double>(13.1, 13.2));
  expected_vals.push_back(std::complex<double>(23.1, 23.2));
  expected_vals.push_back(std::complex<double>(11.3, 11.4));
  expected_vals.push_back(std::complex<double>(21.3, 21.4));
  expected_vals.push_back(std::complex<double>(12.3, 12.4));
  expected_vals.push_back(std::complex<double>(22.3, 22.4));
  expected_vals.push_back(std::complex<double>(13.3, 13.4));
  expected_vals.push_back(std::complex<double>(23.3, 23.4));
  std::vector<size_t> expected_dims;
  expected_dims.push_back(2);
  expected_dims.push_back(3);
  expected_dims.push_back(2);
  test_complex_var(jdata, "foo", expected_vals, expected_dims);
}

TEST(ioJson, jsonData_int_array_3D) {
  std::string txt
      = "{ \"foo\" : [ [ [ 111, 112, 113, 114 ], [ 121, 122, 123, "
        "124 ], [ 131, 132, 133, 134] ],"
        "                            [ [ 211, 212, 213, 214 ], [ "
        "221, 222, 223, 224 ], [ 231, 232, 233, 234] ] ] }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<int> expected_vals;
  expected_vals.push_back(111);
  expected_vals.push_back(211);
  expected_vals.push_back(121);
  expected_vals.push_back(221);
  expected_vals.push_back(131);
  expected_vals.push_back(231);
  expected_vals.push_back(112);
  expected_vals.push_back(212);
  expected_vals.push_back(122);
  expected_vals.push_back(222);
  expected_vals.push_back(132);
  expected_vals.push_back(232);
  expected_vals.push_back(113);
  expected_vals.push_back(213);
  expected_vals.push_back(123);
  expected_vals.push_back(223);
  expected_vals.push_back(133);
  expected_vals.push_back(233);
  expected_vals.push_back(114);
  expected_vals.push_back(214);
  expected_vals.push_back(124);
  expected_vals.push_back(224);
  expected_vals.push_back(134);
  expected_vals.push_back(234);
  std::vector<size_t> expected_dims;
  expected_dims.push_back(2);  // two rows
  expected_dims.push_back(3);  // three cols
  expected_dims.push_back(4);  // four shelves
  test_int_var(jdata, "foo", expected_vals, expected_dims);
}

TEST(ioJson, jsonData_empty_1D_array) {
  std::string txt = "{ \"foo\" : [] }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<int> expected_vals_i;
  std::vector<size_t> expected_dims;
  expected_dims.push_back(0);
  test_int_var(jdata, "foo", expected_vals_i, expected_dims);
}

TEST(ioJson, jsonData_empty_2D_array_0_0) {
  std::string txt = "{ \"foo\" : [] }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<int> expected_vals_i;
  std::vector<size_t> expected_dims;
  expected_dims.push_back(0);
  expected_dims.push_back(0);
  test_empty_int_arr(jdata, "foo", expected_vals_i);
  EXPECT_NO_THROW(jdata.validate_dims("test", "foo", "int", expected_dims));
  EXPECT_NO_THROW(
      jdata.validate_dims("test", "foo.2", "double", expected_dims));
}

TEST(ioJson, jsonData_x_3d_y_2d_z_0d) {
  std::vector<std::string> json_path;
  json_path = {"src", "test", "unit", "io", "test_json_files", "arrays.json"};
  std::string filename = paths_to_fname(json_path);
  std::ifstream in(filename);
  stan::json::json_data jdata(in);

  std::vector<double> expected_vals_x
      = {11.1, 31, 21, 41, 12, 32, 22.2, 42, 13, 33.3, 23, 43};
  std::vector<size_t> expected_dims_x = {2, 2, 3};
  test_real_var(jdata, "x", expected_vals_x, expected_dims_x);

  std::vector<double> expected_vals_y = {1, 3.3, 2.2, 4};
  std::vector<size_t> expected_dims_y = {2, 2};
  test_real_var(jdata, "y", expected_vals_y, expected_dims_y);

  std::vector<double> expected_vals_z = {3.214};
  std::vector<size_t> expected_dims_z;
  test_real_var(jdata, "z", expected_vals_z, expected_dims_z);
}

TEST(ioJson, jsonData_parse_empty_obj) {
  std::string txt = "{}";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<std::string> var_names;
  jdata.names_r(var_names);
  EXPECT_EQ(0U, var_names.size());
  jdata.names_i(var_names);
  EXPECT_EQ(0U, var_names.size());

  EXPECT_THROW(
      jdata.validate_dims("testing", "should_not_exist", "double", {3, 2}),
      std::runtime_error);
  EXPECT_NO_THROW(
      jdata.validate_dims("testing", "zero_dims", "double", {3, 0, 2}));
}

// R: strings "NaN", "Inf", "-Inf"
TEST(ioJson, jsonData_NaN_str) {
  std::string txt = "{ \"foo\" : \"NaN\" }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> vals = jdata.vals_r("foo");
  EXPECT_TRUE(boost::math::isnan(vals[0]));
}

TEST(ioJson, jsonData_unsigned_Inf_str) {
  std::string txt = "{ \"foo\" : \"Inf\" }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals_r;
  expected_vals_r.push_back(std::numeric_limits<double>::infinity());
  std::vector<size_t> expected_dims;
  test_real_var(jdata, "foo", expected_vals_r, expected_dims);
}

TEST(ioJson, jsonData_signed_neg_Inf_str) {
  std::string txt = "{ \"foo\" : \"-Inf\" }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals_r;
  expected_vals_r.push_back(-std::numeric_limits<double>::infinity());
  std::vector<size_t> expected_dims;
  test_real_var(jdata, "foo", expected_vals_r, expected_dims);
}

// python/js:  Infinity, -Infinity, NaN
// test both bare and strings
TEST(ioJson, jsonData_NaN_bare) {
  std::string txt = "{ \"foo\" : NaN }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> vals = jdata.vals_r("foo");
  EXPECT_TRUE(boost::math::isnan(vals[0]));
}

TEST(ioJson, jsonData_unsigned_Infinity_bare) {
  std::string txt = "{ \"foo\" : Infinity }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals_r;
  expected_vals_r.push_back(std::numeric_limits<double>::infinity());
  std::vector<size_t> expected_dims;
  test_real_var(jdata, "foo", expected_vals_r, expected_dims);
}

TEST(ioJson, jsonData_pos_Infinity_bare) {
  std::string txt = "{ \"foo\" : Infinity }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals_r;
  expected_vals_r.push_back(std::numeric_limits<double>::infinity());
  std::vector<size_t> expected_dims;
  test_real_var(jdata, "foo", expected_vals_r, expected_dims);
}

TEST(ioJson, jsonData_signed_neg_Infinity_bare) {
  std::string txt = "{ \"foo\" : -Infinity }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals_r;
  expected_vals_r.push_back(-std::numeric_limits<double>::infinity());
  std::vector<size_t> expected_dims;
  test_real_var(jdata, "foo", expected_vals_r, expected_dims);
}

TEST(ioJson, jsonData_unsigned_Infinity_str) {
  std::string txt = "{ \"foo\" : \"Infinity\" }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals_r;
  expected_vals_r.push_back(std::numeric_limits<double>::infinity());
  std::vector<size_t> expected_dims;
  test_real_var(jdata, "foo", expected_vals_r, expected_dims);
}

TEST(ioJson, jsonData_pos_Infinity_str) {
  std::string txt = "{ \"foo\" : \"Infinity\" }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals_r;
  expected_vals_r.push_back(std::numeric_limits<double>::infinity());
  std::vector<size_t> expected_dims;
  test_real_var(jdata, "foo", expected_vals_r, expected_dims);
}

TEST(ioJson, jsonData_signed_neg_Infinity_str) {
  std::string txt = "{ \"foo\" : \"-Infinity\" }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals_r;
  expected_vals_r.push_back(-std::numeric_limits<double>::infinity());
  std::vector<size_t> expected_dims;
  test_real_var(jdata, "foo", expected_vals_r, expected_dims);
}

TEST(ioJson, jsonData_max_int) {
  std::string txt = "{ \"foo\" : [-2147483648, 2147483647] }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<int> expected_vals_i;
  expected_vals_i.push_back(-2147483648);
  expected_vals_i.push_back(2147483647);
  std::vector<size_t> expected_dims;
  expected_dims.push_back(2);
  test_int_var(jdata, "foo", expected_vals_i, expected_dims);
}

TEST(ioJson, jsonData_promote_large_int_to_double) {
  std::string txt = "{ \"foo\" : -2147483649, \"bar\": 2147483648 }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> foo_vals_r;
  foo_vals_r.push_back(-2147483649.0);
  std::vector<double> bar_vals_r;
  bar_vals_r.push_back(2147483648.0);
  std::vector<size_t> expected_dims;
  test_real_var(jdata, "foo", foo_vals_r, expected_dims);
  test_real_var(jdata, "bar", bar_vals_r, expected_dims);
}

TEST(ioJson, jsonData_promote_extra_large_int_to_double) {
  std::string txt = "{ \"foo\" : 4294967295, \"bar\": 9223372036854775807 }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> foo_vals_r;
  foo_vals_r.push_back(4294967295.0);
  std::vector<double> bar_vals_r;
  bar_vals_r.push_back(9223372036854775807.0);
  std::vector<size_t> expected_dims;
  test_real_var(jdata, "foo", foo_vals_r, expected_dims);
  test_real_var(jdata, "bar", bar_vals_r, expected_dims);
}

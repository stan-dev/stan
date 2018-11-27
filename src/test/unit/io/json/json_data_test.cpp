#include <gtest/gtest.h>

#include <stan/io/json/json_data.hpp>
#include <stan/io/json/json_data_handler.hpp>
#include <stan/io/json/json_error.hpp>
#include <stan/io/json/json_handler.hpp>
#include <stan/io/json/json_parser.hpp>

#include <boost/limits.hpp>
#include <boost/math/concepts/real_concept.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

void test_int_var(stan::json::json_data& jdata,
                  const std::string& text,
                  const std::string& name,
                  const std::vector<int>& expected_vals,
                  const std::vector<size_t>& expected_dims) {
  EXPECT_EQ(true,jdata.contains_i(name));
  std::vector<size_t> dims = jdata.dims_i(name);
  EXPECT_EQ(expected_dims.size(),dims.size());
  for (size_t i = 0; i<dims.size(); i++) 
    EXPECT_EQ(expected_dims[i],dims[i]);
  std::vector<int> vals = jdata.vals_i(name);
  EXPECT_EQ(expected_vals.size(),vals.size());
  for (size_t i = 0; i<vals.size(); i++) 
    EXPECT_EQ(expected_vals[i],vals[i]);
}

void test_real_var(stan::json::json_data& jdata,
                  const std::string& text,
                  const std::string& name,
                  const std::vector<double>& expected_vals,
                  const std::vector<size_t>& expected_dims) {
  EXPECT_EQ(true,jdata.contains_r(name));
  std::vector<size_t> dims = jdata.dims_r(name);
  EXPECT_EQ(expected_dims.size(),dims.size());
  for (size_t i = 0; i<dims.size(); i++) 
    EXPECT_EQ(expected_dims[i],dims[i]);
  std::vector<double> vals = jdata.vals_r(name);
  EXPECT_EQ(expected_vals.size(),vals.size());
  for (size_t i = 0; i<vals.size(); i++) 
    EXPECT_EQ(expected_vals[i],vals[i]);
}

void test_exception(const std::string& input,
                    const std::string& exception_text) {
  try {
    std::stringstream s(input);
    stan::json::json_data jdata(s);
  } catch (const std::exception& e) {
    EXPECT_EQ(e.what(), exception_text);
    return;
  }
  FAIL();  // didn't throw an exception as expected.
}




TEST(ioJson,jsonData_scalar_int) {
  std::string txt = "{ \"foo\" : 1 }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<int> expected_vals;
  expected_vals.push_back(1);
  std::vector<size_t> expected_dims;
  test_int_var(jdata,txt,"foo",expected_vals,expected_dims);
}

TEST(ioJson,jsonData_scalar_real) {
  std::string txt = "{ \"foo\" : 1.1 }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals;
  expected_vals.push_back(1.1);
  std::vector<size_t> expected_dims;
  test_real_var(jdata,txt,"foo",expected_vals,expected_dims);
}

TEST(ioJson,jsonData_mult_vars) {
  std::string txt = "{ \"foo\" : 1, \"bar\" : 0.1 }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<int> expected_vals_i;
  expected_vals_i.push_back(1);
  std::vector<size_t> expected_dims;
  test_int_var(jdata,txt,"foo",expected_vals_i,expected_dims);
  std::vector<double> expected_vals_r;
  expected_vals_r.push_back(0.1);
  test_real_var(jdata,txt,"bar",expected_vals_r,expected_dims);
}

TEST(ioJson,jsonData_mult_vars2) {
  std::string txt = "{ \"foo\" : \"-Inf\", \"bar\" : 0.1 }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals_r;
  expected_vals_r.push_back(-std::numeric_limits<double>::infinity());
  std::vector<size_t> expected_dims;
  test_real_var(jdata,txt,"foo",expected_vals_r,expected_dims);
  expected_vals_r.clear();
  expected_vals_r.push_back(0.1);
  test_real_var(jdata,txt,"bar",expected_vals_r,expected_dims);
}


TEST(ioJson,jsonData_mult_vars3) {
  std::string txt = "{ \"foo\" : \"-Inf\", "
    "                  \"bar\" : 0.1 ," 
    "                  \"baz\" : [ \"-Inf\", 0.1 , 1 ] }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals_r;
  expected_vals_r.push_back(-std::numeric_limits<double>::infinity());
  std::vector<size_t> expected_dims;
  test_real_var(jdata,txt,"foo",expected_vals_r,expected_dims);
  expected_vals_r.clear();
  expected_vals_r.push_back(0.1);
  test_real_var(jdata,txt,"bar",expected_vals_r,expected_dims);
  expected_vals_r.clear();
  expected_vals_r.push_back(-std::numeric_limits<double>::infinity());
  expected_vals_r.push_back(0.1);
  expected_vals_r.push_back(1);
  expected_dims.push_back(3);
  test_real_var(jdata,txt,"baz",expected_vals_r,expected_dims);
}


TEST(ioJson,jsonData_real_array_1D) {
  std::string txt = "{ \"foo\" : [ 1.1, 2.2 ] }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals;
  expected_vals.push_back(1.1);
  expected_vals.push_back(2.2);
  std::vector<size_t> expected_dims;
  expected_dims.push_back(2);
  test_real_var(jdata,txt,"foo",expected_vals,expected_dims);
}


TEST(ioJson,jsonData_array_1D_inf) {
  std::string txt = "{ \"foo\" : [ 1.1, \"Inf\" ] }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals;
  expected_vals.push_back(1.1);
  expected_vals.push_back(std::numeric_limits<double>::infinity());
  std::vector<size_t> expected_dims;
  expected_dims.push_back(2);
  test_real_var(jdata,txt,"foo",expected_vals,expected_dims);
}

TEST(ioJson,jsonData_array_1D_inf2) {
  std::string txt = "{ \"foo\" : [ 1, \"Inf\" ] }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals;
  expected_vals.push_back(1);
  expected_vals.push_back(std::numeric_limits<double>::infinity());
  std::vector<size_t> expected_dims;
  expected_dims.push_back(2);
  test_real_var(jdata,txt,"foo",expected_vals,expected_dims);
}

TEST(ioJson,jsonData_array_1D_neg_inf) {
  std::string txt = "{ \"foo\" : [ 1.1, \"-Inf\" ] }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals;
  expected_vals.push_back(1.1);
  expected_vals.push_back(-std::numeric_limits<double>::infinity());
  std::vector<size_t> expected_dims;
  expected_dims.push_back(2);
  test_real_var(jdata,txt,"foo",expected_vals,expected_dims);
}

TEST(ioJson,jsonData_real_array_2D) {
  std::string txt = "{ \"foo\" : [ [ 1.1, 1.2 ], [ 2.1, 2.2 ], [ 3.1, 3.2] ]  }";
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
  test_real_var(jdata,txt,"foo",expected_vals,expected_dims);
}


TEST(ioJson,jsonData_real_array_3D) {
  std::string txt = "{ \"foo\" : [ [ [ 11.1, 11.2, 11.3, 11.4 ], [ 12.1, 12.2, 12.3, 12.4 ], [ 13.1, 13.2, 13.3, 13.4] ],"
    "                            [ [ 21.1, 21.2, 21.3, 21.4 ], [ 22.1, 22.2, 22.3, 22.4 ], [ 23.1, 23.2, 23.3, 23.4] ] ] }";
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
  test_real_var(jdata,txt,"foo",expected_vals,expected_dims);
}

TEST(ioJson,jsonData_int_array_3D) {
  std::string txt = "{ \"foo\" : [ [ [ 111, 112, 113, 114 ], [ 121, 122, 123, 124 ], [ 131, 132, 133, 134] ],"
    "                            [ [ 211, 212, 213, 214 ], [ 221, 222, 223, 224 ], [ 231, 232, 233, 234] ] ] }";
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
  test_int_var(jdata,txt,"foo",expected_vals,expected_dims);
}

TEST(ioJson,jsonData_array_err1) {
  std::string txt = "{ \"foo\" : [ [ [ 11.1, 11.2, 11.3, 11.4 ], [ 12.1, 12.2, 12.3, 12.4 ], [ 13.1, 13.2, 13.3, 13.4] ],"
    "                            [ [ 21.1, 21.2, 21.3, 21.4 ], [ 666, 22.3, 22.4 ], [ 23.1, 23.2, 23.3, 23.4] ] ] }";
  test_exception(txt,"variable: foo, error: non-rectangular array");
}


TEST(ioJson,jsonData_array_err2) {
  std::string txt = "{ \"foo\" : [ [ [ 11.1, 11.2, 11.3, 11.4 ], [ 12.1, 12.2, 12.3, 12.4 ] ],"
    "                            [ [ 21.1, 21.2, 21.3, 21.4 ], [ 666, 22.3, 22.4 ], [ 23.1, 23.2, 23.3, 23.4] ] ] }";
  test_exception(txt,"variable: foo, error: non-rectangular array");
}

TEST(ioJson,jsonData_array_err3) {
  std::string txt = "{ \"foo\" : [] }";
  test_exception(txt,"variable: foo, error: empty array not allowed");
}

TEST(ioJson,jsonData_array_err4) {
  std::string txt = "{ \"foo\" : [[[]]] }";
  test_exception(txt,"variable: foo, error: empty array not allowed");
}

TEST(ioJson,jsonData_array_err5) {
  std::string txt = "{ \"foo\" : [1, 2, 3, 4, [5], 6, 7] }";
  test_exception(txt,"variable: foo, error: non-scalar array value");
}

TEST(ioJson,jsonData_array_err6) {
  std::string txt = "{ \"foo\" : [[1], 2, 3, 4, 5, 6, 7] }";
  test_exception(txt,"variable: foo, error: non-rectangular array");
}

TEST(ioJson,jsonData_array_err7) {
  std::string txt = "{  \"foo\" : [1, 2, 3, 4, 5, 6, [7]] }";
  test_exception(txt,"variable: foo, error: non-scalar array value");
}


TEST(ioJson,jsonData_array_err8) {
  std::string txt = "{ \"baz\" : [[1.0,2.0,3.0],[4.0,5.0,6]],  \"foo\" : [1, 2, 3, 4, [5], 6, 7] }";
  test_exception(txt,"variable: foo, error: non-scalar array value");
}

TEST(ioJson,jsonData_array_err9) {
  std::string txt = "{ \"baz\":[[1,2],[3,4.0]],  \"foo\" : [[1], 2, 3, 4, 5, 6, 7] }";
  test_exception(txt,"variable: foo, error: non-rectangular array");
}

TEST(ioJson,jsonData_array_err10) {
  std::string txt = "{  \"baz\":[1,2,\"-Inf\"], \"foo\" : [1, 2, 3, 4, 5, 6, [7]] }";
  test_exception(txt,"variable: foo, error: non-scalar array value");
}

TEST(ioJson,jsonData_array_err11) {
  std::string txt = "{\"a\":1,  \"baz\":[1,2,\"-Inf\"], \"b\":2.0, "
    "\"foo\" : [1, 2, 3, 4, 5, 6, [7]] }";
  test_exception(txt,"variable: foo, error: non-scalar array value");
}


TEST(ioJson,jsonData_obj_err1) {
  std::string txt = "{ \"foo\" : { \"bar\" : 1 } }";
  test_exception(txt,"variable: foo, error: nested objects not allowed");
}


TEST(ioJson,jsonData_mult_vars_err1) {
  std::string txt = "{ \"foo\" : 1, \"foo\" : 0.1 }";
  test_exception(txt,"attempt to redefine variable: foo");
}

TEST(ioJson,jsonData_mult_vars_er2) {
  std::string txt = "{ \"foo\" : 1.1, \"foo\" : 0.1 }";
  test_exception(txt,"attempt to redefine variable: foo");
}

TEST(ioJson,jsonData_mult_vars_er3) {
  std::string txt = "{ \"foo\" : [ 1.1, 1 ], \"foo\" : 0.1 }";
  test_exception(txt,"attempt to redefine variable: foo");
}

TEST(ioJson,jsonData_null_value_err) {
  std::string txt = "{ \"foo\" : [ 1.1, 1, null ] }";
  test_exception(txt,"variable: foo, error: null values not allowed");
}

TEST(ioJson,jsonData_bool_value_err1) {
  std::string txt = "{ \"foo\" : [ 1.1, 1, true ] }";
  test_exception(txt,"variable: foo, error: boolean values not allowed");
}

TEST(ioJson,jsonData_bool_value_err2) {
  std::string txt = "{ \"foo\" : [ 1.1, 1, false ] }";
  test_exception(txt,"variable: foo, error: boolean values not allowed");
}


TEST(ioJson,jsonData_string_value_err) {
  std::string txt = "{ \"foo\" : [ 1.1, 1, \"abc\" ] }";
  test_exception(txt,"variable: foo, error: string values not allowed");
}

TEST(ioJson,jsonData_not_an_obj) {
  std::string txt = "[ 1 ]";
  test_exception(txt,"expecting JSON object, found array");
}

// don't allow array of objects 
TEST(ioJson,jsonData_err_array_of_obj) {
  std::string txt = "[ { \"foo\": 1}, { \"bar\": 1 } ]";
  test_exception(txt,"expecting JSON object, found array");
}


TEST(ioJson,jsonData_parse_empty_obj) {
  std::string txt = "{}";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<std::string> var_names;
  jdata.names_r(var_names);
  EXPECT_EQ(0U,var_names.size());
  jdata.names_i(var_names);
  EXPECT_EQ(0U,var_names.size());
}


// parser only reads one top-level object
TEST(ioJson,jsonData_parse_mult_objects) {
  std::string txt = "{ \"foo\": 1}{ \"bar\": 1 }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<std::string> var_names;
  jdata.names_i(var_names);
  EXPECT_EQ(1U,var_names.size());
  EXPECT_EQ("foo",var_names[0]);
}

// R: strings "NaN", "Inf", "-Inf"
TEST(ioJson,jsonData_NaN_str) {
  std::string txt = "{ \"foo\" : \"NaN\" }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> vals = jdata.vals_r("foo");
  EXPECT_TRUE(boost::math::isnan(vals[0]));
}

TEST(ioJson,jsonData_unsigned_Inf_str) {
  std::string txt = "{ \"foo\" : \"Inf\" }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals_r;
  expected_vals_r.push_back(std::numeric_limits<double>::infinity());
  std::vector<size_t> expected_dims;
  test_real_var(jdata,txt,"foo",expected_vals_r,expected_dims);
}

TEST(ioJson,jsonData_signed_neg_Inf_str) {
  std::string txt = "{ \"foo\" : \"-Inf\" }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals_r;
  expected_vals_r.push_back(-std::numeric_limits<double>::infinity());
  std::vector<size_t> expected_dims;
  test_real_var(jdata,txt,"foo",expected_vals_r,expected_dims);
}

// python/js:  Infinity, -Infinity, NaN
// test both bare and strings
TEST(ioJson,jsonData_NaN_bare) {
  std::string txt = "{ \"foo\" : NaN }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> vals = jdata.vals_r("foo");
  EXPECT_TRUE(boost::math::isnan(vals[0]));
}

TEST(ioJson,jsonData_unsigned_Infinity_bare) {
  std::string txt = "{ \"foo\" : Infinity }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals_r;
  expected_vals_r.push_back(std::numeric_limits<double>::infinity());
  std::vector<size_t> expected_dims;
  test_real_var(jdata,txt,"foo",expected_vals_r,expected_dims);
}

TEST(ioJson,jsonData_pos_Infinity_bare) {
  std::string txt = "{ \"foo\" : Infinity }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals_r;
  expected_vals_r.push_back(std::numeric_limits<double>::infinity());
  std::vector<size_t> expected_dims;
  test_real_var(jdata,txt,"foo",expected_vals_r,expected_dims);
}

TEST(ioJson,jsonData_signed_neg_Infinity_bare) {
  std::string txt = "{ \"foo\" : -Infinity }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals_r;
  expected_vals_r.push_back(-std::numeric_limits<double>::infinity());
  std::vector<size_t> expected_dims;
  test_real_var(jdata,txt,"foo",expected_vals_r,expected_dims);
}

TEST(ioJson,jsonData_unsigned_Infinity_str) {
  std::string txt = "{ \"foo\" : \"Infinity\" }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals_r;
  expected_vals_r.push_back(std::numeric_limits<double>::infinity());
  std::vector<size_t> expected_dims;
  test_real_var(jdata,txt,"foo",expected_vals_r,expected_dims);
}

TEST(ioJson,jsonData_pos_Infinity_str) {
  std::string txt = "{ \"foo\" : \"Infinity\" }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals_r;
  expected_vals_r.push_back(std::numeric_limits<double>::infinity());
  std::vector<size_t> expected_dims;
  test_real_var(jdata,txt,"foo",expected_vals_r,expected_dims);
}

TEST(ioJson,jsonData_signed_neg_Infinity_str) {
  std::string txt = "{ \"foo\" : \"-Infinity\" }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals_r;
  expected_vals_r.push_back(-std::numeric_limits<double>::infinity());
  std::vector<size_t> expected_dims;
  test_real_var(jdata,txt,"foo",expected_vals_r,expected_dims);
}

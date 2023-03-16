#include <stan/io/json/json_data.hpp>
#include <stan/io/json/json_data_handler.hpp>
#include <stan/io/json/json_error.hpp>
#include <stan/io/json/json_handler.hpp>
#include <stan/io/json/rapidjson_parser.hpp>

#include <test/unit/util.hpp>
#include <test/unit/io/json/util.hpp>

#include <gtest/gtest.h>
TEST(errJson, jsonData_array_err1) {
  std::string txt
      = "{ \"foo\" : [ [ [ 11.1, 11.2, 11.3, 11.4 ], [ 12.1, 12.2, "
        "12.3, 12.4 ], [ 13.1, 13.2, 13.3, 13.4] ],"
        "                            [ [ 21.1, 21.2, 21.3, 21.4 ], "
        "[ 666, 22.3, 22.4 ], [ 23.1, 23.2, 23.3, 23.4] ] ] }";
  test_exception(txt, "Variable: foo, error: non-rectangular array.");
}

TEST(errJson, jsonData_array_err2) {
  std::string txt
      = "{ \"foo\" : [ [ [ 11.1, 11.2, 11.3, 11.4 ], [ 12.1, 12.2, "
        "12.3, 12.4 ] ],"
        "                            [ [ 21.1, 21.2, 21.3, 21.4 ], "
        "[ 666, 22.3, 22.4 ], [ 23.1, 23.2, 23.3, 23.4] ] ] }";
  test_exception(txt, "Variable: foo, error: non-rectangular array.");
}

TEST(errJson, jsonData_array_err3) {
  std::string txt = "{ \"foo\" : [1, 2, 3, 4, [5], 6, 7] }";
  test_exception(txt, "Variable: foo, error: ill-formed array.");
}

TEST(errJson, jsonData_array_err4) {
  std::string txt = "{ \"foo\" : [[1], 2, 3, 4, 5, 6, 7] }";
  test_exception(txt, "Variable: foo, error: ill-formed array.");
}

TEST(errJson, jsonData_array_err5) {
  std::string txt = "{  \"foo\" : [1, 2, 3, 4, 5, 6, [7]] }";
  test_exception(txt, "Variable: foo, error: ill-formed array.");
}

TEST(errJson, jsonData_array_err6) {
  std::string txt
      = "{ \"baz\" : [[1.0,2.0,3.0],[4.0,5.0,6]],  \"foo\" : [1, "
        "2, 3, 4, [5], 6, 7] }";
  test_exception(txt, "Variable: foo, error: ill-formed array.");
}

TEST(errJson, jsonData_array_err7) {
  std::string txt
      = "{ \"baz\":[[1,2],[3,4.0]],  \"foo\" : [[1], 2, 3, 4, 5, 6, 7] }";
  test_exception(txt, "Variable: foo, error: ill-formed array.");
}

TEST(errJson, jsonData_array_err8) {
  std::string txt
      = "{  \"baz\":[1,2,\"-Inf\"], \"foo\" : [1, 2, 3, 4, 5, 6, [7]] }";
  test_exception(txt, "Variable: foo, error: ill-formed array.");
}

TEST(errJson, jsonData_array_err9) {
  std::string txt
      = "{\"a\":1,  \"baz\":[1,2,\"-Inf\"], \"b\":2.0, "
        "\"foo\" : [1, 2, 3, 4, 5, 6, [7]] }";
  test_exception(txt, "Variable: foo, error: ill-formed array.");
}

TEST(errJson, jsonData_mult_vars_err1) {
  std::string txt = "{ \"foo\" : 1, \"foo\" : 0.1 }";
  test_exception(txt, "Attempt to redefine variable: foo.");
}

TEST(errJson, jsonData_mult_vars_err2) {
  std::string txt = "{ \"foo\" : 1.1, \"foo\" : 0.1 }";
  test_exception(txt, "Attempt to redefine variable: foo.");
}

TEST(errJson, jsonData_mult_vars_err3) {
  std::string txt = "{ \"foo\" : [ 1.1, 1 ], \"foo\" : 0.1 }";
  test_exception(txt, "Attempt to redefine variable: foo.");
}

TEST(errJson, jsonData_null_value_err) {
  std::string txt = "{ \"foo\" : [ 1.1, 1, null ] }";
  test_exception(txt, "Variable: foo, error: null values not allowed.");
}

TEST(errJson, jsonData_bool_value_err1) {
  std::string txt = "{ \"foo\" : [ 1.1, 1, true ] }";
  test_exception(txt, "Variable: foo, error: boolean values not allowed.");
}

TEST(errJson, jsonData_bool_value_err2) {
  std::string txt = "{ \"foo\" : [ 1.1, 1, false ] }";
  test_exception(txt, "Variable: foo, error: boolean values not allowed.");
}

TEST(errJson, jsonData_string_value_err) {
  std::string txt = "{ \"foo\" : [ 1.1, 1, \"abc\" ] }";
  test_exception(txt, "Variable: foo, error: string values not allowed.");
}

TEST(errJson, jsonData_not_an_obj) {
  std::string txt = "[ 1 ]";
  test_exception(txt, "Expecting JSON object, found array.");
}

TEST(errJson, jsonData_err_array_of_obj) {
  std::string txt = "[ { \"foo\": 1}, { \"bar\": 1 } ]";
  test_exception(txt, "Expecting JSON object, found array.");
}

TEST(errJson, jsonData_parse_mult_objects_err) {
  std::string txt = "{ \"foo\": 1}{ \"bar\": 1 }";
  test_exception(txt,
                 "Error in JSON parsing \nat offset 11: \nThe document "
                 "root must not be followed by other values.\n");
}

TEST(errJson, jsonData_empty_2D_array_1_0) {
  std::string txt = "{ \"foo\" : [] }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<int> expected_vals_i;
  std::vector<size_t> expected_dims;
  expected_dims.push_back(1);
  expected_dims.push_back(0);
  test_empty_int_arr(jdata, "foo", expected_vals_i);
  try {
    jdata.validate_dims("test", "foo", "int", expected_dims);
  } catch (const std::exception &e) {
    FAIL();
  }
}

TEST(errJson, jsonData_empty_3D_array_0_0_0) {
  std::string txt = "{ \"foo\" : [] }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<int> expected_vals_i;
  std::vector<size_t> expected_dims;
  expected_dims.push_back(0);
  expected_dims.push_back(0);
  expected_dims.push_back(0);
  test_empty_int_arr(jdata, "foo", expected_vals_i);
  try {
    jdata.validate_dims("test", "foo", "int", expected_dims);
  } catch (const std::exception &e) {
    FAIL();
  }
}

TEST(errJson, jsonData_empty_3D_array_2_1_0) {
  std::string txt = "{ \"foo\" : [] }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<int> expected_vals_i;
  std::vector<size_t> expected_dims;
  expected_dims.push_back(2);
  expected_dims.push_back(1);
  expected_dims.push_back(0);
  test_empty_int_arr(jdata, "foo", expected_vals_i);
  try {
    jdata.validate_dims("test", "foo", "int", expected_dims);
  } catch (const std::exception &e) {
    FAIL();
  }
}

TEST(errJson, jsonData_redefine_var_1) {
  std::vector<std::string> json_path;
  json_path = {
      "src", "test", "unit", "io", "test_json_files", "redefine_vars_1.json"};
  std::string filename = paths_to_fname(json_path);
  std::string txt = file2str(filename);
  test_exception(txt, "Attempt to redefine variable: x.");
}

TEST(errJson, jsonData_redefine_var_2) {
  std::vector<std::string> json_path;
  json_path = {
      "src", "test", "unit", "io", "test_json_files", "redefine_vars_2.json"};
  std::string filename = paths_to_fname(json_path);
  std::string txt = file2str(filename);
  test_exception(txt, "Attempt to redefine variable: x.");
}

TEST(errJson, jsonData_redefine_var_3) {
  std::vector<std::string> json_path;
  json_path = {
      "src", "test", "unit", "io", "test_json_files", "redefine_vars_3.json"};
  std::string filename = paths_to_fname(json_path);
  std::string txt = file2str(filename);
  test_exception(txt, "Attempt to redefine variable: x.");
}

TEST(errJson, jsonData_redefine_var_4) {
  std::vector<std::string> json_path;
  json_path = {
      "src", "test", "unit", "io", "test_json_files", "redefine_vars_4.json"};
  std::string filename = paths_to_fname(json_path);
  std::string txt = file2str(filename);
  test_exception(txt, "Attempt to redefine variable: x.");
}

TEST(errJson, jsonData_redefine_var_5) {
  std::vector<std::string> json_path;
  json_path = {
      "src", "test", "unit", "io", "test_json_files", "redefine_vars_5.json"};
  std::string filename = paths_to_fname(json_path);
  std::string txt = file2str(filename);
  test_exception(txt, "Attempt to redefine variable: x.");
}

TEST(errJson, jsonData_redefine_var_6) {
  std::vector<std::string> json_path;
  json_path = {
      "src", "test", "unit", "io", "test_json_files", "redefine_vars_6.json"};
  std::string filename = paths_to_fname(json_path);
  std::string txt = file2str(filename);
  test_exception(txt, "Attempt to redefine variable: x.");
}

TEST(errJson, jsonData_inconsistent_array_tuples_1) {
  std::vector<std::string> json_path;
  json_path = {"src",
               "test",
               "unit",
               "io",
               "test_json_files",
               "inconsistent_array_tuples_1.json"};
  std::string filename = paths_to_fname(json_path);
  std::string txt = file2str(filename);
  test_exception(txt, "size mismatch between tuple elements.");
}

TEST(errJson, jsonData_inconsistent_array_tuples_2) {
  std::vector<std::string> json_path;
  json_path = {"src",
               "test",
               "unit",
               "io",
               "test_json_files",
               "inconsistent_array_tuples_2.json"};
  std::string filename = paths_to_fname(json_path);
  std::string txt = file2str(filename);
  test_exception(txt, "size mismatch between tuple elements.");
}

TEST(errJson, jsonData_inconsistent_array_tuples_3) {
  std::vector<std::string> json_path;
  json_path = {"src",
               "test",
               "unit",
               "io",
               "test_json_files",
               "inconsistent_array_tuples_3.json"};
  std::string filename = paths_to_fname(json_path);
  std::string txt = file2str(filename);
  test_exception(txt, "size mismatch between tuple elements.");
}

TEST(errJson, jsonData_inconsistent_array_tuples_4) {
  std::vector<std::string> json_path;
  json_path = {"src",
               "test",
               "unit",
               "io",
               "test_json_files",
               "inconsistent_array_tuples_4.json"};
  std::string filename = paths_to_fname(json_path);
  std::string txt = file2str(filename);
  test_exception(txt, "size mismatch between tuple elements.");
}

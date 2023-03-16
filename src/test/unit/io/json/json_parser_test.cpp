#include <stan/io/json/json_data.hpp>
#include <stan/io/json/json_data_handler.hpp>
#include <stan/io/json/json_error.hpp>
#include <stan/io/json/json_handler.hpp>
#include <stan/io/json/rapidjson_parser.hpp>

#include <test/unit/io/json/util.hpp>
#include <gtest/gtest.h>

void test_parser(const std::string &input, const std::string &expected_output) {
  recording_handler handler;
  std::stringstream s(input);
  stan::json::rapidjson_parse(s, handler);
  EXPECT_EQ(expected_output, handler.os_.str());
}

TEST(ioJson, jsonParserA0) {
  test_parser("[0]",
              "S:text"
              "S:arr"
              "U(INT):0"
              "E:arr"
              "E:text");
}

TEST(ioJson, jsonParserA1) {
  test_parser("[5]",
              "S:text"
              "S:arr"
              "U(INT):5"
              "E:arr"
              "E:text");
}

TEST(ioJson, jsonParserA2) {
  test_parser("[5,10]",
              "S:text"
              "S:arr"
              "U(INT):5"
              "U(INT):10"
              "E:arr"
              "E:text");
}

TEST(ioJson, jsonParserA3) {
  test_parser("[]",
              "S:text"
              "S:arr"
              "E:arr"
              "E:text");
}

TEST(ioJson, jsonParserA4) {
  std::stringstream textReport;
  textReport << "S:text"
             << "S:arr"
             << "D(REAL):" << 0.100019 << "E:arr"
             << "E:text";
  test_parser("[ 0.100019 ]", textReport.str());
}

TEST(ioJson, jsonParserA5) {
  std::stringstream textReport;
  textReport << "S:text"
             << "S:arr"
             << "D(REAL):" << 0.10001900 << "E:arr"
             << "E:text";
  test_parser("[ 0.10001900 ]", textReport.str());
}

TEST(ioJson, jsonParserA6) {
  std::stringstream textReport;
  textReport << "S:text"
             << "S:arr"
             << "D(REAL):" << -0.10001900 << "E:arr"
             << "E:text";
  test_parser("[ -0.10001900 ]", textReport.str());
}

TEST(ioJson, jsonParserA7) {
  test_parser("[ -1, -2, \"-Inf\"]",
              "S:text"
              "S:arr"
              "I(INT):-1"
              "I(INT):-2"
              "STR:\"-Inf\""
              "E:arr"
              "E:text");
}

TEST(ioJson, jsonParserA8) {
  test_parser("[ [ -1, -2 ],[ 1, 2 ] ]",
              "S:text"
              "S:arr"
              "S:arr"
              "I(INT):-1"
              "I(INT):-2"
              "E:arr"
              "S:arr"
              "U(INT):1"
              "U(INT):2"
              "E:arr"
              "E:arr"
              "E:text");
}

TEST(ioJson, jsonParserA9) {
  test_parser("[ [ [1, 2 ],[ 3, 4 ] ],[ [5, 6 ],[ 7, 8 ] ] ]",
              "S:text"
              "S:arr"
              "S:arr"
              "S:arr"
              "U(INT):1"
              "U(INT):2"
              "E:arr"
              "S:arr"
              "U(INT):3"
              "U(INT):4"
              "E:arr"
              "E:arr"
              "S:arr"
              "S:arr"
              "U(INT):5"
              "U(INT):6"
              "E:arr"
              "S:arr"
              "U(INT):7"
              "U(INT):8"
              "E:arr"
              "E:arr"
              "E:arr"
              "E:text");
}

TEST(ioJson, jsonParserA10) {
  std::stringstream textReport;
  textReport << "S:text"
             << "S:arr"
             << "I(INT):-2147483648"
             << "U(INT):2147483647"
             << "I64(INT):-2147483649"
             << "U(INT):2147483648"
             << "U(INT):4294967295"
             << "U64(INT):4294967296"
             << "I64(INT):-9223372036854775808"
             << "U64(INT):9223372036854775807"
             << "U64(INT):18446744073709551615"
             << "D(REAL):" << -9223372036854775809.0
             << "U64(INT):9223372036854775808"
             << "D(REAL):" << 18446744073709551616.0 << "E:arr"
             << "E:text";

  test_parser(
      "[ -2147483648, 2147483647, -2147483649, 2147483648, "
      "4294967295, 4294967296, "
      "-9223372036854775808, 9223372036854775807, 18446744073709551615, "
      "-9223372036854775809, 9223372036854775808, 18446744073709551616 ]",
      textReport.str());
}

TEST(ioJson, jsonParserO1) {
  test_parser("{  \"foo\" : 1  }   ",
              "S:text"
              "S:obj"
              "KEY:\"foo\""
              "U(INT):1"
              "E:obj"
              "E:text");
}

TEST(ioJson, jsonParserO11) {
  test_parser("{  \"foo\" : { \"bar\": 1 } }   ",
              "S:text"
              "S:obj"
              "KEY:\"foo\""
              "S:obj"
              "KEY:\"bar\""
              "U(INT):1"
              "E:obj"
              "E:obj"
              "E:text");
}

TEST(ioJson, jsonParserO12) {
  test_parser("{  \"foo\" : { \"bar\": 1, \"baz\": 2 } }   ",
              "S:text"
              "S:obj"
              "KEY:\"foo\""
              "S:obj"
              "KEY:\"bar\""
              "U(INT):1"
              "KEY:\"baz\""
              "U(INT):2"
              "E:obj"
              "E:obj"
              "E:text");
}

TEST(ioJson, jsonParserO13) {
  test_parser("{  \"foo\" : { \"bar\": { \"baz\": [ 1, 2]  } } }  ",
              "S:text"
              "S:obj"
              "KEY:\"foo\""
              "S:obj"
              "KEY:\"bar\""
              "S:obj"
              "KEY:\"baz\""
              "S:arr"
              "U(INT):1"
              "U(INT):2"
              "E:arr"
              "E:obj"
              "E:obj"
              "E:obj"
              "E:text");
}

TEST(ioJson, jsonParserO14) {
  test_parser(
      "{  \"foo\" : [ { \"bar\": { \"baz\": [ 1, 2]  } }, -3, -4.44 ] }  ",
      "S:text"
      "S:obj"
      "KEY:\"foo\""
      "S:arr"
      "S:obj"
      "KEY:\"bar\""
      "S:obj"
      "KEY:\"baz\""
      "S:arr"
      "U(INT):1"
      "U(INT):2"
      "E:arr"
      "E:obj"
      "E:obj"
      "I(INT):-3"
      "D(REAL):-4.44"
      "E:arr"
      "E:obj"
      "E:text");
}

TEST(ioJson, jsonParserO2) {
  test_parser("{  \"foo\"  : 1 ,  \n   \"bar\" : 2 }",
              "S:text"
              "S:obj"
              "KEY:\"foo\""
              "U(INT):1"
              "KEY:\"bar\""
              "U(INT):2"
              "E:obj"
              "E:text");
}

TEST(ioJson, jsonParserO3) {
  test_parser("{\"foo\":1,\"bar\":2,\"baz\":[2]}",
              "S:text"
              "S:obj"
              "KEY:\"foo\""
              "U(INT):1"
              "KEY:\"bar\""
              "U(INT):2"
              "KEY:\"baz\""
              "S:arr"
              "U(INT):2"
              "E:arr"
              "E:obj"
              "E:text");
}

TEST(ioJson, jsonParserO4) {
  std::stringstream textReport;
  textReport << "S:text"
             << "S:obj"
             << "KEY:\"foo\""
             << "U(INT):1"
             << "KEY:\"baz\""
             << "S:arr"
             << "U(INT):2"
             << "U(INT):3"
             << "D(REAL):" << 4.01001 << "E:arr"
             << "E:obj"
             << "E:text";
  test_parser("{ \"foo\" : 1, \n \"baz\" : [ 2, 3, 4.01001 ] }",
              textReport.str());
}

TEST(ioJson, jsonParserO5) {
  std::stringstream textReport;
  textReport << "S:text"
             << "S:obj"
             << "KEY:\"foo\""
             << "D(REAL):" << -0.10001900 << "E:obj"
             << "E:text";
  test_parser("{ \"foo\": -0.10001900 }", textReport.str());
}

TEST(ioJson, jsonParserO6) {
  std::stringstream textReport;
  textReport << "S:text"
             << "S:obj"
             << "KEY:\"foo\""
             << "D(REAL):" << -1.0100e09 << "E:obj"
             << "E:text";
  test_parser("{ \"foo\": -1.0100e09 }", textReport.str());
}

TEST(ioJson, jsonParserO7) {
  std::stringstream textReport;
  textReport << "S:text"
             << "S:obj"
             << "KEY:\"foo\""
             << "D(REAL):" << -1.0100e09 << "E:obj"
             << "E:text";
  test_parser("{ \"foo\": -1.0100e09 }", textReport.str());
}

TEST(ioJson, jsonParserStr0) {
  test_parser("[ \"foo\" ]",
              "S:text"
              "S:arr"
              "STR:\"foo\""
              "E:arr"
              "E:text");
}

TEST(ioJson, jsonParserStr1) {
  test_parser("[ \"\\nfoo\" ]",
              "S:text"
              "S:arr"
              "STR:\"\nfoo\""
              "E:arr"
              "E:text");
}

TEST(ioJson, jsonParserStr2) {
  test_parser("[ \"\\nfoo\\bbar\" ]",
              "S:text"
              "S:arr"
              "STR:\"\nfoo\bbar\""
              "E:arr"
              "E:text");
}

TEST(ioJson, jsonParserStr3) {
  test_parser("[ \"\\bfoo\\nbar\" ]",
              "S:text"
              "S:arr"
              "STR:\"\bfoo\nbar\""
              "E:arr"
              "E:text");
}

TEST(ioJson, jsonParserStr4) {
  test_parser("[ \"\\\"foo\\/bar\" ]",
              "S:text"
              "S:arr"
              "STR:\"\"foo/bar\""
              "E:arr"
              "E:text");
}

TEST(ioJson, jsonParserStr5) {
  test_parser("[ \"\\\"foo\\\\bar\" ]",
              "S:text"
              "S:arr"
              "STR:\"\"foo\\bar\""
              "E:arr"
              "E:text");
}

TEST(ioJson, jsonParserStr6) {
  test_parser("[ \"\\tfoo\\tbar\" ]",
              "S:text"
              "S:arr"
              "STR:\"\tfoo\tbar\""
              "E:arr"
              "E:text");
}

TEST(ioJson, jsonParserStr7) {
  test_parser("[ \"\\nfoo\\nbar\\n\" ]",
              "S:text"
              "S:arr"
              "STR:\"\nfoo\nbar\n\""
              "E:arr"
              "E:text");
}

TEST(ioJson, jsonParserStr8) {
  test_parser("[ \"\\nfoo\\nbar\\\"\" ]",
              "S:text"
              "S:arr"
              "STR:\"\nfoo\nbar\"\""
              "E:arr"
              "E:text");
}

TEST(ioJson, jsonParserStr9) {
  test_parser("[ \"foo\\nbar\\\"\" , \"foo\\nbar\\\"\"  ]",
              "S:text"
              "S:arr"
              "STR:\"foo\nbar\"\""
              "STR:\"foo\nbar\"\""
              "E:arr"
              "E:text");
}

TEST(ioJson, jsonParserStr10) {
  test_parser("[ \"\\u0069\" ]",
              "S:text"
              "S:arr"
              "STR:\"i\""
              "E:arr"
              "E:text");
}

// surrogate pair for G clef character from extended multilingual plane:
// specified here using hex values for non-ASCII UTF-8 bytes
TEST(ioJson, jsonParserStr11) {
  test_parser("[ \"\\uD834\\uDD1E\" ]",
              "S:text"
              "S:arr"
              "STR:\"\xf0\x9D\x84\x9E\""
              "E:arr"
              "E:text");
}

// string w/ two non-ascii Latin 1 chars
TEST(ioJson, jsonParserStr12) {
  test_parser("[ \"D\\u00E9j\\u00E0 vu\" ]",
              "S:text"
              "S:arr"
              "STR:\"D\xc3\xa9j\xc3\xa0 vu\""
              "E:arr"
              "E:text");
}

// same string w/ non-ASCII chars not \u escaped (specified as hex byte values)
TEST(ioJson, jsonParserStr13) {
  test_parser("[ \"D\xc3\xa9j\xc3\xa0 vu\" ]",
              "S:text"
              "S:arr"
              "STR:\"D\xc3\xa9j\xc3\xa0 vu\""
              "E:arr"
              "E:text");
}

// surrogate pair boundary conditions
TEST(ioJson, jsonParserStr14) {
  test_parser("[ \"\\uD800\\uDC00\" ]",
              "S:text"
              "S:arr"
              "STR:\"\xf0\x90\x80\x80\""
              "E:arr"
              "E:text");
}

// surrogate pair boundary conditions
TEST(ioJson, jsonParserStr15) {
  test_parser("[ \"\\uD800\\uDFFF\" ]",
              "S:text"
              "S:arr"
              "STR:\"\xf0\x90\x8F\xBF\""
              "E:arr"
              "E:text");
}

// surrogate pair boundary conditions
TEST(ioJson, jsonParserStr16) {
  test_parser("[ \"\\uDBFF\\uDC00\" ]",
              "S:text"
              "S:arr"
              "STR:\"\xf4\x8f\xb0\x80\""
              "E:arr"
              "E:text");
}

// surrogate pair boundary conditions
TEST(ioJson, jsonParserStr17) {
  test_parser("[ \"a\\uE000\" ]",
              "S:text"
              "S:arr"
              "STR:\"a\xEE\x80\x80\""
              "E:arr"
              "E:text");
}

TEST(ioJson, jsonParserErr01) {
  test_exception(" \n \n   5    ",
                 "\nexpecting start of object ({) or array ([)\n");
}

TEST(ioJson, jsonParserErr02) {
  test_exception("[ .5 ]", "Expecting JSON object, found array.");
}

TEST(ioJson, jsonParserErr02a) {
  test_exception("{ \"x\" : [ 0",
                 "Missing a comma or ']' after an array element or "
                 "found a zero padded number.\n");
}

TEST(ioJson, jsonParserErr02b) {
  test_exception("{ \"x\" : [ 0.", "Missing fraction part in number.\n");
}

TEST(ioJson, jsonParserErr02c) {
  test_exception("{ \"x\": [ 99.9",
                 "Missing a comma or ']' after an array element or "
                 "found a zero padded number.\n");
}

TEST(ioJson, jsonParserErr03) {
  test_exception("{ \"x\": [ 000.005 ]",
                 "Missing a comma or ']' after an array element "
                 "or found a zero padded number.\n");
}

TEST(ioJson, jsonParserErr04) {
  test_exception("{ \"x\": [ 1. ]", "Missing fraction part in number.\n");
}

TEST(ioJson, jsonParserErr05) {
  test_exception("{ \"x\": [ 1.009e ]", "Missing exponent in number.\n");
}

TEST(ioJson, jsonParserErr06b) {
  test_exception("{ \"x\": [ \"\\uD834abc\" ]",
                 "The surrogate pair in string is invalid.\n");
}

TEST(ioJson, jsonParserErr06e) {
  test_exception("{ \"x\": [ \"\\uD834",
                 "The surrogate pair in string is invalid.\n");
}

TEST(ioJson, jsonParserErr06f) {
  test_exception("{ \"x\": [ \"\\uD8",
                 "Incorrect hex digit after \\u escape in string.\n");
}

TEST(ioJson, jsonParserErr06g) {
  test_exception("{ \"x\": [ \"\\uE000\\uD",
                 "Incorrect hex digit after \\u escape in string.\n");
}

TEST(ioJson, jsonParserErr07) {
  test_exception("{ \"x\": [ \"\\aFFFF\" ]",
                 "Invalid escape character in string.\n");
}

TEST(ioJson, jsonParserErr08) {
  std::stringstream ss;
  char c = 11;
  ss << "{ \"x\": [ \"" << c << "\" ]";
  test_exception(ss.str(), "Invalid encoding in string.\n");
}

TEST(ioJson, jsonParserErr09) {
  test_exception("{ \"x\": [ t ]", "Invalid value.\n");
}

TEST(ioJson, jsonParserErr10) {
  test_exception("{ \"x\": [ f ]", "Invalid value.\n");
}

TEST(ioJson, jsonParserErr11) {
  test_exception("{ \"x\": [ n ]", "Invalid value.\n");
}

TEST(ioJson, jsonParserErr12) {
  test_exception("{ \"x\": [5}",
                 "Missing a comma or ']' after an array element or "
                 "found a zero padded number.\n");
}

TEST(ioJson, jsonParserErr12a) {
  test_exception("{ \"x\": [ a ]", "Invalid value.\n");
}

TEST(ioJson, jsonParserErr12b) {
  test_exception("{ \"x\": [ \"a\", a ]",
                 "Variable: x, error: string values not allowed.");
}

TEST(ioJson, jsonParserErr12c) {
  test_exception("{ \"x\": [ \"a\",",
                 "Variable: x, error: string values not allowed.");
}

TEST(ioJson, jsonParserErr12d) {
  test_exception("{ \"a\" : 5 ] }",
                 "Missing a comma or '}' after an object member.\n");
}

TEST(ioJson, jsonParserErr12e) {
  test_exception("{ \"a\" : [ 5 ] ] }",
                 "Missing a comma or '}' after an object member.\n");
}

TEST(ioJson, jsonParserErr13) {
  test_exception("{ hello }", "Missing a name for object member.\n");
}

TEST(ioJson, jsonParserErr14) {
  test_exception("{ \"foo\": -1.0100e09 , }",
                 "Missing a name for object member.\n");
}

TEST(ioJson, jsonParserErr14a) {
  test_exception("{ { \"foo\": -1.0100e09 , }",
                 "Missing a name for object member.\n");
}

TEST(ioJson, jsonParserErr14b) {
  test_exception("{ \"bar\" : { \"foo\": -1.0100e09 , }",
                 "Missing a name for object member.\n");
}

TEST(ioJson, jsonParserErr14c) {
  test_exception("{ \"bar\" : [ \"foo\": -1.0100e09 , }",
                 "Variable: bar, error: string values not allowed.");
}

TEST(ioJson, jsonParseErr14d) {
  test_exception(
      "{  \"foo\" : [ { \"bar\": { \"baz\": [ 1, 2]  } }, -3, -4.44  }  ",
      "Missing a comma or ']' after an array element or found a zero padded "
      "number.\n");
}

TEST(ioJson, jsonParseErr14e) {
  test_exception(
      "{  \"foo\" : [ { \"bar\": { \"baz\": [ 1, 2]  } }, -3, -4.44 "
      "} } } ] }  ",
      "Missing a comma or ']' after an array element or found a "
      "zero padded number.\n");
}

TEST(ioJson, jsonParserErr14f) {
  test_exception("{ \"foo\": -1.0100e09 , ",
                 "Missing a name for object member.\n");
}

TEST(ioJson, jsonParserErr15) {
  test_exception("{ \"5\" 5 }",
                 "Missing a colon after a name of object member.\n");
}

TEST(ioJson, jsonParserErr16) {
  test_exception("{ \"5\" :  5  \"6\" : 6 }",
                 "Missing a comma or '}' after an object member.\n");
}

TEST(ioJson, jsonParserErr17) {
  test_exception("{ \"5\" : ", "Invalid value.\n");
}

TEST(ioJson, jsonParserErr18) {
  test_exception("{ \"x\": [ -1, -2, \"-Inf\", ]", "Invalid value.\n");
}

TEST(ioJson, jsonParserErr19a) {
  test_exception(
      "{ \"x\": [ "
      "111111111111111111111111111111111111111111111111111111111111111111111111"
      "111111111111111111111111111111111111111111111111111111111111111111111111"
      "111111111111111111111111111111111111111111111111111111111111111111111111"
      "111111111111111111111111111111111111111111111111111111111111111111111111"
      "1111111111111111111111 ]",
      "Number too big to be stored in double.\n");
}

TEST(ioJson, jsonParserErr19b) {
  test_exception(
      "{ \"x\": [ "
      "-11111111111111111111111111111111111111111111111111111111111111111111111"
      "111111111111111111111111111111111111111111111111111111111111111111111111"
      "111111111111111111111111111111111111111111111111111111111111111111111111"
      "111111111111111111111111111111111111111111111111111111111111111111111111"
      "11111111111111111111111 ]",
      "Number too big to be stored in double.\n");
}

TEST(ioJson, jsonParserErr19d) {
  test_exception("{ \"x\": [ 9.19191919191919e1000000000000 ]",
                 "Number too big to be stored in double.\n");
}

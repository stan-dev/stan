#include <gtest/gtest.h>

#include <stan/io/json/json_data.hpp>
#include <stan/io/json/json_data_handler.hpp>
#include <stan/io/json/json_error.hpp>
#include <stan/io/json/json_handler.hpp>
#include <stan/io/json/json_parser.hpp>

class recording_handler : public stan::json::json_handler {
public:
  std::stringstream os_;
  recording_handler() : json_handler(), os_() {
  }
  void start_text() {
    os_ << "S:text";
  }
  void end_text() {
    os_ << "E:text";
  }
  void start_array() {
    os_ << "S:arr";
  }
  void end_array() {
    os_ << "E:arr";
  }
  void start_object() {
    os_ << "S:obj";
  }
  void end_object() {
    os_ << "E:obj";
  }
  void null() {
    os_ << "NULL:null";
  }
  void boolean(bool p) {
    os_ << "BOOL:" << p;
  }
  void string(const std::string& s) {
    os_ << "STR:\"" << s << "\"";
  }
  void key(const std::string& key) {
    os_ << "KEY:\"" << key << "\"";
  }
  void number_double(double x) { 
    os_ << "D(REAL):" << x;
  }
  void number_long(long n) { 
    os_ << "L(INT):" << n;
  }
  void number_unsigned_long(unsigned long n) { 
    os_ << "UL(INT):" << n;
  }
};

bool hasEnding(std::string const &fullString, std::string const &ending) {
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
  } else {
    return false;
  }
}

void test_parser(const std::string& input,
                 const std::string& expected_output) {
  recording_handler handler;
  std::stringstream s(input);
  stan::json::parse(s, handler);
  EXPECT_EQ(expected_output, handler.os_.str());
}

void test_exception(const std::string& input,
                    const std::string& exception_text) {
  try {
    recording_handler handler;
    std::stringstream s(input);
    stan::json::parse(s, handler);
  } catch (const std::exception& e) {
    EXPECT_TRUE(hasEnding(e.what(), exception_text));
    return;
  }
  FAIL();  // didn't throw an exception as expected.
}


TEST(ioJson,jsonParserA0) {
  test_parser("[0]",
              "S:text" "S:arr" "UL(INT):0" "E:arr" "E:text");
}

TEST(ioJson,jsonParserA1) {
  test_parser("[5]",
              "S:text" "S:arr" "UL(INT):5" "E:arr" "E:text");
}

TEST(ioJson,jsonParserA2) {
  test_parser("[5,10]",
              "S:text" "S:arr" "UL(INT):5" "UL(INT):10" "E:arr" "E:text");
}

TEST(ioJson,jsonParserA3) {
  test_parser("[]",
              "S:text" "S:arr" "E:arr" "E:text");
}

TEST(ioJson,jsonParserA4) {
  std::stringstream textReport;
  textReport << "S:text" << "S:arr" << "D(REAL):" 
             << 0.100019
             << "E:arr" << "E:text";
  test_parser("[ 0.100019 ]",
              textReport.str());
}

TEST(ioJson,jsonParserA5) {
  std::stringstream textReport;
  textReport << "S:text" << "S:arr" << "D(REAL):" 
             << 0.10001900
             << "E:arr" << "E:text";
  test_parser("[ 0.10001900 ]",
              textReport.str());
}

TEST(ioJson,jsonParserA6) {
  std::stringstream textReport;
  textReport << "S:text" << "S:arr" << "D(REAL):" 
             << -0.10001900
             << "E:arr" << "E:text";
  test_parser("[ -0.10001900 ]",
              textReport.str());
}

TEST(ioJson,jsonParserA7) {
  test_parser("[ -1, -2, \"-inf\"]",
              "S:text" "S:arr" "L(INT):-1" "L(INT):-2"  
              "STR:\"-inf\"" "E:arr" "E:text");
}

TEST(ioJson,jsonParserA8) {
  test_parser("[ [ -1, -2 ],[ 1, 2 ] ]",
              "S:text" "S:arr" 
              "S:arr" "L(INT):-1" "L(INT):-2"  "E:arr"
              "S:arr" "UL(INT):1" "UL(INT):2"  "E:arr"
              "E:arr" "E:text");
}

TEST(ioJson,jsonParserA9) {
  test_parser("[ [ [1, 2 ],[ 3, 4 ] ],[ [5, 6 ],[ 7, 8 ] ] ]",
              "S:text" "S:arr" "S:arr"
              "S:arr" "UL(INT):1" "UL(INT):2"  "E:arr"
              "S:arr" "UL(INT):3" "UL(INT):4"  "E:arr"
              "E:arr" "S:arr"
              "S:arr" "UL(INT):5" "UL(INT):6"  "E:arr"
              "S:arr" "UL(INT):7" "UL(INT):8"  "E:arr"
              "E:arr" "E:arr" "E:text");
}


TEST(ioJson,jsonParserO1) {
  test_parser("{  \"foo\" : 1  }   ",
              "S:text" "S:obj" "KEY:\"foo\"" "UL(INT):1" "E:obj" "E:text");
}

TEST(ioJson,jsonParserO11) {
  test_parser("{  \"foo\" : { \"bar\": 1 } }   ",
              "S:text" "S:obj" "KEY:\"foo\"" "S:obj" "KEY:\"bar\"" 
              "UL(INT):1" "E:obj" "E:obj" "E:text");
}

TEST(ioJson,jsonParserO12) {
  test_parser("{  \"foo\" : { \"bar\": 1, \"baz\": 2 } }   ",
              "S:text" "S:obj" "KEY:\"foo\"" "S:obj" 
              "KEY:\"bar\""  "UL(INT):1" 
              "KEY:\"baz\""  "UL(INT):2" 
              "E:obj" "E:obj" "E:text");
}


TEST(ioJson,jsonParserO13) {
  test_parser("{  \"foo\" : { \"bar\": { \"baz\": [ 1, 2]  } } }  ",
              "S:text" "S:obj" "KEY:\"foo\"" "S:obj" 
              "KEY:\"bar\""  "S:obj" "KEY:\"baz\""  
              "S:arr" "UL(INT):1" "UL(INT):2" "E:arr" 
              "E:obj" "E:obj" "E:obj" "E:text");
}

TEST(ioJson,jsonParserO14) {
  test_parser("{  \"foo\" : [ { \"bar\": { \"baz\": [ 1, 2]  } }, -3, -4.44 ] }  ",
              "S:text" "S:obj" "KEY:\"foo\"" "S:arr" "S:obj" 
              "KEY:\"bar\""  "S:obj" "KEY:\"baz\""  
              "S:arr" "UL(INT):1" "UL(INT):2" "E:arr" "E:obj" "E:obj" 
              "L(INT):-3" "D(REAL):-4.44" "E:arr" "E:obj" "E:text");
}

TEST(ioJson,jsonParserO2) {
  test_parser("{  \"foo\"  : 1 ,  \n   \"bar\" : 2 }",
              "S:text" "S:obj" "KEY:\"foo\"" "UL(INT):1" 
              "KEY:\"bar\"" "UL(INT):2" "E:obj" "E:text");
}

TEST(ioJson,jsonParserO3) {
  test_parser("{\"foo\":1,\"bar\":2,\"baz\":[2]}",
              "S:text" "S:obj" "KEY:\"foo\"" "UL(INT):1"
              "KEY:\"bar\"" "UL(INT):2" 
              "KEY:\"baz\"" "S:arr" "UL(INT):2" "E:arr"
              "E:obj" "E:text"
              );
}

TEST(ioJson,jsonParserO4) {
  std::stringstream textReport;
  textReport << "S:text" << "S:obj" 
             << "KEY:\"foo\"" << "UL(INT):1"
             << "KEY:\"baz\"" 
             << "S:arr" << "UL(INT):2" << "UL(INT):3" << "D(REAL):" 
             << 4.01001 
             << "E:arr" 
             << "E:obj" << "E:text";
  test_parser("{ \"foo\" : 1, \n \"baz\" : [ 2, 3, 4.01001 ] };",
              textReport.str());
}

TEST(ioJson,jsonParserO5) {
  std::stringstream textReport;
  textReport << "S:text" << "S:obj" << "KEY:\"foo\"" << "D(REAL):" 
             << -0.10001900
             << "E:obj" << "E:text";
  test_parser("{ \"foo\": -0.10001900 }",
              textReport.str());
}

TEST(ioJson,jsonParserO6) {
  std::stringstream textReport;
  textReport << "S:text" << "S:obj" << "KEY:\"foo\"" 
             << "D(REAL):" 
             << -1.0100e09
             << "E:obj" << "E:text";
  test_parser("{ \"foo\": -1.0100e09 }",
              textReport.str());
}

TEST(ioJson,jsonParserO7) {
  std::stringstream textReport;
  textReport << "S:text" << "S:obj" << "KEY:\"foo\"" 
             << "D(REAL):" 
             << -1.0100e09
             << "E:obj" << "E:text";
  test_parser("{ \"foo\": -1.0100e09 }",
              textReport.str());
}


TEST(ioJson,jsonParserStr0) {
  test_parser("[ \"foo\" ]",
              "S:text" "S:arr" "STR:\"foo\"" "E:arr" "E:text");
}

TEST(ioJson,jsonParserStr1) {
  test_parser("[ \"\\nfoo\" ]",
              "S:text" "S:arr" "STR:\"\nfoo\"" "E:arr" "E:text");
}

TEST(ioJson,jsonParserStr2) {
  test_parser("[ \"\\nfoo\\bbar\" ]",
              "S:text" "S:arr" "STR:\"\nfoo\bbar\"" "E:arr" "E:text");
}


TEST(ioJson,jsonParserStr3) {
  test_parser("[ \"\\bfoo\\nbar\" ]",
              "S:text" "S:arr" "STR:\"\bfoo\nbar\"" "E:arr" "E:text");
}

TEST(ioJson,jsonParserStr4) {
  test_parser("[ \"\\\"foo\\/bar\" ]",
              "S:text" "S:arr" "STR:\"\"foo/bar\"" "E:arr" "E:text");
}

TEST(ioJson,jsonParserStr5) {
  test_parser("[ \"\\\"foo\\\\bar\" ]",
              "S:text" "S:arr" "STR:\"\"foo\\bar\"" "E:arr" "E:text");
}

TEST(ioJson,jsonParserStr6) {
  test_parser("[ \"\\tfoo\\tbar\" ]",
              "S:text" "S:arr" "STR:\"\tfoo\tbar\"" "E:arr" "E:text");
}

TEST(ioJson,jsonParserStr7) {
  test_parser("[ \"\\nfoo\\nbar\\n\" ]",
              "S:text" "S:arr" "STR:\"\nfoo\nbar\n\"" "E:arr" "E:text");
}

TEST(ioJson,jsonParserStr8) {
  test_parser("[ \"\\nfoo\\nbar\\\"\" ]",
              "S:text" "S:arr" "STR:\"\nfoo\nbar\"\"" "E:arr" "E:text");
}


TEST(ioJson,jsonParserStr9) {
  test_parser("[ \"foo\\nbar\\\"\" , \"foo\\nbar\\\"\"  ]",
              "S:text" "S:arr" 
              "STR:\"foo\nbar\"\"" "STR:\"foo\nbar\"\"" 
              "E:arr" "E:text");
}

TEST(ioJson,jsonParserStr10) {
  test_parser("[ \"\\u0069\" ]",
              "S:text" "S:arr" "STR:\"i\"" "E:arr" "E:text");
}

// surrogate pair for G clef character from extended multilingual plane:
// specified here using hex values for non-ASCII UTF-8 bytes 
TEST(ioJson,jsonParserStr11) {
  test_parser("[ \"\\uD834\\uDD1E\" ]",
              "S:text" "S:arr" "STR:\"\xf0\x9D\x84\x9E\"" "E:arr" "E:text");
}


// string w/ two non-ascii Latin 1 chars
TEST(ioJson,jsonParserStr12) {
  test_parser("[ \"D\\u00E9j\\u00E0 vu\" ]",
              "S:text" "S:arr" "STR:\"D\xc3\xa9j\xc3\xa0 vu\"" "E:arr" "E:text");
}

// same string w/ non-ASCII chars not \u escaped (specified as hex byte values)
TEST(ioJson,jsonParserStr13) {
  test_parser("[ \"D\xc3\xa9j\xc3\xa0 vu\" ]",
              "S:text" "S:arr" "STR:\"D\xc3\xa9j\xc3\xa0 vu\"" "E:arr" "E:text");
}


// surrogate pair boundary conditions
TEST(ioJson,jsonParserStr14) {
  test_parser("[ \"\\uD800\\uDC00\" ]",
              "S:text" "S:arr" "STR:\"\xf0\x90\x80\x80\"" "E:arr" "E:text");
}

// surrogate pair boundary conditions
TEST(ioJson,jsonParserStr15) {
  test_parser("[ \"\\uD800\\uDFFF\" ]",
              "S:text" "S:arr" "STR:\"\xf0\x90\x8F\xBF\"" "E:arr" "E:text");
}

// surrogate pair boundary conditions
TEST(ioJson,jsonParserStr16) {
  test_parser("[ \"\\uDBFF\\uDC00\" ]",
              "S:text" "S:arr" "STR:\"\xf4\x8f\xb0\x80\"" "E:arr" "E:text");
}

// surrogate pair boundary conditions
TEST(ioJson,jsonParserStr17) {
  test_parser("[ \"a\\uE000\" ]",
              "S:text" "S:arr" "STR:\"a\xEE\x80\x80\"" "E:arr" "E:text");
}




TEST(ioJson,jsonParserErr01) {
  test_exception(" \n \n   5    ",
                 "expecting start of object ({) or array ([)\n");
}

TEST(ioJson,jsonParserErr02) {
  test_exception("[ .5 ]",
                 "illegal value, expecting object, array, number, string, or literal true/false/null\n");
}

TEST(ioJson,jsonParserErr02a) {
  test_exception("[ 0",
                 "unexpected end of stream\n");
}

TEST(ioJson,jsonParserErr02b) {
  test_exception("[ 0.",
                 "unexpected end of stream\n");
}


TEST(ioJson,jsonParserErr02c) {
  test_exception("[ 99.9",
                 "unexpected end of stream\n");
}


TEST(ioJson,jsonParserErr03) {
  test_exception("[ 000.005 ]",
                 "zero padded numbers not allowed\n");
}

TEST(ioJson,jsonParserErr04) {
  test_exception("[ 1. ]",
                 "expected digit after decimal\n");
}

TEST(ioJson,jsonParserErr05) {
  test_exception("[ 1.009e ]",
                 "expected digit after e/E\n");
}


/*
TEST(ioJson,jsonParserErr06) {
  test_exception("[ \"\\uFFFF\" ]",
                 "unicode escapes not supported\n");
}
*/

TEST(ioJson,jsonParserErr06a) {
  test_exception("[ \"\\uDD1Eabd \" ]",
                 "illegal unicode values, found low-surrogate, missing high-surrogate\n");
}

TEST(ioJson,jsonParserErr06b) {
  test_exception("[ \"\\uD834abc\" ]",
                 "illegal unicode values, found high-surrogate, expecting low-surrogate\n");
}


// surrogate pair boundary conditions
TEST(ioJson,jsonParserErr06c) {
  test_exception("[ \"\\uDC00\\uDFFFF\" ]",
              "illegal unicode values, found low-surrogate, missing high-surrogate\n");
}


// surrogate pair boundary conditions
TEST(ioJson,jsonParserErr06d) {
  test_exception("[ \"\\uE000\\uDFFF\" ]",
                 "illegal unicode values, found low-surrogate, missing high-surrogate\n");
}

TEST(ioJson,jsonParserErr06e) {
  test_exception("[ \"\\uD834",
                 "unexpected end of stream\n");
}

TEST(ioJson,jsonParserErr06f) {
  test_exception("[ \"\\uD8",
                 "unexpected end of stream\n");
}

TEST(ioJson,jsonParserErr06g) {
  test_exception("[ \"\\uE000\\uD",
                 "unexpected end of stream\n");
}


TEST(ioJson,jsonParserErr07) {
  test_exception("[ \"\\aFFFF\" ]",
                 "expecting legal escape\n");
}

TEST(ioJson,jsonParserErr08) {
  std::stringstream ss;
  char c = 11;
  ss << "[ \"" << c << "\" ]";
  test_exception(ss.str(),
                 "found control character, char values less than U+0020 must be \\u escaped\n");
}

TEST(ioJson,jsonParserErr09) {
  test_exception("[ t ]",
                 "expecting rest of literal: rue\n");
}

TEST(ioJson,jsonParserErr10) {
  test_exception("[ f ]",
                 "expecting rest of literal: alse\n");
}

TEST(ioJson,jsonParserErr11) {
  test_exception("[ n ]",
                 "expecting rest of literal: ull\n");
}

TEST(ioJson,jsonParserErr12) {
  test_exception("[5}",
                 "in array, expecting ] or ,\n");
}

TEST(ioJson,jsonParserErr12a) {
  test_exception("[ a ]",
                 "illegal value, expecting object, array, number, string, or literal true/false/null\n");
}

TEST(ioJson,jsonParserErr12b) {
  test_exception("[ \"a\", a ]",
                 "illegal value, expecting object, array, number, string, or literal true/false/null\n");
}

TEST(ioJson,jsonParserErr12c) {
  test_exception("[ \"a\", ",
                 "unexpected end of stream\n");
}


TEST(ioJson,jsonParserErr12d) {
  test_exception("{ \"a\" : 5 ] }",
                 "expecting end of object } or separator ,\n");
}

TEST(ioJson,jsonParserErr12e) {
  test_exception("{ \"a\" : [ 5 ] ] }",
                 "expecting end of object } or separator ,\n");
}

TEST(ioJson,jsonParserErr13) {
  test_exception("{ hello }",
                 "expecting member key or end of object marker (})\n");
}


TEST(ioJson,jsonParserErr14) {
  test_exception("{ \"foo\": -1.0100e09 , }",
              "expecting member key or end of object marker (})\n");
}


TEST(ioJson,jsonParserErr14a) {
  test_exception("{ { \"foo\": -1.0100e09 , }",
              "expecting member key or end of object marker (})\n");
}


TEST(ioJson,jsonParserErr14b) {
  test_exception("{ \"bar\" : { \"foo\": -1.0100e09 , }",
              "expecting member key or end of object marker (})\n");
}


TEST(ioJson,jsonParserErr14c) {
  test_exception("{ \"bar\" : [ \"foo\": -1.0100e09 , }",
                 "in array, expecting ] or ,\n");
}

TEST(ioJson,jsonParseErr14d) {
  test_exception("{  \"foo\" : [ { \"bar\": { \"baz\": [ 1, 2]  } }, -3, -4.44  }  ",
                 "in array, expecting ] or ,\n");
}

TEST(ioJson,jsonParseErr14e) {
  test_exception("{  \"foo\" : [ { \"bar\": { \"baz\": [ 1, 2]  } }, -3, -4.44 } } } ] }  ",
                 "in array, expecting ] or ,\n");
}

TEST(ioJson,jsonParserErr14f) {
  test_exception("{ \"foo\": -1.0100e09 , ",
              "unexpected end of stream\n");
}


TEST(ioJson,jsonParserErr15) {
  test_exception("{ \"5\" 5 }",
                 "expecting key-value separator :\n");
}

TEST(ioJson,jsonParserErr16) {
  test_exception("{ \"5\" :  5  \"6\" : 6 }",
                 "expecting end of object } or separator ,\n");
}

TEST(ioJson,jsonParserErr17) {
  test_exception("{ \"5\" : ",
                 "unexpected end of stream\n");
}

TEST(ioJson,jsonParserErr18) {
  test_exception("[ -1, -2, \"-inf\", ]",
                 "in array, expecting value\n");
}


TEST(ioJson,jsonParserErr19a) {
  test_exception("[ 1111111111111111111111111111 ]",
                 "number exceeds integer range\n");
}

TEST(ioJson,jsonParserErr19b) {
  test_exception("[ -1111111111111111111111111111 ]",
                 "number exceeds integer range\n");
}


TEST(ioJson,jsonParserErr19c) {
  test_exception("[ 9.9999999e-1000000000000 ]",
                 "number exceeds double range\n");
}

TEST(ioJson,jsonParserErr19d) {
  test_exception("[ 9.19191919191919e1000000000000 ]",
                 "number exceeds double range\n");
}

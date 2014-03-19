#include <gtest/gtest.h>

#include <stan/io/json/json_error.hpp>
#include <stan/io/json/json_handler.hpp>
#include <stan/io/json/json_parser.hpp>


class recording_handler : public stan::json::json_handler {
public:
  std::stringstream os_;
  recording_handler() : json_handler(), os_() { }
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
    std::cout << e.what() << std::endl;
    EXPECT_TRUE(hasEnding(e.what(), exception_text));
    return;
  }
  FAIL();  // didn't throw an exception as expected.
}


TEST(ioJson,0) {
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

TEST(ioJson,jsonParserErr01) {
  test_exception(" \n \n   5    ",
                 "expecting start of object ({) or array ([)\n");
}

TEST(ioJson,jsonParserErr02) {
  test_exception("[ .5 ]",
                 "expecting int part of number\n");
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


TEST(ioJson,jsonParserErr06) {
  test_exception("[ \"\\uFFFF\" ]",
                 "unicode escapes not supported\n");
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
                 "illegal string character with code point less than U+0020\n");
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

TEST(ioJson,jsonParserErr13) {
  test_exception("{ hello }",
                 "expecting member key or end of object marker (})\n");
}


TEST(ioJson,jsonParserErr14) {
  test_exception("{ \"foo\": -1.0100e09 , }",
              "expecting member key or end of object marker (})\n");
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


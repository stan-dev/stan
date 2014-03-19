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
    EXPECT_EQ(exception_text, e.what());
    return;
  }
  FAIL();  // didn't throw an exception as expected.
}

TEST(ioJson,jsonParserA1) {
  test_parser("[" "5" "]",
              "S:text" "S:arr" "UL(INT):5" "E:arr" "E:text");
}

TEST(ioJson,jsonParserA2) {
  test_parser("[" "5" "," "10" "]",
              "S:text" "S:arr" "UL(INT):5" "UL(INT):10" "E:arr" "E:text");
}


TEST(ioJson,jsonParserO1) {
  test_parser("{ " "  \"foo\"  " ":" " 1 " " }   ",
              "S:text" "S:obj" "KEY:\"foo\"" "UL(INT):1" "E:obj" "E:text");
}


TEST(ioJson,jsonParserO2) {
  test_parser("{ " "  \"foo\"  " ":" " 1 " " ,  \n"
              "\"bar\"" ":" "2" "}",
              "S:text" "S:obj" "KEY:\"foo\"" "UL(INT):1" 
              "KEY:\"bar\"" "UL(INT):2" "E:obj" "E:text");
}

TEST(ioJson,jsonParserO3) {
  test_parser(
              "{ " "  \"foo\"  " ":" " 1 " " ,  \n"
              "\"bar\"" ":" "2" ","
              "\"baz\"" ":" "[" "2" "]" "}",

              "S:text" "S:obj" "KEY:\"foo\"" "UL(INT):1"
              "KEY:\"bar\"" "UL(INT):2" 
              "KEY:\"baz\"" "S:arr" "UL(INT):2" "E:arr"
              "E:obj" "E:text"
              );
}

TEST(ioJson,jsonParserO4) {
  test_parser(
              "{ " "  \"foo\"  " ":" " 1 " " ,  \n"
              "\"baz\"" ":" "[" "2" "," "3" "," "4.011" "]" "}",

              "S:text" "S:obj" 
              "KEY:\"foo\"" "UL(INT):1"
              "KEY:\"baz\"" "S:arr" 
              "UL(INT):2" "UL(INT):3" "D(REAL):4.011" 
              "E:arr" 
              "E:obj" "E:text"
              );
}


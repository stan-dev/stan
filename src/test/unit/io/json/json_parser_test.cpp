#include <gtest/gtest.h>

#include <stan/io/json/json_error.hpp>
#include <stan/io/json/json_handler.hpp>
#include <stan/io/json/json_parser.hpp>

class one_two_handler : public stan::json::json_handler {
public:
  std::vector<unsigned long> values_;
  one_two_handler() : json_handler(), values_() { }
  void number_unsigned_long(unsigned long n) {
    values_.push_back(n);
  }
};

TEST(ioJson,jsonParser) {
  std::stringstream s;
  s << "[ 5, 10 ]";
  one_two_handler handler;
  stan::json::parse(s, handler);
  EXPECT_EQ(2U, handler.values_.size());
  EXPECT_EQ(5U, handler.values_[0]);
  EXPECT_EQ(10U, handler.values_[1]);
}
TEST(ioJson,jsonParserException) {
  std::stringstream s;
  s << "[ 5, 10";
  one_two_handler handler;
  EXPECT_THROW(stan::json::parse(s, handler), stan::json::json_error);  
}



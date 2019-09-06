#include <stan/io/validate_zero_buf.hpp>
#include <gtest/gtest.h>


TEST(ioValidateZeroBuf, tester) {
  using stan::io::validate_zero_buf;
  std::string s;

  s = "0.0";
  EXPECT_NO_THROW(validate_zero_buf(s));

  s = "0.00000000000";
  EXPECT_NO_THROW(validate_zero_buf(s));

  s = "0.0e52";
  EXPECT_NO_THROW(validate_zero_buf(s));

  s = "1.0";
  EXPECT_THROW(validate_zero_buf(s), boost::bad_lexical_cast);

  s = "1e1";
  EXPECT_THROW(validate_zero_buf(s), boost::bad_lexical_cast);
}


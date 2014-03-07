#include <stan/common/write_error_msg.hpp>
#include <gtest/gtest.h>
#include <stdexcept>
#include <sstream>

std::string err_msg1 = "<error message 1>";
std::string err_msg2 = "<error message 2>";
std::runtime_error err1(err_msg1);
std::domain_error err2(err_msg2);

TEST(StanUi, write_error_msg_nostream) {
  EXPECT_NO_THROW(stan::common::write_error_msg(0, err1));
  EXPECT_NO_THROW(stan::common::write_error_msg(0, err2));
}

TEST(StanUi, write_error_msg) {
  std::stringstream ss;
  EXPECT_NO_THROW(stan::common::write_error_msg(&ss, err1));
  EXPECT_TRUE(ss.str().find(err_msg1) != std::string::npos)
    << "The message should have err_msg1 inside it";
  
  ss.str("");
  EXPECT_NO_THROW(stan::common::write_error_msg(&ss, err2));
  EXPECT_TRUE(ss.str().find(err_msg2) != std::string::npos)
    << "The message should have err_msg2 inside it";
}

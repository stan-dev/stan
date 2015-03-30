#include <stan/interface_callbacks/writer/stringstream.hpp>
#include <stan/services/io/write_error_msg.hpp>
#include <gtest/gtest.h>
#include <stdexcept>
#include <sstream>

std::string err_msg1 = "<error message 1>";
std::string err_msg2 = "<error message 2>";
std::runtime_error err1(err_msg1);
std::domain_error err2(err_msg2);

typedef stan::interface_callbacks::writer::stringstream writer_t;

TEST(StanUi, write_error_msg_nostream) {
  writer_t writer;
  EXPECT_NO_THROW(stan::services::io::write_error_msg(writer, err1));
  EXPECT_NO_THROW(stan::services::io::write_error_msg(writer, err2));
}

TEST(StanUi, write_error_msg) {
  writer_t writer;
  EXPECT_NO_THROW(stan::services::io::write_error_msg(writer, err1));
  EXPECT_TRUE(writer.contents().find(err_msg1) != std::string::npos)
    << "The message should have err_msg1 inside it";
  
  writer.clear();
  EXPECT_NO_THROW(stan::services::io::write_error_msg(writer, err2));
  EXPECT_TRUE(writer.contents().find(err_msg2) != std::string::npos)
    << "The message should have err_msg2 inside it";
}

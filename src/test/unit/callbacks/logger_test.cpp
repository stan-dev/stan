#include <gtest/gtest.h>
#include <stan/callbacks/logger.hpp>

class StanInterfaceCallbacksLogger: public ::testing::Test {
public:
  void SetUp() { }
  void TearDown() { }
  stan::callbacks::logger logger;
};

TEST_F(StanInterfaceCallbacksLogger, debug_string) {
  EXPECT_NO_THROW(logger.debug("message"));

  std::string msg = "message";
  EXPECT_NO_THROW(logger.debug(msg));
}

TEST_F(StanInterfaceCallbacksLogger, debug_stringstream) {
  std::stringstream msg;
  msg << "message";
  EXPECT_NO_THROW(logger.debug(msg));
}

TEST_F(StanInterfaceCallbacksLogger, info_string) {
  EXPECT_NO_THROW(logger.info("message"));

  std::string msg = "message";
  EXPECT_NO_THROW(logger.info(msg));
}

TEST_F(StanInterfaceCallbacksLogger, info_stringstream) {
  std::stringstream msg;
  msg << "message";
  EXPECT_NO_THROW(logger.info(msg));
}

TEST_F(StanInterfaceCallbacksLogger, warn_string) {
  EXPECT_NO_THROW(logger.warn("message"));

  std::string msg = "message";
  EXPECT_NO_THROW(logger.warn(msg));
}

TEST_F(StanInterfaceCallbacksLogger, warn_stringstream) {
  std::stringstream msg;
  msg << "message";
  EXPECT_NO_THROW(logger.warn(msg));
}

TEST_F(StanInterfaceCallbacksLogger, error_string) {
  EXPECT_NO_THROW(logger.error("message"));

  std::string msg = "message";
  EXPECT_NO_THROW(logger.error(msg));
}

TEST_F(StanInterfaceCallbacksLogger, error_stringstream) {
  std::stringstream msg;
  msg << "message";
  EXPECT_NO_THROW(logger.error(msg));
}

TEST_F(StanInterfaceCallbacksLogger, fatal_string) {
  EXPECT_NO_THROW(logger.fatal("message"));

  std::string msg = "message";
  EXPECT_NO_THROW(logger.fatal(msg));
}

TEST_F(StanInterfaceCallbacksLogger, fatal_stringstream) {
  std::stringstream msg;
  msg << "message";
  EXPECT_NO_THROW(logger.fatal(msg));
}

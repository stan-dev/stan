#include <gtest/gtest.h>
#include <boost/lexical_cast.hpp>
#include <sstream>
#include <stan/callbacks/stream_logger.hpp>

class StanInterfaceCallbacksStreamLogger: public ::testing::Test {
public:
  StanInterfaceCallbacksStreamLogger() :
    message1("message 1"), message2("message 2"),
    logger(debug, info, warn, error, fatal) {}

  void SetUp() {
    debug.str("");
    info.str("");
    warn.str("");
    error.str("");
    fatal.str("");
  }

  void TearDown() { }

  std::string message1;
  std::stringstream message2;
  std::stringstream debug, info, warn, error, fatal;
  stan::callbacks::stream_logger logger;
};

TEST_F(StanInterfaceCallbacksStreamLogger, debug_string) {
  EXPECT_NO_THROW(logger.debug(message1));
  EXPECT_EQ(message1 + "\n", debug.str());

  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST_F(StanInterfaceCallbacksStreamLogger, debug_stringstream) {
  EXPECT_NO_THROW(logger.debug(message2));
  EXPECT_EQ(message2.str() + "\n", debug.str());

  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}


TEST_F(StanInterfaceCallbacksStreamLogger, info_string) {
  EXPECT_NO_THROW(logger.info(message1));
  EXPECT_EQ(message1 + "\n", info.str());

  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST_F(StanInterfaceCallbacksStreamLogger, info_stringstream) {
  EXPECT_NO_THROW(logger.info(message2));
  EXPECT_EQ(message2.str() + "\n", info.str());

  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST_F(StanInterfaceCallbacksStreamLogger, warn_string) {
  EXPECT_NO_THROW(logger.warn(message1));
  EXPECT_EQ(message1 + "\n", warn.str());

  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST_F(StanInterfaceCallbacksStreamLogger, warn_stringstream) {
  EXPECT_NO_THROW(logger.warn(message2));
  EXPECT_EQ(message2.str() + "\n", warn.str());

  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", error.str());
  EXPECT_EQ("", fatal.str());
}

TEST_F(StanInterfaceCallbacksStreamLogger, error_string) {
  EXPECT_NO_THROW(logger.error(message1));
  EXPECT_EQ(message1 + "\n", error.str());

  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", fatal.str());
}

TEST_F(StanInterfaceCallbacksStreamLogger, error_stringstream) {
  EXPECT_NO_THROW(logger.error(message2));
  EXPECT_EQ(message2.str() + "\n", error.str());

  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", fatal.str());
}

TEST_F(StanInterfaceCallbacksStreamLogger, fatal_string) {
  EXPECT_NO_THROW(logger.fatal(message1));
  EXPECT_EQ(message1 + "\n", fatal.str());

  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
}

TEST_F(StanInterfaceCallbacksStreamLogger, fatal_stringstream) {
  EXPECT_NO_THROW(logger.fatal(message2));
  EXPECT_EQ(message2.str() + "\n", fatal.str());

  EXPECT_EQ("", debug.str());
  EXPECT_EQ("", info.str());
  EXPECT_EQ("", warn.str());
  EXPECT_EQ("", error.str());
}

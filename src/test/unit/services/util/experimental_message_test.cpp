#include <stan/services/util/experimental_message.hpp>
#include <gtest/gtest.h>
#include <stan/callbacks/stream_writer.hpp>
#include <sstream>

TEST(ServicesUtil, experimental_message) {
  std::stringstream ss;
  stan::callbacks::stream_writer writer(ss);

  stan::services::util::experimental_message(writer);

  EXPECT_TRUE(ss.str().find("EXPERIMENTAL ALGORITHM") != std::string::npos);
}

#include <stan/services/util/experimental_message.hpp>
#include <gtest/gtest.h>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <sstream>

TEST(ServicesUtil, experimental_message) {
  stan::test::unit::instrumented_logger logger;

  stan::services::util::experimental_message(logger);

  EXPECT_EQ(1, logger.find_info("EXPERIMENTAL ALGORITHM"));
}

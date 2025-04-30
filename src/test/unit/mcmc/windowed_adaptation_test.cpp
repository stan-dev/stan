#include <stan/mcmc/windowed_adaptation.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <gtest/gtest.h>

TEST(McmcWindowedAdaptation, sampler_defaults) {
  stan::test::unit::instrumented_logger logger;
  stan::mcmc::windowed_adaptation adapter("test");

  adapter.set_window_params(1000, 75, 50, 25, logger);
  ASSERT_EQ(0, logger.call_count());
  ASSERT_EQ(0, logger.call_count_info());
  EXPECT_FALSE(adapter.in_phase2_window());
  EXPECT_FALSE(adapter.end_phase2_window());
}

TEST(McmcWindowedAdaptation, num_warmup_10) {
  stan::test::unit::instrumented_logger logger;
  stan::mcmc::windowed_adaptation adapter("test");
  adapter.set_window_params(10, 1, 1, 1, logger);
  ASSERT_EQ(0, logger.call_count());
  ASSERT_EQ(0, logger.call_count_info());

  EXPECT_FALSE(adapter.in_phase2_window());
  EXPECT_FALSE(adapter.end_phase2_window());
}

TEST(McmcWindowedAdaptation, num_warmup_10_reduce) {
  stan::test::unit::instrumented_logger logger;
  stan::mcmc::windowed_adaptation adapter("test");
  adapter.set_window_params(10, 75, 25, 50, logger);
  ASSERT_EQ(logger.call_count(), logger.call_count_info());

  EXPECT_FALSE(adapter.in_phase2_window());
  EXPECT_FALSE(adapter.end_phase2_window());
}

TEST(McmcWindowedAdaptation, num_warmup_1) {
  stan::test::unit::instrumented_logger logger;
  stan::mcmc::windowed_adaptation adapter("test");
  adapter.set_window_params(1, 75, 25, 50, logger);

  EXPECT_FALSE(adapter.in_phase2_window());
  EXPECT_TRUE(adapter.end_phase2_window());
}


TEST(McmcWindowedAdaptation, sampler_defaults_100) {
  stan::test::unit::instrumented_logger logger;
  stan::mcmc::windowed_adaptation adapter("test");

  adapter.set_window_params(100, 75, 50, 25, logger);
  ASSERT_EQ(logger.call_count(), logger.call_count_info());
  EXPECT_EQ(1,
            logger.find_info(
                "WARNING: There aren't enough warmup iterations to fit the"));
  EXPECT_EQ(
      1, logger.find_info(
             "         three stages of adaptation as currently configured."));
  EXPECT_EQ(1,
            logger.find_info(
                "         Reducing each adaptation stage to 15%/75%/10% of"));
  EXPECT_EQ(
      1, logger.find_info("         the given number of warmup iterations:"));
  EXPECT_EQ(1, logger.find_info("           init_buffer = 15"));
  EXPECT_EQ(1, logger.find_info("           adapt_window = 75"));
  EXPECT_EQ(1, logger.find_info("           term_buffer = 10"));
}

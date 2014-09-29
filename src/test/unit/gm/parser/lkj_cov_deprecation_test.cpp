#include <gtest/gtest.h>
#include <test/unit/gm/utility.hpp>

TEST(gmParser, lkjCovDeprecationSampling) {
  test_warning("lkj_cov_deprecation1",
               "Warning: the lkj_cov_log() sampling distribution is deprecated.");
}
TEST(gmParser, lkjCovDeprecationFunction) {
  test_warning("lkj_cov_deprecation2",
               "Warning: the lkj_cov_log() function is deprecated.");
}

#include <stan/common/write_stan.hpp>
#include <gtest/gtest.h>
#include <sstream>
#include <stan/version.hpp>

TEST(StanUi, write_stan_nostream) {
  EXPECT_NO_THROW(stan::common::write_stan(0, ""));
  EXPECT_NO_THROW(stan::common::write_stan(0));
}

TEST(StanUi, write_stan_noprefix) {
  std::stringstream ss;
  std::string expected_output;
  expected_output 
    = " stan_version_major = " + stan::MAJOR_VERSION + "\n"
    + " stan_version_minor = " + stan::MINOR_VERSION + "\n"
    + " stan_version_patch = " + stan::PATCH_VERSION + "\n";

  EXPECT_NO_THROW(stan::common::write_stan(&ss));
  EXPECT_EQ(expected_output, ss.str());
}

TEST(StanUi, write_stan) {
  std::stringstream ss;
  std::string prefix = "123";
  std::string expected_output;
  expected_output 
    = prefix + " stan_version_major = " + stan::MAJOR_VERSION + "\n"
    + prefix + " stan_version_minor = " + stan::MINOR_VERSION + "\n"
    + prefix + " stan_version_patch = " + stan::PATCH_VERSION + "\n";

  EXPECT_NO_THROW(stan::common::write_stan(&ss, prefix));
  EXPECT_EQ(expected_output, ss.str());
}

#include <stan/services/io/write_stan.hpp>
#include <stan/interface_callbacks/writer/stream_writer.hpp>
#include <gtest/gtest.h>
#include <sstream>
#include <stan/version.hpp>

TEST(StanUi, write_stan) {
  std::stringstream ss;
  stan::interface_callbacks::writer::stream_writer writer(ss);
  std::string expected_output;
  expected_output 
    = "stan_version_major = " + stan::MAJOR_VERSION + "\n"
    + "stan_version_minor = " + stan::MINOR_VERSION + "\n"
    + "stan_version_patch = " + stan::PATCH_VERSION + "\n";

  EXPECT_NO_THROW(stan::services::io::write_stan(writer));
  EXPECT_EQ(expected_output, ss.str());
}

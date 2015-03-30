#include <stan/interface_callbacks/writer/stringstream.hpp>
#include <stan/services/io/write_stan.hpp>
#include <gtest/gtest.h>
#include <sstream>
#include <stan/version.hpp>

typedef stan::interface_callbacks::writer::stringstream writer_t;

TEST(StanUi, write_stan_nostream) {
  writer_t writer;
  EXPECT_NO_THROW(stan::services::io::write_stan(writer, ""));
  EXPECT_NO_THROW(stan::services::io::write_stan(writer));
}

TEST(StanUi, write_stan_noprefix) {
  writer_t writer;
  std::string expected_output;
  expected_output 
    = " stan_version_major = " + stan::MAJOR_VERSION + "\n"
    + " stan_version_minor = " + stan::MINOR_VERSION + "\n"
    + " stan_version_patch = " + stan::PATCH_VERSION + "\n";

  EXPECT_NO_THROW(stan::services::io::write_stan(writer));
  EXPECT_EQ(expected_output, writer.contents());
}

TEST(StanUi, write_stan) {
  writer_t writer;
  std::string prefix = "123";
  std::string expected_output;
  expected_output 
    = prefix + " stan_version_major = " + stan::MAJOR_VERSION + "\n"
    + prefix + " stan_version_minor = " + stan::MINOR_VERSION + "\n"
    + prefix + " stan_version_patch = " + stan::PATCH_VERSION + "\n";

  EXPECT_NO_THROW(stan::services::io::write_stan(writer, prefix));
  EXPECT_EQ(expected_output, writer.contents());
}

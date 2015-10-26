#include <stan/interface_callbacks/writer/stream_writer.hpp>
#include <stan/services/io/write_stan.hpp>
#include <gtest/gtest.h>
#include <sstream>
#include <stan/version.hpp>

typedef stan::interface_callbacks::writer::stream_writer writer_t;

TEST(StanUi, write_stan_nostream) {
  std::stringstream writer_ss;
  writer_t writer(writer_ss);
  EXPECT_NO_THROW(stan::services::io::write_stan(writer, ""));
  EXPECT_NO_THROW(stan::services::io::write_stan(writer));
}

TEST(StanUi, write_stan_noprefix) {
  std::stringstream writer_ss;
  writer_t writer(writer_ss);
  std::string expected_output;
  expected_output 
    = " stan_version_major = " + stan::MAJOR_VERSION + "\n"
    + " stan_version_minor = " + stan::MINOR_VERSION + "\n"
    + " stan_version_patch = " + stan::PATCH_VERSION + "\n";

  EXPECT_NO_THROW(stan::services::io::write_stan(writer));
  EXPECT_EQ(expected_output, writer_ss.str());
}

TEST(StanUi, write_stan) {
  std::stringstream writer_ss;
  writer_t writer(writer_ss);
  std::string prefix = "123";
  std::string expected_output;
  expected_output 
    = prefix + " stan_version_major = " + stan::MAJOR_VERSION + "\n"
    + prefix + " stan_version_minor = " + stan::MINOR_VERSION + "\n"
    + prefix + " stan_version_patch = " + stan::PATCH_VERSION + "\n";

  EXPECT_NO_THROW(stan::services::io::write_stan(writer, prefix));
  EXPECT_EQ(expected_output, writer_ss.str());
}

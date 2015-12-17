#include <stan/services/io/write_model.hpp>
#include <stan/interface_callbacks/writer/stream_writer.hpp>
#include <gtest/gtest.h>
#include <sstream>

const std::string model_name = "model name";
const std::string prefix = "prefix";

TEST(StanUi, write_model_noprefix) {
  std::stringstream ss;
  stan::interface_callbacks::writer::stream_writer writer(ss);
  std::string expected_output;
  expected_output = " model = " + model_name + "\n";

  EXPECT_NO_THROW(stan::services::io::write_model(writer, model_name));
  EXPECT_EQ(expected_output, ss.str());
}

TEST(StanUi, write_model) {
  std::stringstream ss;
  stan::interface_callbacks::writer::stream_writer writer(ss);
  std::string expected_output;
  expected_output = prefix + " model = " + model_name + "\n";

  EXPECT_NO_THROW(stan::services::io::write_model(writer, model_name, prefix));
  EXPECT_EQ(expected_output, ss.str());
}


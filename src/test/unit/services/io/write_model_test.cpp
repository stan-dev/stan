#include <stan/services/io/write_model.hpp>
#include <stan/interface_callbacks/writer/stream_writer.hpp>
#include <gtest/gtest.h>
#include <sstream>

const std::string model_name = "model name";

TEST(StanUi, write_model) {
  std::stringstream ss;
  stan::interface_callbacks::writer::stream_writer writer(ss);
  std::string expected_output;
  expected_output = "model = " + model_name + "\n";

  EXPECT_NO_THROW(stan::services::io::write_model(writer, model_name));
  EXPECT_EQ(expected_output, ss.str());
}

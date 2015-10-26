#include <stan/interface_callbacks/writer/stream_writer.hpp>
#include <stan/services/io/write_model.hpp>
#include <gtest/gtest.h>
#include <sstream>

typedef stan::interface_callbacks::writer::stream_writer writer_t;

const std::string model_name = "model name";
const std::string prefix = "prefix";

TEST(StanUi, write_model_nostream) {
  std::stringstream writer_ss;
  writer_t writer(writer_ss);
  EXPECT_NO_THROW(stan::services::io::write_model(writer, model_name, prefix));
  EXPECT_NO_THROW(stan::services::io::write_model(writer, model_name));
}

TEST(StanUi, write_model_noprefix) {
  std::stringstream writer_ss;
  writer_t writer(writer_ss);
  std::string expected_output;
  expected_output = " model = " + model_name + "\n";

  EXPECT_NO_THROW(stan::services::io::write_model(writer, model_name));
  EXPECT_EQ(expected_output, writer_ss.str());
}

TEST(StanUi, write_model) {
  std::stringstream writer_ss;
  writer_t writer(writer_ss);
  std::string expected_output;
  expected_output = prefix + " model = " + model_name + "\n";

  EXPECT_NO_THROW(stan::services::io::write_model(writer, model_name, prefix));
  EXPECT_EQ(expected_output, writer_ss.str());
}


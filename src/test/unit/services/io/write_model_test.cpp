#include <stan/interface_callbacks/writer/stringstream.hpp>
#include <stan/services/io/write_model.hpp>
#include <gtest/gtest.h>
#include <sstream>

typedef stan::interface_callbacks::writer::stringstream writer_t;

const std::string model_name = "model name";
const std::string prefix = "prefix";

TEST(StanUi, write_model_nostream) {
  writer_t writer;
  EXPECT_NO_THROW(stan::services::io::write_model(writer, model_name, prefix));
  EXPECT_NO_THROW(stan::services::io::write_model(writer, model_name));
}

TEST(StanUi, write_model_noprefix) {
  writer_t writer;
  std::string expected_output;
  expected_output = " model = " + model_name + "\n";

  EXPECT_NO_THROW(stan::services::io::write_model(writer, model_name));
  EXPECT_EQ(expected_output, writer.contents());
}

TEST(StanUi, write_model) {
  writer_t writer;
  std::string expected_output;
  expected_output = prefix + " model = " + model_name + "\n";

  EXPECT_NO_THROW(stan::services::io::write_model(writer, model_name, prefix));
  EXPECT_EQ(expected_output, writer.contents());
}


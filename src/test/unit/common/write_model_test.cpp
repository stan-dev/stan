#include <stan/common/write_model.hpp>
#include <gtest/gtest.h>
#include <sstream>

const std::string model_name = "model name";
const std::string prefix = "prefix";

TEST(StanUi, write_model_nostream) {
  EXPECT_NO_THROW(stan::common::write_model(0, model_name, prefix));
  EXPECT_NO_THROW(stan::common::write_model(0, model_name));
}

TEST(StanUi, write_model_noprefix) {
  std::stringstream ss;
  std::string expected_output;
  expected_output = " model = " + model_name + "\n";

  EXPECT_NO_THROW(stan::common::write_model(&ss, model_name));
  EXPECT_EQ(expected_output, ss.str());
}

TEST(StanUi, write_model) {
  std::stringstream ss;
  std::string expected_output;
  expected_output = prefix + " model = " + model_name + "\n";

  EXPECT_NO_THROW(stan::common::write_model(&ss, model_name, prefix));
  EXPECT_EQ(expected_output, ss.str());
}


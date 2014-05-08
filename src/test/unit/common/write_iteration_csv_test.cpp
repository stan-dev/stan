#include <stan/common/write_iteration_csv.hpp>
#include <gtest/gtest.h>
#include <sstream>

TEST(StanUi, write_iteration_csv) {
  std::stringstream stream;
  double lp;
  std::vector<double> model_values;

  stream.str("");
  lp = 0.0;
  model_values.clear();
  stan::common::write_iteration_csv(stream, lp, model_values);
  EXPECT_EQ("0\n", stream.str());

  stream.str("");
  lp = 1.0;
  model_values.clear();
  model_values.push_back(2.0);
  model_values.push_back(3.0);
  stan::common::write_iteration_csv(stream, lp, model_values);
  EXPECT_EQ("1,2,3\n", stream.str());
}

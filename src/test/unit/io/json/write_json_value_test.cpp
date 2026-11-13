#include <stan/io/json/write_json_value.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <string>

class StanIOWriteJsonValue : public ::testing::Test {
 public:
  StanIOWriteJsonValue() : ss() {}

  void SetUp() {
    ss.str(std::string());
    ss.clear();
  }

  void TearDown() { ASSERT_TRUE(stan::test::is_valid_JSON(ss.str())); }

  std::stringstream ss;
};

TEST_F(StanIOWriteJsonValue, write_basic_int) {
  stan::json::write_json_value(&ss, 1234);
  EXPECT_EQ("1234", ss.str());
}

TEST_F(StanIOWriteJsonValue, write_basic_double) {
  stan::json::write_json_value(&ss, 12.34);
  EXPECT_EQ("12.34", ss.str());
}

TEST_F(StanIOWriteJsonValue, write_basic_complex) {
  stan::json::write_json_value(&ss, std::complex{1.2, 3.4});
  EXPECT_EQ("[1.2, 3.4]", ss.str());
}

TEST_F(StanIOWriteJsonValue, write_string) {
  stan::json::write_json_value(&ss, "f00\nb\"a'r");
  EXPECT_EQ("\"f00\\nb\\\"a'r\"", ss.str());
}

TEST_F(StanIOWriteJsonValue, write_vector) {
  stan::json::write_json_value(
      &ss, std::vector{std::vector{1, 2}, std::vector{3, 4}});
  EXPECT_EQ("[ [ 1, 2 ], [ 3, 4 ] ]", ss.str());
}

TEST_F(StanIOWriteJsonValue, write_special_double_vectors) {
  stan::json::write_json_value(
      &ss, std::vector{-0.0, std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()});
  EXPECT_EQ("[ -0, NaN, Inf, -Inf ]", ss.str());
}

TEST_F(StanIOWriteJsonValue, write_nasty_complex) {
  stan::json::write_json_value(
      &ss, std::complex{std::numeric_limits<double>::infinity(),
                        -std::numeric_limits<double>::infinity()});
  EXPECT_EQ("[Inf, -Inf]", ss.str());
}

TEST_F(StanIOWriteJsonValue, write_eigen_vector) {
  stan::json::write_json_value(&ss, Eigen::VectorXd{{1.0, 2.0, 3.0, 4.0}});
  EXPECT_EQ("[ 1, 2, 3, 4 ]", ss.str());
}

TEST_F(StanIOWriteJsonValue, write_eigen_row_vector) {
  stan::json::write_json_value(&ss, Eigen::RowVectorXd{{1.0, 2.0, 3.0, 4.0}});
  EXPECT_EQ("[ 1, 2, 3, 4 ]", ss.str());
}

TEST_F(StanIOWriteJsonValue, write_eigen_matrix) {
  stan::json::write_json_value(
      &ss, Eigen::MatrixXd{{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}});
  EXPECT_EQ("[ [ 1, 2, 3 ], [ 4, 5, 6 ] ]", ss.str());
}

TEST_F(StanIOWriteJsonValue, write_basic_tuple) {
  stan::json::write_json_value(&ss, std::tuple{1, 2.0});
  EXPECT_EQ("{\"1\" : 1, \"2\" : 2}", ss.str());
}

TEST_F(StanIOWriteJsonValue, write_vector_tuple) {
  stan::json::write_json_value(
      &ss, std::vector{std::tuple{1, 2.0}, std::tuple{3, 4.0}});
  EXPECT_EQ("[ {\"1\" : 1, \"2\" : 2}, {\"1\" : 3, \"2\" : 4} ]", ss.str());
}

TEST_F(StanIOWriteJsonValue, write_nasty_tuple) {
  auto value = std::make_tuple(
      3.14, std::complex{1.0, 2.0}, std::tuple{1, .30}, std::vector{4.0, 5.0},
      std::vector{std::make_tuple(1, 2.0), std::make_tuple(3, 4.0)},
      Eigen::MatrixXd::Zero(2, 3));

  stan::json::write_json_value(&ss, value);
  EXPECT_EQ(
      "{\"1\" : 3.14, \"2\" : [1, 2], \"3\" : {  \"1\" : 1,   \"2\" : 0.3}, "
      "\"4\" : [ 4, 5 ], \"5\" : [ {  \"1\" : 1,   \"2\" : 2}, {  \"1\" : 3,   "
      "\"2\" : 4} ], \"6\" : [ [ 0, 0, 0 ], [ 0, 0, 0 ] ]}",
      ss.str());
}

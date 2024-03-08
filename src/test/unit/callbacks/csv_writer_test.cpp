#include <stan/callbacks/csv_writer.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <string>

class StanInterfaceCallbacksCsvWriter : public ::testing::Test {
 public:
  StanInterfaceCallbacksCsvWriter()
      : ss(), writer(std::unique_ptr<std::stringstream, deleter_noop>(&ss)) {}

  void SetUp() {
    ss.str(std::string());
    ss.clear();
  }

  void TearDown() {}

  std::stringstream ss;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer;
};

TEST_F(StanInterfaceCallbacksCsvWriter, header_value) {
  std::vector<std::string> header = {"mu", "sigma", "theta"};
  std::vector<double> values = {1, 2, 3};
  writer.begin_header();
  writer.write_header(header);
  writer.end_header();
  writer.begin_row();
  writer.write_flat(values);
  writer.end_row();
  EXPECT_EQ("mu, sigma, theta\n1, 2, 3\n", ss.str());
}

TEST_F(StanInterfaceCallbacksCsvWriter, header_values) {
  std::vector<std::string> header = {"mu", "sigma", "theta"};
  std::vector<double> values = {1, 2, 3};
  writer.begin_header();
  writer.write_header(header);
  writer.end_header();
  writer.begin_row();
  writer.write_flat(values);
  writer.end_row();
  writer.begin_row();
  writer.write_flat(values);
  writer.end_row();
  EXPECT_EQ("mu, sigma, theta\n1, 2, 3\n1, 2, 3\n", ss.str());
}

TEST_F(StanInterfaceCallbacksCsvWriter, header_no_newline) {
  EXPECT_EQ("", ss.str());
  std::vector<std::string> header = {"mu", "sigma", "theta"};
  writer.write_header(header);
  EXPECT_EQ("mu, sigma, theta", ss.str());
}

TEST_F(StanInterfaceCallbacksCsvWriter, header_newline) {
  std::vector<std::string> header = {"mu", "sigma", "theta"};
  writer.write_header(header);
  writer.end_header();
  EXPECT_EQ("mu, sigma, theta\n", ss.str());
}

TEST_F(StanInterfaceCallbacksCsvWriter, mult_header) {
  std::vector<std::string> header = {"mu", "sigma", "theta"};
  writer.write_header(header);
  writer.write_header(header);
  writer.end_header();
  EXPECT_EQ("mu, sigma, theta, mu, sigma, theta\n", ss.str());
}

TEST_F(StanInterfaceCallbacksCsvWriter, no_header_no_newline) {
  std::vector<double> values = {1, 2, 3};
  writer.write_flat(values);
  EXPECT_EQ("1, 2, 3", ss.str());
}

TEST_F(StanInterfaceCallbacksCsvWriter, no_header) {
  std::vector<double> values = {1, 2, 3};
  writer.write_flat(values);
  writer.end_row();
  EXPECT_EQ("1, 2, 3\n", ss.str());
}

TEST_F(StanInterfaceCallbacksCsvWriter, row_before_header) {
  std::vector<std::string> header = {"mu", "sigma", "theta"};
  std::vector<double> values = {1, 2, 3};
  writer.write_flat(values);
  EXPECT_THROW(writer.begin_header();, std::domain_error);
}

TEST_F(StanInterfaceCallbacksCsvWriter, header_before_header) {
  std::vector<std::string> header = {"mu", "sigma", "theta"};
  std::vector<double> values = {1, 2, 3};
  writer.begin_header();
  writer.write_header(header);
  writer.end_header();
  EXPECT_THROW(writer.begin_header();, std::domain_error);
}

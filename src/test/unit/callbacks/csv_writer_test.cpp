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

  void TearDown() {
  }

  std::stringstream ss;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer;
};

// note:  per-test writers needed to test state

TEST_F(StanInterfaceCallbacksCsvWriter, good) {
  std::vector<std::string> header = {"mu", "sigma", "theta"};
  std::vector<double> values = {1, 2, 3};
  writer.write_header(header);
  writer.write_flat(values);
  EXPECT_EQ("mu, sigma, theta\n1, 2, 3\n", ss.str());
  writer.write_flat(values);
  EXPECT_EQ("mu, sigma, theta\n1, 2, 3\n1, 2, 3\n", ss.str());
}

TEST_F(StanInterfaceCallbacksCsvWriter, header_only) {
  EXPECT_EQ("", ss.str());
  std::vector<std::string> header = {"mu", "sigma", "theta"};
  writer.write_header(header);
  EXPECT_EQ("mu, sigma, theta\n", ss.str());
}

TEST_F(StanInterfaceCallbacksCsvWriter, pad_row) {
  std::vector<std::string> header = {"mu", "sigma", "theta", "zeta"};
  std::vector<double> values = {1, 2, 3};
  writer.write_header(header);
  writer.write_flat_padded(values);
  EXPECT_EQ("mu, sigma, theta, zeta\n1, 2, 3, 0\n", ss.str());
}

TEST_F(StanInterfaceCallbacksCsvWriter, bad_row) {
  std::vector<std::string> header = {"mu", "sigma"};
  std::vector<double> values = {1, 2, 3};
  writer.write_header(header);
  EXPECT_THROW(writer.write_flat(values), std::domain_error);
  EXPECT_THROW(writer.write_flat_padded(values), std::domain_error);
}


TEST_F(StanInterfaceCallbacksCsvWriter, bad_2_header_rows) {
  std::vector<std::string> header = {"mu", "sigma", "theta"};
  writer.write_header(header);
  EXPECT_THROW(writer.write_header(header), std::domain_error);
}

TEST_F(StanInterfaceCallbacksCsvWriter, bad_empty_header) {
  std::vector<std::string> header = {};
  EXPECT_THROW(writer.write_header(header), std::domain_error);
}

TEST_F(StanInterfaceCallbacksCsvWriter, bad_no_header) {
  std::vector<double> values = {1, 2, 3};
  EXPECT_THROW(writer.write_flat(values), std::domain_error);
}

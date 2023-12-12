#include <stan/callbacks/csv_writer.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <string>

struct deleter_noop {
  template <typename T>
  constexpr void operator()(T* arg) const {}
};
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

TEST_F(StanInterfaceCallbacksCsvWriter, header_row) {
  std::vector<std::string> header = {"mu", "sigma", "theta"};
  std::vector<double> values = {1, 2, 3};
  writer.table_header(header);
  writer.table_row(values);
  EXPECT_EQ("mu, sigma, theta\n1, 2, 3\n", ss.str());
}

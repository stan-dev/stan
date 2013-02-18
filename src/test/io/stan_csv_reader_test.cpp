#include <stan/io/stan_csv_reader.hpp>

#include <gtest/gtest.h>
#include <fstream>

class StanIoStanCsvReader : public testing::Test {
  
public:
  void SetUp () {
    blocker0.open("src/test/io/test_csv_files/blocker.0.csv");
  }

  void TearDown() {
    blocker0.close();
  }

  std::ifstream blocker0;
};

TEST_F(StanIoStanCsvReader,DISABLED_Parse) {

}

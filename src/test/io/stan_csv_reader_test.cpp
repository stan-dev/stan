#include <stan/io/stan_csv_reader.hpp>

#include <gtest/gtest.h>
#include <fstream>

class StanIoStanCsvReader : public testing::Test {
  
public:
  void SetUp () {
    blocker0.open("src/test/io/test_csv_files/blocker.0.csv");
    metadata.open("src/test/io/test_csv_files/metadata.csv");
    header.open("src/test/io/test_csv_files/header.csv");
    adaptation.open("src/test/io/test_csv_files/adaptation.csv");
    samples.open("src/test/io/test_csv_files/samples.csv");
  }

  void TearDown() {
    blocker0.close();
  }

  std::ifstream blocker0;
  std::ifstream metadata, header, adaptation, samples;
};

TEST_F(StanIoStanCsvReader,DISABLED_read_metadata) {
  stan::io::stan_csv_reader reader(metadata);
  reader.read_metadata();
}
TEST_F(StanIoStanCsvReader,DISABLED_read_header) { 
  stan::io::stan_csv_reader reader(header);
  reader.read_header();

}
TEST_F(StanIoStanCsvReader,DISABLED_read_adaptation) { 
  stan::io::stan_csv_reader reader(adaptation);
  reader.read_adaptation();

}
TEST_F(StanIoStanCsvReader,DISABLED_read_samples) { 
  stan::io::stan_csv_reader reader(samples);
  reader.read_samples();
}
TEST_F(StanIoStanCsvReader,DISABLED_Parse) {
  stan::io::stan_csv_reader reader(blocker0);
  reader.parse();
}

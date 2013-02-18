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
    metadata.close();
    header.close();
    adaptation.close();
    samples.close();
  }

  std::ifstream blocker0;
  std::ifstream metadata, header, adaptation, samples;
};

TEST_F(StanIoStanCsvReader,read_metadata) {
  stan::io::stan_csv_reader reader(metadata);
  reader.read_metadata();
  
  stan::io::stan_csv_metadata metadata = reader.metadata();
  EXPECT_EQ(1U, metadata.stan_version_major);
  EXPECT_EQ(1U, metadata.stan_version_minor);
  EXPECT_EQ(1U, metadata.stan_version_patch);
  
  EXPECT_EQ("models\\bugs_examples\\vol1\\blocker\\blocker.data.R", metadata.data);
  EXPECT_EQ("models\\bugs_examples\\vol1\\blocker\\blocker.init.R", metadata.init);
  EXPECT_EQ(false, metadata.append_samples);
  EXPECT_EQ(false, metadata.save_warmup);
  EXPECT_EQ(4085885484U, metadata.seed);
  EXPECT_EQ(false, metadata.random_seed);
  EXPECT_EQ(0U, metadata.chain_id);
  EXPECT_EQ(4000U, metadata.iter);
  EXPECT_EQ(2000U, metadata.warmup);
  EXPECT_EQ(2, metadata.thin);
  EXPECT_EQ(false, metadata.equal_step_sizes);
  EXPECT_EQ(-1, metadata.leapfrog_steps);
  EXPECT_EQ(10, metadata.max_treedepth);
  EXPECT_FLOAT_EQ(-1, metadata.epsilon);
  EXPECT_FLOAT_EQ(0, metadata.epsilon_pm);
  EXPECT_FLOAT_EQ(0.5, metadata.delta);
  EXPECT_FLOAT_EQ(0.05, metadata.gamma);
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

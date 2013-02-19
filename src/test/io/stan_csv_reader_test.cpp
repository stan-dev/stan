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
  EXPECT_TRUE(reader.read_metadata());
  
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
TEST_F(StanIoStanCsvReader,read_header) { 
  stan::io::stan_csv_reader reader(header);
  EXPECT_TRUE(reader.read_header());
  
  Eigen::Matrix<std::string, Eigen::Dynamic, 1> header = reader.header();
  ASSERT_EQ(51, header.size());
  EXPECT_EQ("lp__", header(0));
  EXPECT_EQ("treedepth__",header(1));
  EXPECT_EQ("stepsize__",header(2));
  EXPECT_EQ("d",header(3));
  EXPECT_EQ("sigmasq_delta",header(4));
  EXPECT_EQ("mu[1]",header(5));
  EXPECT_EQ("mu[2]",header(6));
  EXPECT_EQ("mu[3]",header(7));
  EXPECT_EQ("mu[4]",header(8));
  EXPECT_EQ("mu[5]",header(9));
  EXPECT_EQ("mu[6]",header(10));
  EXPECT_EQ("mu[7]",header(11));
  EXPECT_EQ("mu[8]",header(12));
  EXPECT_EQ("mu[9]",header(13));
  EXPECT_EQ("mu[10]",header(14));
  EXPECT_EQ("mu[11]",header(15));
  EXPECT_EQ("mu[12]",header(16));
  EXPECT_EQ("mu[13]",header(17));
  EXPECT_EQ("mu[14]",header(18));
  EXPECT_EQ("mu[15]",header(19));
  EXPECT_EQ("mu[16]",header(20));
  EXPECT_EQ("mu[17]",header(21));
  EXPECT_EQ("mu[18]",header(22));
  EXPECT_EQ("mu[19]",header(23));
  EXPECT_EQ("mu[20]",header(24));
  EXPECT_EQ("mu[21]",header(25));
  EXPECT_EQ("mu[22]",header(26));
  EXPECT_EQ("delta[1]",header(27));
  EXPECT_EQ("delta[2]",header(28));
  EXPECT_EQ("delta[3]",header(29));
  EXPECT_EQ("delta[4]",header(30));
  EXPECT_EQ("delta[5]",header(31));
  EXPECT_EQ("delta[6]",header(32));
  EXPECT_EQ("delta[7]",header(33));
  EXPECT_EQ("delta[8]",header(34));
  EXPECT_EQ("delta[9]",header(35));
  EXPECT_EQ("delta[10]",header(36));
  EXPECT_EQ("delta[11]",header(37));
  EXPECT_EQ("delta[12]",header(38));
  EXPECT_EQ("delta[13]",header(39));
  EXPECT_EQ("delta[14]",header(40));
  EXPECT_EQ("delta[15]",header(41));
  EXPECT_EQ("delta[16]",header(42));
  EXPECT_EQ("delta[17]",header(43));
  EXPECT_EQ("delta[18]",header(44));
  EXPECT_EQ("delta[19]",header(45));
  EXPECT_EQ("delta[20]",header(46));
  EXPECT_EQ("delta[21]",header(47));
  EXPECT_EQ("delta[22]",header(48));
  EXPECT_EQ("delta_new",header(49));
  EXPECT_EQ("sigma_delta",header(50));
}

TEST_F(StanIoStanCsvReader,read_adaptation) { 
  stan::io::stan_csv_reader reader(adaptation);
  EXPECT_TRUE(reader.read_adaptation());
  
  stan::io::stan_csv_adaptation adaptation = reader.adaptation();

  EXPECT_EQ("mcmc::nuts_diag", adaptation.sampler);
  EXPECT_FLOAT_EQ(0.104225, adaptation.step_size);
  ASSERT_EQ(47, adaptation.step_size_multipliers.size());
  EXPECT_FLOAT_EQ(0.325114, adaptation.step_size_multipliers(0));
  EXPECT_FLOAT_EQ(3.51248, adaptation.step_size_multipliers(1));
  EXPECT_FLOAT_EQ(1.93143, adaptation.step_size_multipliers(2));
  EXPECT_FLOAT_EQ(1.08468, adaptation.step_size_multipliers(3));
  EXPECT_FLOAT_EQ(1.23781, adaptation.step_size_multipliers(4));
  EXPECT_FLOAT_EQ(0.365398, adaptation.step_size_multipliers(5));
  EXPECT_FLOAT_EQ(0.710215, adaptation.step_size_multipliers(6));
  EXPECT_FLOAT_EQ(1.54178, adaptation.step_size_multipliers(7));
  EXPECT_FLOAT_EQ(0.375339, adaptation.step_size_multipliers(8));
  EXPECT_FLOAT_EQ(0.574698, adaptation.step_size_multipliers(9));
  EXPECT_FLOAT_EQ(0.722729, adaptation.step_size_multipliers(10));
  EXPECT_FLOAT_EQ(0.318383, adaptation.step_size_multipliers(11));
  EXPECT_FLOAT_EQ(0.563756, adaptation.step_size_multipliers(12));
  EXPECT_FLOAT_EQ(0.616147, adaptation.step_size_multipliers(13));
  EXPECT_FLOAT_EQ(0.995688, adaptation.step_size_multipliers(14));
  EXPECT_FLOAT_EQ(0.657117, adaptation.step_size_multipliers(15));
  EXPECT_FLOAT_EQ(0.7757, adaptation.step_size_multipliers(16));
  EXPECT_FLOAT_EQ(0.689116, adaptation.step_size_multipliers(17));
  EXPECT_FLOAT_EQ(0.890119, adaptation.step_size_multipliers(18));
  EXPECT_FLOAT_EQ(1.29209, adaptation.step_size_multipliers(19));
  EXPECT_FLOAT_EQ(1.66928, adaptation.step_size_multipliers(20));
  EXPECT_FLOAT_EQ(0.653107, adaptation.step_size_multipliers(21));
  EXPECT_FLOAT_EQ(0.692784, adaptation.step_size_multipliers(22));
  EXPECT_FLOAT_EQ(0.647013, adaptation.step_size_multipliers(23));
  EXPECT_FLOAT_EQ(0.908997, adaptation.step_size_multipliers(24));
  EXPECT_FLOAT_EQ(0.965595, adaptation.step_size_multipliers(25));
  EXPECT_FLOAT_EQ(0.918005, adaptation.step_size_multipliers(26));
  EXPECT_FLOAT_EQ(0.444468, adaptation.step_size_multipliers(27));
  EXPECT_FLOAT_EQ(0.703633, adaptation.step_size_multipliers(28));
  EXPECT_FLOAT_EQ(0.917533, adaptation.step_size_multipliers(29));
  EXPECT_FLOAT_EQ(0.533222, adaptation.step_size_multipliers(30));
  EXPECT_FLOAT_EQ(0.613215, adaptation.step_size_multipliers(31));
  EXPECT_FLOAT_EQ(0.728461, adaptation.step_size_multipliers(32));
  EXPECT_FLOAT_EQ(0.420359, adaptation.step_size_multipliers(33));
  EXPECT_FLOAT_EQ(0.608283, adaptation.step_size_multipliers(34));
  EXPECT_FLOAT_EQ(0.692184, adaptation.step_size_multipliers(35));
  EXPECT_FLOAT_EQ(0.817112, adaptation.step_size_multipliers(36));
  EXPECT_FLOAT_EQ(0.86689, adaptation.step_size_multipliers(37));
  EXPECT_FLOAT_EQ(0.694276, adaptation.step_size_multipliers(38));
  EXPECT_FLOAT_EQ(0.676297, adaptation.step_size_multipliers(39));
  EXPECT_FLOAT_EQ(0.788411, adaptation.step_size_multipliers(40));
  EXPECT_FLOAT_EQ(0.983241, adaptation.step_size_multipliers(41));
  EXPECT_FLOAT_EQ(0.873982, adaptation.step_size_multipliers(42));
  EXPECT_FLOAT_EQ(0.646372, adaptation.step_size_multipliers(43));
  EXPECT_FLOAT_EQ(0.718841, adaptation.step_size_multipliers(44));
  EXPECT_FLOAT_EQ(0.667207, adaptation.step_size_multipliers(45));
  EXPECT_FLOAT_EQ(1.33936, adaptation.step_size_multipliers(46));
}

TEST_F(StanIoStanCsvReader,read_samples) { 
  stan::io::stan_csv_reader reader(samples);
  EXPECT_TRUE(reader.read_samples());
  
  
}
TEST_F(StanIoStanCsvReader,DISABLED_Parse) {
  stan::io::stan_csv_reader reader(blocker0);
  reader.parse();
}

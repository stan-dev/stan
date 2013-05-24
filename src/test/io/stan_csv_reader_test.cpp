#include <stan/io/stan_csv_reader.hpp>

#include <gtest/gtest.h>
#include <fstream>

class StanIoStanCsvReader : public testing::Test {
  
public:
  void SetUp () {
    blocker0_stream.open("src/test/io/test_csv_files/blocker.0.csv");
    metadata1_stream.open("src/test/io/test_csv_files/metadata1.csv");
    header1_stream.open("src/test/io/test_csv_files/header1.csv");
    adaptation1_stream.open("src/test/io/test_csv_files/adaptation1.csv");
    samples1_stream.open("src/test/io/test_csv_files/samples1.csv");

    epil0_stream.open("src/test/io/test_csv_files/epil.0.csv");
    metadata2_stream.open("src/test/io/test_csv_files/metadata2.csv");
    header2_stream.open("src/test/io/test_csv_files/header2.csv");
    adaptation2_stream.open("src/test/io/test_csv_files/adaptation2.csv");
    samples2_stream.open("src/test/io/test_csv_files/samples2.csv");

    blocker_nondiag0_stream.open("src/test/io/test_csv_files/blocker_nondiag.0.csv");
  }

  void TearDown() {
    blocker0_stream.close();
    metadata1_stream.close();
    header1_stream.close();
    adaptation1_stream.close();
    samples1_stream.close();

    epil0_stream.close();
    metadata2_stream.close();
    header2_stream.close();
    adaptation2_stream.close();
    samples2_stream.close();

    blocker_nondiag0_stream.close();
  }

  std::ifstream blocker0_stream, epil0_stream;
  std::ifstream blocker_nondiag0_stream;
  std::ifstream metadata1_stream, header1_stream, adaptation1_stream, samples1_stream;
  std::ifstream metadata2_stream, header2_stream, adaptation2_stream, samples2_stream;
};

TEST_F(StanIoStanCsvReader,read_metadata1) {
  stan::io::stan_csv_metadata metadata;
  EXPECT_TRUE(stan::io::stan_csv_reader::read_metadata(metadata1_stream, metadata));

  EXPECT_EQ(1, metadata.stan_version_major);
  EXPECT_EQ(3, metadata.stan_version_minor);
  EXPECT_EQ(0, metadata.stan_version_patch);
  
  EXPECT_EQ("src/models/bugs_examples/vol1/blocker/blocker.data.R", metadata.data);
  EXPECT_EQ("src/models/bugs_examples/vol1/blocker/blocker.init.R", metadata.init);
  EXPECT_FALSE(metadata.append_samples);
  EXPECT_FALSE(metadata.save_warmup);
  EXPECT_EQ(4085885484U, metadata.seed);
  EXPECT_FALSE(metadata.random_seed);
  EXPECT_EQ(1U, metadata.chain_id);
  EXPECT_EQ(4000U, metadata.iter);
  EXPECT_EQ(2000U, metadata.warmup);
  EXPECT_EQ(2U, metadata.thin);
  EXPECT_FALSE(metadata.equal_step_sizes);
  EXPECT_EQ(-1, metadata.leapfrog_steps);
  EXPECT_EQ(10, metadata.max_treedepth);
  EXPECT_FLOAT_EQ(-1, metadata.epsilon);
  EXPECT_FLOAT_EQ(0, metadata.epsilon_pm);
  EXPECT_FLOAT_EQ(0.5, metadata.delta);
  EXPECT_FLOAT_EQ(0.05, metadata.gamma);
}
TEST_F(StanIoStanCsvReader,read_header1) { 
  Eigen::Matrix<std::string, Eigen::Dynamic, 1> header;
  EXPECT_TRUE(stan::io::stan_csv_reader::read_header(header1_stream, header));
  
  ASSERT_EQ(51, header.size());
  EXPECT_EQ("lp__", header(0));
  EXPECT_EQ("stepsize__",header(1));
  EXPECT_EQ("depth__",header(2));
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

TEST_F(StanIoStanCsvReader,read_adaptation1) { 
  stan::io::stan_csv_adaptation adaptation;
  EXPECT_TRUE(stan::io::stan_csv_reader::read_adaptation(adaptation1_stream, adaptation));
  
  EXPECT_EQ("NUTS with a diagonal Euclidean metric", adaptation.sampler);
  EXPECT_FLOAT_EQ(0.528659, adaptation.step_size);
  ASSERT_EQ(47, adaptation.step_size_multipliers.size());
  EXPECT_FLOAT_EQ(0.0127071, adaptation.step_size_multipliers(0));
  EXPECT_FLOAT_EQ(1.18497, adaptation.step_size_multipliers(1));
  EXPECT_FLOAT_EQ(0.150737, adaptation.step_size_multipliers(2));
  EXPECT_FLOAT_EQ(0.0578019, adaptation.step_size_multipliers(3));
  EXPECT_FLOAT_EQ(0.105259, adaptation.step_size_multipliers(4));
  EXPECT_FLOAT_EQ(0.0150945, adaptation.step_size_multipliers(5));
  EXPECT_FLOAT_EQ(0.0262535, adaptation.step_size_multipliers(6));
  EXPECT_FLOAT_EQ(0.135112, adaptation.step_size_multipliers(7));
  EXPECT_FLOAT_EQ(0.0131548, adaptation.step_size_multipliers(8));
  EXPECT_FLOAT_EQ(0.0177974, adaptation.step_size_multipliers(9));
  EXPECT_FLOAT_EQ(0.0279516, adaptation.step_size_multipliers(10));
  EXPECT_FLOAT_EQ(0.0119575, adaptation.step_size_multipliers(11));
  EXPECT_FLOAT_EQ(0.0252945, adaptation.step_size_multipliers(12));
  EXPECT_FLOAT_EQ(0.0218081, adaptation.step_size_multipliers(13));
  EXPECT_FLOAT_EQ(0.0498465, adaptation.step_size_multipliers(14));
  EXPECT_FLOAT_EQ(0.022811, adaptation.step_size_multipliers(15));
  EXPECT_FLOAT_EQ(0.0364815, adaptation.step_size_multipliers(16));
  EXPECT_FLOAT_EQ(0.0268602, adaptation.step_size_multipliers(17));
  EXPECT_FLOAT_EQ(0.0495377, adaptation.step_size_multipliers(18));
  EXPECT_FLOAT_EQ(0.0704462, adaptation.step_size_multipliers(19));
  EXPECT_FLOAT_EQ(0.137418, adaptation.step_size_multipliers(20));
  EXPECT_FLOAT_EQ(0.0247963, adaptation.step_size_multipliers(21));
  EXPECT_FLOAT_EQ(0.0359937, adaptation.step_size_multipliers(22));
  EXPECT_FLOAT_EQ(0.0277964, adaptation.step_size_multipliers(23));
  EXPECT_FLOAT_EQ(0.0614465, adaptation.step_size_multipliers(24));
  EXPECT_FLOAT_EQ(0.0380137, adaptation.step_size_multipliers(25));
  EXPECT_FLOAT_EQ(0.0307976, adaptation.step_size_multipliers(26));
  EXPECT_FLOAT_EQ(0.0155192, adaptation.step_size_multipliers(27));
  EXPECT_FLOAT_EQ(0.0257965, adaptation.step_size_multipliers(28));
  EXPECT_FLOAT_EQ(0.0314832, adaptation.step_size_multipliers(29));
  EXPECT_FLOAT_EQ(0.0174462, adaptation.step_size_multipliers(30));
  EXPECT_FLOAT_EQ(0.0197627, adaptation.step_size_multipliers(31));
  EXPECT_FLOAT_EQ(0.0230003, adaptation.step_size_multipliers(32));
  EXPECT_FLOAT_EQ(0.0145179, adaptation.step_size_multipliers(33));
  EXPECT_FLOAT_EQ(0.0207101, adaptation.step_size_multipliers(34));
  EXPECT_FLOAT_EQ(0.026243, adaptation.step_size_multipliers(35));
  EXPECT_FLOAT_EQ(0.0289608, adaptation.step_size_multipliers(36));
  EXPECT_FLOAT_EQ(0.0335959, adaptation.step_size_multipliers(37));
  EXPECT_FLOAT_EQ(0.0236457, adaptation.step_size_multipliers(38));
  EXPECT_FLOAT_EQ(0.026272, adaptation.step_size_multipliers(39));
  EXPECT_FLOAT_EQ(0.0276585, adaptation.step_size_multipliers(40));
  EXPECT_FLOAT_EQ(0.0437609, adaptation.step_size_multipliers(41));
  EXPECT_FLOAT_EQ(0.0404702, adaptation.step_size_multipliers(42));
  EXPECT_FLOAT_EQ(0.0230456, adaptation.step_size_multipliers(43));
  EXPECT_FLOAT_EQ(0.0334405, adaptation.step_size_multipliers(44));
  EXPECT_FLOAT_EQ(0.0236958, adaptation.step_size_multipliers(45));
  EXPECT_FLOAT_EQ(0.0345349, adaptation.step_size_multipliers(46));
}

TEST_F(StanIoStanCsvReader,read_samples1) { 
  Eigen::MatrixXd samples;
  EXPECT_TRUE(stan::io::stan_csv_reader::read_samples(samples1_stream, samples));

  ASSERT_EQ(5, samples.rows());
  ASSERT_EQ(51, samples.cols());

  
  Eigen::MatrixXd expected_samples(5, 51);
  expected_samples <<
  -5920.56,0.528659,3,-0.126865,0.0171491,-2.1068,-1.90301,-2.4257,-2.42028,-2.37917,-2.78062,-1.7215,-2.01923,-1.9047,-2.17854,-2.3345,-1.27208,-2.80227,-2.63606,-1.20054,-1.45414,-2.1107,-3.39025,-3.6262,-1.51217,-2.32518,-3.17131,-0.0496257,-0.541073,-0.194383,-0.136216,-0.0355437,-0.109111,-0.40807,-0.0657383,-0.071448,-0.318889,-0.0375361,-0.250442,-0.267267,-0.143611,-0.264086,-0.114841,-0.0775461,-0.227377,-0.0362438,-0.233862,-0.155884,-0.115541,-0.10498,0.130955,
  -5920.56,0.528659,3,-0.126865,0.0171491,-2.1068,-1.90301,-2.4257,-2.42028,-2.37917,-2.78062,-1.7215,-2.01923,-1.9047,-2.17854,-2.3345,-1.27208,-2.80227,-2.63606,-1.20054,-1.45414,-2.1107,-3.39025,-3.6262,-1.51217,-2.32518,-3.17131,-0.0496257,-0.541073,-0.194383,-0.136216,-0.0355437,-0.109111,-0.40807,-0.0657383,-0.071448,-0.318889,-0.0375361,-0.250442,-0.267267,-0.143611,-0.264086,-0.114841,-0.0775461,-0.227377,-0.0362438,-0.233862,-0.155884,-0.115541,-0.10498,0.130955,
  -5927.27,0.528659,3,-0.183497,0.0263908,-2.65162,-2.29548,-1.81794,-2.27987,-2.31512,-1.71623,-1.69302,-2.11874,-1.76093,-2.22578,-2.13069,-1.42279,-2.67964,-2.98838,-1.37119,-1.56465,-1.82055,-2.59382,-3.46723,-1.42249,-1.98014,-2.68506,-0.0973822,-0.170706,-0.328805,-0.294757,-0.185382,-0.308282,-0.442633,-0.0208355,-0.351909,-0.276284,-0.267614,0.00879373,-1.03867,0.394958,-0.111359,-0.133448,-0.176326,-0.0361666,-0.171098,-0.122819,-0.583553,-0.607785,-0.14541,0.162452,
  -5927.27,0.528659,2,-0.183497,0.0263908,-2.65162,-2.29548,-1.81794,-2.27987,-2.31512,-1.71623,-1.69302,-2.11874,-1.76093,-2.22578,-2.13069,-1.42279,-2.67964,-2.98838,-1.37119,-1.56465,-1.82055,-2.59382,-3.46723,-1.42249,-1.98014,-2.68506,-0.0973822,-0.170706,-0.328805,-0.294757,-0.185382,-0.308282,-0.442633,-0.0208355,-0.351909,-0.276284,-0.267614,0.00879373,-1.03867,0.394958,-0.111359,-0.133448,-0.176326,-0.0361666,-0.171098,-0.122819,-0.583553,-0.607785,-0.14541,0.162452,
  -5929.67,0.528659,3,-0.264279,0.0346307,-2.68101,-2.49765,-1.89686,-2.49734,-2.47776,-2.33716,-1.51465,-2.03101,-2.00187,-2.22189,-2.47179,-1.43117,-2.84742,-2.99686,-1.25123,-1.46619,-2.20088,-2.95166,-3.0407,-1.44049,-2.24141,-2.78273,-0.486993,-0.115857,-0.259522,-0.209478,-0.0701565,-0.154348,-0.635436,-0.215462,-0.162989,-0.295247,-0.382356,-0.353868,-0.906204,0.391292,-0.117671,-0.175015,-0.00656938,-0.302936,-0.248865,-0.290444,-0.364104,-0.824764,-0.101581,0.186093;
  
  for (int i = 0; i < 5; i++) 
    for (int j = 0; j < 51; j++)
      EXPECT_FLOAT_EQ(expected_samples(i,j), samples(i,j))
  << "comparison failed for (" << i << "," << j << ")";
}

TEST_F(StanIoStanCsvReader,ParseBlocker) {
  stan::io::stan_csv blocker0;
  blocker0 = stan::io::stan_csv_reader::parse(blocker0_stream);

  // metadata
  EXPECT_EQ(1, blocker0.metadata.stan_version_major);
  EXPECT_EQ(3, blocker0.metadata.stan_version_minor);
  EXPECT_EQ(0, blocker0.metadata.stan_version_patch);
  
  EXPECT_EQ("src/models/bugs_examples/vol1/blocker/blocker.data.R", blocker0.metadata.data);
  EXPECT_EQ("src/models/bugs_examples/vol1/blocker/blocker.init.R", blocker0.metadata.init);
  EXPECT_FALSE(blocker0.metadata.append_samples);
  EXPECT_FALSE(blocker0.metadata.save_warmup);
  EXPECT_EQ(4085885484U, blocker0.metadata.seed);
  EXPECT_FALSE(blocker0.metadata.random_seed);
  EXPECT_EQ(1U, blocker0.metadata.chain_id);
  EXPECT_EQ(4000U, blocker0.metadata.iter);
  EXPECT_EQ(2000U, blocker0.metadata.warmup);
  EXPECT_EQ(2U, blocker0.metadata.thin);
  EXPECT_FALSE(blocker0.metadata.equal_step_sizes);
  EXPECT_EQ(-1, blocker0.metadata.leapfrog_steps);
  EXPECT_EQ(10, blocker0.metadata.max_treedepth);
  EXPECT_FLOAT_EQ(-1, blocker0.metadata.epsilon);
  EXPECT_FLOAT_EQ(0, blocker0.metadata.epsilon_pm);
  EXPECT_FLOAT_EQ(0.5, blocker0.metadata.delta);
  EXPECT_FLOAT_EQ(0.05, blocker0.metadata.gamma);

  // header
  ASSERT_EQ(51, blocker0.header.size());
  EXPECT_EQ("lp__", blocker0.header(0));
  EXPECT_EQ("stepsize__",blocker0.header(1));
  EXPECT_EQ("depth__",blocker0.header(2));
  EXPECT_EQ("d",blocker0.header(3));
  EXPECT_EQ("sigmasq_delta",blocker0.header(4));
  EXPECT_EQ("mu[1]",blocker0.header(5));
  EXPECT_EQ("mu[2]",blocker0.header(6));
  EXPECT_EQ("mu[3]",blocker0.header(7));
  EXPECT_EQ("mu[4]",blocker0.header(8));
  EXPECT_EQ("mu[5]",blocker0.header(9));
  EXPECT_EQ("mu[6]",blocker0.header(10));
  EXPECT_EQ("mu[7]",blocker0.header(11));
  EXPECT_EQ("mu[8]",blocker0.header(12));
  EXPECT_EQ("mu[9]",blocker0.header(13));
  EXPECT_EQ("mu[10]",blocker0.header(14));
  EXPECT_EQ("mu[11]",blocker0.header(15));
  EXPECT_EQ("mu[12]",blocker0.header(16));
  EXPECT_EQ("mu[13]",blocker0.header(17));
  EXPECT_EQ("mu[14]",blocker0.header(18));
  EXPECT_EQ("mu[15]",blocker0.header(19));
  EXPECT_EQ("mu[16]",blocker0.header(20));
  EXPECT_EQ("mu[17]",blocker0.header(21));
  EXPECT_EQ("mu[18]",blocker0.header(22));
  EXPECT_EQ("mu[19]",blocker0.header(23));
  EXPECT_EQ("mu[20]",blocker0.header(24));
  EXPECT_EQ("mu[21]",blocker0.header(25));
  EXPECT_EQ("mu[22]",blocker0.header(26));
  EXPECT_EQ("delta[1]",blocker0.header(27));
  EXPECT_EQ("delta[2]",blocker0.header(28));
  EXPECT_EQ("delta[3]",blocker0.header(29));
  EXPECT_EQ("delta[4]",blocker0.header(30));
  EXPECT_EQ("delta[5]",blocker0.header(31));
  EXPECT_EQ("delta[6]",blocker0.header(32));
  EXPECT_EQ("delta[7]",blocker0.header(33));
  EXPECT_EQ("delta[8]",blocker0.header(34));
  EXPECT_EQ("delta[9]",blocker0.header(35));
  EXPECT_EQ("delta[10]",blocker0.header(36));
  EXPECT_EQ("delta[11]",blocker0.header(37));
  EXPECT_EQ("delta[12]",blocker0.header(38));
  EXPECT_EQ("delta[13]",blocker0.header(39));
  EXPECT_EQ("delta[14]",blocker0.header(40));
  EXPECT_EQ("delta[15]",blocker0.header(41));
  EXPECT_EQ("delta[16]",blocker0.header(42));
  EXPECT_EQ("delta[17]",blocker0.header(43));
  EXPECT_EQ("delta[18]",blocker0.header(44));
  EXPECT_EQ("delta[19]",blocker0.header(45));
  EXPECT_EQ("delta[20]",blocker0.header(46));
  EXPECT_EQ("delta[21]",blocker0.header(47));
  EXPECT_EQ("delta[22]",blocker0.header(48));
  EXPECT_EQ("delta_new",blocker0.header(49));
  EXPECT_EQ("sigma_delta",blocker0.header(50));

  // adaptation
  EXPECT_EQ("NUTS with a diagonal Euclidean metric", blocker0.adaptation.sampler);
  EXPECT_FLOAT_EQ(0.528659, blocker0.adaptation.step_size);
  ASSERT_EQ(47, blocker0.adaptation.step_size_multipliers.size());
  EXPECT_FLOAT_EQ(0.0127071, blocker0.adaptation.step_size_multipliers(0));
  EXPECT_FLOAT_EQ(1.18497, blocker0.adaptation.step_size_multipliers(1));
  EXPECT_FLOAT_EQ(0.150737, blocker0.adaptation.step_size_multipliers(2));
  EXPECT_FLOAT_EQ(0.0578019, blocker0.adaptation.step_size_multipliers(3));
  EXPECT_FLOAT_EQ(0.105259, blocker0.adaptation.step_size_multipliers(4));
  EXPECT_FLOAT_EQ(0.0150945, blocker0.adaptation.step_size_multipliers(5));
  EXPECT_FLOAT_EQ(0.0262535, blocker0.adaptation.step_size_multipliers(6));
  EXPECT_FLOAT_EQ(0.135112, blocker0.adaptation.step_size_multipliers(7));
  EXPECT_FLOAT_EQ(0.0131548, blocker0.adaptation.step_size_multipliers(8));
  EXPECT_FLOAT_EQ(0.0177974, blocker0.adaptation.step_size_multipliers(9));
  EXPECT_FLOAT_EQ(0.0279516, blocker0.adaptation.step_size_multipliers(10));
  EXPECT_FLOAT_EQ(0.0119575, blocker0.adaptation.step_size_multipliers(11));
  EXPECT_FLOAT_EQ(0.0252945, blocker0.adaptation.step_size_multipliers(12));
  EXPECT_FLOAT_EQ(0.0218081, blocker0.adaptation.step_size_multipliers(13));
  EXPECT_FLOAT_EQ(0.0498465, blocker0.adaptation.step_size_multipliers(14));
  EXPECT_FLOAT_EQ(0.022811, blocker0.adaptation.step_size_multipliers(15));
  EXPECT_FLOAT_EQ(0.0364815, blocker0.adaptation.step_size_multipliers(16));
  EXPECT_FLOAT_EQ(0.0268602, blocker0.adaptation.step_size_multipliers(17));
  EXPECT_FLOAT_EQ(0.0495377, blocker0.adaptation.step_size_multipliers(18));
  EXPECT_FLOAT_EQ(0.0704462, blocker0.adaptation.step_size_multipliers(19));
  EXPECT_FLOAT_EQ(0.137418, blocker0.adaptation.step_size_multipliers(20));
  EXPECT_FLOAT_EQ(0.0247963, blocker0.adaptation.step_size_multipliers(21));
  EXPECT_FLOAT_EQ(0.0359937, blocker0.adaptation.step_size_multipliers(22));
  EXPECT_FLOAT_EQ(0.0277964, blocker0.adaptation.step_size_multipliers(23));
  EXPECT_FLOAT_EQ(0.0614465, blocker0.adaptation.step_size_multipliers(24));
  EXPECT_FLOAT_EQ(0.0380137, blocker0.adaptation.step_size_multipliers(25));
  EXPECT_FLOAT_EQ(0.0307976, blocker0.adaptation.step_size_multipliers(26));
  EXPECT_FLOAT_EQ(0.0155192, blocker0.adaptation.step_size_multipliers(27));
  EXPECT_FLOAT_EQ(0.0257965, blocker0.adaptation.step_size_multipliers(28));
  EXPECT_FLOAT_EQ(0.0314832, blocker0.adaptation.step_size_multipliers(29));
  EXPECT_FLOAT_EQ(0.0174462, blocker0.adaptation.step_size_multipliers(30));
  EXPECT_FLOAT_EQ(0.0197627, blocker0.adaptation.step_size_multipliers(31));
  EXPECT_FLOAT_EQ(0.0230003, blocker0.adaptation.step_size_multipliers(32));
  EXPECT_FLOAT_EQ(0.0145179, blocker0.adaptation.step_size_multipliers(33));
  EXPECT_FLOAT_EQ(0.0207101, blocker0.adaptation.step_size_multipliers(34));
  EXPECT_FLOAT_EQ(0.026243, blocker0.adaptation.step_size_multipliers(35));
  EXPECT_FLOAT_EQ(0.0289608, blocker0.adaptation.step_size_multipliers(36));
  EXPECT_FLOAT_EQ(0.0335959, blocker0.adaptation.step_size_multipliers(37));
  EXPECT_FLOAT_EQ(0.0236457, blocker0.adaptation.step_size_multipliers(38));
  EXPECT_FLOAT_EQ(0.026272, blocker0.adaptation.step_size_multipliers(39));
  EXPECT_FLOAT_EQ(0.0276585, blocker0.adaptation.step_size_multipliers(40));
  EXPECT_FLOAT_EQ(0.0437609, blocker0.adaptation.step_size_multipliers(41));
  EXPECT_FLOAT_EQ(0.0404702, blocker0.adaptation.step_size_multipliers(42));
  EXPECT_FLOAT_EQ(0.0230456, blocker0.adaptation.step_size_multipliers(43));
  EXPECT_FLOAT_EQ(0.0334405, blocker0.adaptation.step_size_multipliers(44));
  EXPECT_FLOAT_EQ(0.0236958, blocker0.adaptation.step_size_multipliers(45));
  EXPECT_FLOAT_EQ(0.0345349, blocker0.adaptation.step_size_multipliers(46));
  
  // samples
  ASSERT_EQ(1000, blocker0.samples.rows());
  ASSERT_EQ(51, blocker0.samples.cols());

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> expected_samples(6, 51);
  expected_samples <<
  -5920.56,0.528659,3,-0.126865,0.0171491,-2.1068,-1.90301,-2.4257,-2.42028,-2.37917,-2.78062,-1.7215,-2.01923,-1.9047,-2.17854,-2.3345,-1.27208,-2.80227,-2.63606,-1.20054,-1.45414,-2.1107,-3.39025,-3.6262,-1.51217,-2.32518,-3.17131,-0.0496257,-0.541073,-0.194383,-0.136216,-0.0355437,-0.109111,-0.40807,-0.0657383,-0.071448,-0.318889,-0.0375361,-0.250442,-0.267267,-0.143611,-0.264086,-0.114841,-0.0775461,-0.227377,-0.0362438,-0.233862,-0.155884,-0.115541,-0.10498,0.130955,
  -5920.56,0.528659,3,-0.126865,0.0171491,-2.1068,-1.90301,-2.4257,-2.42028,-2.37917,-2.78062,-1.7215,-2.01923,-1.9047,-2.17854,-2.3345,-1.27208,-2.80227,-2.63606,-1.20054,-1.45414,-2.1107,-3.39025,-3.6262,-1.51217,-2.32518,-3.17131,-0.0496257,-0.541073,-0.194383,-0.136216,-0.0355437,-0.109111,-0.40807,-0.0657383,-0.071448,-0.318889,-0.0375361,-0.250442,-0.267267,-0.143611,-0.264086,-0.114841,-0.0775461,-0.227377,-0.0362438,-0.233862,-0.155884,-0.115541,-0.10498,0.130955,
  -5927.27,0.528659,3,-0.183497,0.0263908,-2.65162,-2.29548,-1.81794,-2.27987,-2.31512,-1.71623,-1.69302,-2.11874,-1.76093,-2.22578,-2.13069,-1.42279,-2.67964,-2.98838,-1.37119,-1.56465,-1.82055,-2.59382,-3.46723,-1.42249,-1.98014,-2.68506,-0.0973822,-0.170706,-0.328805,-0.294757,-0.185382,-0.308282,-0.442633,-0.0208355,-0.351909,-0.276284,-0.267614,0.00879373,-1.03867,0.394958,-0.111359,-0.133448,-0.176326,-0.0361666,-0.171098,-0.122819,-0.583553,-0.607785,-0.14541,0.162452,
  -5927.27,0.528659,2,-0.183497,0.0263908,-2.65162,-2.29548,-1.81794,-2.27987,-2.31512,-1.71623,-1.69302,-2.11874,-1.76093,-2.22578,-2.13069,-1.42279,-2.67964,-2.98838,-1.37119,-1.56465,-1.82055,-2.59382,-3.46723,-1.42249,-1.98014,-2.68506,-0.0973822,-0.170706,-0.328805,-0.294757,-0.185382,-0.308282,-0.442633,-0.0208355,-0.351909,-0.276284,-0.267614,0.00879373,-1.03867,0.394958,-0.111359,-0.133448,-0.176326,-0.0361666,-0.171098,-0.122819,-0.583553,-0.607785,-0.14541,0.162452,
  -5929.67,0.528659,3,-0.264279,0.0346307,-2.68101,-2.49765,-1.89686,-2.49734,-2.47776,-2.33716,-1.51465,-2.03101,-2.00187,-2.22189,-2.47179,-1.43117,-2.84742,-2.99686,-1.25123,-1.46619,-2.20088,-2.95166,-3.0407,-1.44049,-2.24141,-2.78273,-0.486993,-0.115857,-0.259522,-0.209478,-0.0701565,-0.154348,-0.635436,-0.215462,-0.162989,-0.295247,-0.382356,-0.353868,-0.906204,0.391292,-0.117671,-0.175015,-0.00656938,-0.302936,-0.248865,-0.290444,-0.364104,-0.824764,-0.101581,0.186093,
  -5924.89,0.528659,2,-0.19173,0.033674,-2.90768,-2.52024,-2.17333,-2.37579,-2.42717,-2.22547,-1.6139,-2.11519,-1.88145,-2.17443,-2.21292,-1.56056,-2.79641,-3.06299,-1.23863,-1.44349,-1.84912,-2.93344,-3.48824,-1.28341,-2.31082,-2.75587,0.00905893,-0.343741,-0.584244,-0.234466,-0.229939,-0.135327,-0.6485,-0.300543,-0.309818,-0.360624,-0.304393,-0.114683,-0.60295,0.404441,-0.112307,-0.286742,-0.192521,-0.222329,-0.227109,-0.308601,-0.156497,-0.697791,-0.242174,0.183505;
  
  for (int i = 0; i < 6; i++) 
    for (int j = 0; j < 51; j++)
      EXPECT_FLOAT_EQ(expected_samples(i,j), blocker0.samples(i,j))
  << "comparison failed for (" << i << "," << j << ")";
}

TEST_F(StanIoStanCsvReader,read_metadata2) {
  stan::io::stan_csv_metadata metadata;
  EXPECT_TRUE(stan::io::stan_csv_reader::read_metadata(metadata2_stream, metadata));

  EXPECT_EQ(1, metadata.stan_version_major);
  EXPECT_EQ(3, metadata.stan_version_minor);
  EXPECT_EQ(0, metadata.stan_version_patch);
  
  EXPECT_EQ("src/models/bugs_examples/vol1/epil/epil.data.R", metadata.data);
  EXPECT_EQ("random initialization", metadata.init);
  EXPECT_FALSE(metadata.append_samples);
  EXPECT_FALSE(metadata.save_warmup);
  EXPECT_EQ(4258844633U, metadata.seed);
  EXPECT_FALSE(metadata.random_seed);
  EXPECT_EQ(1U, metadata.chain_id);
  EXPECT_EQ(2000U, metadata.iter);
  EXPECT_EQ(1000U, metadata.warmup);
  EXPECT_EQ(1U, metadata.thin);
  EXPECT_FALSE(metadata.equal_step_sizes);
  EXPECT_EQ(-1, metadata.leapfrog_steps);
  EXPECT_EQ(10, metadata.max_treedepth);
  EXPECT_FLOAT_EQ(-1, metadata.epsilon);
  EXPECT_FLOAT_EQ(0, metadata.epsilon_pm);
  EXPECT_FLOAT_EQ(0.5, metadata.delta);
  EXPECT_FLOAT_EQ(0.05, metadata.gamma);
}

TEST_F(StanIoStanCsvReader,read_header2) { 
  Eigen::Matrix<std::string, Eigen::Dynamic, 1> header;
  EXPECT_TRUE(stan::io::stan_csv_reader::read_header(header2_stream, header));
  
  ASSERT_EQ(309, header.size());
  EXPECT_EQ("lp__", header(0));
  EXPECT_EQ("stepsize__", header(1));
  EXPECT_EQ("depth__", header(2));
  EXPECT_EQ("a0", header(3));
  EXPECT_EQ("alpha_Base", header(4));
  EXPECT_EQ("alpha_Trt", header(5));
  EXPECT_EQ("alpha_BT", header(6));
  EXPECT_EQ("alpha_Age", header(7));
  EXPECT_EQ("alpha_V4", header(8));
  EXPECT_EQ("b1[1]", header(9));
  EXPECT_EQ("b1[2]", header(10));
  // ...
  EXPECT_EQ("b1[59]", header(67));
  EXPECT_EQ("b[1,1]", header(68));
  EXPECT_EQ("b[1,2]", header(69));
  EXPECT_EQ("b[1,3]", header(70));
  EXPECT_EQ("b[1,4]", header(71));
  EXPECT_EQ("b[2,1]", header(72));
  //...
  EXPECT_EQ("b[59,4]", header(303));
  EXPECT_EQ("sigmasq_b", header(304));
  EXPECT_EQ("sigmasq_b1", header(305));
  EXPECT_EQ("sigma_b", header(306));
  EXPECT_EQ("sigma_b1", header(307));
  EXPECT_EQ("alpha0", header(308));
}

TEST_F(StanIoStanCsvReader,read_adaptation2) { 
  stan::io::stan_csv_adaptation adaptation;
  EXPECT_TRUE(stan::io::stan_csv_reader::read_adaptation(adaptation2_stream, adaptation));
  
  EXPECT_EQ("NUTS with a diagonal Euclidean metric", adaptation.sampler);
  EXPECT_FLOAT_EQ(0.115015, adaptation.step_size);
  ASSERT_EQ(303, adaptation.step_size_multipliers.size());
  EXPECT_FLOAT_EQ(0.0219139, adaptation.step_size_multipliers(0));
  EXPECT_FLOAT_EQ(0.0356712, adaptation.step_size_multipliers(1));
  EXPECT_FLOAT_EQ(0.167986, adaptation.step_size_multipliers(2));
  // ...
  EXPECT_FLOAT_EQ(0.0679679, adaptation.step_size_multipliers(301));
  EXPECT_FLOAT_EQ(0.0899625, adaptation.step_size_multipliers(302));
}

TEST_F(StanIoStanCsvReader,read_samples2) { 
  Eigen::MatrixXd samples;
  EXPECT_TRUE(stan::io::stan_csv_reader::read_samples(samples2_stream, samples));

  ASSERT_EQ(3, samples.rows());
  ASSERT_EQ(309, samples.cols());

  Eigen::MatrixXd expected_samples(3, 309);
  expected_samples <<
  3413.09,0.115015,10,1.51484,1.01923,-0.621049,0.11056,-0.0691901,-0.294049,-0.243524,0.286699,-0.0531019,0.0263536,-0.603631,-0.0642254,0.0740205,0.530753,0.0344619,0.371297,-0.0897412,0.139706,-0.0395585,0.0255589,-0.493957,-1.0139,-1.4199,-0.00603114,-0.154987,-0.415264,0.662395,0.195613,0.0207539,0.209297,0.814185,-0.461153,0.230644,0.189312,-0.447868,-0.132626,-0.820277,0.384938,0.0630201,-0.387381,1.15444,0.0539652,0.732476,-0.863075,-0.08559,0.149062,-1.49257,0.515946,0.597099,0.0901956,0.400621,0.241568,0.0145009,-0.195782,0.737078,-0.00558925,0.0736978,-0.47623,0.30932,-0.32276,0.204327,1.39926,-0.43366,-1.05893,0.276954,0.125897,0.283081,0.186213,0.405639,-0.244769,0.493048,0.18552,-0.191094,-0.220089,0.549975,-0.200403,0.348472,0.0281709,-0.2602,-0.0637835,0.163992,0.204242,0.569281,-0.130111,0.575737,-0.143522,0.00102818,0.119173,-0.0484052,0.039245,-0.306881,-0.193284,-0.815046,0.492887,-0.0266505,0.120316,-0.269449,-0.184118,-0.569225,0.069866,-0.306601,0.430054,0.740888,0.455852,-0.391087,0.329897,-0.0344271,-0.35123,0.634805,-0.0295406,-0.0780164,-0.161517,-0.0492904,0.216342,0.0526798,0.601599,-0.275683,-0.560885,-0.0667728,-0.0148923,0.301657,-0.400817,0.301339,-0.216875,-0.18502,0.683748,-0.565575,-0.406418,-0.118362,0.716222,0.476707,0.110029,0.18534,0.202582,-0.0342085,0.493463,0.343436,0.0264412,0.239573,-0.33767,-0.124536,-0.267385,0.392923,-0.0455884,0.391077,-0.309277,-0.474835,-0.429355,-0.229551,-0.762364,0.192139,-0.236683,0.331836,-0.458372,-0.254388,-0.157175,0.507077,-0.0775531,0.0952105,0.0618797,-0.229648,-0.486909,-0.12972,0.644137,0.212288,0.561578,-0.626232,0.12649,0.500182,-0.224069,-0.165955,0.44267,-0.287308,0.0554345,0.11069,-0.168803,-0.105017,-0.0444611,-0.0629354,0.219351,0.426591,-0.184656,0.0484864,0.0594359,-0.0201588,0.278213,0.496346,0.316751,0.00616912,-0.251901,0.194444,-0.277984,0.263856,0.278469,-0.260093,-0.158784,0.100765,-0.0975716,0.318326,0.182902,-0.316445,0.0908969,0.127613,-0.166602,0.465639,0.536211,-0.0599409,-0.0421435,-0.000409483,0.14308,-0.0697533,-0.318968,0.472799,-0.0550308,0.00659994,0.0188792,0.177798,0.342248,0.452412,-0.416214,0.538627,0.0442812,-0.270144,-0.155452,-0.716189,0.66221,-0.0341583,0.565198,-0.355651,0.670199,0.696711,-0.0606457,0.283589,-0.272628,-0.0130886,0.264074,0.558448,0.249178,0.0290381,-0.302273,0.367832,0.569707,-0.00164709,-0.363105,-0.0343198,-0.286051,0.519618,0.314297,0.212841,-0.0220061,-0.331056,-0.0698947,0.0378684,0.0585447,0.204862,-0.47421,-0.758469,0.519175,-0.0593454,-0.144341,-0.0567177,0.351822,0.0293629,0.276361,-0.104907,-0.0296142,-0.259349,-0.0429548,-0.172361,0.117219,0.253988,0.294024,0.0828804,0.186616,-0.235631,0.344248,0.0303802,-0.437464,0.122078,-0.405162,-0.0707254,0.226482,-0.433298,-0.759245,0.776597,-1.08299,0.400145,0.0573294,0.00874768,-0.0938766,0.0213305,-0.355149,-0.136094,-0.231676,0.109353,-0.25185,-0.359561,-0.0399986,-0.231807,0.120122,0.242671,0.109609,0.295455,0.331072,0.543558,0.23756,
  3413.09,0.115015,6,1.51484,1.01923,-0.621049,0.11056,-0.0691901,-0.294049,-0.243524,0.286699,-0.0531019,0.0263536,-0.603631,-0.0642254,0.0740205,0.530753,0.0344619,0.371297,-0.0897412,0.139706,-0.0395585,0.0255589,-0.493957,-1.0139,-1.4199,-0.00603114,-0.154987,-0.415264,0.662395,0.195613,0.0207539,0.209297,0.814185,-0.461153,0.230644,0.189312,-0.447868,-0.132626,-0.820277,0.384938,0.0630201,-0.387381,1.15444,0.0539652,0.732476,-0.863075,-0.08559,0.149062,-1.49257,0.515946,0.597099,0.0901956,0.400621,0.241568,0.0145009,-0.195782,0.737078,-0.00558925,0.0736978,-0.47623,0.30932,-0.32276,0.204327,1.39926,-0.43366,-1.05893,0.276954,0.125897,0.283081,0.186213,0.405639,-0.244769,0.493048,0.18552,-0.191094,-0.220089,0.549975,-0.200403,0.348472,0.0281709,-0.2602,-0.0637835,0.163992,0.204242,0.569281,-0.130111,0.575737,-0.143522,0.00102818,0.119173,-0.0484052,0.039245,-0.306881,-0.193284,-0.815046,0.492887,-0.0266505,0.120316,-0.269449,-0.184118,-0.569225,0.069866,-0.306601,0.430054,0.740888,0.455852,-0.391087,0.329897,-0.0344271,-0.35123,0.634805,-0.0295406,-0.0780164,-0.161517,-0.0492904,0.216342,0.0526798,0.601599,-0.275683,-0.560885,-0.0667728,-0.0148923,0.301657,-0.400817,0.301339,-0.216875,-0.18502,0.683748,-0.565575,-0.406418,-0.118362,0.716222,0.476707,0.110029,0.18534,0.202582,-0.0342085,0.493463,0.343436,0.0264412,0.239573,-0.33767,-0.124536,-0.267385,0.392923,-0.0455884,0.391077,-0.309277,-0.474835,-0.429355,-0.229551,-0.762364,0.192139,-0.236683,0.331836,-0.458372,-0.254388,-0.157175,0.507077,-0.0775531,0.0952105,0.0618797,-0.229648,-0.486909,-0.12972,0.644137,0.212288,0.561578,-0.626232,0.12649,0.500182,-0.224069,-0.165955,0.44267,-0.287308,0.0554345,0.11069,-0.168803,-0.105017,-0.0444611,-0.0629354,0.219351,0.426591,-0.184656,0.0484864,0.0594359,-0.0201588,0.278213,0.496346,0.316751,0.00616912,-0.251901,0.194444,-0.277984,0.263856,0.278469,-0.260093,-0.158784,0.100765,-0.0975716,0.318326,0.182902,-0.316445,0.0908969,0.127613,-0.166602,0.465639,0.536211,-0.0599409,-0.0421435,-0.000409483,0.14308,-0.0697533,-0.318968,0.472799,-0.0550308,0.00659994,0.0188792,0.177798,0.342248,0.452412,-0.416214,0.538627,0.0442812,-0.270144,-0.155452,-0.716189,0.66221,-0.0341583,0.565198,-0.355651,0.670199,0.696711,-0.0606457,0.283589,-0.272628,-0.0130886,0.264074,0.558448,0.249178,0.0290381,-0.302273,0.367832,0.569707,-0.00164709,-0.363105,-0.0343198,-0.286051,0.519618,0.314297,0.212841,-0.0220061,-0.331056,-0.0698947,0.0378684,0.0585447,0.204862,-0.47421,-0.758469,0.519175,-0.0593454,-0.144341,-0.0567177,0.351822,0.0293629,0.276361,-0.104907,-0.0296142,-0.259349,-0.0429548,-0.172361,0.117219,0.253988,0.294024,0.0828804,0.186616,-0.235631,0.344248,0.0303802,-0.437464,0.122078,-0.405162,-0.0707254,0.226482,-0.433298,-0.759245,0.776597,-1.08299,0.400145,0.0573294,0.00874768,-0.0938766,0.0213305,-0.355149,-0.136094,-0.231676,0.109353,-0.25185,-0.359561,-0.0399986,-0.231807,0.120122,0.242671,0.109609,0.295455,0.331072,0.543558,0.23756,
  3413.09,0.115015,5,1.51484,1.01923,-0.621049,0.11056,-0.0691901,-0.294049,-0.243524,0.286699,-0.0531019,0.0263536,-0.603631,-0.0642254,0.0740205,0.530753,0.0344619,0.371297,-0.0897412,0.139706,-0.0395585,0.0255589,-0.493957,-1.0139,-1.4199,-0.00603114,-0.154987,-0.415264,0.662395,0.195613,0.0207539,0.209297,0.814185,-0.461153,0.230644,0.189312,-0.447868,-0.132626,-0.820277,0.384938,0.0630201,-0.387381,1.15444,0.0539652,0.732476,-0.863075,-0.08559,0.149062,-1.49257,0.515946,0.597099,0.0901956,0.400621,0.241568,0.0145009,-0.195782,0.737078,-0.00558925,0.0736978,-0.47623,0.30932,-0.32276,0.204327,1.39926,-0.43366,-1.05893,0.276954,0.125897,0.283081,0.186213,0.405639,-0.244769,0.493048,0.18552,-0.191094,-0.220089,0.549975,-0.200403,0.348472,0.0281709,-0.2602,-0.0637835,0.163992,0.204242,0.569281,-0.130111,0.575737,-0.143522,0.00102818,0.119173,-0.0484052,0.039245,-0.306881,-0.193284,-0.815046,0.492887,-0.0266505,0.120316,-0.269449,-0.184118,-0.569225,0.069866,-0.306601,0.430054,0.740888,0.455852,-0.391087,0.329897,-0.0344271,-0.35123,0.634805,-0.0295406,-0.0780164,-0.161517,-0.0492904,0.216342,0.0526798,0.601599,-0.275683,-0.560885,-0.0667728,-0.0148923,0.301657,-0.400817,0.301339,-0.216875,-0.18502,0.683748,-0.565575,-0.406418,-0.118362,0.716222,0.476707,0.110029,0.18534,0.202582,-0.0342085,0.493463,0.343436,0.0264412,0.239573,-0.33767,-0.124536,-0.267385,0.392923,-0.0455884,0.391077,-0.309277,-0.474835,-0.429355,-0.229551,-0.762364,0.192139,-0.236683,0.331836,-0.458372,-0.254388,-0.157175,0.507077,-0.0775531,0.0952105,0.0618797,-0.229648,-0.486909,-0.12972,0.644137,0.212288,0.561578,-0.626232,0.12649,0.500182,-0.224069,-0.165955,0.44267,-0.287308,0.0554345,0.11069,-0.168803,-0.105017,-0.0444611,-0.0629354,0.219351,0.426591,-0.184656,0.0484864,0.0594359,-0.0201588,0.278213,0.496346,0.316751,0.00616912,-0.251901,0.194444,-0.277984,0.263856,0.278469,-0.260093,-0.158784,0.100765,-0.0975716,0.318326,0.182902,-0.316445,0.0908969,0.127613,-0.166602,0.465639,0.536211,-0.0599409,-0.0421435,-0.000409483,0.14308,-0.0697533,-0.318968,0.472799,-0.0550308,0.00659994,0.0188792,0.177798,0.342248,0.452412,-0.416214,0.538627,0.0442812,-0.270144,-0.155452,-0.716189,0.66221,-0.0341583,0.565198,-0.355651,0.670199,0.696711,-0.0606457,0.283589,-0.272628,-0.0130886,0.264074,0.558448,0.249178,0.0290381,-0.302273,0.367832,0.569707,-0.00164709,-0.363105,-0.0343198,-0.286051,0.519618,0.314297,0.212841,-0.0220061,-0.331056,-0.0698947,0.0378684,0.0585447,0.204862,-0.47421,-0.758469,0.519175,-0.0593454,-0.144341,-0.0567177,0.351822,0.0293629,0.276361,-0.104907,-0.0296142,-0.259349,-0.0429548,-0.172361,0.117219,0.253988,0.294024,0.0828804,0.186616,-0.235631,0.344248,0.0303802,-0.437464,0.122078,-0.405162,-0.0707254,0.226482,-0.433298,-0.759245,0.776597,-1.08299,0.400145,0.0573294,0.00874768,-0.0938766,0.0213305,-0.355149,-0.136094,-0.231676,0.109353,-0.25185,-0.359561,-0.0399986,-0.231807,0.120122,0.242671,0.109609,0.295455,0.331072,0.543558,0.23756;

  for (int i = 0; i < 3; i++) 
    for (int j = 0; j < 309; j++)
      EXPECT_FLOAT_EQ(expected_samples(i,j), samples(i,j))
  << "comparison failed for (" << i << "," << j << ")";
}

TEST_F(StanIoStanCsvReader,ParseEpil) {
  stan::io::stan_csv epil0;
  epil0 = stan::io::stan_csv_reader::parse(epil0_stream);

  // metadata
  EXPECT_EQ(1, epil0.metadata.stan_version_major);
  EXPECT_EQ(3, epil0.metadata.stan_version_minor);
  EXPECT_EQ(0, epil0.metadata.stan_version_patch);
  
  EXPECT_EQ("src/models/bugs_examples/vol1/epil/epil.data.R", epil0.metadata.data);
  EXPECT_EQ("random initialization", epil0.metadata.init);
  EXPECT_FALSE(epil0.metadata.append_samples);
  EXPECT_FALSE(epil0.metadata.save_warmup);
  EXPECT_EQ(4258844633U, epil0.metadata.seed);
  EXPECT_FALSE(epil0.metadata.random_seed);
  EXPECT_EQ(1U, epil0.metadata.chain_id);
  EXPECT_EQ(2000U, epil0.metadata.iter);
  EXPECT_EQ(1000U, epil0.metadata.warmup);
  EXPECT_EQ(1U, epil0.metadata.thin);
  EXPECT_FALSE(epil0.metadata.equal_step_sizes);
  EXPECT_EQ(-1, epil0.metadata.leapfrog_steps);
  EXPECT_EQ(10, epil0.metadata.max_treedepth);
  EXPECT_FLOAT_EQ(-1, epil0.metadata.epsilon);
  EXPECT_FLOAT_EQ(0, epil0.metadata.epsilon_pm);
  EXPECT_FLOAT_EQ(0.5, epil0.metadata.delta);
  EXPECT_FLOAT_EQ(0.05, epil0.metadata.gamma);

  // header
  ASSERT_EQ(309, epil0.header.size());
  EXPECT_EQ("lp__", epil0.header(0));
  EXPECT_EQ("stepsize__", epil0.header(1));
  EXPECT_EQ("depth__", epil0.header(2));
  EXPECT_EQ("a0", epil0.header(3));
  EXPECT_EQ("alpha_Base", epil0.header(4));
  EXPECT_EQ("alpha_Trt", epil0.header(5));
  EXPECT_EQ("alpha_BT", epil0.header(6));
  EXPECT_EQ("alpha_Age", epil0.header(7));
  EXPECT_EQ("alpha_V4", epil0.header(8));
  EXPECT_EQ("b1[1]", epil0.header(9));
  EXPECT_EQ("b1[2]", epil0.header(10));
  // ...
  EXPECT_EQ("b1[59]", epil0.header(67));
  EXPECT_EQ("b[1,1]", epil0.header(68));
  EXPECT_EQ("b[1,2]", epil0.header(69));
  EXPECT_EQ("b[1,3]", epil0.header(70));
  EXPECT_EQ("b[1,4]", epil0.header(71));
  EXPECT_EQ("b[2,1]", epil0.header(72));
  //...
  EXPECT_EQ("b[59,4]", epil0.header(303));
  EXPECT_EQ("sigmasq_b", epil0.header(304));
  EXPECT_EQ("sigmasq_b1", epil0.header(305));
  EXPECT_EQ("sigma_b", epil0.header(306));
  EXPECT_EQ("sigma_b1", epil0.header(307));
  EXPECT_EQ("alpha0", epil0.header(308));

  // adaptation
  EXPECT_EQ("NUTS with a diagonal Euclidean metric", epil0.adaptation.sampler);
  EXPECT_FLOAT_EQ(0.115015, epil0.adaptation.step_size);
  ASSERT_EQ(303, epil0.adaptation.step_size_multipliers.size());
  EXPECT_FLOAT_EQ(0.0219139, epil0.adaptation.step_size_multipliers(0));
  EXPECT_FLOAT_EQ(0.0356712, epil0.adaptation.step_size_multipliers(1));
  EXPECT_FLOAT_EQ(0.167986, epil0.adaptation.step_size_multipliers(2));
  // ...
  EXPECT_FLOAT_EQ(0.0679679, epil0.adaptation.step_size_multipliers(301));
  EXPECT_FLOAT_EQ(0.0899625, epil0.adaptation.step_size_multipliers(302));

  // samples
  ASSERT_EQ(1000, epil0.samples.rows());
  ASSERT_EQ(309, epil0.samples.cols());

  Eigen::MatrixXd expected_samples(3, 309);
  expected_samples <<
  3413.09,0.115015,10,1.51484,1.01923,-0.621049,0.11056,-0.0691901,-0.294049,-0.243524,0.286699,-0.0531019,0.0263536,-0.603631,-0.0642254,0.0740205,0.530753,0.0344619,0.371297,-0.0897412,0.139706,-0.0395585,0.0255589,-0.493957,-1.0139,-1.4199,-0.00603114,-0.154987,-0.415264,0.662395,0.195613,0.0207539,0.209297,0.814185,-0.461153,0.230644,0.189312,-0.447868,-0.132626,-0.820277,0.384938,0.0630201,-0.387381,1.15444,0.0539652,0.732476,-0.863075,-0.08559,0.149062,-1.49257,0.515946,0.597099,0.0901956,0.400621,0.241568,0.0145009,-0.195782,0.737078,-0.00558925,0.0736978,-0.47623,0.30932,-0.32276,0.204327,1.39926,-0.43366,-1.05893,0.276954,0.125897,0.283081,0.186213,0.405639,-0.244769,0.493048,0.18552,-0.191094,-0.220089,0.549975,-0.200403,0.348472,0.0281709,-0.2602,-0.0637835,0.163992,0.204242,0.569281,-0.130111,0.575737,-0.143522,0.00102818,0.119173,-0.0484052,0.039245,-0.306881,-0.193284,-0.815046,0.492887,-0.0266505,0.120316,-0.269449,-0.184118,-0.569225,0.069866,-0.306601,0.430054,0.740888,0.455852,-0.391087,0.329897,-0.0344271,-0.35123,0.634805,-0.0295406,-0.0780164,-0.161517,-0.0492904,0.216342,0.0526798,0.601599,-0.275683,-0.560885,-0.0667728,-0.0148923,0.301657,-0.400817,0.301339,-0.216875,-0.18502,0.683748,-0.565575,-0.406418,-0.118362,0.716222,0.476707,0.110029,0.18534,0.202582,-0.0342085,0.493463,0.343436,0.0264412,0.239573,-0.33767,-0.124536,-0.267385,0.392923,-0.0455884,0.391077,-0.309277,-0.474835,-0.429355,-0.229551,-0.762364,0.192139,-0.236683,0.331836,-0.458372,-0.254388,-0.157175,0.507077,-0.0775531,0.0952105,0.0618797,-0.229648,-0.486909,-0.12972,0.644137,0.212288,0.561578,-0.626232,0.12649,0.500182,-0.224069,-0.165955,0.44267,-0.287308,0.0554345,0.11069,-0.168803,-0.105017,-0.0444611,-0.0629354,0.219351,0.426591,-0.184656,0.0484864,0.0594359,-0.0201588,0.278213,0.496346,0.316751,0.00616912,-0.251901,0.194444,-0.277984,0.263856,0.278469,-0.260093,-0.158784,0.100765,-0.0975716,0.318326,0.182902,-0.316445,0.0908969,0.127613,-0.166602,0.465639,0.536211,-0.0599409,-0.0421435,-0.000409483,0.14308,-0.0697533,-0.318968,0.472799,-0.0550308,0.00659994,0.0188792,0.177798,0.342248,0.452412,-0.416214,0.538627,0.0442812,-0.270144,-0.155452,-0.716189,0.66221,-0.0341583,0.565198,-0.355651,0.670199,0.696711,-0.0606457,0.283589,-0.272628,-0.0130886,0.264074,0.558448,0.249178,0.0290381,-0.302273,0.367832,0.569707,-0.00164709,-0.363105,-0.0343198,-0.286051,0.519618,0.314297,0.212841,-0.0220061,-0.331056,-0.0698947,0.0378684,0.0585447,0.204862,-0.47421,-0.758469,0.519175,-0.0593454,-0.144341,-0.0567177,0.351822,0.0293629,0.276361,-0.104907,-0.0296142,-0.259349,-0.0429548,-0.172361,0.117219,0.253988,0.294024,0.0828804,0.186616,-0.235631,0.344248,0.0303802,-0.437464,0.122078,-0.405162,-0.0707254,0.226482,-0.433298,-0.759245,0.776597,-1.08299,0.400145,0.0573294,0.00874768,-0.0938766,0.0213305,-0.355149,-0.136094,-0.231676,0.109353,-0.25185,-0.359561,-0.0399986,-0.231807,0.120122,0.242671,0.109609,0.295455,0.331072,0.543558,0.23756,
  3413.09,0.115015,6,1.51484,1.01923,-0.621049,0.11056,-0.0691901,-0.294049,-0.243524,0.286699,-0.0531019,0.0263536,-0.603631,-0.0642254,0.0740205,0.530753,0.0344619,0.371297,-0.0897412,0.139706,-0.0395585,0.0255589,-0.493957,-1.0139,-1.4199,-0.00603114,-0.154987,-0.415264,0.662395,0.195613,0.0207539,0.209297,0.814185,-0.461153,0.230644,0.189312,-0.447868,-0.132626,-0.820277,0.384938,0.0630201,-0.387381,1.15444,0.0539652,0.732476,-0.863075,-0.08559,0.149062,-1.49257,0.515946,0.597099,0.0901956,0.400621,0.241568,0.0145009,-0.195782,0.737078,-0.00558925,0.0736978,-0.47623,0.30932,-0.32276,0.204327,1.39926,-0.43366,-1.05893,0.276954,0.125897,0.283081,0.186213,0.405639,-0.244769,0.493048,0.18552,-0.191094,-0.220089,0.549975,-0.200403,0.348472,0.0281709,-0.2602,-0.0637835,0.163992,0.204242,0.569281,-0.130111,0.575737,-0.143522,0.00102818,0.119173,-0.0484052,0.039245,-0.306881,-0.193284,-0.815046,0.492887,-0.0266505,0.120316,-0.269449,-0.184118,-0.569225,0.069866,-0.306601,0.430054,0.740888,0.455852,-0.391087,0.329897,-0.0344271,-0.35123,0.634805,-0.0295406,-0.0780164,-0.161517,-0.0492904,0.216342,0.0526798,0.601599,-0.275683,-0.560885,-0.0667728,-0.0148923,0.301657,-0.400817,0.301339,-0.216875,-0.18502,0.683748,-0.565575,-0.406418,-0.118362,0.716222,0.476707,0.110029,0.18534,0.202582,-0.0342085,0.493463,0.343436,0.0264412,0.239573,-0.33767,-0.124536,-0.267385,0.392923,-0.0455884,0.391077,-0.309277,-0.474835,-0.429355,-0.229551,-0.762364,0.192139,-0.236683,0.331836,-0.458372,-0.254388,-0.157175,0.507077,-0.0775531,0.0952105,0.0618797,-0.229648,-0.486909,-0.12972,0.644137,0.212288,0.561578,-0.626232,0.12649,0.500182,-0.224069,-0.165955,0.44267,-0.287308,0.0554345,0.11069,-0.168803,-0.105017,-0.0444611,-0.0629354,0.219351,0.426591,-0.184656,0.0484864,0.0594359,-0.0201588,0.278213,0.496346,0.316751,0.00616912,-0.251901,0.194444,-0.277984,0.263856,0.278469,-0.260093,-0.158784,0.100765,-0.0975716,0.318326,0.182902,-0.316445,0.0908969,0.127613,-0.166602,0.465639,0.536211,-0.0599409,-0.0421435,-0.000409483,0.14308,-0.0697533,-0.318968,0.472799,-0.0550308,0.00659994,0.0188792,0.177798,0.342248,0.452412,-0.416214,0.538627,0.0442812,-0.270144,-0.155452,-0.716189,0.66221,-0.0341583,0.565198,-0.355651,0.670199,0.696711,-0.0606457,0.283589,-0.272628,-0.0130886,0.264074,0.558448,0.249178,0.0290381,-0.302273,0.367832,0.569707,-0.00164709,-0.363105,-0.0343198,-0.286051,0.519618,0.314297,0.212841,-0.0220061,-0.331056,-0.0698947,0.0378684,0.0585447,0.204862,-0.47421,-0.758469,0.519175,-0.0593454,-0.144341,-0.0567177,0.351822,0.0293629,0.276361,-0.104907,-0.0296142,-0.259349,-0.0429548,-0.172361,0.117219,0.253988,0.294024,0.0828804,0.186616,-0.235631,0.344248,0.0303802,-0.437464,0.122078,-0.405162,-0.0707254,0.226482,-0.433298,-0.759245,0.776597,-1.08299,0.400145,0.0573294,0.00874768,-0.0938766,0.0213305,-0.355149,-0.136094,-0.231676,0.109353,-0.25185,-0.359561,-0.0399986,-0.231807,0.120122,0.242671,0.109609,0.295455,0.331072,0.543558,0.23756,
  3413.09,0.115015,5,1.51484,1.01923,-0.621049,0.11056,-0.0691901,-0.294049,-0.243524,0.286699,-0.0531019,0.0263536,-0.603631,-0.0642254,0.0740205,0.530753,0.0344619,0.371297,-0.0897412,0.139706,-0.0395585,0.0255589,-0.493957,-1.0139,-1.4199,-0.00603114,-0.154987,-0.415264,0.662395,0.195613,0.0207539,0.209297,0.814185,-0.461153,0.230644,0.189312,-0.447868,-0.132626,-0.820277,0.384938,0.0630201,-0.387381,1.15444,0.0539652,0.732476,-0.863075,-0.08559,0.149062,-1.49257,0.515946,0.597099,0.0901956,0.400621,0.241568,0.0145009,-0.195782,0.737078,-0.00558925,0.0736978,-0.47623,0.30932,-0.32276,0.204327,1.39926,-0.43366,-1.05893,0.276954,0.125897,0.283081,0.186213,0.405639,-0.244769,0.493048,0.18552,-0.191094,-0.220089,0.549975,-0.200403,0.348472,0.0281709,-0.2602,-0.0637835,0.163992,0.204242,0.569281,-0.130111,0.575737,-0.143522,0.00102818,0.119173,-0.0484052,0.039245,-0.306881,-0.193284,-0.815046,0.492887,-0.0266505,0.120316,-0.269449,-0.184118,-0.569225,0.069866,-0.306601,0.430054,0.740888,0.455852,-0.391087,0.329897,-0.0344271,-0.35123,0.634805,-0.0295406,-0.0780164,-0.161517,-0.0492904,0.216342,0.0526798,0.601599,-0.275683,-0.560885,-0.0667728,-0.0148923,0.301657,-0.400817,0.301339,-0.216875,-0.18502,0.683748,-0.565575,-0.406418,-0.118362,0.716222,0.476707,0.110029,0.18534,0.202582,-0.0342085,0.493463,0.343436,0.0264412,0.239573,-0.33767,-0.124536,-0.267385,0.392923,-0.0455884,0.391077,-0.309277,-0.474835,-0.429355,-0.229551,-0.762364,0.192139,-0.236683,0.331836,-0.458372,-0.254388,-0.157175,0.507077,-0.0775531,0.0952105,0.0618797,-0.229648,-0.486909,-0.12972,0.644137,0.212288,0.561578,-0.626232,0.12649,0.500182,-0.224069,-0.165955,0.44267,-0.287308,0.0554345,0.11069,-0.168803,-0.105017,-0.0444611,-0.0629354,0.219351,0.426591,-0.184656,0.0484864,0.0594359,-0.0201588,0.278213,0.496346,0.316751,0.00616912,-0.251901,0.194444,-0.277984,0.263856,0.278469,-0.260093,-0.158784,0.100765,-0.0975716,0.318326,0.182902,-0.316445,0.0908969,0.127613,-0.166602,0.465639,0.536211,-0.0599409,-0.0421435,-0.000409483,0.14308,-0.0697533,-0.318968,0.472799,-0.0550308,0.00659994,0.0188792,0.177798,0.342248,0.452412,-0.416214,0.538627,0.0442812,-0.270144,-0.155452,-0.716189,0.66221,-0.0341583,0.565198,-0.355651,0.670199,0.696711,-0.0606457,0.283589,-0.272628,-0.0130886,0.264074,0.558448,0.249178,0.0290381,-0.302273,0.367832,0.569707,-0.00164709,-0.363105,-0.0343198,-0.286051,0.519618,0.314297,0.212841,-0.0220061,-0.331056,-0.0698947,0.0378684,0.0585447,0.204862,-0.47421,-0.758469,0.519175,-0.0593454,-0.144341,-0.0567177,0.351822,0.0293629,0.276361,-0.104907,-0.0296142,-0.259349,-0.0429548,-0.172361,0.117219,0.253988,0.294024,0.0828804,0.186616,-0.235631,0.344248,0.0303802,-0.437464,0.122078,-0.405162,-0.0707254,0.226482,-0.433298,-0.759245,0.776597,-1.08299,0.400145,0.0573294,0.00874768,-0.0938766,0.0213305,-0.355149,-0.136094,-0.231676,0.109353,-0.25185,-0.359561,-0.0399986,-0.231807,0.120122,0.242671,0.109609,0.295455,0.331072,0.543558,0.23756;

  for (int i = 0; i < 3; i++) 
    for (int j = 0; j < 309; j++)
      EXPECT_FLOAT_EQ(expected_samples(i,j), epil0.samples(i,j))
  << "comparison failed for (" << i << "," << j << ")";

}



TEST_F(StanIoStanCsvReader,ParseBlockerNondiag) {
  stan::io::stan_csv blocker_nondiag;
  blocker_nondiag = stan::io::stan_csv_reader::parse(blocker_nondiag0_stream);

  // metadata
  EXPECT_EQ(1, blocker_nondiag.metadata.stan_version_major);
  EXPECT_EQ(3, blocker_nondiag.metadata.stan_version_minor);
  EXPECT_EQ(0, blocker_nondiag.metadata.stan_version_patch);
  
  EXPECT_EQ("src/models/bugs_examples/vol1/blocker/blocker.data.R", blocker_nondiag.metadata.data);
  EXPECT_EQ("src/models/bugs_examples/vol1/blocker/blocker.init.R", blocker_nondiag.metadata.init);
  EXPECT_FALSE(blocker_nondiag.metadata.append_samples);
  EXPECT_FALSE(blocker_nondiag.metadata.save_warmup);
  EXPECT_EQ(4085885484U, blocker_nondiag.metadata.seed);
  EXPECT_FALSE(blocker_nondiag.metadata.random_seed);
  EXPECT_EQ(1U, blocker_nondiag.metadata.chain_id);
  EXPECT_EQ(4000U, blocker_nondiag.metadata.iter);
  EXPECT_EQ(2000U, blocker_nondiag.metadata.warmup);
  EXPECT_EQ(2U, blocker_nondiag.metadata.thin);
  EXPECT_FALSE(blocker_nondiag.metadata.equal_step_sizes);
  EXPECT_EQ(-1, blocker_nondiag.metadata.leapfrog_steps);
  EXPECT_EQ(10, blocker_nondiag.metadata.max_treedepth);
  EXPECT_FLOAT_EQ(-1, blocker_nondiag.metadata.epsilon);
  EXPECT_FLOAT_EQ(0, blocker_nondiag.metadata.epsilon_pm);
  EXPECT_FLOAT_EQ(0.5, blocker_nondiag.metadata.delta);
  EXPECT_FLOAT_EQ(0.05, blocker_nondiag.metadata.gamma);

  // header
  ASSERT_EQ(51, blocker_nondiag.header.size());
  EXPECT_EQ("lp__", blocker_nondiag.header(0));
  EXPECT_EQ("stepsize__",blocker_nondiag.header(1));
  EXPECT_EQ("depth__",blocker_nondiag.header(2));
  EXPECT_EQ("d",blocker_nondiag.header(3));
  EXPECT_EQ("sigmasq_delta",blocker_nondiag.header(4));
  EXPECT_EQ("mu[1]",blocker_nondiag.header(5));
  EXPECT_EQ("mu[2]",blocker_nondiag.header(6));
  EXPECT_EQ("mu[3]",blocker_nondiag.header(7));
  EXPECT_EQ("mu[4]",blocker_nondiag.header(8));
  EXPECT_EQ("mu[5]",blocker_nondiag.header(9));
  EXPECT_EQ("mu[6]",blocker_nondiag.header(10));
  EXPECT_EQ("mu[7]",blocker_nondiag.header(11));
  EXPECT_EQ("mu[8]",blocker_nondiag.header(12));
  EXPECT_EQ("mu[9]",blocker_nondiag.header(13));
  EXPECT_EQ("mu[10]",blocker_nondiag.header(14));
  EXPECT_EQ("mu[11]",blocker_nondiag.header(15));
  EXPECT_EQ("mu[12]",blocker_nondiag.header(16));
  EXPECT_EQ("mu[13]",blocker_nondiag.header(17));
  EXPECT_EQ("mu[14]",blocker_nondiag.header(18));
  EXPECT_EQ("mu[15]",blocker_nondiag.header(19));
  EXPECT_EQ("mu[16]",blocker_nondiag.header(20));
  EXPECT_EQ("mu[17]",blocker_nondiag.header(21));
  EXPECT_EQ("mu[18]",blocker_nondiag.header(22));
  EXPECT_EQ("mu[19]",blocker_nondiag.header(23));
  EXPECT_EQ("mu[20]",blocker_nondiag.header(24));
  EXPECT_EQ("mu[21]",blocker_nondiag.header(25));
  EXPECT_EQ("mu[22]",blocker_nondiag.header(26));
  EXPECT_EQ("delta[1]",blocker_nondiag.header(27));
  EXPECT_EQ("delta[2]",blocker_nondiag.header(28));
  EXPECT_EQ("delta[3]",blocker_nondiag.header(29));
  EXPECT_EQ("delta[4]",blocker_nondiag.header(30));
  EXPECT_EQ("delta[5]",blocker_nondiag.header(31));
  EXPECT_EQ("delta[6]",blocker_nondiag.header(32));
  EXPECT_EQ("delta[7]",blocker_nondiag.header(33));
  EXPECT_EQ("delta[8]",blocker_nondiag.header(34));
  EXPECT_EQ("delta[9]",blocker_nondiag.header(35));
  EXPECT_EQ("delta[10]",blocker_nondiag.header(36));
  EXPECT_EQ("delta[11]",blocker_nondiag.header(37));
  EXPECT_EQ("delta[12]",blocker_nondiag.header(38));
  EXPECT_EQ("delta[13]",blocker_nondiag.header(39));
  EXPECT_EQ("delta[14]",blocker_nondiag.header(40));
  EXPECT_EQ("delta[15]",blocker_nondiag.header(41));
  EXPECT_EQ("delta[16]",blocker_nondiag.header(42));
  EXPECT_EQ("delta[17]",blocker_nondiag.header(43));
  EXPECT_EQ("delta[18]",blocker_nondiag.header(44));
  EXPECT_EQ("delta[19]",blocker_nondiag.header(45));
  EXPECT_EQ("delta[20]",blocker_nondiag.header(46));
  EXPECT_EQ("delta[21]",blocker_nondiag.header(47));
  EXPECT_EQ("delta[22]",blocker_nondiag.header(48));
  EXPECT_EQ("delta_new",blocker_nondiag.header(49));
  EXPECT_EQ("sigma_delta",blocker_nondiag.header(50));

  // adaptation
  EXPECT_EQ("NUTS with a dense Euclidean metric", blocker_nondiag.adaptation.sampler);
  EXPECT_FLOAT_EQ(0.246724, blocker_nondiag.adaptation.step_size);
  ASSERT_EQ(47*47, blocker_nondiag.adaptation.step_size_multipliers.size());
  EXPECT_FLOAT_EQ(0.0161828, blocker_nondiag.adaptation.step_size_multipliers(0,0));
  EXPECT_FLOAT_EQ(0.00974695, blocker_nondiag.adaptation.step_size_multipliers(0,1));
  EXPECT_FLOAT_EQ(0.00974695, blocker_nondiag.adaptation.step_size_multipliers(1,0));
  EXPECT_FLOAT_EQ(0.604547, blocker_nondiag.adaptation.step_size_multipliers(1,1));
  EXPECT_FLOAT_EQ(-0.00832875, blocker_nondiag.adaptation.step_size_multipliers(4,0));
  EXPECT_FLOAT_EQ(-0.016723, blocker_nondiag.adaptation.step_size_multipliers(4,46));
  EXPECT_FLOAT_EQ(0.00947164, blocker_nondiag.adaptation.step_size_multipliers(46,0));
  EXPECT_FLOAT_EQ(0.097000301, blocker_nondiag.adaptation.step_size_multipliers(46,46));

  // samples
  ASSERT_EQ(1000, blocker_nondiag.samples.rows());
  ASSERT_EQ(51, blocker_nondiag.samples.cols());

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> expected_samples(6, 51);
  expected_samples <<
  -5926.48,0.246724,3,-0.311157,0.0677484,-2.51166,-1.89596,-2.3258,-2.43266,-2.27532,-2.62758,-1.70583,-2.07853,-1.65004,-2.23559,-2.21202,-1.47587,-3.14668,-2.84015,-1.49442,-1.61735,-1.86787,-3.24885,-3.10812,-1.43662,-2.12735,-2.86932,-0.372761,-0.445298,-0.294804,-0.23084,-0.303822,-0.408726,-0.550839,-0.30455,-0.355906,-0.450826,-0.359745,-0.233544,-0.298835,0.193832,-0.196433,-0.12851,-0.332956,-0.0739457,-0.102053,-0.248046,-0.23991,-0.631952,0.201986,0.260285,
  -5924.05,0.246724,4,-0.309216,0.0148971,-1.83876,-2.65203,-1.92405,-2.30055,-2.49994,-1.82688,-1.60359,-2.0743,-1.86007,-2.22869,-2.50874,-1.52494,-3.33068,-2.91206,-1.46873,-1.36817,-1.95352,-2.68484,-3.18343,-1.44816,-2.08514,-2.80663,-0.0289075,-0.349799,-0.280604,-0.399029,-0.0488077,-0.242368,-0.412373,-0.0479823,-0.483247,-0.241592,0.0148917,-0.260925,-0.193791,0.231194,-0.457137,-0.289194,-0.181036,-0.318246,-0.332496,-0.299544,-0.456323,-0.312878,-0.536928,0.122054,
  -5933.6,0.246724,3,-0.348648,0.00918932,-1.93763,-2.44556,-1.89617,-2.23204,-2.39474,-1.72472,-1.6123,-1.9754,-1.89048,-2.25858,-2.5002,-1.46339,-3.2449,-2.82741,-1.33053,-1.23517,-1.7694,-2.43524,-2.86177,-1.43275,-1.99451,-2.82894,-0.104495,-0.468131,-0.349876,-0.37416,-0.16621,-0.368697,-0.370763,-0.148633,-0.476764,-0.276522,-0.0226429,-0.306111,-0.358191,0.0606187,-0.695668,-0.513489,-0.501577,-0.375382,-0.574766,-0.328659,-0.590392,-0.259226,-0.555894,0.0958609,
  -5923.07,0.246724,3,-0.18745,0.0297742,-2.13678,-2.37113,-2.29729,-2.39945,-2.54504,-2.22042,-1.64334,-2.20388,-1.85085,-2.34376,-2.52741,-1.55955,-2.77941,-2.95713,-1.49301,-1.42427,-2.12399,-3.42147,-2.80567,-1.58293,-2.03855,-2.8704,-0.105782,-0.107105,-0.0882,-0.212905,-0.232545,-0.252355,-0.4613,-0.113463,-0.357907,-0.141676,-0.100957,0.0265875,-0.317955,0.247121,0.0341783,-0.129578,-0.0114132,0.182162,-0.288887,-0.154355,-0.435679,-0.781506,-0.105477,0.172552,
  -5920.11,0.246724,4,-0.2387,0.0216922,-2.33578,-2.16573,-2.20488,-2.44396,-2.27906,-1.73085,-1.74242,-2.12548,-2.03616,-2.12423,-2.09649,-1.46484,-3.28672,-2.658,-1.31113,-1.4693,-1.9785,-3.15059,-3.72531,-1.4769,-1.95725,-3.04267,-0.253021,-0.137333,-0.373072,-0.108776,-0.146891,-0.495243,-0.374383,-0.200987,-0.198054,-0.30627,-0.446166,-0.278447,-0.186493,-0.104948,-0.261358,-0.379767,-0.400571,-0.0494108,-0.220698,-0.318966,-0.624246,-0.134914,-0.387797,0.147283,
  -5937.93,0.246724,3,-0.0414201,0.0925441,-1.72967,-1.82705,-2.64366,-2.4884,-2.54227,-2.02259,-1.79489,-2.06505,-2.08317,-2.26308,-2.37307,-1.55654,-3.12123,-2.76548,-1.46115,-1.48075,-1.92367,-3.25369,-3.52751,-1.64151,-1.9305,-3.03405,0.192683,-0.210269,-0.0664352,-0.184564,0.241455,-0.13075,-0.386244,-0.0704897,-0.026866,-0.268924,0.00155751,0.341495,-0.0471777,0.0955306,-0.154971,-0.0205135,-0.475991,0.145137,0.163358,0.0313725,-0.447384,-0.533345,-0.0621271,0.304211;
  
  for (int i = 0; i < 6; i++) 
    for (int j = 0; j < 51; j++)
      EXPECT_FLOAT_EQ(expected_samples(i,j), blocker_nondiag.samples(i,j))
  << "comparison failed for (" << i << "," << j << ")";
}

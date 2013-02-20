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
  }

  std::ifstream blocker0_stream, epil0_stream;
  std::ifstream metadata1_stream, header1_stream, adaptation1_stream, samples1_stream;
  std::ifstream metadata2_stream, header2_stream, adaptation2_stream, samples2_stream;
};

TEST_F(StanIoStanCsvReader,read_metadata1) {
  stan::io::stan_csv_metadata metadata;
  EXPECT_TRUE(stan::io::stan_csv_reader::read_metadata(metadata1_stream, metadata));

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
TEST_F(StanIoStanCsvReader,read_header1) { 
  Eigen::Matrix<std::string, Eigen::Dynamic, 1> header;
  EXPECT_TRUE(stan::io::stan_csv_reader::read_header(header1_stream, header));
  
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

TEST_F(StanIoStanCsvReader,read_adaptation1) { 
  stan::io::stan_csv_adaptation adaptation;
  EXPECT_TRUE(stan::io::stan_csv_reader::read_adaptation(adaptation1_stream, adaptation));
  
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

TEST_F(StanIoStanCsvReader,read_samples1) { 
  Eigen::MatrixXd samples;
  EXPECT_TRUE(stan::io::stan_csv_reader::read_samples(samples1_stream, samples));

  ASSERT_EQ(5, samples.rows());
  ASSERT_EQ(51, samples.cols());

  
  Eigen::MatrixXd expected_samples(5, 51);
  expected_samples <<
    -5923.71,3,0.104225,-0.31639,0.0064363,-2.0354,-1.94524,-2.06081,-2.31731,-2.54922,-2.50787,-1.74407,-2.04826,-1.75427,-2.16927,-2.31893,-1.37094,-2.8661,-2.67123,-1.36898,-1.43593,-1.8091,-2.92111,-3.98671,-1.4164,-1.89509,-2.94966,-0.557794,-0.483134,-0.156347,-0.250073,-0.121718,-0.243917,-0.327415,-0.127961,-0.238188,-0.425699,-0.280189,-0.294234,-0.442824,-0.30521,0.0666663,-0.457187,-0.270417,-0.247473,-0.402816,-0.20402,-0.346646,-0.306836,-0.436561,0.0802265,
    -5931.85,3,0.104225,-0.372371,0.0143758,-2.53617,-1.9066,-2.69784,-2.34748,-2.40404,-1.84627,-1.76547,-1.95062,-2.02309,-2.26443,-2.35281,-1.65016,-2.81412,-2.59655,-1.3871,-1.36819,-2.27108,-2.82077,-2.94598,-1.14072,-2.0336,-3.24552,-0.429167,-0.345602,-0.255696,-0.40927,-0.207543,-0.556137,-0.462445,-0.204487,-0.216857,-0.324489,-0.230653,-0.117995,-0.603386,-0.276107,-0.256322,-0.491629,0.252045,-0.199305,-0.638357,-0.285059,-0.364338,-0.269418,-0.470453,0.119899,
    -5930.95,3,0.104225,-0.27131,0.0189956,-3.00112,-2.24849,-2.64399,-2.31772,-2.80533,-2.01866,-1.69406,-2.29215,-2.03636,-2.20395,-2.24996,-1.46979,-3.08446,-2.61183,-1.34088,-1.42055,-1.83887,-2.9487,-3.28892,-1.21848,-1.89679,-3.31541,-0.349814,-0.380662,-0.317758,-0.19059,0.326377,-0.526384,-0.276535,-0.226859,-0.124268,-0.325952,-0.291204,-0.2831,-0.241056,-0.241745,-0.232814,-0.319929,-0.260222,-0.0643623,-0.797857,-0.255868,-0.423334,-0.21906,-0.292107,0.137825,
    -5920.6,3,0.104225,-0.181149,0.0084497,-2.60067,-2.36891,-1.98918,-2.42548,-2.45579,-1.98555,-1.53862,-2.31263,-2.00816,-2.24248,-2.27819,-1.34942,-2.98162,-2.9084,-1.38994,-1.53782,-2.20171,-3.36967,-3.30242,-1.27175,-1.89589,-3.19499,-0.242842,-0.254323,-0.336628,-0.0500155,-0.0773821,-0.266518,-0.462303,-0.158688,-0.193312,-0.274016,-0.192628,-0.276757,-0.20334,0.153512,-0.00977478,-0.094097,-0.0122078,-0.0125588,-0.226709,-0.179547,-0.466747,-0.262438,-0.303859,0.0919223,
    -5926.75,3,0.104225,-0.138456,0.0120685,-2.44305,-2.24172,-2.23265,-2.50717,-2.46876,-2.00241,-1.57722,-2.27684,-1.85228,-2.2156,-2.35399,-1.41582,-2.7912,-2.94016,-1.36884,-1.50599,-2.2534,-3.50275,-3.27105,-1.38539,-1.93125,-3.11444,-0.297937,-0.324996,-0.314406,-0.0338911,-0.0923396,-0.0247583,-0.499841,-0.18307,-0.179302,-0.258559,-0.174569,-0.251664,-0.230094,0.0998433,-0.0475798,-0.0392007,-0.158131,-0.161068,-0.402353,-0.295153,-0.568905,-0.37115,-0.438521,0.109857;
  
  for (int i = 0; i < 5; i++) 
    for (int j = 0; j < 51; j++)
      EXPECT_FLOAT_EQ(expected_samples(i,j), samples(i,j))
	<< "comparison failed for (" << i << "," << j << ")";
}

TEST_F(StanIoStanCsvReader,ParseBlocker) {
  stan::io::stan_csv blocker0;
  blocker0 = stan::io::stan_csv_reader::parse(blocker0_stream);

  // metadata
  EXPECT_EQ(1U, blocker0.metadata.stan_version_major);
  EXPECT_EQ(1U, blocker0.metadata.stan_version_minor);
  EXPECT_EQ(1U, blocker0.metadata.stan_version_patch);
  
  EXPECT_EQ("models\\bugs_examples\\vol1\\blocker\\blocker.data.R", blocker0.metadata.data);
  EXPECT_EQ("models\\bugs_examples\\vol1\\blocker\\blocker.init.R", blocker0.metadata.init);
  EXPECT_EQ(false, blocker0.metadata.append_samples);
  EXPECT_EQ(false, blocker0.metadata.save_warmup);
  EXPECT_EQ(4085885484U, blocker0.metadata.seed);
  EXPECT_EQ(false, blocker0.metadata.random_seed);
  EXPECT_EQ(0U, blocker0.metadata.chain_id);
  EXPECT_EQ(4000U, blocker0.metadata.iter);
  EXPECT_EQ(2000U, blocker0.metadata.warmup);
  EXPECT_EQ(2, blocker0.metadata.thin);
  EXPECT_EQ(false, blocker0.metadata.equal_step_sizes);
  EXPECT_EQ(-1, blocker0.metadata.leapfrog_steps);
  EXPECT_EQ(10, blocker0.metadata.max_treedepth);
  EXPECT_FLOAT_EQ(-1, blocker0.metadata.epsilon);
  EXPECT_FLOAT_EQ(0, blocker0.metadata.epsilon_pm);
  EXPECT_FLOAT_EQ(0.5, blocker0.metadata.delta);
  EXPECT_FLOAT_EQ(0.05, blocker0.metadata.gamma);

  // header
  ASSERT_EQ(51, blocker0.header.size());
  EXPECT_EQ("lp__", blocker0.header(0));
  EXPECT_EQ("treedepth__",blocker0.header(1));
  EXPECT_EQ("stepsize__",blocker0.header(2));
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
  EXPECT_EQ("mcmc::nuts_diag", blocker0.adaptation.sampler);
  EXPECT_FLOAT_EQ(0.104225, blocker0.adaptation.step_size);
  ASSERT_EQ(47, blocker0.adaptation.step_size_multipliers.size());
  EXPECT_FLOAT_EQ(0.325114, blocker0.adaptation.step_size_multipliers(0));
  EXPECT_FLOAT_EQ(3.51248, blocker0.adaptation.step_size_multipliers(1));
  EXPECT_FLOAT_EQ(1.93143, blocker0.adaptation.step_size_multipliers(2));
  EXPECT_FLOAT_EQ(1.08468, blocker0.adaptation.step_size_multipliers(3));
  EXPECT_FLOAT_EQ(1.23781, blocker0.adaptation.step_size_multipliers(4));
  EXPECT_FLOAT_EQ(0.365398, blocker0.adaptation.step_size_multipliers(5));
  EXPECT_FLOAT_EQ(0.710215, blocker0.adaptation.step_size_multipliers(6));
  EXPECT_FLOAT_EQ(1.54178, blocker0.adaptation.step_size_multipliers(7));
  EXPECT_FLOAT_EQ(0.375339, blocker0.adaptation.step_size_multipliers(8));
  EXPECT_FLOAT_EQ(0.574698, blocker0.adaptation.step_size_multipliers(9));
  EXPECT_FLOAT_EQ(0.722729, blocker0.adaptation.step_size_multipliers(10));
  EXPECT_FLOAT_EQ(0.318383, blocker0.adaptation.step_size_multipliers(11));
  EXPECT_FLOAT_EQ(0.563756, blocker0.adaptation.step_size_multipliers(12));
  EXPECT_FLOAT_EQ(0.616147, blocker0.adaptation.step_size_multipliers(13));
  EXPECT_FLOAT_EQ(0.995688, blocker0.adaptation.step_size_multipliers(14));
  EXPECT_FLOAT_EQ(0.657117, blocker0.adaptation.step_size_multipliers(15));
  EXPECT_FLOAT_EQ(0.7757, blocker0.adaptation.step_size_multipliers(16));
  EXPECT_FLOAT_EQ(0.689116, blocker0.adaptation.step_size_multipliers(17));
  EXPECT_FLOAT_EQ(0.890119, blocker0.adaptation.step_size_multipliers(18));
  EXPECT_FLOAT_EQ(1.29209, blocker0.adaptation.step_size_multipliers(19));
  EXPECT_FLOAT_EQ(1.66928, blocker0.adaptation.step_size_multipliers(20));
  EXPECT_FLOAT_EQ(0.653107, blocker0.adaptation.step_size_multipliers(21));
  EXPECT_FLOAT_EQ(0.692784, blocker0.adaptation.step_size_multipliers(22));
  EXPECT_FLOAT_EQ(0.647013, blocker0.adaptation.step_size_multipliers(23));
  EXPECT_FLOAT_EQ(0.908997, blocker0.adaptation.step_size_multipliers(24));
  EXPECT_FLOAT_EQ(0.965595, blocker0.adaptation.step_size_multipliers(25));
  EXPECT_FLOAT_EQ(0.918005, blocker0.adaptation.step_size_multipliers(26));
  EXPECT_FLOAT_EQ(0.444468, blocker0.adaptation.step_size_multipliers(27));
  EXPECT_FLOAT_EQ(0.703633, blocker0.adaptation.step_size_multipliers(28));
  EXPECT_FLOAT_EQ(0.917533, blocker0.adaptation.step_size_multipliers(29));
  EXPECT_FLOAT_EQ(0.533222, blocker0.adaptation.step_size_multipliers(30));
  EXPECT_FLOAT_EQ(0.613215, blocker0.adaptation.step_size_multipliers(31));
  EXPECT_FLOAT_EQ(0.728461, blocker0.adaptation.step_size_multipliers(32));
  EXPECT_FLOAT_EQ(0.420359, blocker0.adaptation.step_size_multipliers(33));
  EXPECT_FLOAT_EQ(0.608283, blocker0.adaptation.step_size_multipliers(34));
  EXPECT_FLOAT_EQ(0.692184, blocker0.adaptation.step_size_multipliers(35));
  EXPECT_FLOAT_EQ(0.817112, blocker0.adaptation.step_size_multipliers(36));
  EXPECT_FLOAT_EQ(0.86689, blocker0.adaptation.step_size_multipliers(37));
  EXPECT_FLOAT_EQ(0.694276, blocker0.adaptation.step_size_multipliers(38));
  EXPECT_FLOAT_EQ(0.676297, blocker0.adaptation.step_size_multipliers(39));
  EXPECT_FLOAT_EQ(0.788411, blocker0.adaptation.step_size_multipliers(40));
  EXPECT_FLOAT_EQ(0.983241, blocker0.adaptation.step_size_multipliers(41));
  EXPECT_FLOAT_EQ(0.873982, blocker0.adaptation.step_size_multipliers(42));
  EXPECT_FLOAT_EQ(0.646372, blocker0.adaptation.step_size_multipliers(43));
  EXPECT_FLOAT_EQ(0.718841, blocker0.adaptation.step_size_multipliers(44));
  EXPECT_FLOAT_EQ(0.667207, blocker0.adaptation.step_size_multipliers(45));
  EXPECT_FLOAT_EQ(1.33936, blocker0.adaptation.step_size_multipliers(46));

  // samples
  ASSERT_EQ(1000, blocker0.samples.rows());
  ASSERT_EQ(51, blocker0.samples.cols());

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> expected_samples(6, 51);
  expected_samples <<
    -5923.71,3,0.104225,-0.31639,0.0064363,-2.0354,-1.94524,-2.06081,-2.31731,-2.54922,-2.50787,-1.74407,-2.04826,-1.75427,-2.16927,-2.31893,-1.37094,-2.8661,-2.67123,-1.36898,-1.43593,-1.8091,-2.92111,-3.98671,-1.4164,-1.89509,-2.94966,-0.557794,-0.483134,-0.156347,-0.250073,-0.121718,-0.243917,-0.327415,-0.127961,-0.238188,-0.425699,-0.280189,-0.294234,-0.442824,-0.30521,0.0666663,-0.457187,-0.270417,-0.247473,-0.402816,-0.20402,-0.346646,-0.306836,-0.436561,0.0802265,
    -5931.85,3,0.104225,-0.372371,0.0143758,-2.53617,-1.9066,-2.69784,-2.34748,-2.40404,-1.84627,-1.76547,-1.95062,-2.02309,-2.26443,-2.35281,-1.65016,-2.81412,-2.59655,-1.3871,-1.36819,-2.27108,-2.82077,-2.94598,-1.14072,-2.0336,-3.24552,-0.429167,-0.345602,-0.255696,-0.40927,-0.207543,-0.556137,-0.462445,-0.204487,-0.216857,-0.324489,-0.230653,-0.117995,-0.603386,-0.276107,-0.256322,-0.491629,0.252045,-0.199305,-0.638357,-0.285059,-0.364338,-0.269418,-0.470453,0.119899,
    -5930.95,3,0.104225,-0.27131,0.0189956,-3.00112,-2.24849,-2.64399,-2.31772,-2.80533,-2.01866,-1.69406,-2.29215,-2.03636,-2.20395,-2.24996,-1.46979,-3.08446,-2.61183,-1.34088,-1.42055,-1.83887,-2.9487,-3.28892,-1.21848,-1.89679,-3.31541,-0.349814,-0.380662,-0.317758,-0.19059,0.326377,-0.526384,-0.276535,-0.226859,-0.124268,-0.325952,-0.291204,-0.2831,-0.241056,-0.241745,-0.232814,-0.319929,-0.260222,-0.0643623,-0.797857,-0.255868,-0.423334,-0.21906,-0.292107,0.137825,
    -5920.6,3,0.104225,-0.181149,0.0084497,-2.60067,-2.36891,-1.98918,-2.42548,-2.45579,-1.98555,-1.53862,-2.31263,-2.00816,-2.24248,-2.27819,-1.34942,-2.98162,-2.9084,-1.38994,-1.53782,-2.20171,-3.36967,-3.30242,-1.27175,-1.89589,-3.19499,-0.242842,-0.254323,-0.336628,-0.0500155,-0.0773821,-0.266518,-0.462303,-0.158688,-0.193312,-0.274016,-0.192628,-0.276757,-0.20334,0.153512,-0.00977478,-0.094097,-0.0122078,-0.0125588,-0.226709,-0.179547,-0.466747,-0.262438,-0.303859,0.0919223,
    -5926.75,3,0.104225,-0.138456,0.0120685,-2.44305,-2.24172,-2.23265,-2.50717,-2.46876,-2.00241,-1.57722,-2.27684,-1.85228,-2.2156,-2.35399,-1.41582,-2.7912,-2.94016,-1.36884,-1.50599,-2.2534,-3.50275,-3.27105,-1.38539,-1.93125,-3.11444,-0.297937,-0.324996,-0.314406,-0.0338911,-0.0923396,-0.0247583,-0.499841,-0.18307,-0.179302,-0.258559,-0.174569,-0.251664,-0.230094,0.0998433,-0.0475798,-0.0392007,-0.158131,-0.161068,-0.402353,-0.295153,-0.568905,-0.37115,-0.438521,0.109857,
    -5930.32,3,0.104225,-0.33959,0.0082748,-2.24292,-2.59292,-2.14996,-2.29561,-2.23006,-2.22206,-1.73513,-2.14952,-1.91364,-2.40552,-2.05,-1.62903,-2.90922,-2.73344,-1.44031,-1.71372,-2.07111,-3.34253,-2.94646,-1.53898,-2.11676,-2.88289,-0.347802,-0.166085,-0.380007,-0.272417,-0.439907,-0.372669,-0.644076,-0.200165,-0.309358,-0.223815,-0.362217,-0.366416,-0.196985,-0.197497,-0.293263,-0.189633,-0.413598,-0.214359,0.139252,-0.198864,-0.297768,-0.396228,-0.359652,0.0909659;
  
  for (int i = 0; i < 6; i++) 
    for (int j = 0; j < 51; j++)
      EXPECT_FLOAT_EQ(expected_samples(i,j), blocker0.samples(i,j))
	<< "comparison failed for (" << i << "," << j << ")";
}

TEST_F(StanIoStanCsvReader,read_metadata2) {
  stan::io::stan_csv_metadata metadata;
  EXPECT_TRUE(stan::io::stan_csv_reader::read_metadata(metadata2_stream, metadata));

  EXPECT_EQ(1U, metadata.stan_version_major);
  EXPECT_EQ(1U, metadata.stan_version_minor);
  EXPECT_EQ(1U, metadata.stan_version_patch);
  
  EXPECT_EQ("models\\bugs_examples\\vol1\\epil\\epil.data.R", metadata.data);
  EXPECT_EQ("random initialization", metadata.init);
  EXPECT_EQ(false, metadata.append_samples);
  EXPECT_EQ(false, metadata.save_warmup);
  EXPECT_EQ(4258844633, metadata.seed);
  EXPECT_EQ(false, metadata.random_seed);
  EXPECT_EQ(0U, metadata.chain_id);
  EXPECT_EQ(2000, metadata.iter);
  EXPECT_EQ(1000, metadata.warmup);
  EXPECT_EQ(1, metadata.thin);
  EXPECT_EQ(false, metadata.equal_step_sizes);
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
  EXPECT_EQ("treedepth__", header(1));
  EXPECT_EQ("stepsize__", header(2));
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
  
  EXPECT_EQ("mcmc::nuts_diag", adaptation.sampler);
  EXPECT_FLOAT_EQ(0.0329391, adaptation.step_size);
  ASSERT_EQ(303, adaptation.step_size_multipliers.size());
  EXPECT_FLOAT_EQ(0.24468, adaptation.step_size_multipliers(0));
  EXPECT_FLOAT_EQ(0.475192, adaptation.step_size_multipliers(1));
  EXPECT_FLOAT_EQ(1.40967, adaptation.step_size_multipliers(2));
  // ...
  EXPECT_FLOAT_EQ(0.783541, adaptation.step_size_multipliers(301));
  EXPECT_FLOAT_EQ(0.897251, adaptation.step_size_multipliers(302));
}

TEST_F(StanIoStanCsvReader,read_samples2) { 
  Eigen::MatrixXd samples;
  EXPECT_TRUE(stan::io::stan_csv_reader::read_samples(samples2_stream, samples));

  ASSERT_EQ(3, samples.rows());
  ASSERT_EQ(309, samples.cols());

  Eigen::MatrixXd expected_samples(3, 309);
  expected_samples <<
    3424.22,5,0.0329391,1.61039,0.826045,-1.39472,0.687673,0.762373,0.022646,0.0826075,0.0508744,0.391101,-0.0956092,0.350164,0.206973,0.00714559,0.365607,-0.29502,0.803028,0.31197,0.0420494,0.241144,-0.0247595,-0.233713,-1.36818,-1.13302,0.269616,-0.0764419,0.0631729,0.294498,0.237991,0.0140363,-0.351328,0.743448,-0.939778,0.33465,0.499292,-0.722415,-0.544638,-0.616131,1.05877,0.938631,-1.03576,0.640407,0.866087,0.42525,-1.2794,-0.565656,0.20039,-0.434551,-0.332087,0.0471453,-0.219348,0.0452685,0.0436157,0.165612,-0.484468,-0.0239643,-0.409058,-0.236811,-1.13149,0.156999,-0.598726,-0.429529,1.1218,-0.320064,-0.980336,-0.0731693,0.429689,0.365287,0.0518742,0.587699,0.236905,-0.116221,-0.092645,0.392289,0.145638,-0.251316,-0.0752843,0.474348,-0.0618829,-0.174935,-0.0676768,-0.0698,0.0088803,-0.0655355,0.0283244,0.27969,0.0462874,0.0205615,0.0304115,0.430673,0.155516,0.159107,-0.593034,-0.149458,0.537684,-0.0795061,-0.0649155,-0.373151,-0.366243,0.0237465,0.288847,-0.721316,0.452319,0.211053,-0.434077,-0.423056,0.473969,0.082728,-0.514061,-0.0278643,0.427484,-0.122786,-0.144033,-0.196445,0.153959,-0.391194,0.0900982,-0.401386,-0.443562,-0.279207,-0.193079,0.0912218,0.722528,0.477155,-0.208542,-0.393619,0.569678,-0.189731,-0.223391,0.837297,-0.162696,0.0684816,-0.334646,0.237708,0.139365,-0.0251705,-0.0413567,-0.140664,0.0832137,0.219908,-0.518884,0.075642,0.00975988,-0.121598,0.129981,0.342547,-0.417353,0.146226,0.0197883,-0.483455,-0.0831054,0.421645,0.0155905,0.2955,-0.64083,-0.0512395,0.140705,0.0248991,0.328748,0.480913,0.136294,0.241574,-0.0420278,-0.227889,0.817389,-0.254315,0.07929,-0.183466,0.327748,-0.528722,0.022682,-0.135913,-0.0612773,-0.0180904,0.0658847,-0.0124776,0.162385,-0.209024,-0.354235,-0.0253943,-0.288547,-0.32335,-0.170661,0.382418,0.295373,-0.360842,0.401828,0.777741,0.0583004,-0.539385,0.163524,-0.0361224,0.302313,-0.120457,-1.01838,-0.327161,-0.0125101,0.0518289,-0.0346203,0.636714,-0.18493,0.631269,0.136574,-0.0183379,0.644956,-0.229762,0.0712465,-0.235251,-0.089891,-0.217998,0.293582,-0.054304,0.0552874,-0.0395218,-0.724418,0.397748,-0.164199,-0.172541,-0.227426,0.622009,-0.0218048,0.418244,0.163945,-0.186162,0.155522,-0.204957,-0.850852,0.346638,0.0461378,-0.586416,0.443563,0.131019,-0.453361,0.234799,-0.1108,0.152843,0.451918,-0.295081,-0.0300201,-0.0272349,0.0967838,-0.0163309,0.610888,-0.0363076,-0.404594,-0.162652,-0.30748,-0.614577,0.407039,0.204888,-0.269772,0.038154,0.0382237,-0.364989,0.640229,0.182812,-0.205286,-0.0705009,0.37174,-0.00227053,0.13982,-0.148527,0.116377,0.155179,-0.423113,0.363509,0.134141,-0.273117,-0.291586,0.140179,-0.372191,-0.032141,-0.522246,0.592227,0.417149,-0.294915,0.102469,-0.51684,0.801336,0.0632711,0.381657,-0.634146,-0.265461,0.102963,0.19102,0.187851,-0.970572,0.502434,0.775024,-0.0859577,-0.599943,-0.0662822,-0.167819,0.322584,-0.351152,-0.0513993,0.0468622,0.0257681,0.408019,-0.0749289,0.0779259,0.704707,0.120187,0.318675,0.34668,0.564513,-2.30595,
    3418.74,5,0.0329391,1.61571,0.933598,-1.21919,0.546408,0.750927,-0.0449031,0.0831532,0.490216,0.324493,-0.0836644,-0.165352,-1.00373,0.00932553,0.100307,0.0330245,0.91682,-0.214042,0.243671,0.0804622,0.263925,-0.432691,-1.13664,-0.321077,-0.201329,-0.55494,-0.435819,0.18106,-0.21245,0.212442,0.0554528,0.717025,0.131089,-0.133229,0.152722,-0.187442,-0.398495,-0.187646,0.192597,0.662316,-0.5264,0.63355,0.306176,0.6208,-1.00058,-0.221678,-0.40203,-1.10454,0.370739,0.118384,0.10698,-0.427095,0.544655,0.0552674,-0.830099,0.244045,-0.576995,-0.142457,-0.754028,0.357272,-0.33775,-0.0781952,0.306807,-1.67354,-0.341974,0.193405,-0.20332,-0.755089,-0.131213,-0.396147,0.00055802,-0.137188,-0.0619917,-0.550208,0.219521,-0.262356,0.110996,0.323668,-0.110075,0.44392,-0.317418,0.641339,-0.610778,0.228727,-0.672223,0.441519,0.347973,0.0457679,0.0357819,0.338266,0.613677,-0.294188,-0.146891,-0.0803564,0.623128,0.259229,0.165599,-0.0991599,-0.182051,-0.184303,-0.0660666,0.243814,0.630669,0.496878,0.145177,-0.793429,0.643221,-0.0271669,-0.513524,0.287752,-0.0926585,-0.10634,0.0722246,-0.455539,0.0173928,-0.393675,-0.0105273,0.226448,-0.271765,-0.0747771,-0.272545,-0.0307679,0.100748,0.528199,-0.159555,-0.0771378,0.689079,-0.569188,-0.51079,0.120335,0.0162313,-0.243872,-0.359517,0.160259,0.389814,0.391497,0.0693341,0.389439,-0.0453418,-0.225082,-0.309103,0.195721,0.0593156,0.181809,-0.0436763,-0.164509,-0.262134,-0.222312,0.170494,0.224611,0.732533,-0.383062,0.651477,0.147027,-0.260691,-0.550937,-0.860379,-0.12767,0.434868,0.443083,-0.357304,-0.0221471,-0.428198,-0.0942307,0.857596,-0.0568024,0.142052,-0.181839,0.421561,-0.425749,0.411209,0.262863,0.269932,0.0945732,-0.0607488,0.0396529,0.116448,-0.260526,-0.215878,0.200856,-0.42874,-0.474039,-0.0585186,-0.136097,0.153867,0.185534,-0.345233,-0.134012,0.160127,0.210495,0.0857351,0.410708,-0.235702,0.0568755,-0.143165,0.188295,0.271012,-0.158548,0.317397,-0.408797,-0.232688,-0.0767663,0.463491,0.291604,0.189074,0.178333,0.127813,0.0756293,0.444773,0.464329,0.0866102,0.0956736,0.0645217,-0.57326,-0.332839,0.192414,0.651095,0.2649,-0.557747,0.848309,-0.292117,-0.109691,0.27357,0.566901,0.213398,0.561347,-0.444745,0.372279,0.30686,-0.316519,0.0208837,0.256118,-0.810658,0.312575,-0.70536,-0.399529,0.57846,-0.15973,0.368443,-0.604666,-0.250912,0.0349324,0.353727,-0.364628,-0.00804545,0.116765,-0.410575,0.49229,-0.18064,0.598558,-0.201898,-0.0300683,-0.641877,0.345767,-0.310557,0.22957,0.2209,0.340372,0.263762,-0.191375,-0.151514,-0.0942615,0.402104,0.0629431,0.451281,0.186138,-0.0058844,-0.286493,-0.405829,-0.111944,-0.272497,-0.34497,0.0870521,0.102918,-0.376425,-0.69523,0.397752,-0.0907688,-0.387629,-0.162276,-0.642227,0.182955,-0.426664,0.282094,0.608335,-0.338179,-0.296611,1.0051,0.439554,-0.00711828,-0.0373566,0.618951,-0.439179,-0.158118,-0.0839996,0.237265,-0.455057,0.30245,-0.51878,0.0520716,0.237671,-0.0941439,0.123822,0.272011,0.351884,0.521547,-2.39414,
    3407.21,5,0.0329391,1.55163,1.08668,-0.185311,0.0516193,0.540505,-0.123652,0.7124,0.562882,0.823631,0.197424,0.153752,-0.0533528,0.00648923,0.0426531,0.0465048,0.84167,0.318667,-0.27806,0.488991,-0.512559,-0.485717,-1.21997,-0.972358,0.192378,0.536718,-0.0109068,-0.0116751,0.536098,0.433199,-0.0881594,0.752565,-0.635814,0.11942,0.609688,-0.333269,-0.261483,-0.701251,0.409382,0.176735,-0.215629,0.55182,0.24315,0.36769,-0.780017,-0.289419,0.0890808,-0.537387,0.0279583,0.538032,-0.125365,0.267132,0.143937,0.173131,-0.19215,0.818874,-0.688349,-0.315856,-0.795018,0.694019,0.00163254,-0.183884,0.787485,-0.587735,-1.51426,-0.120957,0.24376,0.00429144,-0.263666,-0.484695,0.145582,-0.463825,-0.164282,-0.511264,0.0876946,-0.0623181,-0.535772,0.577174,0.132861,0.0920713,-0.580456,0.31792,-0.120866,0.0838642,0.0561963,0.0688509,-0.316184,-0.293046,0.154516,0.530937,-0.318074,-0.285955,-0.491936,-0.304913,0.590882,-0.0946603,0.328245,0.0487597,-0.361439,0.0493364,-0.0905128,-0.6335,0.433625,0.418482,-0.0553955,-0.106758,0.115915,0.00917687,-0.507166,-0.103701,0.677595,0.0577508,0.0969565,-0.075613,-0.341538,0.335934,0.106096,-0.425771,-0.0814507,-0.249472,0.370185,0.504796,0.0647769,0.343328,0.249871,-0.373504,0.957196,-1.02245,0.295785,0.222688,0.312174,-0.208276,0.0821213,-0.11995,0.100902,-0.243075,-0.249825,-0.223344,-0.424526,0.184163,-0.0649065,-0.568508,-0.127637,-0.0789986,0.127397,0.811828,0.368516,-0.245377,0.219046,-0.158219,0.238531,0.308506,-0.0642331,0.0465033,-0.496083,0.0319447,-0.555712,-0.377563,0.275885,-0.195162,0.215085,-0.00616127,-0.443122,0.0462086,0.726816,-0.2386,-0.273993,-0.403529,0.0117012,-0.386468,0.10618,0.0893576,0.540108,-0.235931,-0.298039,0.0165023,-0.46146,-0.0846554,-0.27603,0.147042,-0.285387,-0.481777,0.193712,0.0652229,0.0663355,-0.0147696,-0.349264,-0.0459635,-0.0977288,-1.00047,0.300684,0.812489,-0.942193,0.021367,-0.0090619,-0.0840107,0.280483,0.246136,0.0766418,-0.572347,0.156612,-0.0596813,0.548515,0.0620647,0.305397,0.443155,0.062549,0.0440643,0.689189,0.1375,0.117455,0.541275,-0.0367812,0.30266,-0.428283,0.320655,-0.168203,0.183626,-0.202138,0.907439,-0.767922,0.248143,0.398126,0.315522,-0.195069,-0.719402,-0.329534,0.42175,0.790238,-0.601592,0.292544,0.105631,-0.505753,-0.0627191,-0.210633,0.148667,-0.0930104,0.0965733,0.0421983,-0.126469,0.152541,-0.467016,0.172419,0.246155,-0.509015,-0.114483,-0.501521,0.0727728,0.223696,0.511898,-0.300001,-0.269342,0.31238,0.418857,0.863455,-0.220907,-0.542542,-0.328938,0.267997,-0.132501,-0.0185591,-0.053648,0.0247002,-0.146737,0.121626,0.465647,0.299831,0.0323686,0.00462897,-0.132831,-0.296215,-0.302456,-0.276987,-0.766698,0.140129,-0.21649,-0.25657,-0.504698,0.349671,-0.354638,0.13319,-0.415817,0.443977,0.238312,0.0929591,0.0945705,-0.00317493,0.731805,1.04992,-0.137632,-0.0317397,0.502012,-0.587553,0.0902527,-0.188334,0.0630904,-0.30483,0.0903956,0.242313,-0.0368947,-0.0440031,-0.168151,0.12522,0.270894,0.353864,0.520475,-2.08461;

  for (int i = 0; i < 3; i++) 
    for (int j = 0; j < 309; j++)
      EXPECT_FLOAT_EQ(expected_samples(i,j), samples(i,j))
	<< "comparison failed for (" << i << "," << j << ")";
}

TEST_F(StanIoStanCsvReader,ParseEpil) {
  stan::io::stan_csv epil0;
  epil0 = stan::io::stan_csv_reader::parse(epil0_stream);

  // metadata
  EXPECT_EQ(1U, epil0.metadata.stan_version_major);
  EXPECT_EQ(1U, epil0.metadata.stan_version_minor);
  EXPECT_EQ(1U, epil0.metadata.stan_version_patch);
  
  EXPECT_EQ("models\\bugs_examples\\vol1\\epil\\epil.data.R", epil0.metadata.data);
  EXPECT_EQ("random initialization", epil0.metadata.init);
  EXPECT_EQ(false, epil0.metadata.append_samples);
  EXPECT_EQ(false, epil0.metadata.save_warmup);
  EXPECT_EQ(4258844633, epil0.metadata.seed);
  EXPECT_EQ(false, epil0.metadata.random_seed);
  EXPECT_EQ(0U, epil0.metadata.chain_id);
  EXPECT_EQ(2000, epil0.metadata.iter);
  EXPECT_EQ(1000, epil0.metadata.warmup);
  EXPECT_EQ(1, epil0.metadata.thin);
  EXPECT_EQ(false, epil0.metadata.equal_step_sizes);
  EXPECT_EQ(-1, epil0.metadata.leapfrog_steps);
  EXPECT_EQ(10, epil0.metadata.max_treedepth);
  EXPECT_FLOAT_EQ(-1, epil0.metadata.epsilon);
  EXPECT_FLOAT_EQ(0, epil0.metadata.epsilon_pm);
  EXPECT_FLOAT_EQ(0.5, epil0.metadata.delta);
  EXPECT_FLOAT_EQ(0.05, epil0.metadata.gamma);

  // header
  ASSERT_EQ(309, epil0.header.size());
  EXPECT_EQ("lp__", epil0.header(0));
  EXPECT_EQ("treedepth__", epil0.header(1));
  EXPECT_EQ("stepsize__", epil0.header(2));
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
  EXPECT_EQ("mcmc::nuts_diag", epil0.adaptation.sampler);
  EXPECT_FLOAT_EQ(0.0329391, epil0.adaptation.step_size);
  ASSERT_EQ(303, epil0.adaptation.step_size_multipliers.size());
  EXPECT_FLOAT_EQ(0.24468, epil0.adaptation.step_size_multipliers(0));
  EXPECT_FLOAT_EQ(0.475192, epil0.adaptation.step_size_multipliers(1));
  EXPECT_FLOAT_EQ(1.40967, epil0.adaptation.step_size_multipliers(2));
  // ...
  EXPECT_FLOAT_EQ(0.783541, epil0.adaptation.step_size_multipliers(301));
  EXPECT_FLOAT_EQ(0.897251, epil0.adaptation.step_size_multipliers(302));

  // samples
  ASSERT_EQ(1000, epil0.samples.rows());
  ASSERT_EQ(309, epil0.samples.cols());

  Eigen::MatrixXd expected_samples(3, 309);
  expected_samples <<
    3424.22,5,0.0329391,1.61039,0.826045,-1.39472,0.687673,0.762373,0.022646,0.0826075,0.0508744,0.391101,-0.0956092,0.350164,0.206973,0.00714559,0.365607,-0.29502,0.803028,0.31197,0.0420494,0.241144,-0.0247595,-0.233713,-1.36818,-1.13302,0.269616,-0.0764419,0.0631729,0.294498,0.237991,0.0140363,-0.351328,0.743448,-0.939778,0.33465,0.499292,-0.722415,-0.544638,-0.616131,1.05877,0.938631,-1.03576,0.640407,0.866087,0.42525,-1.2794,-0.565656,0.20039,-0.434551,-0.332087,0.0471453,-0.219348,0.0452685,0.0436157,0.165612,-0.484468,-0.0239643,-0.409058,-0.236811,-1.13149,0.156999,-0.598726,-0.429529,1.1218,-0.320064,-0.980336,-0.0731693,0.429689,0.365287,0.0518742,0.587699,0.236905,-0.116221,-0.092645,0.392289,0.145638,-0.251316,-0.0752843,0.474348,-0.0618829,-0.174935,-0.0676768,-0.0698,0.0088803,-0.0655355,0.0283244,0.27969,0.0462874,0.0205615,0.0304115,0.430673,0.155516,0.159107,-0.593034,-0.149458,0.537684,-0.0795061,-0.0649155,-0.373151,-0.366243,0.0237465,0.288847,-0.721316,0.452319,0.211053,-0.434077,-0.423056,0.473969,0.082728,-0.514061,-0.0278643,0.427484,-0.122786,-0.144033,-0.196445,0.153959,-0.391194,0.0900982,-0.401386,-0.443562,-0.279207,-0.193079,0.0912218,0.722528,0.477155,-0.208542,-0.393619,0.569678,-0.189731,-0.223391,0.837297,-0.162696,0.0684816,-0.334646,0.237708,0.139365,-0.0251705,-0.0413567,-0.140664,0.0832137,0.219908,-0.518884,0.075642,0.00975988,-0.121598,0.129981,0.342547,-0.417353,0.146226,0.0197883,-0.483455,-0.0831054,0.421645,0.0155905,0.2955,-0.64083,-0.0512395,0.140705,0.0248991,0.328748,0.480913,0.136294,0.241574,-0.0420278,-0.227889,0.817389,-0.254315,0.07929,-0.183466,0.327748,-0.528722,0.022682,-0.135913,-0.0612773,-0.0180904,0.0658847,-0.0124776,0.162385,-0.209024,-0.354235,-0.0253943,-0.288547,-0.32335,-0.170661,0.382418,0.295373,-0.360842,0.401828,0.777741,0.0583004,-0.539385,0.163524,-0.0361224,0.302313,-0.120457,-1.01838,-0.327161,-0.0125101,0.0518289,-0.0346203,0.636714,-0.18493,0.631269,0.136574,-0.0183379,0.644956,-0.229762,0.0712465,-0.235251,-0.089891,-0.217998,0.293582,-0.054304,0.0552874,-0.0395218,-0.724418,0.397748,-0.164199,-0.172541,-0.227426,0.622009,-0.0218048,0.418244,0.163945,-0.186162,0.155522,-0.204957,-0.850852,0.346638,0.0461378,-0.586416,0.443563,0.131019,-0.453361,0.234799,-0.1108,0.152843,0.451918,-0.295081,-0.0300201,-0.0272349,0.0967838,-0.0163309,0.610888,-0.0363076,-0.404594,-0.162652,-0.30748,-0.614577,0.407039,0.204888,-0.269772,0.038154,0.0382237,-0.364989,0.640229,0.182812,-0.205286,-0.0705009,0.37174,-0.00227053,0.13982,-0.148527,0.116377,0.155179,-0.423113,0.363509,0.134141,-0.273117,-0.291586,0.140179,-0.372191,-0.032141,-0.522246,0.592227,0.417149,-0.294915,0.102469,-0.51684,0.801336,0.0632711,0.381657,-0.634146,-0.265461,0.102963,0.19102,0.187851,-0.970572,0.502434,0.775024,-0.0859577,-0.599943,-0.0662822,-0.167819,0.322584,-0.351152,-0.0513993,0.0468622,0.0257681,0.408019,-0.0749289,0.0779259,0.704707,0.120187,0.318675,0.34668,0.564513,-2.30595,
    3418.74,5,0.0329391,1.61571,0.933598,-1.21919,0.546408,0.750927,-0.0449031,0.0831532,0.490216,0.324493,-0.0836644,-0.165352,-1.00373,0.00932553,0.100307,0.0330245,0.91682,-0.214042,0.243671,0.0804622,0.263925,-0.432691,-1.13664,-0.321077,-0.201329,-0.55494,-0.435819,0.18106,-0.21245,0.212442,0.0554528,0.717025,0.131089,-0.133229,0.152722,-0.187442,-0.398495,-0.187646,0.192597,0.662316,-0.5264,0.63355,0.306176,0.6208,-1.00058,-0.221678,-0.40203,-1.10454,0.370739,0.118384,0.10698,-0.427095,0.544655,0.0552674,-0.830099,0.244045,-0.576995,-0.142457,-0.754028,0.357272,-0.33775,-0.0781952,0.306807,-1.67354,-0.341974,0.193405,-0.20332,-0.755089,-0.131213,-0.396147,0.00055802,-0.137188,-0.0619917,-0.550208,0.219521,-0.262356,0.110996,0.323668,-0.110075,0.44392,-0.317418,0.641339,-0.610778,0.228727,-0.672223,0.441519,0.347973,0.0457679,0.0357819,0.338266,0.613677,-0.294188,-0.146891,-0.0803564,0.623128,0.259229,0.165599,-0.0991599,-0.182051,-0.184303,-0.0660666,0.243814,0.630669,0.496878,0.145177,-0.793429,0.643221,-0.0271669,-0.513524,0.287752,-0.0926585,-0.10634,0.0722246,-0.455539,0.0173928,-0.393675,-0.0105273,0.226448,-0.271765,-0.0747771,-0.272545,-0.0307679,0.100748,0.528199,-0.159555,-0.0771378,0.689079,-0.569188,-0.51079,0.120335,0.0162313,-0.243872,-0.359517,0.160259,0.389814,0.391497,0.0693341,0.389439,-0.0453418,-0.225082,-0.309103,0.195721,0.0593156,0.181809,-0.0436763,-0.164509,-0.262134,-0.222312,0.170494,0.224611,0.732533,-0.383062,0.651477,0.147027,-0.260691,-0.550937,-0.860379,-0.12767,0.434868,0.443083,-0.357304,-0.0221471,-0.428198,-0.0942307,0.857596,-0.0568024,0.142052,-0.181839,0.421561,-0.425749,0.411209,0.262863,0.269932,0.0945732,-0.0607488,0.0396529,0.116448,-0.260526,-0.215878,0.200856,-0.42874,-0.474039,-0.0585186,-0.136097,0.153867,0.185534,-0.345233,-0.134012,0.160127,0.210495,0.0857351,0.410708,-0.235702,0.0568755,-0.143165,0.188295,0.271012,-0.158548,0.317397,-0.408797,-0.232688,-0.0767663,0.463491,0.291604,0.189074,0.178333,0.127813,0.0756293,0.444773,0.464329,0.0866102,0.0956736,0.0645217,-0.57326,-0.332839,0.192414,0.651095,0.2649,-0.557747,0.848309,-0.292117,-0.109691,0.27357,0.566901,0.213398,0.561347,-0.444745,0.372279,0.30686,-0.316519,0.0208837,0.256118,-0.810658,0.312575,-0.70536,-0.399529,0.57846,-0.15973,0.368443,-0.604666,-0.250912,0.0349324,0.353727,-0.364628,-0.00804545,0.116765,-0.410575,0.49229,-0.18064,0.598558,-0.201898,-0.0300683,-0.641877,0.345767,-0.310557,0.22957,0.2209,0.340372,0.263762,-0.191375,-0.151514,-0.0942615,0.402104,0.0629431,0.451281,0.186138,-0.0058844,-0.286493,-0.405829,-0.111944,-0.272497,-0.34497,0.0870521,0.102918,-0.376425,-0.69523,0.397752,-0.0907688,-0.387629,-0.162276,-0.642227,0.182955,-0.426664,0.282094,0.608335,-0.338179,-0.296611,1.0051,0.439554,-0.00711828,-0.0373566,0.618951,-0.439179,-0.158118,-0.0839996,0.237265,-0.455057,0.30245,-0.51878,0.0520716,0.237671,-0.0941439,0.123822,0.272011,0.351884,0.521547,-2.39414,
    3407.21,5,0.0329391,1.55163,1.08668,-0.185311,0.0516193,0.540505,-0.123652,0.7124,0.562882,0.823631,0.197424,0.153752,-0.0533528,0.00648923,0.0426531,0.0465048,0.84167,0.318667,-0.27806,0.488991,-0.512559,-0.485717,-1.21997,-0.972358,0.192378,0.536718,-0.0109068,-0.0116751,0.536098,0.433199,-0.0881594,0.752565,-0.635814,0.11942,0.609688,-0.333269,-0.261483,-0.701251,0.409382,0.176735,-0.215629,0.55182,0.24315,0.36769,-0.780017,-0.289419,0.0890808,-0.537387,0.0279583,0.538032,-0.125365,0.267132,0.143937,0.173131,-0.19215,0.818874,-0.688349,-0.315856,-0.795018,0.694019,0.00163254,-0.183884,0.787485,-0.587735,-1.51426,-0.120957,0.24376,0.00429144,-0.263666,-0.484695,0.145582,-0.463825,-0.164282,-0.511264,0.0876946,-0.0623181,-0.535772,0.577174,0.132861,0.0920713,-0.580456,0.31792,-0.120866,0.0838642,0.0561963,0.0688509,-0.316184,-0.293046,0.154516,0.530937,-0.318074,-0.285955,-0.491936,-0.304913,0.590882,-0.0946603,0.328245,0.0487597,-0.361439,0.0493364,-0.0905128,-0.6335,0.433625,0.418482,-0.0553955,-0.106758,0.115915,0.00917687,-0.507166,-0.103701,0.677595,0.0577508,0.0969565,-0.075613,-0.341538,0.335934,0.106096,-0.425771,-0.0814507,-0.249472,0.370185,0.504796,0.0647769,0.343328,0.249871,-0.373504,0.957196,-1.02245,0.295785,0.222688,0.312174,-0.208276,0.0821213,-0.11995,0.100902,-0.243075,-0.249825,-0.223344,-0.424526,0.184163,-0.0649065,-0.568508,-0.127637,-0.0789986,0.127397,0.811828,0.368516,-0.245377,0.219046,-0.158219,0.238531,0.308506,-0.0642331,0.0465033,-0.496083,0.0319447,-0.555712,-0.377563,0.275885,-0.195162,0.215085,-0.00616127,-0.443122,0.0462086,0.726816,-0.2386,-0.273993,-0.403529,0.0117012,-0.386468,0.10618,0.0893576,0.540108,-0.235931,-0.298039,0.0165023,-0.46146,-0.0846554,-0.27603,0.147042,-0.285387,-0.481777,0.193712,0.0652229,0.0663355,-0.0147696,-0.349264,-0.0459635,-0.0977288,-1.00047,0.300684,0.812489,-0.942193,0.021367,-0.0090619,-0.0840107,0.280483,0.246136,0.0766418,-0.572347,0.156612,-0.0596813,0.548515,0.0620647,0.305397,0.443155,0.062549,0.0440643,0.689189,0.1375,0.117455,0.541275,-0.0367812,0.30266,-0.428283,0.320655,-0.168203,0.183626,-0.202138,0.907439,-0.767922,0.248143,0.398126,0.315522,-0.195069,-0.719402,-0.329534,0.42175,0.790238,-0.601592,0.292544,0.105631,-0.505753,-0.0627191,-0.210633,0.148667,-0.0930104,0.0965733,0.0421983,-0.126469,0.152541,-0.467016,0.172419,0.246155,-0.509015,-0.114483,-0.501521,0.0727728,0.223696,0.511898,-0.300001,-0.269342,0.31238,0.418857,0.863455,-0.220907,-0.542542,-0.328938,0.267997,-0.132501,-0.0185591,-0.053648,0.0247002,-0.146737,0.121626,0.465647,0.299831,0.0323686,0.00462897,-0.132831,-0.296215,-0.302456,-0.276987,-0.766698,0.140129,-0.21649,-0.25657,-0.504698,0.349671,-0.354638,0.13319,-0.415817,0.443977,0.238312,0.0929591,0.0945705,-0.00317493,0.731805,1.04992,-0.137632,-0.0317397,0.502012,-0.587553,0.0902527,-0.188334,0.0630904,-0.30483,0.0903956,0.242313,-0.0368947,-0.0440031,-0.168151,0.12522,0.270894,0.353864,0.520475,-2.08461;

  for (int i = 0; i < 3; i++) 
    for (int j = 0; j < 309; j++)
      EXPECT_FLOAT_EQ(expected_samples(i,j), epil0.samples(i,j))
	<< "comparison failed for (" << i << "," << j << ")";

}


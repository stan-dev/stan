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

  ASSERT_EQ(6, samples.rows());
  ASSERT_EQ(51, samples.cols());

  
  Eigen::MatrixXd expected_samples(6, 51);
  expected_samples <<
    -5923.71,3,0.104225,-0.31639,0.0064363,-2.0354,-1.94524,-2.06081,-2.31731,-2.54922,-2.50787,-1.74407,-2.04826,-1.75427,-2.16927,-2.31893,-1.37094,-2.8661,-2.67123,-1.36898,-1.43593,-1.8091,-2.92111,-3.98671,-1.4164,-1.89509,-2.94966,-0.557794,-0.483134,-0.156347,-0.250073,-0.121718,-0.243917,-0.327415,-0.127961,-0.238188,-0.425699,-0.280189,-0.294234,-0.442824,-0.30521,0.0666663,-0.457187,-0.270417,-0.247473,-0.402816,-0.20402,-0.346646,-0.306836,-0.436561,0.0802265,
    -5931.85,3,0.104225,-0.372371,0.0143758,-2.53617,-1.9066,-2.69784,-2.34748,-2.40404,-1.84627,-1.76547,-1.95062,-2.02309,-2.26443,-2.35281,-1.65016,-2.81412,-2.59655,-1.3871,-1.36819,-2.27108,-2.82077,-2.94598,-1.14072,-2.0336,-3.24552,-0.429167,-0.345602,-0.255696,-0.40927,-0.207543,-0.556137,-0.462445,-0.204487,-0.216857,-0.324489,-0.230653,-0.117995,-0.603386,-0.276107,-0.256322,-0.491629,0.252045,-0.199305,-0.638357,-0.285059,-0.364338,-0.269418,-0.470453,0.119899,
    -5930.95,3,0.104225,-0.27131,0.0189956,-3.00112,-2.24849,-2.64399,-2.31772,-2.80533,-2.01866,-1.69406,-2.29215,-2.03636,-2.20395,-2.24996,-1.46979,-3.08446,-2.61183,-1.34088,-1.42055,-1.83887,-2.9487,-3.28892,-1.21848,-1.89679,-3.31541,-0.349814,-0.380662,-0.317758,-0.19059,0.326377,-0.526384,-0.276535,-0.226859,-0.124268,-0.325952,-0.291204,-0.2831,-0.241056,-0.241745,-0.232814,-0.319929,-0.260222,-0.0643623,-0.797857,-0.255868,-0.423334,-0.21906,-0.292107,0.137825,
    -5920.6,3,0.104225,-0.181149,0.0084497,-2.60067,-2.36891,-1.98918,-2.42548,-2.45579,-1.98555,-1.53862,-2.31263,-2.00816,-2.24248,-2.27819,-1.34942,-2.98162,-2.9084,-1.38994,-1.53782,-2.20171,-3.36967,-3.30242,-1.27175,-1.89589,-3.19499,-0.242842,-0.254323,-0.336628,-0.0500155,-0.0773821,-0.266518,-0.462303,-0.158688,-0.193312,-0.274016,-0.192628,-0.276757,-0.20334,0.153512,-0.00977478,-0.094097,-0.0122078,-0.0125588,-0.226709,-0.179547,-0.466747,-0.262438,-0.303859,0.0919223,
    -5926.75,3,0.104225,-0.138456,0.0120685,-2.44305,-2.24172,-2.23265,-2.50717,-2.46876,-2.00241,-1.57722,-2.27684,-1.85228,-2.2156,-2.35399,-1.41582,-2.7912,-2.94016,-1.36884,-1.50599,-2.2534,-3.50275,-3.27105,-1.38539,-1.93125,-3.11444,-0.297937,-0.324996,-0.314406,-0.0338911,-0.0923396,-0.0247583,-0.499841,-0.18307,-0.179302,-0.258559,-0.174569,-0.251664,-0.230094,0.0998433,-0.0475798,-0.0392007,-0.158131,-0.161068,-0.402353,-0.295153,-0.568905,-0.37115,-0.438521,0.109857,
    -5926.75,3,0.104225,-0.138456,0.0120685,-2.44305,-2.24172,-2.23265,-2.50717,-2.46876,-2.00241,-1.57722,-2.27684,-1.85228,-2.2156,-2.35399,-1.41582,-2.7912,-2.94016,-1.36884,-1.50599,-2.2534,-3.50275,-3.27105,-1.38539,-1.93125,-3.11444,-0.297937,-0.324996,-0.314406,-0.0338911,-0.0923396,-0.0247583,-0.499841,-0.18307,-0.179302,-0.258559,-0.174569,-0.251664,-0.230094,0.0998433,-0.0475798,-0.0392007,-0.158131,-0.161068,-0.402353,-0.295153,-0.568905,-0.37115,-0.438521,0.109857;
  
  for (int i = 0; i < 6; i++) 
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
  ASSERT_EQ(1001, blocker0.samples.rows());
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

/*
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

  ASSERT_EQ(6, samples.rows());
  ASSERT_EQ(51, samples.cols());

  
  Eigen::MatrixXd expected_samples(6, 51);
  expected_samples <<
    -5923.71,3,0.104225,-0.31639,0.0064363,-2.0354,-1.94524,-2.06081,-2.31731,-2.54922,-2.50787,-1.74407,-2.04826,-1.75427,-2.16927,-2.31893,-1.37094,-2.8661,-2.67123,-1.36898,-1.43593,-1.8091,-2.92111,-3.98671,-1.4164,-1.89509,-2.94966,-0.557794,-0.483134,-0.156347,-0.250073,-0.121718,-0.243917,-0.327415,-0.127961,-0.238188,-0.425699,-0.280189,-0.294234,-0.442824,-0.30521,0.0666663,-0.457187,-0.270417,-0.247473,-0.402816,-0.20402,-0.346646,-0.306836,-0.436561,0.0802265,
    -5931.85,3,0.104225,-0.372371,0.0143758,-2.53617,-1.9066,-2.69784,-2.34748,-2.40404,-1.84627,-1.76547,-1.95062,-2.02309,-2.26443,-2.35281,-1.65016,-2.81412,-2.59655,-1.3871,-1.36819,-2.27108,-2.82077,-2.94598,-1.14072,-2.0336,-3.24552,-0.429167,-0.345602,-0.255696,-0.40927,-0.207543,-0.556137,-0.462445,-0.204487,-0.216857,-0.324489,-0.230653,-0.117995,-0.603386,-0.276107,-0.256322,-0.491629,0.252045,-0.199305,-0.638357,-0.285059,-0.364338,-0.269418,-0.470453,0.119899,
    -5930.95,3,0.104225,-0.27131,0.0189956,-3.00112,-2.24849,-2.64399,-2.31772,-2.80533,-2.01866,-1.69406,-2.29215,-2.03636,-2.20395,-2.24996,-1.46979,-3.08446,-2.61183,-1.34088,-1.42055,-1.83887,-2.9487,-3.28892,-1.21848,-1.89679,-3.31541,-0.349814,-0.380662,-0.317758,-0.19059,0.326377,-0.526384,-0.276535,-0.226859,-0.124268,-0.325952,-0.291204,-0.2831,-0.241056,-0.241745,-0.232814,-0.319929,-0.260222,-0.0643623,-0.797857,-0.255868,-0.423334,-0.21906,-0.292107,0.137825,
    -5920.6,3,0.104225,-0.181149,0.0084497,-2.60067,-2.36891,-1.98918,-2.42548,-2.45579,-1.98555,-1.53862,-2.31263,-2.00816,-2.24248,-2.27819,-1.34942,-2.98162,-2.9084,-1.38994,-1.53782,-2.20171,-3.36967,-3.30242,-1.27175,-1.89589,-3.19499,-0.242842,-0.254323,-0.336628,-0.0500155,-0.0773821,-0.266518,-0.462303,-0.158688,-0.193312,-0.274016,-0.192628,-0.276757,-0.20334,0.153512,-0.00977478,-0.094097,-0.0122078,-0.0125588,-0.226709,-0.179547,-0.466747,-0.262438,-0.303859,0.0919223,
    -5926.75,3,0.104225,-0.138456,0.0120685,-2.44305,-2.24172,-2.23265,-2.50717,-2.46876,-2.00241,-1.57722,-2.27684,-1.85228,-2.2156,-2.35399,-1.41582,-2.7912,-2.94016,-1.36884,-1.50599,-2.2534,-3.50275,-3.27105,-1.38539,-1.93125,-3.11444,-0.297937,-0.324996,-0.314406,-0.0338911,-0.0923396,-0.0247583,-0.499841,-0.18307,-0.179302,-0.258559,-0.174569,-0.251664,-0.230094,0.0998433,-0.0475798,-0.0392007,-0.158131,-0.161068,-0.402353,-0.295153,-0.568905,-0.37115,-0.438521,0.109857,
    -5926.75,3,0.104225,-0.138456,0.0120685,-2.44305,-2.24172,-2.23265,-2.50717,-2.46876,-2.00241,-1.57722,-2.27684,-1.85228,-2.2156,-2.35399,-1.41582,-2.7912,-2.94016,-1.36884,-1.50599,-2.2534,-3.50275,-3.27105,-1.38539,-1.93125,-3.11444,-0.297937,-0.324996,-0.314406,-0.0338911,-0.0923396,-0.0247583,-0.499841,-0.18307,-0.179302,-0.258559,-0.174569,-0.251664,-0.230094,0.0998433,-0.0475798,-0.0392007,-0.158131,-0.161068,-0.402353,-0.295153,-0.568905,-0.37115,-0.438521,0.109857;
  
  for (int i = 0; i < 6; i++) 
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
  ASSERT_EQ(1001, blocker0.samples.rows());
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
*/

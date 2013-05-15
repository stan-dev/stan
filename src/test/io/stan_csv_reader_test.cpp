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
  
  EXPECT_EQ("blocker_model", metadata.model);
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
  EXPECT_EQ("NUTS with a diagonal Euclidean metric", metadata.algorithm);
}
TEST_F(StanIoStanCsvReader,read_header1) { 
  Eigen::Matrix<std::string, Eigen::Dynamic, 1> header;
  EXPECT_TRUE(stan::io::stan_csv_reader::read_header(header1_stream, header));
  
  ASSERT_EQ(52, header.size());
  EXPECT_EQ("log_post", header(0));
  EXPECT_EQ("accept_stat", header(1));
  EXPECT_EQ("stepsize__", header(2));
  EXPECT_EQ("depth__", header(3));
  EXPECT_EQ("d", header(4));
  EXPECT_EQ("sigmasq_delta", header(5));
  EXPECT_EQ("mu[1]", header(6));
  EXPECT_EQ("mu[2]", header(7));
  EXPECT_EQ("mu[3]", header(8));
  EXPECT_EQ("mu[4]", header(9));
  EXPECT_EQ("mu[5]", header(10));
  EXPECT_EQ("mu[6]", header(11));
  EXPECT_EQ("mu[7]", header(12));
  EXPECT_EQ("mu[8]", header(13));
  EXPECT_EQ("mu[9]", header(14));
  EXPECT_EQ("mu[10]", header(15));
  EXPECT_EQ("mu[11]", header(16));
  EXPECT_EQ("mu[12]", header(17));
  EXPECT_EQ("mu[13]", header(18));
  EXPECT_EQ("mu[14]", header(19));
  EXPECT_EQ("mu[15]", header(20));
  EXPECT_EQ("mu[16]", header(21));
  EXPECT_EQ("mu[17]", header(22));
  EXPECT_EQ("mu[18]", header(23));
  EXPECT_EQ("mu[19]", header(24));
  EXPECT_EQ("mu[20]", header(25));
  EXPECT_EQ("mu[21]", header(26));
  EXPECT_EQ("mu[22]", header(27));
  EXPECT_EQ("delta[1]", header(28));
  EXPECT_EQ("delta[2]", header(29));
  EXPECT_EQ("delta[3]", header(30));
  EXPECT_EQ("delta[4]", header(31));
  EXPECT_EQ("delta[5]", header(32));
  EXPECT_EQ("delta[6]", header(33));
  EXPECT_EQ("delta[7]", header(34));
  EXPECT_EQ("delta[8]", header(35));
  EXPECT_EQ("delta[9]", header(36));
  EXPECT_EQ("delta[10]", header(37));
  EXPECT_EQ("delta[11]", header(38));
  EXPECT_EQ("delta[12]", header(39));
  EXPECT_EQ("delta[13]", header(40));
  EXPECT_EQ("delta[14]", header(41));
  EXPECT_EQ("delta[15]", header(42));
  EXPECT_EQ("delta[16]", header(43));
  EXPECT_EQ("delta[17]", header(44));
  EXPECT_EQ("delta[18]", header(45));
  EXPECT_EQ("delta[19]", header(46));
  EXPECT_EQ("delta[20]", header(47));
  EXPECT_EQ("delta[21]", header(48));
  EXPECT_EQ("delta[22]", header(49));
  EXPECT_EQ("delta_new", header(50));
  EXPECT_EQ("sigma_delta", header(51));
}


TEST_F(StanIoStanCsvReader,read_adaptation1) { 
  stan::io::stan_csv_adaptation adaptation;
  EXPECT_TRUE(stan::io::stan_csv_reader::read_adaptation(adaptation1_stream, adaptation));
  
  EXPECT_FLOAT_EQ(0.311368, adaptation.step_size);
  ASSERT_EQ(47, adaptation.metric.size());
  EXPECT_FLOAT_EQ(0.00381324, adaptation.metric(0));
  EXPECT_FLOAT_EQ(0.805729, adaptation.metric(1));
  EXPECT_FLOAT_EQ(0.202616, adaptation.metric(2));
  EXPECT_FLOAT_EQ(0.0602331, adaptation.metric(3));
  EXPECT_FLOAT_EQ(0.0703091, adaptation.metric(4));
  EXPECT_FLOAT_EQ(0.00715755, adaptation.metric(5));
  EXPECT_FLOAT_EQ(0.0245927, adaptation.metric(6));
  EXPECT_FLOAT_EQ(0.107405, adaptation.metric(7));
  EXPECT_FLOAT_EQ(0.00698799, adaptation.metric(8));
  EXPECT_FLOAT_EQ(0.0153979, adaptation.metric(9));
  EXPECT_FLOAT_EQ(0.0205916, adaptation.metric(10));
  EXPECT_FLOAT_EQ(0.0045571, adaptation.metric(11));
  EXPECT_FLOAT_EQ(0.0127992, adaptation.metric(12));
  EXPECT_FLOAT_EQ(0.0167977, adaptation.metric(13));
  EXPECT_FLOAT_EQ(0.042275, adaptation.metric(14));
  EXPECT_FLOAT_EQ(0.0172858, adaptation.metric(15));
  EXPECT_FLOAT_EQ(0.0226093, adaptation.metric(16));
  EXPECT_FLOAT_EQ(0.0187731, adaptation.metric(17));
  EXPECT_FLOAT_EQ(0.0357831, adaptation.metric(18));
  EXPECT_FLOAT_EQ(0.0829232, adaptation.metric(19));
  EXPECT_FLOAT_EQ(0.121351, adaptation.metric(20));
  EXPECT_FLOAT_EQ(0.0178708, adaptation.metric(21));
  EXPECT_FLOAT_EQ(0.0185714, adaptation.metric(22));
  EXPECT_FLOAT_EQ(0.0213618, adaptation.metric(23));
  EXPECT_FLOAT_EQ(0.0300144, adaptation.metric(24));
  EXPECT_FLOAT_EQ(0.0236461, adaptation.metric(25));
  EXPECT_FLOAT_EQ(0.0237509, adaptation.metric(26));
  EXPECT_FLOAT_EQ(0.00879464, adaptation.metric(27));
  EXPECT_FLOAT_EQ(0.022751, adaptation.metric(28));
  EXPECT_FLOAT_EQ(0.0259289, adaptation.metric(29));
  EXPECT_FLOAT_EQ(0.0127472, adaptation.metric(30));
  EXPECT_FLOAT_EQ(0.0150928, adaptation.metric(31));
  EXPECT_FLOAT_EQ(0.0158485, adaptation.metric(32));
  EXPECT_FLOAT_EQ(0.00790296, adaptation.metric(33));
  EXPECT_FLOAT_EQ(0.0122545, adaptation.metric(34));
  EXPECT_FLOAT_EQ(0.0155033, adaptation.metric(35));
  EXPECT_FLOAT_EQ(0.0225776, adaptation.metric(36));
  EXPECT_FLOAT_EQ(0.0301693, adaptation.metric(37));
  EXPECT_FLOAT_EQ(0.0188162, adaptation.metric(38));
  EXPECT_FLOAT_EQ(0.0146871, adaptation.metric(39));
  EXPECT_FLOAT_EQ(0.0219563, adaptation.metric(40));
  EXPECT_FLOAT_EQ(0.0251297, adaptation.metric(41));
  EXPECT_FLOAT_EQ(0.0277353, adaptation.metric(42));
  EXPECT_FLOAT_EQ(0.0130705, adaptation.metric(43));
  EXPECT_FLOAT_EQ(0.0198774, adaptation.metric(44));
  EXPECT_FLOAT_EQ(0.0217701, adaptation.metric(45));
  EXPECT_FLOAT_EQ(0.0318291, adaptation.metric(46));
}

TEST_F(StanIoStanCsvReader,read_samples1) {
  
  Eigen::MatrixXd samples;
  stan::io::stan_csv_timing timing;
  
  EXPECT_TRUE(stan::io::stan_csv_reader::read_samples(samples1_stream, samples, timing));

  ASSERT_EQ(5, samples.rows());
  ASSERT_EQ(51, samples.cols());

  Eigen::MatrixXd expected_samples(5, 51);
  expected_samples <<
  -5911.64,0.967054,0.311368,3,-0.238118,0.00305286,-2.42684,-2.29093,-1.99263,-2.26299,-2.31944,-2.96034,-1.70959,-2.29562,-2.03781,-2.18972,-2.27694,-1.4621,-3.07621,-2.86434,-1.35515,-1.47218,-1.89231,-2.70779,-3.4288,-1.28393,-2.18253,-2.89665,-0.147604,-0.233922,-0.274866,-0.253127,-0.195783,-0.102417,-0.386614,-0.136222,-0.162793,-0.251523,-0.317926,-0.259203,-0.391585,-0.174969,-0.258695,-0.350346,-0.136897,-0.219393,-0.214675,-0.257814,-0.17696,-0.303203,-0.420529,
  -5913.77,0.531379,0.311368,2,-0.239172,0.00487776,-1.88534,-1.51993,-2.34457,-2.51106,-2.39591,-2.19711,-1.80561,-2.06836,-1.80934,-2.22759,-2.32382,-1.4244,-2.93167,-2.7447,-1.40988,-1.42141,-1.7524,-2.84511,-3.36012,-1.40374,-2.09072,-3.11558,-0.151872,-0.245106,-0.221913,-0.19077,-0.131126,-0.361577,-0.187243,-0.249345,-0.21893,-0.18255,-0.116192,-0.257595,-0.353543,-0.170421,-0.370093,-0.259884,-0.348797,-0.289304,-0.230848,-0.34023,-0.22912,-0.279577,-0.400804,
  -5922.35,0.917981,0.311368,2,-0.237376,0.0192928,-1.83782,-2.12284,-2.29383,-2.44835,-2.47654,-1.73429,-1.51217,-1.96536,-1.8924,-2.329,-2.20581,-1.43235,-2.724,-2.84092,-1.56597,-1.30692,-2.14435,-2.73964,-3.80843,-1.39313,-2.36626,-2.87865,-0.0515039,-0.168673,-0.270707,-0.257859,-0.0812945,-0.269081,-0.70406,-0.226624,-0.170277,-0.100679,-0.200387,-0.207395,-0.334092,0.0389646,-0.356326,-0.292223,-0.217492,-0.306135,-0.268407,-0.340284,-0.12705,-0.414931,-0.321806,
  -5917.02,0.923539,0.311368,3,-0.267992,0.00750267,-1.95283,-2.35033,-1.5138,-2.42368,-2.52995,-2.02857,-1.65281,-2.12492,-2.06527,-2.39034,-2.34571,-1.47664,-3.35229,-2.74107,-1.30918,-1.45469,-2.08893,-2.67505,-3.63169,-1.41763,-2.00212,-2.85479,-0.411062,-0.337641,-0.364414,-0.234828,-0.11843,-0.268575,-0.422397,-0.19208,-0.207241,-0.215091,-0.185168,-0.535715,-0.32755,-0.219064,-0.257552,-0.273324,-0.255703,-0.355741,-0.364587,-0.278556,-0.249763,-0.472751,-0.156428,
  -5920.19,0.851519,0.311368,2,-0.2313,0.00607974,-1.68649,-2.51337,-2.04341,-2.50502,-2.44801,-2.66995,-1.71671,-2.0592,-2.25767,-2.23898,-2.31161,-1.47557,-3.00854,-2.57867,-1.13958,-1.27922,-1.87026,-2.85338,-3.30177,-1.40046,-1.89427,-2.94869,-0.0406589,-0.28921,-0.273337,-0.155008,-0.256302,-0.269896,-0.328092,-0.225172,-0.153995,-0.283121,-0.213809,-0.350999,-0.332991,-0.304812,-0.401796,-0.267212,-0.26457,-0.238125,-0.376133,-0.0517001,-0.204209,-0.413372,-0.101746;
  
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 51; j++)
      EXPECT_FLOAT_EQ(expected_samples(i,j), samples(i,j));
      
  EXPECT_FLOAT_EQ(0.310768, timing.warmup);
  EXPECT_FLOAT_EQ(0.354467, timing.sampling);
  
}

TEST_F(StanIoStanCsvReader,ParseBlocker) {
  stan::io::stan_csv blocker0;
  blocker0 = stan::io::stan_csv_reader::parse(blocker0_stream);

  // metadata
  EXPECT_EQ(1, blocker0.metadata.stan_version_major);
  EXPECT_EQ(3, blocker0.metadata.stan_version_minor);
  EXPECT_EQ(0, blocker0.metadata.stan_version_patch);
  
  EXPECT_EQ("blocker_model", blocker0.metadata.model);
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
  EXPECT_EQ("NUTS with a diagonal Euclidean metric", blocker0.metadata.algorithm);

  // header
  ASSERT_EQ(52, blocker0.header.size());
  EXPECT_EQ("log_post", blocker0.header(0));
  EXPECT_EQ("accept_stat", blocker0.header(1));
  EXPECT_EQ("stepsize__", blocker0.header(2));
  EXPECT_EQ("depth__", blocker0.header(3));
  EXPECT_EQ("d", blocker0.header(4));
  EXPECT_EQ("sigmasq_delta", blocker0.header(5));
  EXPECT_EQ("mu[1]", blocker0.header(6));
  EXPECT_EQ("mu[2]", blocker0.header(7));
  EXPECT_EQ("mu[3]", blocker0.header(8));
  EXPECT_EQ("mu[4]", blocker0.header(9));
  EXPECT_EQ("mu[5]", blocker0.header(10));
  EXPECT_EQ("mu[6]", blocker0.header(11));
  EXPECT_EQ("mu[7]", blocker0.header(12));
  EXPECT_EQ("mu[8]", blocker0.header(13));
  EXPECT_EQ("mu[9]", blocker0.header(14));
  EXPECT_EQ("mu[10]", blocker0.header(15));
  EXPECT_EQ("mu[11]", blocker0.header(16));
  EXPECT_EQ("mu[12]", blocker0.header(17));
  EXPECT_EQ("mu[13]", blocker0.header(18));
  EXPECT_EQ("mu[14]", blocker0.header(19));
  EXPECT_EQ("mu[15]", blocker0.header(20));
  EXPECT_EQ("mu[16]", blocker0.header(21));
  EXPECT_EQ("mu[17]", blocker0.header(22));
  EXPECT_EQ("mu[18]", blocker0.header(23));
  EXPECT_EQ("mu[19]", blocker0.header(24));
  EXPECT_EQ("mu[20]", blocker0.header(25));
  EXPECT_EQ("mu[21]", blocker0.header(26));
  EXPECT_EQ("mu[22]", blocker0.header(27));
  EXPECT_EQ("delta[1]", blocker0.header(28));
  EXPECT_EQ("delta[2]", blocker0.header(29));
  EXPECT_EQ("delta[3]", blocker0.header(30));
  EXPECT_EQ("delta[4]", blocker0.header(31));
  EXPECT_EQ("delta[5]", blocker0.header(32));
  EXPECT_EQ("delta[6]", blocker0.header(33));
  EXPECT_EQ("delta[7]", blocker0.header(34));
  EXPECT_EQ("delta[8]", blocker0.header(35));
  EXPECT_EQ("delta[9]", blocker0.header(36));
  EXPECT_EQ("delta[10]", blocker0.header(37));
  EXPECT_EQ("delta[11]", blocker0.header(38));
  EXPECT_EQ("delta[12]", blocker0.header(39));
  EXPECT_EQ("delta[13]", blocker0.header(40));
  EXPECT_EQ("delta[14]", blocker0.header(41));
  EXPECT_EQ("delta[15]", blocker0.header(42));
  EXPECT_EQ("delta[16]", blocker0.header(43));
  EXPECT_EQ("delta[17]", blocker0.header(44));
  EXPECT_EQ("delta[18]", blocker0.header(45));
  EXPECT_EQ("delta[19]", blocker0.header(46));
  EXPECT_EQ("delta[20]", blocker0.header(47));
  EXPECT_EQ("delta[21]", blocker0.header(48));
  EXPECT_EQ("delta[22]", blocker0.header(49));
  EXPECT_EQ("delta_new", blocker0.header(50));
  EXPECT_EQ("sigma_delta", blocker0.header(51));

  // adaptation
  EXPECT_FLOAT_EQ(0.311368, blocker0.adaptation.step_size);
  ASSERT_EQ(47, blocker0.adaptation.metric.size());
  EXPECT_FLOAT_EQ(0.00381324, blocker0.adaptation.metric(0));
  EXPECT_FLOAT_EQ(0.805729, blocker0.adaptation.metric(1));
  EXPECT_FLOAT_EQ(0.202616, blocker0.adaptation.metric(2));
  EXPECT_FLOAT_EQ(0.0602331, blocker0.adaptation.metric(3));
  EXPECT_FLOAT_EQ(0.0703091, blocker0.adaptation.metric(4));
  EXPECT_FLOAT_EQ(0.00715755, blocker0.adaptation.metric(5));
  EXPECT_FLOAT_EQ(0.0245927, blocker0.adaptation.metric(6));
  EXPECT_FLOAT_EQ(0.107405, blocker0.adaptation.metric(7));
  EXPECT_FLOAT_EQ(0.00698799, blocker0.adaptation.metric(8));
  EXPECT_FLOAT_EQ(0.0153979, blocker0.adaptation.metric(9));
  EXPECT_FLOAT_EQ(0.0205916, blocker0.adaptation.metric(10));
  EXPECT_FLOAT_EQ(0.0045571, blocker0.adaptation.metric(11));
  EXPECT_FLOAT_EQ(0.0127992, blocker0.adaptation.metric(12));
  EXPECT_FLOAT_EQ(0.0167977, blocker0.adaptation.metric(13));
  EXPECT_FLOAT_EQ(0.042275, blocker0.adaptation.metric(14));
  EXPECT_FLOAT_EQ(0.0172858, blocker0.adaptation.metric(15));
  EXPECT_FLOAT_EQ(0.0226093, blocker0.adaptation.metric(16));
  EXPECT_FLOAT_EQ(0.0187731, blocker0.adaptation.metric(17));
  EXPECT_FLOAT_EQ(0.0357831, blocker0.adaptation.metric(18));
  EXPECT_FLOAT_EQ(0.0829232, blocker0.adaptation.metric(19));
  EXPECT_FLOAT_EQ(0.121351, blocker0.adaptation.metric(20));
  EXPECT_FLOAT_EQ(0.0178708, blocker0.adaptation.metric(21));
  EXPECT_FLOAT_EQ(0.0185714, blocker0.adaptation.metric(22));
  EXPECT_FLOAT_EQ(0.0213618, blocker0.adaptation.metric(23));
  EXPECT_FLOAT_EQ(0.0300144, blocker0.adaptation.metric(24));
  EXPECT_FLOAT_EQ(0.0236461, blocker0.adaptation.metric(25));
  EXPECT_FLOAT_EQ(0.0237509, blocker0.adaptation.metric(26));
  EXPECT_FLOAT_EQ(0.00879464, blocker0.adaptation.metric(27));
  EXPECT_FLOAT_EQ(0.022751, blocker0.adaptation.metric(28));
  EXPECT_FLOAT_EQ(0.0259289, blocker0.adaptation.metric(29));
  EXPECT_FLOAT_EQ(0.0127472, blocker0.adaptation.metric(30));
  EXPECT_FLOAT_EQ(0.0150928, blocker0.adaptation.metric(31));
  EXPECT_FLOAT_EQ(0.0158485, blocker0.adaptation.metric(32));
  EXPECT_FLOAT_EQ(0.00790296, blocker0.adaptation.metric(33));
  EXPECT_FLOAT_EQ(0.0122545, blocker0.adaptation.metric(34));
  EXPECT_FLOAT_EQ(0.0155033, blocker0.adaptation.metric(35));
  EXPECT_FLOAT_EQ(0.0225776, blocker0.adaptation.metric(36));
  EXPECT_FLOAT_EQ(0.0301693, blocker0.adaptation.metric(37));
  EXPECT_FLOAT_EQ(0.0188162, blocker0.adaptation.metric(38));
  EXPECT_FLOAT_EQ(0.0146871, blocker0.adaptation.metric(39));
  EXPECT_FLOAT_EQ(0.0219563, blocker0.adaptation.metric(40));
  EXPECT_FLOAT_EQ(0.0251297, blocker0.adaptation.metric(41));
  EXPECT_FLOAT_EQ(0.0277353, blocker0.adaptation.metric(42));
  EXPECT_FLOAT_EQ(0.0130705, blocker0.adaptation.metric(43));
  EXPECT_FLOAT_EQ(0.0198774, blocker0.adaptation.metric(44));
  EXPECT_FLOAT_EQ(0.0217701, blocker0.adaptation.metric(45));
  EXPECT_FLOAT_EQ(0.0318291, blocker0.adaptation.metric(46));
  
  // samples
  ASSERT_EQ(1000, blocker0.samples.rows());
  ASSERT_EQ(51, blocker0.samples.cols());

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> expected_samples(6, 51);
  expected_samples <<
  -5911.64,0.967054,0.311368,3,-0.238118,0.00305286,-2.42684,-2.29093,-1.99263,-2.26299,-2.31944,-2.96034,-1.70959,-2.29562,-2.03781,-2.18972,-2.27694,-1.4621,-3.07621,-2.86434,-1.35515,-1.47218,-1.89231,-2.70779,-3.4288,-1.28393,-2.18253,-2.89665,-0.147604,-0.233922,-0.274866,-0.253127,-0.195783,-0.102417,-0.386614,-0.136222,-0.162793,-0.251523,-0.317926,-0.259203,-0.391585,-0.174969,-0.258695,-0.350346,-0.136897,-0.219393,-0.214675,-0.257814,-0.17696,-0.303203,-0.420529,
  -5913.77,0.531379,0.311368,2,-0.239172,0.00487776,-1.88534,-1.51993,-2.34457,-2.51106,-2.39591,-2.19711,-1.80561,-2.06836,-1.80934,-2.22759,-2.32382,-1.4244,-2.93167,-2.7447,-1.40988,-1.42141,-1.7524,-2.84511,-3.36012,-1.40374,-2.09072,-3.11558,-0.151872,-0.245106,-0.221913,-0.19077,-0.131126,-0.361577,-0.187243,-0.249345,-0.21893,-0.18255,-0.116192,-0.257595,-0.353543,-0.170421,-0.370093,-0.259884,-0.348797,-0.289304,-0.230848,-0.34023,-0.22912,-0.279577,-0.400804,
  -5922.35,0.917981,0.311368,2,-0.237376,0.0192928,-1.83782,-2.12284,-2.29383,-2.44835,-2.47654,-1.73429,-1.51217,-1.96536,-1.8924,-2.329,-2.20581,-1.43235,-2.724,-2.84092,-1.56597,-1.30692,-2.14435,-2.73964,-3.80843,-1.39313,-2.36626,-2.87865,-0.0515039,-0.168673,-0.270707,-0.257859,-0.0812945,-0.269081,-0.70406,-0.226624,-0.170277,-0.100679,-0.200387,-0.207395,-0.334092,0.0389646,-0.356326,-0.292223,-0.217492,-0.306135,-0.268407,-0.340284,-0.12705,-0.414931,-0.321806,
  -5917.02,0.923539,0.311368,3,-0.267992,0.00750267,-1.95283,-2.35033,-1.5138,-2.42368,-2.52995,-2.02857,-1.65281,-2.12492,-2.06527,-2.39034,-2.34571,-1.47664,-3.35229,-2.74107,-1.30918,-1.45469,-2.08893,-2.67505,-3.63169,-1.41763,-2.00212,-2.85479,-0.411062,-0.337641,-0.364414,-0.234828,-0.11843,-0.268575,-0.422397,-0.19208,-0.207241,-0.215091,-0.185168,-0.535715,-0.32755,-0.219064,-0.257552,-0.273324,-0.255703,-0.355741,-0.364587,-0.278556,-0.249763,-0.472751,-0.156428,
  -5920.19,0.851519,0.311368,2,-0.2313,0.00607974,-1.68649,-2.51337,-2.04341,-2.50502,-2.44801,-2.66995,-1.71671,-2.0592,-2.25767,-2.23898,-2.31161,-1.47557,-3.00854,-2.57867,-1.13958,-1.27922,-1.87026,-2.85338,-3.30177,-1.40046,-1.89427,-2.94869,-0.0406589,-0.28921,-0.273337,-0.155008,-0.256302,-0.269896,-0.328092,-0.225172,-0.153995,-0.283121,-0.213809,-0.350999,-0.332991,-0.304812,-0.401796,-0.267212,-0.26457,-0.238125,-0.376133,-0.0517001,-0.204209,-0.413372,-0.101746,
  -5913.25,0.956678,0.311368,2,-0.273456,0.0120474,-2.75399,-2.33464,-2.16181,-2.49723,-2.3648,-1.52517,-1.74961,-1.98552,-1.98194,-2.25525,-2.36602,-1.49934,-2.82758,-2.66482,-1.39502,-1.53034,-1.87157,-2.83383,-3.34516,-1.63761,-2.2678,-2.99005,-0.253124,-0.272473,-0.261209,-0.251645,-0.332537,-0.355065,-0.209884,-0.278575,-0.306856,-0.33466,-0.267162,-0.130665,-0.29008,-0.0606014,-0.249043,-0.288622,-0.221687,0.00783253,-0.262921,-0.359394,-0.487682,-0.158997,-0.244443;
  
  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 51; j++)
      EXPECT_FLOAT_EQ(expected_samples(i,j), blocker0.samples(i,j));
  
  EXPECT_FLOAT_EQ(0.310768, blocker0.timing.warmup);
  EXPECT_FLOAT_EQ(0.354467, blocker0.timing.sampling);
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
  
  ASSERT_EQ(312, header.size());
  EXPECT_EQ("log_post", header(0));
  EXPECT_EQ("accept_stat", header(1));
  EXPECT_EQ("stepsize__", header(2));
  EXPECT_EQ("depth__", header(3));
  EXPECT_EQ("a0", header(4));
  EXPECT_EQ("alpha_Base", header(5));
  EXPECT_EQ("alpha_Trt", header(6));
  EXPECT_EQ("alpha_BT", header(7));
  EXPECT_EQ("alpha_Age", header(8));
  EXPECT_EQ("alpha_V4", header(9));
  EXPECT_EQ("b1[1]", header(10));
  EXPECT_EQ("b1[2]", header(11));
  EXPECT_EQ("b1[59]", header(68));
  EXPECT_EQ("b[1,1]", header(69));
  EXPECT_EQ("b[1,2]", header(70));
  EXPECT_EQ("b[1,3]", header(71));
  EXPECT_EQ("b[1,4]", header(72));
  EXPECT_EQ("b[2,1]", header(73));
  EXPECT_EQ("b[59,4]", header(304));
  EXPECT_EQ("sigmasq_b", header(305));
  EXPECT_EQ("sigmasq_", header(306));
  EXPECT_EQ("b1", header(307));
  EXPECT_EQ("sigma_b", header(308));
  EXPECT_EQ("sigma_", header(309));
  EXPECT_EQ("b1", header(310));
  EXPECT_EQ("alpha0", header(311));
  
}

TEST_F(StanIoStanCsvReader,read_adaptation2) { 
  stan::io::stan_csv_adaptation adaptation;
  EXPECT_TRUE(stan::io::stan_csv_reader::read_adaptation(adaptation2_stream, adaptation));
  
  EXPECT_FLOAT_EQ(0.0822187, adaptation.step_size);
  ASSERT_EQ(303, adaptation.metric.size());
  EXPECT_FLOAT_EQ(0.00554969, adaptation.metric(0));
  EXPECT_FLOAT_EQ(0.0187568, adaptation.metric(1));
  EXPECT_FLOAT_EQ(0.18556, adaptation.metric(2));
  EXPECT_FLOAT_EQ(0.0517218, adaptation.metric(301));
  EXPECT_FLOAT_EQ(0.0704762, adaptation.metric(302));
}

TEST_F(StanIoStanCsvReader,read_samples2) { 
  
  Eigen::MatrixXd samples;
  stan::io::stan_csv_timing timing;
  
  EXPECT_TRUE(stan::io::stan_csv_reader::read_samples(samples2_stream, samples, timing));

  ASSERT_EQ(3, samples.rows());
  ASSERT_EQ(307, samples.cols());

  Eigen::MatrixXd expected_samples(3, 307);
  expected_samples <<
  3421.23,0.937286,0.0822187,4,1.52358,0.832142,-1.75186,0.732349,0.581719,-0.0956847,0.129881,0.291981,0.206332,-0.0914288,-0.0511028,0.00257955,-0.464127,0.532475,-0.470424,0.871308,0.0709595,0.0635587,-0.369269,0.0126262,0.0547285,-0.813157,-1.0469,0.478363,0.130403,-0.0133173,-0.0596814,0.0222047,0.0283259,-0.0643246,0.761457,-0.0149241,-0.113361,0.382672,-0.506165,-0.151217,-0.299224,1.08806,0.224676,-0.608304,0.837489,0.677715,0.268503,-0.638102,-0.163803,-0.466867,-0.101927,0.0718306,-0.0400695,0.523616,-0.0386449,0.672028,0.0243993,-0.270021,0.257947,-0.0915172,-0.1089,-0.653684,0.128279,0.0611797,0.126549,0.992298,-0.432765,-0.378834,-0.277551,-0.660989,-0.986074,0.0900664,0.0292324,-0.034784,-0.131666,0.695082,0.208848,-0.770174,0.298827,0.312659,-0.307014,-0.00634741,-0.251043,-0.230869,0.537833,0.341158,-0.00420492,-0.491781,-0.112962,-0.390608,0.363788,-0.604658,0.606161,-0.197107,-0.276075,0.130583,-0.0712164,0.18376,-0.499806,-0.66026,0.370261,0.00854728,-0.0868798,0.260967,0.406198,-0.170811,-0.443757,-0.676608,-0.152996,0.0196129,0.368349,-0.000206977,-0.28317,0.510604,0.493545,-0.492945,-0.0877427,0.231765,0.0384101,-0.047576,-0.578215,0.259233,0.114818,0.378702,-0.57977,0.135902,0.259441,-0.0529871,-0.614916,-0.263011,0.120037,0.0373222,0.387723,-0.50651,0.291969,-0.322418,0.266892,0.29166,-0.20604,-0.453557,0.120064,-0.353737,0.41626,-0.349312,-0.472675,-0.176935,-0.103863,-0.144145,0.0127681,0.22459,-0.478262,0.217294,-0.368937,-0.280888,-0.196748,-0.108028,0.489689,0.0599641,0.103548,0.324713,0.845389,-0.128266,0.443205,0.173609,0.656571,0.201881,1.11719,0.453658,-0.545862,0.304403,0.310417,-0.378691,-0.138672,-0.626345,0.0504052,-0.0207573,-0.344184,-0.547217,-0.0476474,-0.0287413,-0.202042,-0.140236,0.262365,0.929479,-0.116882,-0.138981,-0.432823,0.00308807,-0.0907176,0.411141,-0.363322,-0.501115,0.0648479,-0.835257,-0.0786177,0.514753,-0.402192,-0.0697823,0.327783,-0.10135,-0.175465,-0.213098,-0.734818,-0.511434,-0.161901,-0.579105,-0.0349863,0.508914,0.293413,0.345071,-0.305739,1.16372,-0.119605,-0.0449234,0.219932,0.158583,-0.0232387,-0.300874,0.202966,0.462589,0.145201,-0.0410295,-0.312652,-0.28884,-0.47489,0.0066505,0.131026,-0.21777,0.390337,0.529508,-0.623533,-0.442618,0.526806,-0.567202,-0.22499,0.0423771,-0.375834,0.166689,0.0396714,0.190636,-0.287924,0.377336,0.557626,-0.245169,-0.114383,0.725871,-0.0533289,-0.489468,0.0813491,0.0099482,0.588994,-0.142369,0.497675,-0.78904,-0.131813,-0.49381,0.148018,-0.470343,-0.0343832,0.0152691,-0.0325922,0.171025,0.206433,-0.275871,-0.0124194,0.17076,0.354938,-0.483457,-0.0382223,0.208386,0.0300638,0.033795,0.450908,0.00942022,-0.0973246,0.127447,-0.863126,0.288728,0.23081,-0.0634318,0.312909,-0.101682,-0.0681568,0.192532,-0.0347934,0.0881257,0.108717,0.271307,0.352674,-0.00539426,-0.183513,0.290762,0.496757,0.439495,0.0152755,0.0906724,0.350073,-0.0978165,0.115204,-0.41276,0.523454,-0.367346,-0.103687,-0.454961,0.183484,0.137012,0.196364,
  3397.02,0.941767,0.0822187,4,1.51305,0.825715,-1.68143,0.663567,0.388199,-0.149882,0.37911,0.010886,0.261191,0.348686,-0.118497,-0.00466986,-0.00347264,0.519305,-0.411866,0.90656,0.254771,-0.0309118,-0.875493,0.298829,-0.289179,-0.802027,-0.56928,0.28425,0.157443,0.307647,0.295864,0.0935333,-0.315905,-0.0858896,0.73041,-0.0527516,0.215841,0.0178213,-0.598824,-0.146196,0.249497,0.53796,0.674642,-0.318004,0.985593,0.805007,0.654743,-0.581854,-0.174526,-0.557663,-0.754649,-0.0389194,0.284117,0.294737,0.119617,0.70946,-0.328479,0.168332,0.527801,-0.205642,-0.385518,-0.0472791,0.202826,-0.210838,0.432783,1.14151,-0.61708,-0.362993,0.237458,-0.38081,-0.892512,0.429856,0.364895,-0.196814,0.0215446,0.390724,0.638664,-0.281547,0.4581,0.593331,0.219361,0.272306,-0.532622,-0.481165,1.04258,0.151891,0.265983,-0.474189,-0.420614,-0.335478,-0.472147,-0.670693,0.568867,-0.612168,-0.500194,0.242462,-0.251436,-0.167323,-0.451188,-0.539589,0.215186,0.41185,0.135504,0.32792,0.470283,0.35338,-0.78284,-0.812597,-0.551357,-0.936678,0.768143,0.062984,-0.555425,0.238985,0.737339,-0.841306,-0.44347,0.0657396,-0.0131058,0.283186,-0.313479,0.147675,0.352154,0.42557,-0.353327,-0.183986,-0.217387,0.051668,0.125799,0.371878,-0.320861,-0.347693,0.51696,-0.235152,0.103033,0.23597,-0.00894705,0.27772,-0.113662,-0.555051,0.375178,-0.264982,0.316754,-0.301573,-0.55025,-0.024334,-0.308318,-0.189975,0.0861153,0.176677,-0.336233,0.360163,-0.0785847,0.0229804,-0.202919,0.249383,0.468756,0.181024,0.273788,0.270709,0.451499,-0.352163,0.371524,-0.723423,0.562147,-0.303412,0.968129,0.390592,-0.177668,0.506314,-0.204481,-0.61938,-0.528678,-0.103722,0.227317,-0.0531273,-0.424148,-0.494589,-0.178856,-0.234547,-0.0282887,-0.0672468,0.594921,0.630496,-0.106009,-0.181793,0.0915175,0.108272,0.0281998,0.054067,0.0956321,-0.192231,0.515981,-0.415932,-0.158857,0.624149,0.0445678,-0.210834,0.250824,-0.111617,-0.0911613,-0.252763,-0.538996,-0.0913585,-0.110107,-0.422959,0.274027,0.00715785,-0.195433,-0.116996,-0.196794,0.82386,-0.46129,0.288575,0.319932,-0.0715003,-0.0932481,-0.075741,-0.000285035,0.504914,-0.433218,0.196166,0.406808,-0.524464,-0.645385,-0.635314,-0.184473,-0.357757,-0.243751,0.649615,-0.967931,-0.172174,0.681664,-0.0330975,-0.118462,-0.254771,-0.347457,-0.123273,-0.138805,0.901447,0.307826,-0.0743004,0.450242,-0.538529,-0.250628,0.792936,-0.313868,-0.340155,0.0259574,-0.12572,0.484013,-0.302442,0.0229207,-0.86264,-0.298396,-0.673519,-0.198077,-0.124125,-0.193288,0.19701,0.0745229,0.0699828,0.549073,-0.212579,0.203637,-0.0646874,0.12099,0.262915,-0.136803,0.330657,0.190778,-0.323766,-0.108034,0.0652954,-0.218773,-0.168416,-1.11049,0.233882,-0.0404487,0.160961,0.112797,0.108568,0.359711,-0.184669,-0.00700956,-0.404508,-0.837475,0.140129,0.453764,-0.68715,0.0757106,0.0525023,0.358078,0.426377,-0.262996,0.428706,0.177705,0.0506844,0.0949841,-0.276855,0.521643,0.0397819,-0.00298384,-0.760282,0.0106922,0.165432,0.189066,
  3411.27,0.9801,0.0822187,4,1.46061,0.901133,-1.17656,0.493489,0.264105,-0.0707901,0.0590835,0.549689,0.468383,0.616112,-0.290877,-0.31117,-0.472996,0.705924,0.101407,1.2017,0.107521,0.154349,-0.0471367,0.389302,0.0196745,-1.13094,-0.581542,0.181555,-0.483917,0.034513,0.406151,0.269273,-0.281302,-0.0991023,1.04036,0.0290517,-0.0114168,0.180341,-0.548892,0.0234969,-0.209533,0.406801,0.232694,-0.434189,0.862561,0.942837,0.581544,-0.147207,-0.596117,-0.289137,-0.00132144,-0.123397,0.0834492,0.630263,0.420227,0.518569,0.147478,-0.255709,0.492913,0.0347787,-0.204272,-0.375392,-0.211274,-0.13881,0.563653,1.08563,-0.162108,-1.07124,0.0428944,0.0862678,-0.415749,-0.0634762,0.691086,-0.398317,-0.182618,0.546518,0.428545,-0.519203,0.317519,0.412505,0.340957,0.0455202,-0.603371,0.127888,0.731095,0.0108715,0.187426,-0.570161,-0.43852,-0.429801,-0.155631,-0.244029,-0.0983598,-0.282688,-0.38967,-0.367717,0.214383,-0.083689,0.10821,-0.396959,-0.00558305,0.129039,0.67935,0.349208,0.116352,-0.188436,-0.552793,-0.478643,0.256106,-0.726712,0.787821,-0.0579392,-0.148597,0.43004,0.330341,-0.214137,-0.0804792,0.126813,0.36545,0.386561,-0.528215,0.161322,-0.0706694,0.139329,-0.611884,0.242689,-0.296643,0.141254,0.068957,0.605866,-0.015601,0.444034,0.12718,-0.217594,0.535984,-0.206416,0.444351,-0.25472,-0.179522,-0.758611,0.620879,-0.0906023,0.330193,-0.350035,-0.543876,0.0799522,-0.0187512,-0.61047,-0.352462,0.174016,0.079501,0.857283,-0.442247,0.115087,-0.136165,-0.0600432,0.230183,0.199496,0.632995,0.150932,0.245845,-0.182998,0.11839,-0.411396,0.260937,-0.410454,0.808554,-0.112858,-0.326782,0.115556,0.288897,-0.534174,0.223673,0.286936,0.282982,0.201681,-0.275551,-0.283837,0.405197,-0.467146,0.0908445,0.186189,-0.212239,0.592556,-0.0645539,-0.461593,0.0221325,0.447698,-0.107631,-0.0074328,-0.786743,0.0440519,0.207132,-0.186647,0.217685,0.191607,-0.0116842,-0.566419,-0.481796,0.327231,-0.152874,-0.179597,-1.06552,-0.0119357,0.12522,-0.121305,-0.115398,-0.25774,-0.476504,-0.183359,-0.22164,0.881884,-0.161745,0.0259716,-0.0146291,-0.121366,0.0905673,0.522375,0.522542,-0.166705,-0.253245,0.00384188,0.433051,-0.47542,-0.545012,-1.00211,0.345601,-0.737647,-0.686605,0.697853,-0.535058,-0.360423,0.450114,0.283179,-0.668309,-0.189784,0.450038,-0.191166,-0.390235,0.449319,0.367355,0.0922884,0.447165,-0.717364,0.360873,0.140975,-0.213782,-0.0869214,0.764262,0.0823513,1.01171,0.405438,-0.186777,-0.553427,0.2189,-0.671612,0.539175,-0.0998238,0.0242638,-0.00704223,-0.240099,0.372918,0.591623,-0.238087,-0.30286,0.306051,-0.787538,0.612637,0.164627,0.0723437,0.0309391,0.134583,-0.103042,0.184017,0.0341284,-0.43441,-0.714739,0.111872,0.441788,0.328767,0.307117,0.0643992,0.116684,-1.11166,0.448797,-0.158111,0.225782,0.316813,0.0991554,-0.382125,-0.176967,-0.277192,0.0149808,0.077943,-0.180398,0.0813666,0.285277,0.0806504,0.423675,-0.13178,0.023832,-0.169381,-0.337127,-0.343572,-0.532462,0.1439,0.26762;

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 309; j++)
      EXPECT_FLOAT_EQ(expected_samples(i,j), samples(i,j));
      
  EXPECT_FLOAT_EQ(0.310768, timing.warmup);
  EXPECT_FLOAT_EQ(0.354467, timing.sampling);
  
}

/*
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
*/
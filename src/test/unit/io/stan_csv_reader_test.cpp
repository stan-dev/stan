#include <stan/io/stan_csv_reader.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <fstream>
#include <sstream>

class StanIoStanCsvReader : public testing::Test {
  
public:
  void SetUp () {
    blocker0_stream.open("src/test/unit/io/test_csv_files/blocker.0.csv");
    metadata1_stream.open("src/test/unit/io/test_csv_files/metadata1.csv");
    metadata3_stream.open("src/test/unit/io/test_csv_files/metadata3.csv");
    header1_stream.open("src/test/unit/io/test_csv_files/header1.csv");
    adaptation1_stream.open("src/test/unit/io/test_csv_files/adaptation1.csv");
    samples1_stream.open("src/test/unit/io/test_csv_files/samples1.csv");
    
    epil0_stream.open("src/test/unit/io/test_csv_files/epil.0.csv");
    metadata2_stream.open("src/test/unit/io/test_csv_files/metadata2.csv");
    header2_stream.open("src/test/unit/io/test_csv_files/header2.csv");
    adaptation2_stream.open("src/test/unit/io/test_csv_files/adaptation2.csv");
    samples2_stream.open("src/test/unit/io/test_csv_files/samples2.csv");
    
    blocker_nondiag0_stream.open("src/test/unit/io/test_csv_files/blocker_nondiag.0.csv");
    eight_schools_stream.open("src/test/unit/io/test_csv_files/eight_schools.csv");
  }
  
  void TearDown() {
    blocker0_stream.close();
    metadata1_stream.close();
    metadata3_stream.close();
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
  std::ifstream metadata3_stream;
  std::ifstream metadata2_stream, header2_stream, adaptation2_stream, samples2_stream;
  std::ifstream eight_schools_stream;
};

TEST_F(StanIoStanCsvReader,read_metadata1) {
  stan::io::stan_csv_metadata metadata;
  EXPECT_TRUE(stan::io::stan_csv_reader::read_metadata(metadata1_stream, metadata, 0));
  
  EXPECT_EQ(2, metadata.stan_version_major);
  EXPECT_EQ(9, metadata.stan_version_minor);
  EXPECT_EQ(0, metadata.stan_version_patch);
  
  EXPECT_EQ("blocker_model", metadata.model);
  EXPECT_EQ("../example-models/bugs_examples/vol1/blocker/blocker.data.R", metadata.data);
  EXPECT_EQ("../example-models/bugs_examples/vol1/blocker/blocker.init.R", metadata.init);
  EXPECT_FALSE(metadata.append_samples);
  EXPECT_FALSE(metadata.save_warmup);
  EXPECT_EQ(4085885484U, metadata.seed);
  EXPECT_FALSE(metadata.random_seed);
  EXPECT_EQ(0U, metadata.chain_id);
  EXPECT_EQ(2000U, metadata.num_samples);
  EXPECT_EQ(2000U, metadata.num_warmup);
  EXPECT_EQ(2U, metadata.thin);
  EXPECT_EQ("hmc", metadata.algorithm);
  EXPECT_EQ("nuts", metadata.engine);
  EXPECT_EQ(10, metadata.max_depth);
}
TEST_F(StanIoStanCsvReader,read_metadata3) {
  stan::io::stan_csv_metadata metadata;
  EXPECT_TRUE(stan::io::stan_csv_reader::read_metadata(metadata3_stream, metadata, 0));
  
  EXPECT_EQ(2, metadata.stan_version_major);
  EXPECT_EQ(9, metadata.stan_version_minor);
  EXPECT_EQ(0, metadata.stan_version_patch);
  
  EXPECT_EQ("blocker_model", metadata.model);
  EXPECT_EQ("../example-models/bugs_examples/vol1/blocker/blocker.data.R", metadata.data);
  EXPECT_EQ("../example-models/bugs_examples/vol1/blocker/blocker.init.R", metadata.init);
  EXPECT_FALSE(metadata.append_samples);
  EXPECT_FALSE(metadata.save_warmup);
  EXPECT_EQ(4085885484U, metadata.seed);
  EXPECT_FALSE(metadata.random_seed);
  EXPECT_EQ(0U, metadata.chain_id);
  EXPECT_EQ(2000U, metadata.num_samples);
  EXPECT_EQ(2000U, metadata.num_warmup);
  EXPECT_EQ(2U, metadata.thin);
  EXPECT_EQ("hmc", metadata.algorithm);
  EXPECT_EQ("nuts", metadata.engine);
  EXPECT_EQ(15, metadata.max_depth);
}
TEST_F(StanIoStanCsvReader,read_header1) {
  Eigen::Matrix<std::string, Eigen::Dynamic, 1> header;
  EXPECT_TRUE(stan::io::stan_csv_reader::read_header(header1_stream, header, 0));

  ASSERT_EQ(55, header.size());
  EXPECT_EQ("lp__", header(0));
  EXPECT_EQ("accept_stat__", header(1));
  EXPECT_EQ("stepsize__", header(2));
  EXPECT_EQ("treedepth__", header(3));
  EXPECT_EQ("n_leapfrog__", header(4));
  EXPECT_EQ("divergent__", header(5));
  EXPECT_EQ("energy__", header(6));
  EXPECT_EQ("d", header(7));
  EXPECT_EQ("sigmasq_delta", header(8));
  EXPECT_EQ("mu[1]", header(9));
  EXPECT_EQ("mu[2]", header(10));
  EXPECT_EQ("mu[3]", header(11));
  EXPECT_EQ("mu[4]", header(12));
  EXPECT_EQ("mu[5]", header(13));
  EXPECT_EQ("mu[6]", header(14));
  EXPECT_EQ("mu[7]", header(15));
  EXPECT_EQ("mu[8]", header(16));
  EXPECT_EQ("mu[9]", header(17));
  EXPECT_EQ("mu[10]", header(18));
  EXPECT_EQ("mu[11]", header(19));
  EXPECT_EQ("mu[12]", header(20));
  EXPECT_EQ("mu[13]", header(21));
  EXPECT_EQ("mu[14]", header(22));
  EXPECT_EQ("mu[15]", header(23));
  EXPECT_EQ("mu[16]", header(24));
  EXPECT_EQ("mu[17]", header(25));
  EXPECT_EQ("mu[18]", header(26));
  EXPECT_EQ("mu[19]", header(27));
  EXPECT_EQ("mu[20]", header(28));
  EXPECT_EQ("mu[21]", header(29));
  EXPECT_EQ("mu[22]", header(30));
  EXPECT_EQ("delta[1]", header(31));
  EXPECT_EQ("delta[2]", header(32));
  EXPECT_EQ("delta[3]", header(33));
  EXPECT_EQ("delta[4]", header(34));
  EXPECT_EQ("delta[5]", header(35));
  EXPECT_EQ("delta[6]", header(36));
  EXPECT_EQ("delta[7]", header(37));
  EXPECT_EQ("delta[8]", header(38));
  EXPECT_EQ("delta[9]", header(39));
  EXPECT_EQ("delta[10]", header(40));
  EXPECT_EQ("delta[11]", header(41));
  EXPECT_EQ("delta[12]", header(42));
  EXPECT_EQ("delta[13]", header(43));
  EXPECT_EQ("delta[14]", header(44));
  EXPECT_EQ("delta[15]", header(45));
  EXPECT_EQ("delta[16]", header(46));
  EXPECT_EQ("delta[17]", header(47));
  EXPECT_EQ("delta[18]", header(48));
  EXPECT_EQ("delta[19]", header(49));
  EXPECT_EQ("delta[20]", header(50));
  EXPECT_EQ("delta[21]", header(51));
  EXPECT_EQ("delta[22]", header(52));
  EXPECT_EQ("delta_new", header(53));
  EXPECT_EQ("sigma_delta", header(54));
}

TEST_F(StanIoStanCsvReader,read_adaptation1) {
  stan::io::stan_csv_adaptation adaptation;
  EXPECT_TRUE(stan::io::stan_csv_reader::read_adaptation(adaptation1_stream, adaptation, 0));

  EXPECT_FLOAT_EQ(0.118745, adaptation.step_size);
  ASSERT_EQ(47, adaptation.metric.size());
  EXPECT_FLOAT_EQ(0.00376431, adaptation.metric(0));
  EXPECT_FLOAT_EQ(1.38212, adaptation.metric(1));
  EXPECT_FLOAT_EQ(0.216607, adaptation.metric(2));
  EXPECT_FLOAT_EQ(0.0588362, adaptation.metric(3));
  EXPECT_FLOAT_EQ(0.0773243, adaptation.metric(4));
  EXPECT_FLOAT_EQ(0.00576153, adaptation.metric(5));
  EXPECT_FLOAT_EQ(0.0226168, adaptation.metric(6));
  EXPECT_FLOAT_EQ(0.140826, adaptation.metric(7));
  EXPECT_FLOAT_EQ(0.00691007, adaptation.metric(8));
  EXPECT_FLOAT_EQ(0.0144645, adaptation.metric(9));
  EXPECT_FLOAT_EQ(0.0185795, adaptation.metric(10));
  EXPECT_FLOAT_EQ(0.00443264, adaptation.metric(11));
  EXPECT_FLOAT_EQ(0.0141671, adaptation.metric(12));
  EXPECT_FLOAT_EQ(0.0165921, adaptation.metric(13));
  EXPECT_FLOAT_EQ(0.0443114, adaptation.metric(14));
  EXPECT_FLOAT_EQ(0.0180591, adaptation.metric(15));
  EXPECT_FLOAT_EQ(0.0222813, adaptation.metric(16));
  EXPECT_FLOAT_EQ(0.019491, adaptation.metric(17));
  EXPECT_FLOAT_EQ(0.0372924, adaptation.metric(18));
  EXPECT_FLOAT_EQ(0.0841709, adaptation.metric(19));
  EXPECT_FLOAT_EQ(0.122628, adaptation.metric(20));
  EXPECT_FLOAT_EQ(0.020642, adaptation.metric(21));
  EXPECT_FLOAT_EQ(0.0174555, adaptation.metric(22));
  EXPECT_FLOAT_EQ(0.0190478, adaptation.metric(23));
  EXPECT_FLOAT_EQ(0.0194389, adaptation.metric(24));
  EXPECT_FLOAT_EQ(0.0208765, adaptation.metric(25));
  EXPECT_FLOAT_EQ(0.0266566, adaptation.metric(26));
  EXPECT_FLOAT_EQ(0.00803359, adaptation.metric(27));
  EXPECT_FLOAT_EQ(0.0179588, adaptation.metric(28));
  EXPECT_FLOAT_EQ(0.0297546, adaptation.metric(29));
  EXPECT_FLOAT_EQ(0.0114378, adaptation.metric(30));
  EXPECT_FLOAT_EQ(0.0135584, adaptation.metric(31));
  EXPECT_FLOAT_EQ(0.0149708, adaptation.metric(32));
  EXPECT_FLOAT_EQ(0.00658085, adaptation.metric(33));
  EXPECT_FLOAT_EQ(0.0125537, adaptation.metric(34));
  EXPECT_FLOAT_EQ(0.0166118, adaptation.metric(35));
  EXPECT_FLOAT_EQ(0.023786, adaptation.metric(36));
  EXPECT_FLOAT_EQ(0.0308986, adaptation.metric(37));
  EXPECT_FLOAT_EQ(0.013756, adaptation.metric(38));
  EXPECT_FLOAT_EQ(0.0145632, adaptation.metric(39));
  EXPECT_FLOAT_EQ(0.0241766, adaptation.metric(40));
  EXPECT_FLOAT_EQ(0.0227147, adaptation.metric(41));
  EXPECT_FLOAT_EQ(0.0247547, adaptation.metric(42));
  EXPECT_FLOAT_EQ(0.0142955, adaptation.metric(43));
  EXPECT_FLOAT_EQ(0.0167622, adaptation.metric(44));
  EXPECT_FLOAT_EQ(0.0187646, adaptation.metric(45));
  EXPECT_FLOAT_EQ(0.0248957, adaptation.metric(46));

}

TEST_F(StanIoStanCsvReader,read_samples1) {

  Eigen::MatrixXd samples;
  stan::io::stan_csv_timing timing;

  EXPECT_TRUE(stan::io::stan_csv_reader::read_samples(samples1_stream, samples, timing, 0));

  ASSERT_EQ(5, samples.rows());
  ASSERT_EQ(55, samples.cols());

  Eigen::MatrixXd expected_samples(5, 55);
  expected_samples <<
-5919.76,0.946838,0.118745,5,31,0,5950.76,-0.28158,0.0044619,-2.43441,-2.43628,-2.27778,-2.20654,-2.30954,-2.5927,-1.81483,-2.0766,-1.91096,-2.16673,-2.40527,-1.50555,-2.78592,-2.79865,-1.4619,-1.37719,-2.28927,-2.94799,-2.95409,-1.50565,-1.99425,-3.01008,-0.262722,-0.292787,-0.192026,-0.603011,-0.448924,-0.235658,-0.321412,-0.115794,-0.270572,-0.357005,-0.299995,-0.222147,-0.368037,0.0721705,-0.306608,-0.220126,-0.263808,-0.224855,-0.257224,-0.174363,-0.319416,-0.426591,-0.447132,0.0667975, -5913.75,0.978966,0.118745,5,31,0,5933.73,-0.165669,0.00152746,-2.83556,-2.4689,-1.9264,-2.54126,-2.45274,-2.32033,-1.91113,-1.88637,-2.04474,-2.15813,-2.39328,-1.40665,-2.7231,-2.71618,-1.40794,-1.27969,-2.17681,-3.09021,-3.26601,-1.58883,-2.02172,-2.92058,-0.212157,-0.101097,-0.244538,-0.152758,-0.116112,-0.199575,-0.126594,-0.142169,-0.240593,-0.357698,-0.113754,-0.199293,-0.179466,-0.291602,-0.214974,-0.232257,-0.157207,-0.127607,-0.223854,-0.193368,-0.196059,-0.254329,-0.106494,0.0390827, -5907.28,0.984132,0.118745,5,31,0,5934.33,-0.25545,0.00448755,-1.69069,-2.01335,-1.77163,-2.30603,-2.49152,-1.57809,-1.8542,-2.10274,-2.01238,-2.27824,-2.22447,-1.42891,-2.87708,-2.82301,-1.27687,-1.50763,-2.09728,-2.88422,-3.04933,-1.62906,-2.1219,-2.67806,-0.274857,-0.282542,-0.251405,-0.232108,-0.234986,-0.235362,-0.23886,-0.315712,-0.38353,-0.261716,-0.290438,-0.173991,-0.237273,-0.0588792,-0.313305,-0.32741,-0.148365,-0.232223,-0.156899,-0.324278,-0.321059,-0.371524,-0.25568,0.0669892, -5898.49,0.943724,0.118745,5,31,0,5924.27,-0.226014,0.00140102,-2.83712,-2.31534,-2.10846,-2.51197,-2.31247,-3.0448,-1.74887,-2.24496,-2.05242,-2.23807,-2.35051,-1.46592,-3.05366,-2.73498,-1.46668,-1.55851,-2.19039,-3.03883,-3.46062,-1.71741,-2.07787,-2.99703,-0.26175,-0.256368,-0.17223,-0.228403,-0.260527,-0.22473,-0.222987,-0.215522,-0.241519,-0.264097,-0.245378,-0.202356,-0.166893,-0.281911,-0.263059,-0.285411,-0.112129,-0.238267,-0.20906,-0.310847,-0.224357,-0.190043,-0.258069,0.0374301, -5896.11,0.940995,0.118745,5,31,0,5920.79,-0.291872,0.00144837,-2.35093,-1.86718,-2.04535,-2.30298,-2.38943,-2.06397,-1.81984,-1.94986,-2.08656,-2.40705,-2.33068,-1.3395,-3.04947,-2.68795,-1.34923,-1.61601,-2.11929,-2.35461,-3.51377,-1.41752,-2.08607,-2.83186,-0.299988,-0.350616,-0.233389,-0.313407,-0.254554,-0.362427,-0.320068,-0.310309,-0.30733,-0.30774,-0.314381,-0.27577,-0.290035,-0.280467,-0.293124,-0.220124,-0.251525,-0.274673,-0.314283,-0.242625,-0.275404,-0.260346,-0.255923,0.0380574;

  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 55; j++)
      EXPECT_FLOAT_EQ(expected_samples(i,j), samples(i,j));

  EXPECT_FLOAT_EQ(0.391415, timing.warmup);
  EXPECT_FLOAT_EQ(0.648336, timing.sampling);

}
TEST_F(StanIoStanCsvReader,ParseBlocker) {
  stan::io::stan_csv blocker0;
  std::stringstream out;
  blocker0 = stan::io::stan_csv_reader::parse(blocker0_stream, &out);

  // metadata
  EXPECT_EQ(2, blocker0.metadata.stan_version_major);
  EXPECT_EQ(9, blocker0.metadata.stan_version_minor);
  EXPECT_EQ(0, blocker0.metadata.stan_version_patch);

  EXPECT_EQ("blocker_model", blocker0.metadata.model);
  EXPECT_EQ("../example-models/bugs_examples/vol1/blocker/blocker.data.R", blocker0.metadata.data);
  EXPECT_EQ("../example-models/bugs_examples/vol1/blocker/blocker.init.R", blocker0.metadata.init);
  EXPECT_FALSE(blocker0.metadata.append_samples);
  EXPECT_FALSE(blocker0.metadata.save_warmup);
  EXPECT_EQ(4085885484U, blocker0.metadata.seed);
  EXPECT_FALSE(blocker0.metadata.random_seed);
  EXPECT_EQ(0U, blocker0.metadata.chain_id);
  EXPECT_EQ(2000U, blocker0.metadata.num_samples);
  EXPECT_EQ(2000U, blocker0.metadata.num_warmup);
  EXPECT_EQ(2U, blocker0.metadata.thin);
  EXPECT_EQ("hmc", blocker0.metadata.algorithm);
  EXPECT_EQ("nuts", blocker0.metadata.engine);

  // header
  ASSERT_EQ(55, blocker0.header.size());
  EXPECT_EQ("lp__", blocker0.header(0));
  EXPECT_EQ("accept_stat__", blocker0.header(1));
  EXPECT_EQ("stepsize__", blocker0.header(2));
  EXPECT_EQ("treedepth__", blocker0.header(3));
  EXPECT_EQ("n_leapfrog__", blocker0.header(4));
  EXPECT_EQ("divergent__", blocker0.header(5));
  EXPECT_EQ("energy__", blocker0.header(6));
  EXPECT_EQ("d", blocker0.header(7));
  EXPECT_EQ("sigmasq_delta", blocker0.header(8));
  EXPECT_EQ("mu[1]", blocker0.header(9));
  EXPECT_EQ("mu[2]", blocker0.header(10));
  EXPECT_EQ("mu[3]", blocker0.header(11));
  EXPECT_EQ("mu[4]", blocker0.header(12));
  EXPECT_EQ("mu[5]", blocker0.header(13));
  EXPECT_EQ("mu[6]", blocker0.header(14));
  EXPECT_EQ("mu[7]", blocker0.header(15));
  EXPECT_EQ("mu[8]", blocker0.header(16));
  EXPECT_EQ("mu[9]", blocker0.header(17));
  EXPECT_EQ("mu[10]", blocker0.header(18));
  EXPECT_EQ("mu[11]", blocker0.header(19));
  EXPECT_EQ("mu[12]", blocker0.header(20));
  EXPECT_EQ("mu[13]", blocker0.header(21));
  EXPECT_EQ("mu[14]", blocker0.header(22));
  EXPECT_EQ("mu[15]", blocker0.header(23));
  EXPECT_EQ("mu[16]", blocker0.header(24));
  EXPECT_EQ("mu[17]", blocker0.header(25));
  EXPECT_EQ("mu[18]", blocker0.header(26));
  EXPECT_EQ("mu[19]", blocker0.header(27));
  EXPECT_EQ("mu[20]", blocker0.header(28));
  EXPECT_EQ("mu[21]", blocker0.header(29));
  EXPECT_EQ("mu[22]", blocker0.header(30));
  EXPECT_EQ("delta[1]", blocker0.header(31));
  EXPECT_EQ("delta[2]", blocker0.header(32));
  EXPECT_EQ("delta[3]", blocker0.header(33));
  EXPECT_EQ("delta[4]", blocker0.header(34));
  EXPECT_EQ("delta[5]", blocker0.header(35));
  EXPECT_EQ("delta[6]", blocker0.header(36));
  EXPECT_EQ("delta[7]", blocker0.header(37));
  EXPECT_EQ("delta[8]", blocker0.header(38));
  EXPECT_EQ("delta[9]", blocker0.header(39));
  EXPECT_EQ("delta[10]", blocker0.header(40));
  EXPECT_EQ("delta[11]", blocker0.header(41));
  EXPECT_EQ("delta[12]", blocker0.header(42));
  EXPECT_EQ("delta[13]", blocker0.header(43));
  EXPECT_EQ("delta[14]", blocker0.header(44));
  EXPECT_EQ("delta[15]", blocker0.header(45));
  EXPECT_EQ("delta[16]", blocker0.header(46));
  EXPECT_EQ("delta[17]", blocker0.header(47));
  EXPECT_EQ("delta[18]", blocker0.header(48));
  EXPECT_EQ("delta[19]", blocker0.header(49));
  EXPECT_EQ("delta[20]", blocker0.header(50));
  EXPECT_EQ("delta[21]", blocker0.header(51));
  EXPECT_EQ("delta[22]", blocker0.header(52));
  EXPECT_EQ("delta_new", blocker0.header(53));
  EXPECT_EQ("sigma_delta", blocker0.header(54));

  // adaptation
  EXPECT_FLOAT_EQ(0.118745, blocker0.adaptation.step_size);
  ASSERT_EQ(47, blocker0.adaptation.metric.size());
  EXPECT_FLOAT_EQ(0.00376431, blocker0.adaptation.metric(0));
  EXPECT_FLOAT_EQ(1.38212, blocker0.adaptation.metric(1));
  EXPECT_FLOAT_EQ(0.216607, blocker0.adaptation.metric(2));
  EXPECT_FLOAT_EQ(0.0588362, blocker0.adaptation.metric(3));
  EXPECT_FLOAT_EQ(0.0773243, blocker0.adaptation.metric(4));
  EXPECT_FLOAT_EQ(0.00576153, blocker0.adaptation.metric(5));
  EXPECT_FLOAT_EQ(0.0226168, blocker0.adaptation.metric(6));
  EXPECT_FLOAT_EQ(0.140826, blocker0.adaptation.metric(7));
  EXPECT_FLOAT_EQ(0.00691007, blocker0.adaptation.metric(8));
  EXPECT_FLOAT_EQ(0.0144645, blocker0.adaptation.metric(9));
  EXPECT_FLOAT_EQ(0.0185795, blocker0.adaptation.metric(10));
  EXPECT_FLOAT_EQ(0.00443264, blocker0.adaptation.metric(11));
  EXPECT_FLOAT_EQ(0.0141671, blocker0.adaptation.metric(12));
  EXPECT_FLOAT_EQ(0.0165921, blocker0.adaptation.metric(13));
  EXPECT_FLOAT_EQ(0.0443114, blocker0.adaptation.metric(14));
  EXPECT_FLOAT_EQ(0.0180591, blocker0.adaptation.metric(15));
  EXPECT_FLOAT_EQ(0.0222813, blocker0.adaptation.metric(16));
  EXPECT_FLOAT_EQ(0.019491, blocker0.adaptation.metric(17));
  EXPECT_FLOAT_EQ(0.0372924, blocker0.adaptation.metric(18));
  EXPECT_FLOAT_EQ(0.0841709, blocker0.adaptation.metric(19));
  EXPECT_FLOAT_EQ(0.122628, blocker0.adaptation.metric(20));
  EXPECT_FLOAT_EQ(0.020642, blocker0.adaptation.metric(21));
  EXPECT_FLOAT_EQ(0.0174555, blocker0.adaptation.metric(22));
  EXPECT_FLOAT_EQ(0.0190478, blocker0.adaptation.metric(23));
  EXPECT_FLOAT_EQ(0.0194389, blocker0.adaptation.metric(24));
  EXPECT_FLOAT_EQ(0.0208765, blocker0.adaptation.metric(25));
  EXPECT_FLOAT_EQ(0.0266566, blocker0.adaptation.metric(26));
  EXPECT_FLOAT_EQ(0.00803359, blocker0.adaptation.metric(27));
  EXPECT_FLOAT_EQ(0.0179588, blocker0.adaptation.metric(28));
  EXPECT_FLOAT_EQ(0.0297546, blocker0.adaptation.metric(29));
  EXPECT_FLOAT_EQ(0.0114378, blocker0.adaptation.metric(30));
  EXPECT_FLOAT_EQ(0.0135584, blocker0.adaptation.metric(31));
  EXPECT_FLOAT_EQ(0.0149708, blocker0.adaptation.metric(32));
  EXPECT_FLOAT_EQ(0.00658085, blocker0.adaptation.metric(33));
  EXPECT_FLOAT_EQ(0.0125537, blocker0.adaptation.metric(34));
  EXPECT_FLOAT_EQ(0.0166118, blocker0.adaptation.metric(35));
  EXPECT_FLOAT_EQ(0.023786, blocker0.adaptation.metric(36));
  EXPECT_FLOAT_EQ(0.0308986, blocker0.adaptation.metric(37));
  EXPECT_FLOAT_EQ(0.013756, blocker0.adaptation.metric(38));
  EXPECT_FLOAT_EQ(0.0145632, blocker0.adaptation.metric(39));
  EXPECT_FLOAT_EQ(0.0241766, blocker0.adaptation.metric(40));
  EXPECT_FLOAT_EQ(0.0227147, blocker0.adaptation.metric(41));
  EXPECT_FLOAT_EQ(0.0247547, blocker0.adaptation.metric(42));
  EXPECT_FLOAT_EQ(0.0142955, blocker0.adaptation.metric(43));
  EXPECT_FLOAT_EQ(0.0167622, blocker0.adaptation.metric(44));
  EXPECT_FLOAT_EQ(0.0187646, blocker0.adaptation.metric(45));
  EXPECT_FLOAT_EQ(0.0248957, blocker0.adaptation.metric(46));

  // samples
  ASSERT_EQ(1000, blocker0.samples.rows());
  ASSERT_EQ(55, blocker0.samples.cols());

  Eigen::MatrixXd expected_samples(6, 55);
  expected_samples <<
-5919.76,0.946838,0.118745,5,31,0,5950.76,-0.28158,0.0044619,-2.43441,-2.43628,-2.27778,-2.20654,-2.30954,-2.5927,-1.81483,-2.0766,-1.91096,-2.16673,-2.40527,-1.50555,-2.78592,-2.79865,-1.4619,-1.37719,-2.28927,-2.94799,-2.95409,-1.50565,-1.99425,-3.01008,-0.262722,-0.292787,-0.192026,-0.603011,-0.448924,-0.235658,-0.321412,-0.115794,-0.270572,-0.357005,-0.299995,-0.222147,-0.368037,0.0721705,-0.306608,-0.220126,-0.263808,-0.224855,-0.257224,-0.174363,-0.319416,-0.426591,-0.447132,0.0667975, -5913.75,0.978966,0.118745,5,31,0,5933.73,-0.165669,0.00152746,-2.83556,-2.4689,-1.9264,-2.54126,-2.45274,-2.32033,-1.91113,-1.88637,-2.04474,-2.15813,-2.39328,-1.40665,-2.7231,-2.71618,-1.40794,-1.27969,-2.17681,-3.09021,-3.26601,-1.58883,-2.02172,-2.92058,-0.212157,-0.101097,-0.244538,-0.152758,-0.116112,-0.199575,-0.126594,-0.142169,-0.240593,-0.357698,-0.113754,-0.199293,-0.179466,-0.291602,-0.214974,-0.232257,-0.157207,-0.127607,-0.223854,-0.193368,-0.196059,-0.254329,-0.106494,0.0390827, -5907.28,0.984132,0.118745,5,31,0,5934.33,-0.25545,0.00448755,-1.69069,-2.01335,-1.77163,-2.30603,-2.49152,-1.57809,-1.8542,-2.10274,-2.01238,-2.27824,-2.22447,-1.42891,-2.87708,-2.82301,-1.27687,-1.50763,-2.09728,-2.88422,-3.04933,-1.62906,-2.1219,-2.67806,-0.274857,-0.282542,-0.251405,-0.232108,-0.234986,-0.235362,-0.23886,-0.315712,-0.38353,-0.261716,-0.290438,-0.173991,-0.237273,-0.0588792,-0.313305,-0.32741,-0.148365,-0.232223,-0.156899,-0.324278,-0.321059,-0.371524,-0.25568,0.0669892, -5898.49,0.943724,0.118745,5,31,0,5924.27,-0.226014,0.00140102,-2.83712,-2.31534,-2.10846,-2.51197,-2.31247,-3.0448,-1.74887,-2.24496,-2.05242,-2.23807,-2.35051,-1.46592,-3.05366,-2.73498,-1.46668,-1.55851,-2.19039,-3.03883,-3.46062,-1.71741,-2.07787,-2.99703,-0.26175,-0.256368,-0.17223,-0.228403,-0.260527,-0.22473,-0.222987,-0.215522,-0.241519,-0.264097,-0.245378,-0.202356,-0.166893,-0.281911,-0.263059,-0.285411,-0.112129,-0.238267,-0.20906,-0.310847,-0.224357,-0.190043,-0.258069,0.0374301, -5896.11,0.940995,0.118745,5,31,0,5920.79,-0.291872,0.00144837,-2.35093,-1.86718,-2.04535,-2.30298,-2.38943,-2.06397,-1.81984,-1.94986,-2.08656,-2.40705,-2.33068,-1.3395,-3.04947,-2.68795,-1.34923,-1.61601,-2.11929,-2.35461,-3.51377,-1.41752,-2.08607,-2.83186,-0.299988,-0.350616,-0.233389,-0.313407,-0.254554,-0.362427,-0.320068,-0.310309,-0.30733,-0.30774,-0.314381,-0.27577,-0.290035,-0.280467,-0.293124,-0.220124,-0.251525,-0.274673,-0.314283,-0.242625,-0.275404,-0.260346,-0.255923,0.0380574, -5880.49,0.914003,0.118745,5,31,0,5911.21,-0.319224,0.00044364,-2.20964,-1.96473,-2.07892,-2.38091,-2.2818,-2.07341,-1.81693,-1.98129,-1.89555,-2.15034,-2.29751,-1.33719,-2.75326,-2.63368,-1.04212,-1.58997,-2.0156,-2.54265,-3.23691,-1.45021,-2.11466,-3.06222,-0.329029,-0.305347,-0.337222,-0.317096,-0.332441,-0.270107,-0.343368,-0.328734,-0.300179,-0.311261,-0.326503,-0.303854,-0.335826,-0.292654,-0.294823,-0.283162,-0.346726,-0.33714,-0.331064,-0.314595,-0.322634,-0.32064,-0.309401,0.0210628;

  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 55; j++)
      EXPECT_FLOAT_EQ(expected_samples(i,j), blocker0.samples(i,j));

  EXPECT_FLOAT_EQ(0.391415, blocker0.timing.warmup);
  EXPECT_FLOAT_EQ(0.648336, blocker0.timing.sampling);

  EXPECT_EQ("", out.str());
}


TEST_F(StanIoStanCsvReader,ParseEightSchools) {
  stan::io::stan_csv eight_schools;
  std::stringstream out;
  eight_schools = stan::io::stan_csv_reader::parse(eight_schools_stream, &out);

  // metadata
  EXPECT_EQ(2, eight_schools.metadata.stan_version_major);
  EXPECT_EQ(18, eight_schools.metadata.stan_version_minor);
  EXPECT_EQ(0, eight_schools.metadata.stan_version_patch);

  EXPECT_EQ("eight_schools_model", eight_schools.metadata.model);
  EXPECT_EQ("eight_schools.data.R", eight_schools.metadata.data);
  EXPECT_EQ("2", eight_schools.metadata.init);
  EXPECT_FALSE(eight_schools.metadata.append_samples);
  EXPECT_FALSE(eight_schools.metadata.save_warmup);
  EXPECT_EQ(2212811313U, eight_schools.metadata.seed);
  EXPECT_FALSE(eight_schools.metadata.random_seed);
  EXPECT_EQ(3U, eight_schools.metadata.chain_id);
  EXPECT_EQ(1000U, eight_schools.metadata.num_samples);
  EXPECT_EQ(1000U, eight_schools.metadata.num_warmup);
  EXPECT_EQ(1U, eight_schools.metadata.thin);
  EXPECT_EQ("hmc", eight_schools.metadata.algorithm);
  EXPECT_EQ("nuts", eight_schools.metadata.engine);

    // header
  ASSERT_EQ(25, eight_schools.header.size());
  EXPECT_EQ("lp__", eight_schools.header(0));
  EXPECT_EQ("accept_stat__", eight_schools.header(1));
  EXPECT_EQ("stepsize__", eight_schools.header(2));
  EXPECT_EQ("treedepth__", eight_schools.header(3));
  EXPECT_EQ("n_leapfrog__", eight_schools.header(4));
  EXPECT_EQ("divergent__", eight_schools.header(5));
  EXPECT_EQ("energy__", eight_schools.header(6));
  EXPECT_EQ("mu", eight_schools.header(7));
  EXPECT_EQ("tau", eight_schools.header(8));
  EXPECT_EQ("eta[1]", eight_schools.header(9));
  EXPECT_EQ("eta[2]", eight_schools.header(10));
  EXPECT_EQ("eta[3]", eight_schools.header(11));
  EXPECT_EQ("eta[4]", eight_schools.header(12));
  EXPECT_EQ("eta[5]", eight_schools.header(13));
  EXPECT_EQ("eta[6]", eight_schools.header(14));
  EXPECT_EQ("eta[7]", eight_schools.header(15));
  EXPECT_EQ("eta[8]", eight_schools.header(16));
  EXPECT_EQ("theta[1]", eight_schools.header(17));
  EXPECT_EQ("theta[2]", eight_schools.header(18));
  EXPECT_EQ("theta[3]", eight_schools.header(19));
  EXPECT_EQ("theta[4]", eight_schools.header(20));
  EXPECT_EQ("theta[5]", eight_schools.header(21));
  EXPECT_EQ("theta[6]", eight_schools.header(22));
  EXPECT_EQ("theta[7]", eight_schools.header(23));
  EXPECT_EQ("theta[8]", eight_schools.header(24));

  // adaptation
  EXPECT_FLOAT_EQ(0.400175, eight_schools.adaptation.step_size);
  ASSERT_EQ(10, eight_schools.adaptation.metric.size());
  EXPECT_FLOAT_EQ(33.4344, eight_schools.adaptation.metric(0));
  EXPECT_FLOAT_EQ(1.06806, eight_schools.adaptation.metric(1));
  EXPECT_FLOAT_EQ(0.875304, eight_schools.adaptation.metric(2));
  EXPECT_FLOAT_EQ(0.881766, eight_schools.adaptation.metric(3));
  EXPECT_FLOAT_EQ(0.956387, eight_schools.adaptation.metric(4));
  EXPECT_FLOAT_EQ(0.704904, eight_schools.adaptation.metric(5));
  EXPECT_FLOAT_EQ(0.838633, eight_schools.adaptation.metric(6));
  EXPECT_FLOAT_EQ(0.785699, eight_schools.adaptation.metric(7));
  EXPECT_FLOAT_EQ(0.701917, eight_schools.adaptation.metric(8));
  EXPECT_FLOAT_EQ(0.886245, eight_schools.adaptation.metric(9));

  ASSERT_EQ(1000, eight_schools.samples.rows());
  ASSERT_EQ(25, eight_schools.samples.cols());

  Eigen::VectorXd first_draw(25);
  first_draw << -39.3871,0.998172,0.400175,3,7,0,41.5619,10.1835,2.02267,0.792196,-0.113107,1.23841,0.271166,-1.71864,0.0893772,0.689502,-0.977715,11.7858,9.95471,12.6884,10.732,6.70724,10.3643,11.5781,8.20589;
  for (int i = 0; i < 25; ++i)
    EXPECT_FLOAT_EQ(first_draw(i), eight_schools.samples(0, i));

  EXPECT_FLOAT_EQ(0.053314, eight_schools.timing.warmup);
  EXPECT_FLOAT_EQ(0.063405, eight_schools.timing.sampling);
  
  EXPECT_EQ("", out.str());
}

#include <stan/mcmc/chains.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <gtest/gtest.h>
#include <boost/random/additive_combine.hpp>
#include <set>
#include <exception>
#include <utility>
#include <fstream>



class McmcChains : public testing::Test {
public:
  
  void SetUp() {
    blocker1_stream.open("src/test/mcmc/test_csv_files/blocker.1.csv");
    blocker2_stream.open("src/test/mcmc/test_csv_files/blocker.2.csv");
    epil1_stream.open("src/test/mcmc/test_csv_files/epil.1.csv");
    epil2_stream.open("src/test/mcmc/test_csv_files/epil.2.csv");
  }
  
  void TearDown() {
    blocker1_stream.close();
    blocker2_stream.close();
    epil1_stream.close();
    epil2_stream.close();
  }
  std::ifstream blocker1_stream, blocker2_stream, epil1_stream, epil2_stream;
};

TEST_F(McmcChains, constructor) {
  stan::io::stan_csv blocker1 = stan::io::stan_csv_reader::parse(blocker1_stream);
  // construct with Eigen::Vector
  stan::mcmc::chains<> chains1(blocker1.header);
  EXPECT_EQ(0, chains1.num_chains());
  EXPECT_EQ(blocker1.header.size(), chains1.num_params());
  for (int i = 0; i < blocker1.header.size(); i++)
    EXPECT_EQ(blocker1.header(i), chains1.param_name(i));
  
  
  // construct with stan_csv
  stan::mcmc::chains<> chains2(blocker1);
  EXPECT_EQ(1, chains2.num_chains());
  EXPECT_EQ(blocker1.header.size(), chains2.num_params());
  for (int i = 0; i < blocker1.header.size(); i++)
    EXPECT_EQ(blocker1.header(i), chains2.param_name(i));
  EXPECT_EQ(0, chains2.warmup(0));
  EXPECT_EQ(1000, chains2.num_samples(0));
}

TEST_F(McmcChains, add) {
  stan::io::stan_csv blocker1 = stan::io::stan_csv_reader::parse(blocker1_stream);
  
  // construct with Eigen::Vector
  stan::mcmc::chains<> chains(blocker1.header);
  EXPECT_EQ(0, chains.num_chains());
  EXPECT_EQ(0, chains.num_samples());

  Eigen::RowVectorXd theta = blocker1.samples.row(0);
  EXPECT_NO_THROW(chains.add(1, theta)) 
    << "adding a single sample to a new chain";
  EXPECT_EQ(2, chains.num_chains());
  EXPECT_EQ(0, chains.num_samples(0));
  EXPECT_EQ(1, chains.num_samples(1));
  EXPECT_EQ(1, chains.num_samples());

  theta = blocker1.samples.row(1);
  EXPECT_NO_THROW(chains.add(1, theta)) 
    << "adding a single sample to an existing chain";
  EXPECT_EQ(2, chains.num_chains());
  EXPECT_EQ(0, chains.num_samples(0));
  EXPECT_EQ(2, chains.num_samples(1));
  EXPECT_EQ(2, chains.num_samples());

  EXPECT_NO_THROW(chains.add(3, blocker1.samples))
    << "adding multiple samples to a new chain";
  EXPECT_EQ(4, chains.num_chains());
  EXPECT_EQ(0, chains.num_samples(0));
  EXPECT_EQ(2, chains.num_samples(1));
  EXPECT_EQ(0, chains.num_samples(2));
  EXPECT_EQ(1000, chains.num_samples(3));
  EXPECT_EQ(1002, chains.num_samples());

  EXPECT_NO_THROW(chains.add(3, blocker1.samples))
    << "adding multiple samples to an existing chain";
  EXPECT_EQ(4, chains.num_chains());
  EXPECT_EQ(0, chains.num_samples(0));
  EXPECT_EQ(2, chains.num_samples(1));
  EXPECT_EQ(0, chains.num_samples(2));
  EXPECT_EQ(2000, chains.num_samples(3));
  EXPECT_EQ(2002, chains.num_samples());


  EXPECT_NO_THROW(chains.add(blocker1.samples))
    << "adding multiple samples, adds new chain";
  EXPECT_EQ(5, chains.num_chains());
  EXPECT_EQ(0, chains.num_samples(0));
  EXPECT_EQ(2, chains.num_samples(1));
  EXPECT_EQ(0, chains.num_samples(2));
  EXPECT_EQ(2000, chains.num_samples(3));
  EXPECT_EQ(1000, chains.num_samples(4));
  EXPECT_EQ(3002, chains.num_samples());

  stan::io::stan_csv epil1 = stan::io::stan_csv_reader::parse(epil1_stream);
  theta.resize(epil1.samples.cols());
  theta = epil1.samples.row(0);
  EXPECT_THROW(chains.add(1, theta), std::invalid_argument)
    << "adding mismatched sample to an existing chain";
  EXPECT_THROW(chains.add(10, theta), std::invalid_argument)
    << "adding mismatched sample to a new chain";
  EXPECT_THROW(chains.add(3, epil1.samples), std::invalid_argument)
    << "adding mismatched samples to an existing chain";
  EXPECT_THROW(chains.add(10, epil1.samples), std::invalid_argument)
    << "adding mismatched samples to a new chain";
  EXPECT_THROW(chains.add(epil1), std::invalid_argument)
    << "adding mismatched sample";

  EXPECT_EQ(5, chains.num_chains())
    << "validate state is identical to before";
  EXPECT_EQ(0, chains.num_samples(0))
    << "validate state is identical to before";
  EXPECT_EQ(2, chains.num_samples(1))
    << "validate state is identical to before";
  EXPECT_EQ(0, chains.num_samples(2))
    << "validate state is identical to before";
  EXPECT_EQ(2000, chains.num_samples(3))
    << "validate state is identical to before";
  EXPECT_EQ(1000, chains.num_samples(4))
    << "validate state is identical to before";
  EXPECT_EQ(3002, chains.num_samples())
    << "validate state is identical to before";
}

TEST_F(McmcChains, blocker1_num_chains) {
  stan::io::stan_csv blocker1 = stan::io::stan_csv_reader::parse(blocker1_stream);
  
  stan::mcmc::chains<> chains(blocker1);
  
  EXPECT_EQ(1, chains.num_chains());
}

TEST_F(McmcChains, blocker1_num_samples) {
  stan::io::stan_csv blocker1 = stan::io::stan_csv_reader::parse(blocker1_stream);
  
  stan::mcmc::chains<> chains(blocker1);
  
  EXPECT_EQ(1000, chains.num_samples());
}

TEST_F(McmcChains, blocker2_num_samples) {
  stan::io::stan_csv blocker2 = stan::io::stan_csv_reader::parse(blocker2_stream);
  
  stan::mcmc::chains<> chains(blocker2);
  
  EXPECT_EQ(1000, chains.num_samples());
}

TEST_F(McmcChains, blocker_num_samples) {
  stan::io::stan_csv blocker1 = stan::io::stan_csv_reader::parse(blocker1_stream);
  stan::io::stan_csv blocker2 = stan::io::stan_csv_reader::parse(blocker2_stream);
  
  stan::mcmc::chains<> chains(blocker1);
  chains.add(blocker2);
  
  EXPECT_EQ(2000, chains.num_samples());
  EXPECT_EQ(1000, chains.num_samples(0));
  EXPECT_EQ(1000, chains.num_samples(1));
}


TEST_F(McmcChains, blocker1_param_names) {
  stan::io::stan_csv blocker1 = stan::io::stan_csv_reader::parse(blocker1_stream);
  
  stan::mcmc::chains<> chains(blocker1);
  ASSERT_EQ(blocker1.header.size(), chains.num_params());
  ASSERT_EQ(blocker1.header.size(), chains.param_names().size());
  for (int i = 0; i < blocker1.header.size(); i++) {
    EXPECT_EQ(blocker1.header(i), chains.param_names()(i));
  }
}
TEST_F(McmcChains, blocker1_param_name) {
  stan::io::stan_csv blocker1 = stan::io::stan_csv_reader::parse(blocker1_stream);
  
  stan::mcmc::chains<> chains(blocker1);
  ASSERT_EQ(blocker1.header.size(), chains.num_params());
  for (int i = 0; i < blocker1.header.size(); i++) {
    EXPECT_EQ(blocker1.header(i), chains.param_name(i));
  }
}
TEST_F(McmcChains, blocker1_index) {
  stan::io::stan_csv blocker1 = stan::io::stan_csv_reader::parse(blocker1_stream);
  
  stan::mcmc::chains<> chains(blocker1);
  ASSERT_EQ(blocker1.header.size(), chains.num_params());
  for (int i = 0; i < blocker1.header.size(); i++)
    EXPECT_EQ(i, chains.index(blocker1.header(i)));
}
TEST_F(McmcChains, blocker1_warmup) {
  stan::io::stan_csv blocker1 = stan::io::stan_csv_reader::parse(blocker1_stream);
  
  stan::mcmc::chains<> chains(blocker1);

  ASSERT_EQ(1, chains.warmup().size());
  EXPECT_EQ(0, chains.warmup()(0));
  EXPECT_EQ(0, chains.warmup(0));

  chains.set_warmup(10);
  ASSERT_EQ(1, chains.warmup().size());
  EXPECT_EQ(10, chains.warmup()(0));
  EXPECT_EQ(10, chains.warmup(0));


  chains.set_warmup(100);
  ASSERT_EQ(1, chains.warmup().size());
  EXPECT_EQ(100, chains.warmup()(0));
  EXPECT_EQ(100, chains.warmup(0));
 
  chains.add(blocker1);
  ASSERT_EQ(2, chains.warmup().size());
  EXPECT_EQ(100, chains.warmup()(0));
  EXPECT_EQ(100, chains.warmup(0));
  EXPECT_EQ(0, chains.warmup()(1));
  EXPECT_EQ(0, chains.warmup(1));


  chains.set_warmup(1, 20);
  ASSERT_EQ(2, chains.warmup().size());
  EXPECT_EQ(100, chains.warmup()(0));
  EXPECT_EQ(100, chains.warmup(0));
  EXPECT_EQ(20, chains.warmup()(1));
  EXPECT_EQ(20, chains.warmup(1));


  chains.set_warmup(50);
  ASSERT_EQ(2, chains.warmup().size());
  EXPECT_EQ(50, chains.warmup()(0));
  EXPECT_EQ(50, chains.warmup(0));
  EXPECT_EQ(50, chains.warmup()(1));
  EXPECT_EQ(50, chains.warmup(1));
}
TEST_F(McmcChains, blocker_mean) {
  stan::io::stan_csv blocker1 = stan::io::stan_csv_reader::parse(blocker1_stream);
  stan::io::stan_csv blocker2 = stan::io::stan_csv_reader::parse(blocker2_stream);
  
  stan::mcmc::chains<> chains(blocker1);

  Eigen::VectorXd means1 = blocker1.samples.colwise().mean();
  for (int j = 0; j < chains.num_params(); j++) {
    ASSERT_FLOAT_EQ(means1(j), chains.mean(0,j))
      << "1: chain, param mean";
    ASSERT_FLOAT_EQ(means1(j), chains.mean(j))
      << "1: param mean";
  }

  chains.add(blocker2);
  Eigen::VectorXd means2 = blocker2.samples.colwise().mean();
  for (int j = 0; j < chains.num_params(); j++) {
    ASSERT_FLOAT_EQ(means2(j), chains.mean(1,j))
      << "2: chain, param mean";
    ASSERT_FLOAT_EQ((means1(j) + means2(j)) * 0.5, chains.mean(j))
      << "2: param mean";
  }


  chains.set_warmup(500);
  means1 = blocker1.samples.bottomRows(500).colwise().mean();
  means2 = blocker2.samples.bottomRows(500).colwise().mean();
  for (int j = 0; j < chains.num_params(); j++) {
    ASSERT_FLOAT_EQ(means1(j), chains.mean(0,j))
      << "3: chain mean 1 with warmup";
    ASSERT_FLOAT_EQ(means2(j), chains.mean(1,j))
      << "3: chain mean 2 with warmup";
    ASSERT_FLOAT_EQ((means1(j) + means2(j)) * 0.5, chains.mean(j))
      << "3: param mean with warmup";
  }

  for (int j = 0; j < chains.num_params(); j++) {
    std::string param_name = chains.param_name(j);
    ASSERT_FLOAT_EQ(chains.mean(0,j), chains.mean(0,param_name))
      << "4: chain mean 0 called with string name: " << param_name;
    ASSERT_FLOAT_EQ(chains.mean(1,j), chains.mean(1,param_name))
      << "4: chain mean 1 called with string name: " << param_name;
    ASSERT_FLOAT_EQ(chains.mean(j), chains.mean(param_name))
      << "4: mean called with string name: " << param_name;
  }
}

double sd(Eigen::VectorXd x) {
  return sqrt((x.array() - x.mean()).square().sum() / (x.rows() - 1));
}

TEST_F(McmcChains, blocker_sd) {
  stan::io::stan_csv blocker1 = stan::io::stan_csv_reader::parse(blocker1_stream);
  stan::io::stan_csv blocker2 = stan::io::stan_csv_reader::parse(blocker2_stream);
  
  stan::mcmc::chains<> chains(blocker1);
  
  using std::sqrt;
  for (int j = 0; j < chains.num_params(); j++) {
    ASSERT_NEAR(sd(blocker1.samples.col(j)), chains.sd(0,j), 1e-8)
      << "1: chain, param sd. index: " << j;
    ASSERT_NEAR(sd(blocker1.samples.col(j)), chains.sd(j), 1e-8)
      << "1: param sd. index: " << j;
  }

  chains.add(blocker2);
  for (int j = 0; j < chains.num_params(); j++) {
    ASSERT_NEAR(sd(blocker2.samples.col(j)), chains.sd(1,j), 1e-8)
      << "2: chain, param sd. index: " << j;
    Eigen::VectorXd x(blocker1.samples.rows() + blocker2.samples.rows());
    x << blocker1.samples.col(j), blocker2.samples.col(j);
    ASSERT_NEAR(sd(x), chains.sd(j), 1e-8)
      << "2: param sd. index: " << j;
  }
  
  chains.set_warmup(500);
  for (int j = 0; j < chains.num_params(); j++) {
    Eigen::VectorXd x1(500), x2(500), x(1000);
    x1 << blocker1.samples.col(j).bottomRows(500);
    x2 << blocker2.samples.col(j).bottomRows(500);
    x << x1, x2;
    
    ASSERT_NEAR(sd(x1), chains.sd(0,j), 1e-8)
      << "3: chain sd 1 with warmup";
    ASSERT_NEAR(sd(x2), chains.sd(1,j), 1e-8)
      << "3: chain sd 2 with warmup";
    ASSERT_NEAR(sd(x), chains.sd(j), 1e-8)
      << "3: param sd with warmup";
  }

  for (int j = 0; j < chains.num_params(); j++) {
    std::string param_name = chains.param_name(j);
    ASSERT_FLOAT_EQ(chains.sd(0,j), chains.sd(0,param_name))
      << "4: chain sd 0 called with string name: " << param_name;
    ASSERT_FLOAT_EQ(chains.sd(1,j), chains.sd(1,param_name))
      << "4: chain sd 1 called with string name: " << param_name;
    ASSERT_FLOAT_EQ(chains.sd(j), chains.sd(param_name))
      << "4: sd called with string name: " << param_name;
  }

}


double variance(Eigen::VectorXd x) {
  return (x.array() - x.mean()).square().sum() / (x.rows() - 1);
}

TEST_F(McmcChains, blocker_variance) {
  stan::io::stan_csv blocker1 = stan::io::stan_csv_reader::parse(blocker1_stream);
  stan::io::stan_csv blocker2 = stan::io::stan_csv_reader::parse(blocker2_stream);
  
  stan::mcmc::chains<> chains(blocker1);
  
  using std::sqrt;
  for (int j = 0; j < chains.num_params(); j++) {
    ASSERT_NEAR(variance(blocker1.samples.col(j)), chains.variance(0,j), 1e-8)
      << "1: chain, param variance. index: " << j;
    ASSERT_NEAR(variance(blocker1.samples.col(j)), chains.variance(j), 1e-8)
      << "1: param variance. index: " << j;
  }

  chains.add(blocker2);
  for (int j = 0; j < chains.num_params(); j++) {
    ASSERT_NEAR(variance(blocker2.samples.col(j)), chains.variance(1,j), 1e-8)
      << "2: chain, param variance. index: " << j;
    Eigen::VectorXd x(blocker1.samples.rows() + blocker2.samples.rows());
    x << blocker1.samples.col(j), blocker2.samples.col(j);
    ASSERT_NEAR(variance(x), chains.variance(j), 1e-8)
      << "2: param variance. index: " << j;
  }
  
  chains.set_warmup(500);
  for (int j = 0; j < chains.num_params(); j++) {
    Eigen::VectorXd x1(500), x2(500), x(1000);
    x1 << blocker1.samples.col(j).bottomRows(500);
    x2 << blocker2.samples.col(j).bottomRows(500);
    x << x1, x2;
    
    ASSERT_NEAR(variance(x1), chains.variance(0,j), 1e-8)
      << "3: chain variance 1 with warmup";
    ASSERT_NEAR(variance(x2), chains.variance(1,j), 1e-8)
      << "3: chain variance 2 with warmup";
    ASSERT_NEAR(variance(x), chains.variance(j), 1e-8)
      << "3: param variance with warmup";
  }

  for (int j = 0; j < chains.num_params(); j++) {
    std::string param_name = chains.param_name(j);
    ASSERT_FLOAT_EQ(chains.variance(0,j), chains.variance(0,param_name))
      << "4: chain variance 0 called with string name: " << param_name;
    ASSERT_FLOAT_EQ(chains.variance(1,j), chains.variance(1,param_name))
      << "4: chain variance 1 called with string name: " << param_name;
    ASSERT_FLOAT_EQ(chains.variance(j), chains.variance(param_name))
      << "4: variance called with string name: " << param_name;
  }

}

double covariance(Eigen::VectorXd x, Eigen::VectorXd y) {
  double x_mean = x.mean();
  double y_mean = y.mean();
  return ((x.array() - x_mean) * (y.array() - y_mean)).sum() / (x.rows() - 1);
}

TEST_F(McmcChains, blocker_covariance) {
  stan::io::stan_csv blocker1 = stan::io::stan_csv_reader::parse(blocker1_stream);
  stan::io::stan_csv blocker2 = stan::io::stan_csv_reader::parse(blocker2_stream);
  
  stan::mcmc::chains<> chains(blocker1);
  chains.add(blocker2);
  
  int n = 0;
  for (int i = 0; i < chains.num_params(); i++) {
    for (int j = i; j < chains.num_params(); j++) {
      if (++n % 13 == 0) { // test every 13th value
  Eigen::VectorXd x1(1000), x2(1000), x(2000);
  Eigen::VectorXd y1(1000), y2(1000), y(2000);
  x1 << blocker1.samples.col(i);
  x2 << blocker1.samples.col(j);

  y1 << blocker2.samples.col(i);
  y2 << blocker2.samples.col(j);

  x << x1, y1;
  y << x2, y2;
      
  double cov1 = covariance(x1, x2);
  double cov2 = covariance(y1, y2);
  double cov = covariance(x, y);
      
  ASSERT_NEAR(cov1, chains.covariance(0,i,j), 1e-8);
  ASSERT_NEAR(cov2, chains.covariance(1,i,j), 1e-8);
  ASSERT_NEAR(cov, chains.covariance(i,j), 1e-8);

  ASSERT_NEAR(cov1, chains.covariance(0,j,i), 1e-8);
  ASSERT_NEAR(cov2, chains.covariance(1,j,i), 1e-8);
  ASSERT_NEAR(cov, chains.covariance(j,i), 1e-8);

  std::string name1 = chains.param_name(i);
  std::string name2 = chains.param_name(j);
  ASSERT_FLOAT_EQ(chains.covariance(0,i,j), chains.covariance(0,name1,name2));
  ASSERT_FLOAT_EQ(chains.covariance(1,j,i), chains.covariance(1,name2,name1));
  ASSERT_FLOAT_EQ(chains.covariance(i,j), chains.covariance(name1,name2));
      }
    }
  }
}

TEST_F(McmcChains, blocker_correlation) {
  stan::io::stan_csv blocker1 = stan::io::stan_csv_reader::parse(blocker1_stream);
  stan::io::stan_csv blocker2 = stan::io::stan_csv_reader::parse(blocker2_stream);
  
  stan::mcmc::chains<> chains(blocker1);
  chains.add(blocker2);
  
  int n = 0;
  for (int i = 0; i < chains.num_params(); i++) {
    for (int j = i; j < chains.num_params(); j++) {
      if (++n % 13 == 0) { // test every 13th value
  Eigen::VectorXd x1(1000), x2(1000), x(2000);
  Eigen::VectorXd y1(1000), y2(1000), y(2000);
  x1 << blocker1.samples.col(i);
  x2 << blocker1.samples.col(j);

  y1 << blocker2.samples.col(i);
  y2 << blocker2.samples.col(j);

  x << x1, y1;
  y << x2, y2;
      

  double cov1 = covariance(x1, x2);
  double cov2 = covariance(y1, y2);
  double cov = covariance(x, y);

  double corr1 = 0;
  double corr2 = 0;
  double corr = 0;

  if (std::fabs(cov1) > 1e-8)
    corr1 = cov1 / sd(x1) / sd(x2);
  if (std::fabs(cov2) > 1e-8)
    corr2 = cov2 / sd(y1) / sd(y2);
  if (std::fabs(cov) > 1e-8)
    corr = cov / sd(x) / sd(y);
    
  ASSERT_NEAR(corr1, chains.correlation(0,i,j), 1e-8)
    << "(" << i << ", " << j << ")";
  ASSERT_NEAR(corr2, chains.correlation(1,i,j), 1e-8)
    << "(" << i << ", " << j << ")";
  ASSERT_NEAR(corr, chains.correlation(i,j), 1e-8)
    << "(" << i << ", " << j << ")";

  ASSERT_NEAR(corr1, chains.correlation(0,j,i), 1e-8)
    << "(" << i << ", " << j << ")";
  ASSERT_NEAR(corr2, chains.correlation(1,j,i), 1e-8)
    << "(" << i << ", " << j << ")";
  ASSERT_NEAR(corr, chains.correlation(j,i), 1e-8)
    << "(" << i << ", " << j << ")";
  
  std::string name1 = chains.param_name(i);
  std::string name2 = chains.param_name(j);
  ASSERT_FLOAT_EQ(chains.correlation(0,i,j), chains.correlation(0,name1,name2));
  ASSERT_FLOAT_EQ(chains.correlation(1,j,i), chains.correlation(1,name2,name1));
  ASSERT_FLOAT_EQ(chains.correlation(i,j), chains.correlation(name1,name2));
      }
    }
  }
}

TEST_F(McmcChains, blocker_quantile) {
  stan::io::stan_csv blocker1 = stan::io::stan_csv_reader::parse(blocker1_stream);
  
  stan::mcmc::chains<> chains(blocker1);
  
  int index = 5;
  
  // R's quantile function
  EXPECT_NEAR(-2.93776, chains.quantile(0,index,0.1), 1e-2);
  EXPECT_NEAR(-2.71573, chains.quantile(0,index,0.2), 1e-2);
  EXPECT_NEAR(-2.58469, chains.quantile(0,index,0.3), 1e-2);
  EXPECT_NEAR(-2.57960, chains.quantile(0,index,0.4), 1e-2);
  EXPECT_NEAR(-2.49539, chains.quantile(0,index,0.5), 1e-2);
  EXPECT_NEAR(-2.37983, chains.quantile(0,index,0.6), 1e-2);
  EXPECT_NEAR(-2.22837, chains.quantile(0,index,0.7), 1e-2);
  EXPECT_NEAR(-2.08649, chains.quantile(0,index,0.8), 1e-2);
  EXPECT_NEAR(-1.90839, chains.quantile(0,index,0.9), 1e-2);

  EXPECT_NEAR(-2.93776, chains.quantile(index,0.1), 1e-2);
  EXPECT_NEAR(-2.71573, chains.quantile(index,0.2), 1e-2);
  EXPECT_NEAR(-2.58469, chains.quantile(index,0.3), 1e-2);
  EXPECT_NEAR(-2.57960, chains.quantile(index,0.4), 1e-2);
  EXPECT_NEAR(-2.49539, chains.quantile(index,0.5), 1e-2);
  EXPECT_NEAR(-2.37983, chains.quantile(index,0.6), 1e-2);
  EXPECT_NEAR(-2.22837, chains.quantile(index,0.7), 1e-2);
  EXPECT_NEAR(-2.08649, chains.quantile(index,0.8), 1e-2);
  EXPECT_NEAR(-1.90839, chains.quantile(index,0.9), 1e-2);

  std::string name = chains.param_name(index);
  EXPECT_FLOAT_EQ(chains.quantile(0,index,0.1), chains.quantile(0,name,0.1));
  EXPECT_FLOAT_EQ(chains.quantile(0,index,0.3), chains.quantile(0,name,0.3));  
  EXPECT_FLOAT_EQ(chains.quantile(0,index,0.5), chains.quantile(0,name,0.5));

  EXPECT_FLOAT_EQ(chains.quantile(index,0.2), chains.quantile(name,0.2));
  EXPECT_FLOAT_EQ(chains.quantile(index,0.4), chains.quantile(name,0.4));  
  EXPECT_FLOAT_EQ(chains.quantile(index,0.6), chains.quantile(name,0.6));
}

TEST_F(McmcChains, blocker_quantiles) {
  stan::io::stan_csv blocker1 = stan::io::stan_csv_reader::parse(blocker1_stream);
  
  stan::mcmc::chains<> chains(blocker1);
  
  int index = 5;

  Eigen::VectorXd probs(9);
  probs << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9;
  
  Eigen::VectorXd quantiles;
  
  quantiles = chains.quantiles(0,index,probs);
  // R's quantile function
  ASSERT_EQ(9, quantiles.size());
  EXPECT_NEAR(-2.93776, quantiles(0), 1e-2);
  EXPECT_NEAR(-2.71573, quantiles(1), 1e-2);
  EXPECT_NEAR(-2.58469, quantiles(2), 1e-2);
  EXPECT_NEAR(-2.57960, quantiles(3), 1e-2);
  EXPECT_NEAR(-2.49539, quantiles(4), 1e-2);
  EXPECT_NEAR(-2.37983, quantiles(5), 1e-2);
  EXPECT_NEAR(-2.22837, quantiles(6), 1e-2);
  EXPECT_NEAR(-2.08649, quantiles(7), 1e-2);
  EXPECT_NEAR(-1.90839, quantiles(8), 1e-2);


  quantiles = chains.quantiles(index,probs);
  // R's quantile function
  ASSERT_EQ(9, quantiles.size());
  EXPECT_NEAR(-2.93776, quantiles(0), 1e-2);
  EXPECT_NEAR(-2.71573, quantiles(1), 1e-2);
  EXPECT_NEAR(-2.58469, quantiles(2), 1e-2);
  EXPECT_NEAR(-2.57960, quantiles(3), 1e-2);
  EXPECT_NEAR(-2.49539, quantiles(4), 1e-2);
  EXPECT_NEAR(-2.37983, quantiles(5), 1e-2);
  EXPECT_NEAR(-2.22837, quantiles(6), 1e-2);
  EXPECT_NEAR(-2.08649, quantiles(7), 1e-2);
  EXPECT_NEAR(-1.90839, quantiles(8), 1e-2);


  std::string name = chains.param_name(index);
  Eigen::VectorXd quantiles_by_name;
  quantiles = chains.quantiles(0,index,probs);
  quantiles_by_name = chains.quantiles(0,name,probs);
  
  ASSERT_EQ(quantiles.size(), quantiles_by_name.size());
  for (int i = 0; i < quantiles.size(); i++) {
    EXPECT_FLOAT_EQ(quantiles(i), quantiles_by_name(i));
  }

  quantiles = chains.quantiles(index,probs);
  quantiles_by_name = chains.quantiles(name,probs);
  
  ASSERT_EQ(quantiles.size(), quantiles_by_name.size());
  for (int i = 0; i < quantiles.size(); i++) {
    EXPECT_FLOAT_EQ(quantiles(i), quantiles_by_name(i));
  }

}
TEST_F(McmcChains, blocker_central_interval) {
  stan::io::stan_csv blocker1 = stan::io::stan_csv_reader::parse(blocker1_stream);
  
  stan::mcmc::chains<> chains(blocker1);
  
  int index = 5;
  
  Eigen::Vector2d interval;

  interval = chains.central_interval(0,index,0.6);
  // R's quantile function
  EXPECT_NEAR(-2.71573, interval(0), 1e-2); // 0.2
  EXPECT_NEAR(-2.08649, interval(1), 1e-2); // 0.8

  interval = chains.central_interval(index,0.6);
  // R's quantile function
  EXPECT_NEAR(-2.71573, interval(0), 1e-2); // 0.2
  EXPECT_NEAR(-2.08649, interval(1), 1e-2); // 0.8

  std::string name = chains.param_name(index);
  Eigen::VectorXd interval_by_name;
  interval = chains.central_interval(0,index,0.6);
  interval_by_name = chains.central_interval(0,name,0.6);
  ASSERT_EQ(2, interval_by_name.size());
  EXPECT_FLOAT_EQ(interval(0), interval_by_name(0));
  EXPECT_FLOAT_EQ(interval(1), interval_by_name(1));

  interval = chains.central_interval(index,0.6);
  interval_by_name = chains.central_interval(name,0.6);
  ASSERT_EQ(2, interval_by_name.size());
  EXPECT_FLOAT_EQ(interval(0), interval_by_name(0));
  EXPECT_FLOAT_EQ(interval(1), interval_by_name(1));
}
TEST_F(McmcChains, blocker_autocorrelation) {
  stan::io::stan_csv blocker1 = stan::io::stan_csv_reader::parse(blocker1_stream);
  
  stan::mcmc::chains<> chains(blocker1);
  Eigen::VectorXd ac;
  EXPECT_NO_THROW(ac = chains.autocorrelation(0,5));

  EXPECT_NEAR(1.0, ac[0], 0.01);  
  EXPECT_NEAR(0.277, ac[1], 0.01);
  EXPECT_NEAR(0.197, ac[2], 0.01);
  EXPECT_NEAR(0.123, ac[3], 0.01);
  EXPECT_NEAR(0.087, ac[4], 0.01);
  EXPECT_NEAR(0.096, ac[5], 0.01);
  EXPECT_NEAR(0.100, ac[6], 0.01);
  EXPECT_NEAR(0.007, ac[7], 0.01);
  EXPECT_NEAR(0.052, ac[8], 0.01);
  EXPECT_NEAR(0.038, ac[9], 0.01);
  EXPECT_NEAR(-0.007, ac[10], 0.01);
  EXPECT_NEAR(0.070, ac[11], 0.01);
  EXPECT_NEAR(0.004, ac[12], 0.01);
  EXPECT_NEAR(0.019, ac[13], 0.01);
  EXPECT_NEAR(0.003, ac[14], 0.01);
  EXPECT_NEAR(-0.035, ac[15], 0.01);
  EXPECT_NEAR(-0.076, ac[16], 0.01);
  EXPECT_NEAR(-0.048, ac[17], 0.01);
  EXPECT_NEAR(-0.097, ac[18], 0.01);
  EXPECT_NEAR(-0.014, ac[19], 0.01);
  EXPECT_NEAR(-0.065, ac[20], 0.01);
  EXPECT_NEAR(-0.069, ac[21], 0.01);
  EXPECT_NEAR(-0.004, ac[22], 0.01);
  EXPECT_NEAR(-0.084, ac[23], 0.01);
  EXPECT_NEAR(-0.005, ac[24], 0.01);
  EXPECT_NEAR(0.031, ac[25], 0.01);
  EXPECT_NEAR(0.002, ac[26], 0.01);
  EXPECT_NEAR(-0.019, ac[27], 0.01);
  EXPECT_NEAR(0.002, ac[28], 0.01);
  EXPECT_NEAR(-0.011, ac[29], 0.01);
  EXPECT_NEAR(-0.016, ac[30], 0.01);

  std::string name = chains.param_name(5);
  Eigen::VectorXd ac_by_name;
  EXPECT_NO_THROW(ac_by_name = chains.autocorrelation(0,name));
  ASSERT_EQ(ac.size(), ac_by_name.size());
  for (int i = 0; i < ac.size(); i++) {
    EXPECT_FLOAT_EQ(ac(i), ac_by_name(i));
  }
}
TEST_F(McmcChains, blocker_autocovariance) {
  stan::io::stan_csv blocker1 = stan::io::stan_csv_reader::parse(blocker1_stream);
  
  stan::mcmc::chains<> chains(blocker1);
  Eigen::VectorXd ac;
  EXPECT_NO_THROW(ac = chains.autocovariance(0,5));

  EXPECT_NEAR( 0.1728635075, ac[0], 0.01);  
  EXPECT_NEAR( 0.0479453913, ac[1], 0.01);
  EXPECT_NEAR( 0.0339712275, ac[2], 0.01);
  EXPECT_NEAR( 0.0213133322, ac[3], 0.01);
  EXPECT_NEAR( 0.0150613410, ac[4], 0.01);
  EXPECT_NEAR( 0.0166353592, ac[5], 0.01);
  EXPECT_NEAR( 0.0173524816, ac[6], 0.01);
  EXPECT_NEAR( 0.0011691359, ac[7], 0.01);
  EXPECT_NEAR( 0.0089205594, ac[8], 0.01);
  EXPECT_NEAR( 0.0065516382, ac[9], 0.01);
  EXPECT_NEAR(-0.0012707460, ac[10], 0.01);
  EXPECT_NEAR( 0.0120453780, ac[11], 0.01);
  EXPECT_NEAR( 0.0007551419, ac[12], 0.01);
  EXPECT_NEAR( 0.0032945508, ac[13], 0.01);
  EXPECT_NEAR( 0.0005656385, ac[14], 0.01);
  EXPECT_NEAR(-0.0060412611, ac[15], 0.01);
  EXPECT_NEAR(-0.0131231022, ac[16], 0.01);
  EXPECT_NEAR(-0.0083577880, ac[17], 0.01);
  EXPECT_NEAR(-0.0168234290, ac[18], 0.01);
  EXPECT_NEAR(-0.0024401327, ac[19], 0.01);
  EXPECT_NEAR(-0.0112285003, ac[20], 0.01);
  EXPECT_NEAR(-0.0120031605, ac[21], 0.01);
  EXPECT_NEAR(-0.0007499154, ac[22], 0.01);
  EXPECT_NEAR(-0.0144821063, ac[23], 0.01);
  EXPECT_NEAR(-0.0008450844, ac[24], 0.01);
  EXPECT_NEAR( 0.0054121234, ac[25], 0.01);
  EXPECT_NEAR( 0.0003861726, ac[26], 0.01);
  EXPECT_NEAR(-0.0032979440, ac[27], 0.01);
  EXPECT_NEAR( 0.0003806655, ac[28], 0.01);
  EXPECT_NEAR(-0.0019447624, ac[29], 0.01);
  EXPECT_NEAR(-0.0028139251, ac[30], 0.01);

  std::string name = chains.param_name(5);
  Eigen::VectorXd ac_by_name;
  EXPECT_NO_THROW(ac_by_name = chains.autocovariance(0,name));
  ASSERT_EQ(ac.size(), ac_by_name.size());
  for (int i = 0; i < ac.size(); i++) {
    EXPECT_FLOAT_EQ(ac(i), ac_by_name(i));
  }
}

TEST_F(McmcChains,blocker_effective_sample_size) {
  stan::io::stan_csv blocker1 = stan::io::stan_csv_reader::parse(blocker1_stream);
  stan::io::stan_csv blocker2 = stan::io::stan_csv_reader::parse(blocker2_stream);
  
  stan::mcmc::chains<> chains(blocker1);
  chains.add(blocker2);

  Eigen::VectorXd n_eff(48);
  n_eff << 
    431,136,450,121,258,
    202,287,615,107,131,
    39,573,608,538,412,
    32,199,106,153,298,
    105,155,340,475,806,
    518,657,277,149,979,
    150,741,923,647,473,
    638,796,62,709,551,
    687,933,798,574,377,
    457,650,44;

  for (int index = 3; index < chains.num_params(); index++) {
    ASSERT_NEAR(n_eff(index - 3), chains.effective_sample_size(index), 1.0)
      << "n_effective for index: " << index << ", parameter: " 
      << chains.param_name(index);
  }

  for (int index = 0; index < chains.num_params(); index++) {
    std::string name = chains.param_name(index);
    ASSERT_EQ(chains.effective_sample_size(index), 
        chains.effective_sample_size(name));
  }
}

TEST_F(McmcChains,blocker_split_potential_scale_reduction) {
  stan::io::stan_csv blocker1 = stan::io::stan_csv_reader::parse(blocker1_stream);
  stan::io::stan_csv blocker2 = stan::io::stan_csv_reader::parse(blocker2_stream);
  
  stan::mcmc::chains<> chains(blocker1);
  chains.add(blocker2);

  Eigen::VectorXd rhat(48);
  rhat << 
    1.0044119, 1.0130149, 1.0024355, 1.0123774, 1.0091136,
    1.0046465, 1.0030522, 1.0025938, 1.0016099, 1.0218149, 
    1.0397644, 1.0087594, 1.0017704, 1.0048593, 1.0113208,
    1.0401753, 1.0040739, 1.0151312, 1.0120381, 1.0034498, 
    1.0179943, 1.0160549, 1.0107518, 1.0050004, 0.9997756, 
    1.0014585, 1.0002573, 1.0113728, 1.0368905, 1.0020098,
    1.0058891, 1.0008469, 1.0031100, 1.0071503, 1.0083617,
    1.0062931, 0.9997813, 1.0169052, 0.9998256, 0.9994163,
    1.0029211, 1.0013409, 1.0073875, 1.0023928, 1.0029867,
    1.0002895, 1.0018717, 1.0289164;

  for (int index = 3; index < chains.num_params(); index++) {
    ASSERT_FLOAT_EQ(rhat(index - 3), chains.split_potential_scale_reduction(index))
      << "rhat for index: " << index << ", parameter: " 
      << chains.param_name(index);
  }

  for (int index = 0; index < chains.num_params(); index++) {
    std::string name = chains.param_name(index);
    ASSERT_EQ(chains.split_potential_scale_reduction(index), 
        chains.split_potential_scale_reduction(name));
  }

}

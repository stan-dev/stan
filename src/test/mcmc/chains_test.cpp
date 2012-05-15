#include <stan/mcmc/chains.hpp>
#include <gtest/gtest.h>

TEST(McmcChains,ctor) {
  using std::vector;
  using std::string;
  using stan::mcmc::chains;

  size_t K = 4;

  // b(), a(2), d(3,4,5), c(6,7)

  vector<string> names;
  names.push_back("b");
  names.push_back("a");
  names.push_back("d");
  names.push_back("c");

  vector<size_t> b_dims;

  vector<size_t> a_dims;
  a_dims.push_back(2);

  vector<size_t> d_dims;
  d_dims.push_back(3);
  d_dims.push_back(4);
  d_dims.push_back(5);

  vector<size_t> c_dims;
  c_dims.push_back(6);
  c_dims.push_back(7);

  vector<vector<size_t> > dimss;
  dimss.push_back(b_dims);
  dimss.push_back(a_dims);
  dimss.push_back(d_dims);
  dimss.push_back(c_dims);
  chains c(K,names,dimss);

  EXPECT_EQ(4U, c.num_chains());

  EXPECT_EQ(1 + 2 + 3*4*5 + 6*7, c.num_params());

  EXPECT_EQ(4U, c.num_param_names());

  EXPECT_EQ(4U, c.param_names().size());
  EXPECT_EQ("b", c.param_names()[0]);
  EXPECT_EQ("a", c.param_names()[1]);
  EXPECT_EQ("d", c.param_names()[2]);
  EXPECT_EQ("c", c.param_names()[3]);

  EXPECT_EQ(0, c.param_start(0));
  EXPECT_EQ(1, c.param_start(1));
  EXPECT_EQ(3, c.param_start(2));
  EXPECT_EQ(63, c.param_start(3));
  EXPECT_EQ(0, c.param_starts()[0]);
  EXPECT_EQ(1, c.param_starts()[1]);
  EXPECT_EQ(3, c.param_starts()[2]);
  EXPECT_EQ(63, c.param_starts()[3]);

  EXPECT_EQ(1U, c.param_size(0));
  EXPECT_EQ(1U, c.param_sizes()[0]);
  EXPECT_EQ(2U, c.param_size(1));
  EXPECT_EQ(2U, c.param_sizes()[1]);
  EXPECT_EQ(60U, c.param_size(2));
  EXPECT_EQ(60U, c.param_sizes()[2]);
  EXPECT_EQ(42U, c.param_size(3));
  EXPECT_EQ(42U, c.param_sizes()[3]);
  EXPECT_EQ(4U, c.param_sizes().size());

  EXPECT_EQ(0U, c.param_name_to_index("b"));
  EXPECT_EQ(1U, c.param_name_to_index("a"));
  EXPECT_EQ(2U, c.param_name_to_index("d"));
  EXPECT_EQ(3U, c.param_name_to_index("c"));

  EXPECT_EQ(4U, c.param_dimss().size());

  EXPECT_EQ(0U, c.param_dimss()[0].size());
  EXPECT_EQ(0U, c.param_dims(0).size());
  EXPECT_EQ(1U, c.param_dimss()[1].size());
  EXPECT_EQ(1U, c.param_dims(1).size());
  EXPECT_EQ(3U, c.param_dimss()[2].size());
  EXPECT_EQ(3U, c.param_dims(2).size());
  EXPECT_EQ(2U, c.param_dimss()[3].size());
  EXPECT_EQ(2U, c.param_dims(3).size());

  EXPECT_EQ(2U, c.param_dimss()[1][0]);
  EXPECT_EQ(2U, c.param_dims(1)[0]);

  EXPECT_EQ(4U, c.param_dimss()[2][1]);
  EXPECT_EQ(4U, c.param_dims(2)[1]);
  

  EXPECT_THROW(c.param_dims(5), std::out_of_range);

  EXPECT_EQ(0U, c.warmup());
  c.set_warmup(1000U);
  EXPECT_EQ(1000U,c.warmup());
}

TEST(McmcChains,get_offset) {
  using std::vector;
  vector<size_t> idxs(3);
  idxs[0] = 0;
  idxs[1] = 0;
  idxs[2] = 0;
  vector<size_t> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 4;
  size_t offset = 0;
  for (size_t c = 0; c < 4; ++c) {
    for (size_t b = 0; b < 3; ++b) {
      for (size_t a = 0; a < 2; ++a) {
        idxs[0] = a;
        idxs[1] = b;
        idxs[2] = c;
        EXPECT_EQ(offset,
                  stan::mcmc::chains::get_offset(dims,idxs));
        ++offset;
      }
    }
  }
}


TEST(McmcChains,increment_indexes) {
  using std::vector;
  vector<size_t> idxs(3);
  idxs[0] = 0;
  idxs[1] = 0;
  idxs[2] = 0;
  vector<size_t> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 4;
  for (size_t c = 0; c < 4; ++c) {
    for (size_t b = 0; b < 3; ++b) {
      for (size_t a = 0; a < 2; ++a) {
        EXPECT_FLOAT_EQ(a,idxs[0]);
        EXPECT_FLOAT_EQ(b,idxs[1]);
        EXPECT_FLOAT_EQ(c,idxs[2]);
        if (a != 1 || b != 2 || c != 3)
          stan::mcmc::chains::increment_indexes(dims,idxs);
      }
    }
  }

  stan::mcmc::chains::increment_indexes(dims,idxs);
  EXPECT_FLOAT_EQ(0.0,idxs[0]);
  EXPECT_FLOAT_EQ(0.0,idxs[1]);
  EXPECT_FLOAT_EQ(0.0,idxs[2]);
  
  vector<size_t> dims4(4,5);
  vector<size_t> idxs4(4,0); 
  EXPECT_THROW(stan::mcmc::chains::increment_indexes(dims4,idxs),
               std::invalid_argument);
  EXPECT_THROW(stan::mcmc::chains::increment_indexes(dims,idxs4),
               std::invalid_argument);
  EXPECT_NO_THROW(stan::mcmc::chains::increment_indexes(dims4,idxs4));

  idxs4[3] = 12; // now out of range
  EXPECT_THROW(stan::mcmc::chains::increment_indexes(dims4,idxs4),
               std::out_of_range);
}

TEST(McmcChains,add_samples) {
  using std::vector;
  using std::string;
  using stan::mcmc::chains;

  size_t K = 4; // num chains

  vector<string> names;
  names.push_back("b");
  names.push_back("a");
  names.push_back("c");

  vector<size_t> b_dims;

  vector<size_t> a_dims;
  a_dims.push_back(2);
  a_dims.push_back(3);

  vector<size_t> c_dims;
  c_dims.push_back(4);

  vector<vector<size_t> > dimss;
  dimss.push_back(b_dims);
  dimss.push_back(a_dims);
  dimss.push_back(c_dims);
  chains c(K,names,dimss);

  size_t N = 1 + 2*3 + 4;

  vector<double> theta(N);
  for (size_t n = 0; n < N; ++n)
    theta[n] = n;
  

  EXPECT_EQ(0U,c.num_samples());
  EXPECT_EQ(0U,c.num_samples(0));
  EXPECT_EQ(0U,c.num_samples(1));

  c.add_sample(0,theta);
  
  EXPECT_EQ(1U,c.num_samples());
  EXPECT_EQ(1U,c.num_samples(0));
  EXPECT_EQ(0U,c.num_samples(1));

  for (size_t n = 0; n < N; ++n)
    theta[n] *= 2.0;

  c.add_sample(0,theta);

  EXPECT_EQ(2U,c.num_samples());
  EXPECT_EQ(2U,c.num_samples(0));
  EXPECT_EQ(0U,c.num_samples(1));

  c.add_sample(1,theta);

  EXPECT_EQ(3U,c.num_samples());
  EXPECT_EQ(2U,c.num_samples(0));
  EXPECT_EQ(1U,c.num_samples(1));

  vector<double> rho;
  
  c.get_samples(0,rho);
  EXPECT_EQ(3U, rho.size());
}

#include <boost/random/additive_combine.hpp>

void test_permutation(size_t N) {
  int seed = 187049587;
  boost::random::ecuyer1988 rng(seed);
  std::vector<size_t> pi;
  stan::mcmc::permutation(pi,N,rng);

  EXPECT_EQ(N,pi.size());
  for (size_t i = 0; i < N; ++i)
    EXPECT_TRUE(pi[i] < N);
  int match_count = 0;
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      if (pi[j] == i) ++match_count;
  EXPECT_EQ(N,match_count);

}

TEST(McmcChains,permutation) {
  test_permutation(0);
  test_permutation(1);
  test_permutation(2);
  test_permutation(3);
  test_permutation(15);
  test_permutation(1024);
  test_permutation(1023);
  test_permutation(1025);
}


void test_permute(size_t N) {
  int seed = 187049587;
  boost::random::ecuyer1988 rng(seed);
  std::vector<size_t> pi;
  stan::mcmc::permutation(pi,N,rng);

  std::vector<double> x(N);
  for (size_t n = 0; n < N; ++n)
    x[n] = 1.0 + n / 2.0;
  std::vector<double> x_pi;
  
  stan::mcmc::permute(pi,x,x_pi);
  EXPECT_EQ(N,x_pi.size());
  if (N < 1) return;

  double sum = 0.0;
  double sum_pi = 0.0;
  for (size_t n = 0; n < N; ++n) {
    sum += x[n];
    sum_pi += x_pi[n];
  }
  EXPECT_GT(sum,0.0);
  EXPECT_EQ(sum,sum_pi);
}

TEST(McmcChains, permute) {
  test_permute(0);
  test_permute(1);  
  test_permute(2);  
  test_permute(3);  
  test_permute(4);  
  test_permute(5);  
  test_permute(2055);
  test_permute(2056);
  test_permute(2057);
}

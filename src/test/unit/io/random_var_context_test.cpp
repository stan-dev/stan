#include <stan/io/random_var_context.hpp>
#include <stan/io/empty_var_context.hpp>
#include <gtest/gtest.h>
#include <boost/random/additive_combine.hpp>  // L'Ecuyer RNG
#include <boost/random/uniform_real_distribution.hpp>
#include <test/test-models/good/services/test_lp.hpp>

class random_var_context : public testing::Test {
public:
  random_var_context()
    : empty_context(),
      model(empty_context, 0),
      rng(0) { }

  stan::io::empty_var_context empty_context;
  stan_model model;
  boost::ecuyer1988 rng;
};

TEST_F(random_var_context, contains_r) {
  stan::io::random_var_context context(model, rng, 2, false);
  EXPECT_FALSE(context.contains_r(""));
  EXPECT_TRUE(context.contains_r("y"));
}

TEST_F(random_var_context, vals_r) {
  stan::io::random_var_context context(model, rng, 2, false);
  std::vector<double> vals_r;
  EXPECT_NO_THROW(vals_r = context.vals_r(""));
  EXPECT_EQ(0, vals_r.size());

  EXPECT_NO_THROW(vals_r = context.vals_r("y"));
  ASSERT_EQ(2, vals_r.size());
  EXPECT_GT(vals_r[0], -10);
  EXPECT_LT(vals_r[0], 10);
  EXPECT_GT(vals_r[1], -100);
  EXPECT_LT(vals_r[1], 10);  
}

TEST_F(random_var_context, dims_r) {
  stan::io::random_var_context context(model, rng, 2, false);
  std::vector<size_t> dims_r;
  EXPECT_NO_THROW(dims_r = context.dims_r(""));
  EXPECT_EQ(0, dims_r.size());


  EXPECT_NO_THROW(dims_r = context.dims_r("y"));
  ASSERT_EQ(1, dims_r.size());
  EXPECT_EQ(2, dims_r[0]);
}

TEST_F(random_var_context, contains_i) {
  stan::io::random_var_context context(model, rng, 2, false);
  EXPECT_FALSE(context.contains_i(""));
}

TEST_F(random_var_context, vals_i) {
  stan::io::random_var_context context(model, rng, 2, false);
  std::vector<int> vals_i;
  EXPECT_NO_THROW(vals_i = context.vals_i(""));
  EXPECT_EQ(0, vals_i.size());
}

TEST_F(random_var_context, dims_i) {
  stan::io::random_var_context context(model, rng, 2, false);
  std::vector<size_t> dims_i;
  EXPECT_NO_THROW(dims_i = context.dims_i(""));
  EXPECT_EQ(0, dims_i.size());
}

TEST_F(random_var_context, names_r) {
  stan::io::random_var_context context(model, rng, 2, false);
  std::vector<std::string> names_r;
  EXPECT_NO_THROW(context.names_r(names_r));
  EXPECT_EQ(1, names_r.size());
}

TEST_F(random_var_context, names_i) {
  stan::io::random_var_context context(model, rng, 2, false);
  std::vector<std::string> names_i;
  EXPECT_NO_THROW(context.names_i(names_i));
  EXPECT_EQ(0, names_i.size());
}

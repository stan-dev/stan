#include <stan/io/random_init_var_context.hpp>
#include <test/test-models/good/random_init_var_context.cpp>
#include <gtest/gtest.h>

int N_repeat = 100;

stan_model create_model(std::stringstream& output) {
  // Model
  std::fstream data_stream("", std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();
  
  return stan_model(data_var_context, &output);
}

class mock_rng {
public:
  typedef double result_type;

  mock_rng() :
    calls(0) { }
    
  void reset() {
    calls = 0;
  }

  result_type operator()() {
    calls++;
    return calls / 10000.0;
  }

  static result_type max() {
    return 1.0;
  }

  static result_type min() {
    return -1.0;
  }

  int calls;
};

TEST(IoRandomInitVarContext, constructor) {
  std::stringstream output;
  stan_model model = create_model(output);
  mock_rng rng;
  
  stan::io::random_init_var_context<stan_model, mock_rng> context(model, rng);
  
  EXPECT_EQ(0, rng.calls);
}

TEST(IoRandomInitVarContext, contains_r) {
  std::stringstream output;
  stan_model model = create_model(output);
  mock_rng rng;
  
  stan::io::random_init_var_context<stan_model, mock_rng> context(model, rng);
  
  EXPECT_TRUE(context.contains_r("mu1"));
  EXPECT_TRUE(context.contains_r("mu2"));
  EXPECT_TRUE(context.contains_r("theta"));
  EXPECT_FALSE(context.contains_r("y"));
  EXPECT_FALSE(context.contains_r("foo"));
  EXPECT_FALSE(context.contains_r("bar"));
  EXPECT_FALSE(context.contains_r(""));

  EXPECT_EQ(0, rng.calls);
}

TEST(IoRandomInitVarContext, vals_r) {
  std::stringstream output;
  stan_model model = create_model(output);
  mock_rng rng;

  stan::io::random_init_var_context<stan_model, mock_rng> context(model, rng);

  std::vector<double> vals;

  vals = context.vals_r("mu1");
  ASSERT_EQ(1, vals.size());
  double last_mu1 = vals[0];
  for (int n = 0; n < N_repeat; n++) {
    vals = context.vals_r("mu1");
    ASSERT_EQ(1, vals.size());
    EXPECT_NE(last_mu1, vals[0]);
    last_mu1 = vals[0];
  }
  EXPECT_EQ(N_repeat + 1, rng.calls);
  rng.reset();

  vals = context.vals_r("mu2");
  ASSERT_EQ(1, vals.size());
  double last_mu2 = vals[0];
  for (int n = 0; n < N_repeat; n++) {
    vals = context.vals_r("mu2");
    ASSERT_EQ(1, vals.size());
    EXPECT_NE(last_mu2, vals[0]);
    last_mu2 = vals[0];
  }
  EXPECT_EQ(N_repeat + 1, rng.calls);
  rng.reset();

  vals = context.vals_r("theta");
  ASSERT_EQ(6, vals.size());
  double last_theta = vals[0];
  for (int n = 0; n < N_repeat; n++) {
    vals = context.vals_r("theta");
    ASSERT_EQ(6, vals.size());
    EXPECT_NE(last_theta, vals[0]);
    last_theta = vals[0];
  }
  EXPECT_EQ((N_repeat + 1) * 6, rng.calls);
  rng.reset();

  
  EXPECT_NO_THROW(vals = context.vals_r("foo"));
  EXPECT_EQ(0, vals.size());
}

TEST(IoRandomInitVarContext, dims_r) {
  std::stringstream output;
  stan_model model = create_model(output);
  mock_rng rng;

  stan::io::random_init_var_context<stan_model, mock_rng> context(model, rng);
  
  std::vector<size_t> dims_r;
  
  dims_r = context.dims_r("mu1");
  EXPECT_EQ(0U, dims_r.size());

  dims_r = context.dims_r("mu2");
  EXPECT_EQ(0U, dims_r.size());

  dims_r = context.dims_r("theta");
  ASSERT_EQ(2U, dims_r.size());
  EXPECT_EQ(2U, dims_r[0]);
  EXPECT_EQ(3U, dims_r[1]);

  dims_r = context.dims_r("foo");
  EXPECT_EQ(0U, dims_r.size());
}

TEST(IoRandomInitVarContext, contains_i) {
  std::stringstream output;
  stan_model model = create_model(output);
  mock_rng rng;

  stan::io::random_init_var_context<stan_model, mock_rng> context(model, rng);

  EXPECT_FALSE(context.contains_i(""));
  EXPECT_FALSE(context.contains_i("mu1"));
  EXPECT_FALSE(context.contains_i("theta"));
  EXPECT_FALSE(context.contains_i("foo"));
}

TEST(IoRandomInitVarContext, vals_i) {
  std::stringstream output;
  stan_model model = create_model(output);
  mock_rng rng;

  stan::io::random_init_var_context<stan_model, mock_rng> context(model, rng);
  
  std::vector<int> vals;
  vals = context.vals_i("foo");
  ASSERT_EQ(0, vals.size());

  vals = context.vals_i("");
  ASSERT_EQ(0, vals.size());

  vals = context.vals_i("mu1");
  ASSERT_EQ(0, vals.size());
}

TEST(IoRandomInitVarContext, dims_i) {
  std::stringstream output;
  stan_model model = create_model(output);
  mock_rng rng;

  stan::io::random_init_var_context<stan_model, mock_rng> context(model, rng);
  
  std::vector<size_t> dims;
  
  dims = context.dims_i("");
  ASSERT_EQ(0, dims.size());

  dims = context.dims_i("foo");
  ASSERT_EQ(0, dims.size());

  dims = context.dims_i("mu1");
  ASSERT_EQ(0, dims.size());

  dims = context.dims_i("theta");
  ASSERT_EQ(0, dims.size());
}

TEST(IoRandomInitVarContext, names_r) {
  std::stringstream output;
  stan_model model = create_model(output);
  mock_rng rng;

  stan::io::random_init_var_context<stan_model, mock_rng> context(model, rng);
  
  std::vector<std::string> names;
  context.names_r(names);
  
  ASSERT_EQ(3, names.size());
  EXPECT_EQ("mu1", names[0]);
  EXPECT_EQ("mu2", names[1]);
  EXPECT_EQ("theta", names[2]);
}

TEST(IoRandomInitVarContext, names_i) {
  std::stringstream output;
  stan_model model = create_model(output);
  mock_rng rng;

  stan::io::random_init_var_context<stan_model, mock_rng> context(model, rng);
  
  std::vector<std::string> names;
  context.names_i(names);
  
  ASSERT_EQ(0, names.size());
}

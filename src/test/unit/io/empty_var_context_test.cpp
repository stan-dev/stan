#include <stan/io/empty_var_context.hpp>
#include <gtest/gtest.h>

TEST(empty_var_context, contains_r) {
  stan::io::empty_var_context context;
  EXPECT_FALSE(context.contains_r(""));
}

TEST(empty_var_context, vals_r) {
  stan::io::empty_var_context context;
  std::vector<double> vals_r;
  EXPECT_NO_THROW(vals_r = context.vals_r(""));
  EXPECT_EQ(0, vals_r.size());
}

TEST(empty_var_context, dims_r) {
  stan::io::empty_var_context context;
  std::vector<size_t> dims_r;
  EXPECT_NO_THROW(dims_r = context.dims_r(""));
  EXPECT_EQ(0, dims_r.size());
}

TEST(empty_var_context, contains_i) {
  stan::io::empty_var_context context;
  EXPECT_FALSE(context.contains_i(""));
}

TEST(empty_var_context, vals_i) {
  stan::io::empty_var_context context;
  std::vector<int> vals_i;
  EXPECT_NO_THROW(vals_i = context.vals_i(""));
  EXPECT_EQ(0, vals_i.size());
}

TEST(empty_var_context, dims_i) {
  stan::io::empty_var_context context;
  std::vector<size_t> dims_i;
  EXPECT_NO_THROW(dims_i = context.dims_i(""));
  EXPECT_EQ(0, dims_i.size());
}

TEST(empty_var_context, names_r) {
  stan::io::empty_var_context context;
  std::vector<std::string> names_r;
  EXPECT_NO_THROW(context.names_r(names_r));
  EXPECT_EQ(0, names_r.size());
}

TEST(empty_var_context, names_i) {
  stan::io::empty_var_context context;
  std::vector<std::string> names_i;
  EXPECT_NO_THROW(context.names_i(names_i));
  EXPECT_EQ(0, names_i.size());
}

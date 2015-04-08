#include <stan/io/chained_var_context.hpp>
#include <stan/io/array_var_context.hpp>
#include <gtest/gtest.h>

TEST(chained_var_context, ctor) {
  std::vector<double> v;
  for (size_t i = 0; i < 16; i++) v.push_back(1.0 * i);
  std::vector<std::vector<size_t> > dims;
  std::vector<size_t> scalar_dim;
  std::vector<size_t> vec_dim;
  vec_dim.push_back(3);
  std::vector<size_t> array_dim;
  array_dim.push_back(3);
  array_dim.push_back(4);
  dims.push_back(scalar_dim);
  dims.push_back(vec_dim);
  dims.push_back(array_dim);
  std::vector<std::string> names;
  names.push_back("alpha");
  names.push_back("beta");
  names.push_back("gamma");
  stan::io::array_var_context avc(names, v, dims);

  std::vector<double> v2;
  for (size_t i = 1; i < 21; i++) v2.push_back(10 * i);
  std::vector<std::vector<size_t> > dims2;
  std::vector<size_t> vec_dim2;
  vec_dim2.push_back(4);
  std::vector<size_t> array_dim2;
  array_dim2.push_back(2);
  array_dim2.push_back(7);
  dims2.push_back(scalar_dim); // 1
  dims2.push_back(vec_dim2);   // 4 
  dims2.push_back(array_dim2); // 14 
  dims2.push_back(scalar_dim); // 1
  std::vector<std::string> names2;
  names2.push_back("alpha");
  names2.push_back("b");
  names2.push_back("c");
  names2.push_back("d");
  stan::io::array_var_context avc2(names2, v2, dims2);
  stan::io::chained_var_context vcc(avc, avc2);
  EXPECT_TRUE(avc2.contains_r("d"));
  EXPECT_TRUE(vcc.contains_r("alpha"));
  EXPECT_TRUE(vcc.contains_r("beta"));
  EXPECT_TRUE(vcc.contains_r("gamma"));
  EXPECT_TRUE(vcc.contains_r("b"));
  EXPECT_TRUE(vcc.contains_r("c"));
  EXPECT_TRUE(vcc.contains_r("d"));

  std::vector<double> alpha(1, 0);
  EXPECT_EQ(alpha, vcc.vals_r("alpha"));
}

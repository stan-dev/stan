
#include <limits>
#include <stan/io/array_var_context.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>

TEST(array_var_context, ctor) {
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
  EXPECT_TRUE(avc.contains_r("alpha"));
}

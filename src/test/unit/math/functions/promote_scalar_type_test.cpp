#include <gtest/gtest.h>
#include <test/unit/math/functions/util.hpp>

// E is expected value of promote_scalar_type<T,S>::type
template <typename E, typename T, typename S>
void expect_promote_type() {
  using stan::math::promote_scalar_type;
  return expect_same_type<E,typename promote_scalar_type<T,S>::type>();
}

TEST(MathFunctions,PromoteScalarType) {
  using std::vector;
  expect_promote_type<double,
                      double, double>();
  expect_promote_type<double,
                      double, int>();
  expect_promote_type<vector<double>,
                      double, vector<int> >();
  expect_promote_type<vector<vector<double> >, 
                    double, vector<vector<int> > >();
  expect_promote_type<vector<vector<double> >, 
                    double, vector<vector<double> > >();
  
}

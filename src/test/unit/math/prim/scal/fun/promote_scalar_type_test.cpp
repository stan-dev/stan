#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/promote_scalar_type.hpp>
#include <test/unit/math/prim/scal/fun/promote_type_test_util.hpp>

TEST(MathFunctionsPromoteScalarType,primitive) {
  using std::vector;
  expect_promote_type<double,
                      double, double>();
  expect_promote_type<double,
                      double, int>();
  expect_promote_type<vector<double>,
                      double, vector<int> >();
}

TEST(MathFunctionsPromoteScalarType,StdVector) {
  using std::vector;
  expect_promote_type<vector<vector<double> >, 
                    double, vector<vector<int> > >();
  expect_promote_type<vector<vector<double> >, 
                    double, vector<vector<double> > >();
}


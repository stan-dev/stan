#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/promote_scalar_type.hpp>
#include <test/unit/math/prim/scal/fun/promote_type_test_util.hpp>

TEST(MathFunctionsPromoteScalar,TypeMatrix) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  expect_promote_type<Matrix<double,Dynamic,Dynamic>,
                      double, Matrix<int,Dynamic,Dynamic> >();
  
  expect_promote_type<Matrix<double,Dynamic,Dynamic>,
                      double, Matrix<double,Dynamic,Dynamic> >();

  expect_promote_type<vector<Matrix<double,Dynamic,Dynamic> >,
                      double,  vector<Matrix<int,Dynamic,Dynamic> > >();
}

TEST(MathFunctionsPromoteScalar,TypeVector) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  expect_promote_type<Matrix<double,Dynamic,1>,
                      double, Matrix<int,Dynamic,1> >();
  
  expect_promote_type<Matrix<double,Dynamic,1>,
                      double, Matrix<double,Dynamic,1> >();

  expect_promote_type<vector<Matrix<double,Dynamic,1> >,
                      double,  vector<Matrix<int,Dynamic,1> > >();
}

TEST(MathFunctionsPromoteScalar,TypeRowVector) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  expect_promote_type<Matrix<double,1,Dynamic>,
                      double, Matrix<int,1,Dynamic> >();
  
  expect_promote_type<Matrix<double,1,Dynamic>,
                      double, Matrix<double,1,Dynamic> >();

  expect_promote_type<vector<Matrix<double,1,Dynamic> >,
                      double,  vector<Matrix<int,1,Dynamic> > >();
}

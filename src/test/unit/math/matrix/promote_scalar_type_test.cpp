#include <gtest/gtest.h>
#include <stan/math/matrix/promote_scalar_type.hpp>
#include <test/unit/math/matrix/promote_type_test_util.hpp>

TEST(MathFunctions,PromoteScalarType) {
  using std::vector;
  expect_promote_type<double,
                      double, double>();
  expect_promote_type<double,
                      double, int>();
  expect_promote_type<vector<double>,
                      double, vector<int> >();
}

TEST(MathFunctions,PromoteScalarTypeStdVector) {
  using std::vector;
  expect_promote_type<vector<vector<double> >, 
                    double, vector<vector<int> > >();
  expect_promote_type<vector<vector<double> >, 
                    double, vector<vector<double> > >();
}

TEST(MathFunctions,PromoteScalarTypeMatrix) {
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

TEST(MathFunctions,PromoteScalarTypeVector) {
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

TEST(MathFunctions,PromoteScalarTypeRowVector) {
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

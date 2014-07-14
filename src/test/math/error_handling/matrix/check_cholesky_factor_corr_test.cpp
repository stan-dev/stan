#include <cmath>
#include <stan/math/error_handling/matrix/check_cholesky_factor_corr.hpp>
#include <gtest/gtest.h>

TEST(MathErrorHandlingMatrix, checkCorrCholeskyMatrix) {
  using stan::math::check_cholesky_factor_corr;
  using Eigen::MatrixXd;
  using std::vector;
  using std::sqrt;

  vector<MatrixXd> Lg;

  Lg.push_back(MatrixXd(2,2));
  Lg[0] << 
    1, 0, 
    0, 1;

  Lg.push_back(MatrixXd(2,2));
  Lg[1] << 
    1, 0, 
    sqrt(0.5), sqrt(0.5);

  for (size_t i = 0; i < Lg.size(); ++i) {
    double init_result = -1.3 * i;
    double result = init_result; 
    bool pass = check_cholesky_factor_corr("foo<%1%>",
                                           Lg[i],
                                           &result);
    EXPECT_TRUE(pass);
    EXPECT_FLOAT_EQ(init_result, result);
  }

  vector<MatrixXd> Lb;
  Lb.push_back(MatrixXd(2,3));  // not square
  Lb[0] << 
    1, 0, 0,
    0, 1, 0;

  Lb.push_back(MatrixXd(2,2));  // first row not unit length
  Lb[1] <<
    1.01, 0,
    0, 1; 

  Lb.push_back(MatrixXd(2,2));  // second row not unit length
  Lb[2] <<
    1, 0,
    1, 1;

  Lb.push_back(MatrixXd(2,2)); // not lower-triangular
  Lb[3] <<
    sqrt(0.5), sqrt(0.5),
    sqrt(0.5), sqrt(0.5);

  Lb.push_back(MatrixXd(2,2)); // not positive diag
  Lb[4] <<
    -1, 0,
     0, 1;

  Lb.push_back(MatrixXd(3,2));  // not square
  Lb[5] << 
    1, 0,
    0, 1,
    0, 0;

  


  for (size_t i = 0; i < Lb.size(); ++i) {
    double init_result = 1232 * i;
    double result = init_result;
    EXPECT_THROW(check_cholesky_factor_corr("foo<%1%>",
                                            Lb[i],
                                            &result),
                 std::domain_error);
  }

    
}



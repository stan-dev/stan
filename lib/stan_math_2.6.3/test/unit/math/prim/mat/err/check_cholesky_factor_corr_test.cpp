#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <cmath>
#include <stan/math/prim/mat/err/check_cholesky_factor_corr.hpp>
#include <gtest/gtest.h>

TEST(ErrorHandlingMatrix, checkCorrCholeskyMatrix) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;

  using stan::math::check_cholesky_factor_corr;
  using std::sqrt;

  y.resize(1,1);
  y << 1;
  EXPECT_TRUE(check_cholesky_factor_corr("checkCorrCholeskyMatrix",
                                         "y", y));
  
  y.resize(3,3);
  y << 
    1, 0, 0,
    sqrt(0.5), sqrt(0.5), 0,
    sqrt(0.25), sqrt(0.25), sqrt(0.5);
  EXPECT_TRUE(check_cholesky_factor_corr("checkCorrCholeskyMatrix",
                                         "y", y));

  // not positive
  y.resize(1,1);
  y << -1;
  EXPECT_THROW(check_cholesky_factor_corr("checkCorrCholeskyMatrix", 
                                          "y", y), 
               std::domain_error);

  // not lower triangular
  y.resize(3,3);
  y << 
    1, 2, 3, 
    0, 5, 6, 
    0, 0, 9;
  EXPECT_THROW(check_cholesky_factor_corr("checkCorrCholeskyMatrix", 
                                          "y", y), 
               std::domain_error);

  // not positive
  y.resize(3,3);
  y <<
    1, 0, 0, 
    2, -1, 0,
    1, 2, 3;
  EXPECT_THROW(check_cholesky_factor_corr("checkCorrCholeskyMatrix", 
                                          "y", y), 
               std::domain_error);

  // not rectangular
  y.resize(2,3);
  y << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(check_cholesky_factor_corr("checkCorrCholeskyMatrix", 
                                          "y", y),
               std::invalid_argument);
  y.resize(3,2);
  y << 
    1, 0,
    2, 3,
    4, 5;
  EXPECT_THROW(check_cholesky_factor_corr("checkCorrCholeskyMatrix",
                                          "y", y),
               std::invalid_argument);

  // not unit vectors
  y.resize(3,3);
  y << 
    1, 0, 0,
    1, 1, 0,
    1, 1, 1;
  EXPECT_THROW(check_cholesky_factor_corr("checkCorrCholeskyMatrix", 
                                          "y", y),
               std::domain_error);
}



TEST(ErrorHandlingMatrix, checkCorrCholeskyMatrix_nan) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  double nan = std::numeric_limits<double>::quiet_NaN();

  using stan::math::check_cholesky_factor_corr;
  using std::sqrt;

  y.resize(1,1);
  y << nan;
  EXPECT_THROW(check_cholesky_factor_corr("checkCorrCholeskyMatrix",
                                          "y", y),
               std::domain_error);
  
  y.resize(3,3);
  y << 
    1, 0, 0,
    sqrt(0.5), sqrt(0.5), 0,
    sqrt(0.25), sqrt(0.25), sqrt(0.5);
  EXPECT_TRUE(check_cholesky_factor_corr("checkCorrCholeskyMatrix",
                                         "y", y));

  for (int i = 0 ; i < y.size(); i++) {
    y(i) = nan;
    EXPECT_THROW(check_cholesky_factor_corr("checkCorrCholeskyMatrix",
                                            "y", y),
                 std::domain_error);
    y << 
      1, 0, 0,
      sqrt(0.5), sqrt(0.5), 0,
      sqrt(0.25), sqrt(0.25), sqrt(0.5);
  }

}



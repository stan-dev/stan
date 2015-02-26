#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/prim/mat/err/check_lower_triangular.hpp>
#include <gtest/gtest.h>

TEST(ErrorHandlingMatrix, checkLowerTriangular) {
  using stan::math::check_lower_triangular;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  
  y.resize(1,1);
  y << 1;
  EXPECT_TRUE(check_lower_triangular("checkLowerTriangular", "y", y));

  y.resize(1,2);
  y << 1, 0;
  EXPECT_TRUE(check_lower_triangular("checkLowerTriangular", "y", y));

  y(0,1) = 1;
  EXPECT_THROW(check_lower_triangular("checkLowerTriangular", "y", y), 
               std::domain_error);
  
  

  y.resize(2,2);
  y << 1, 0, 2, 3;
  EXPECT_TRUE(check_lower_triangular("checkLowerTriangular", "y", y));

  y << 1, 2, 3, 4;
  EXPECT_THROW(check_lower_triangular("checkLowerTriangular", "y", y), 
               std::domain_error);

  y.resize(3,2);
  y << 1, 0,
    2, 3,
    4, 5;
  EXPECT_TRUE(check_lower_triangular("checkLowerTriangular", "y", y));
  
  y(0,1) = 1.5;
  EXPECT_THROW(check_lower_triangular("checkLowerTriangular", "y", y), 
               std::domain_error);
  
  y.resize(2,3);
  y << 
    1, 0, 0,
    4, 5, 0;
  EXPECT_TRUE(check_lower_triangular("checkLowerTriangular", "y", y));
  y(0,2) = 3;
}


TEST(ErrorHandlingMatrix, checkLowerTriangular_one_indexed_message) {
  using stan::math::check_lower_triangular;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  std::string message;

  y.resize(2,3);
  y << 
    1, 0, 3,
    4, 5, 0;
  try {
    check_lower_triangular("checkLowerTriangular", "y", y);
    FAIL() << "should have thrown";
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    FAIL() << "threw the wrong error";
  }

  EXPECT_NE(std::string::npos, message.find("[1,3]"))
    << message;
}

TEST(ErrorHandlingMatrix, checkLowerTriangular_nan) {
  using stan::math::check_lower_triangular;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
    double nan = std::numeric_limits<double>::quiet_NaN();

  y.resize(1,1);
  y << nan;
  EXPECT_TRUE(check_lower_triangular("checkLowerTriangular", "y", y));

  y.resize(1,2);
  y << nan, 0;
  EXPECT_TRUE(check_lower_triangular("checkLowerTriangular", "y", y));

  y(0,1) = nan;
  EXPECT_THROW(check_lower_triangular("checkLowerTriangular", "y", y), 
               std::domain_error);
  
  

  y.resize(2,2);
  y << nan, 0, nan, nan;
  EXPECT_TRUE(check_lower_triangular("checkLowerTriangular", "y", y));

  y << 1, nan, nan, 4;
              EXPECT_THROW(check_lower_triangular("checkLowerTriangular", "y", y), 
               std::domain_error);

  y.resize(3,2);
  y << nan, 0,
    2, nan,
    4, 5;
  EXPECT_TRUE(check_lower_triangular("checkLowerTriangular", "y", y));
  
  y(0,1) = nan;
  EXPECT_THROW(check_lower_triangular("checkLowerTriangular", "y", y), 
               std::domain_error);
  
  y.resize(2,3);
  y << 
    nan, 0, 0,
    4, nan, 0;
  EXPECT_TRUE(check_lower_triangular("checkLowerTriangular", "y", y));

  y(0,2) = nan;
  EXPECT_THROW(check_lower_triangular("checkLowerTriangular", "y", y), 
               std::domain_error);
}

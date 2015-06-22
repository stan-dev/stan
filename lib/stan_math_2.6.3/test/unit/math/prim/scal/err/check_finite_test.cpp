#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/arr/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/meta/value_type.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/length.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <gtest/gtest.h>

using stan::math::check_finite;

TEST(ErrorHandlingScalar,CheckFinite) {
  const char* function = "check_finite";
  double x = 0;
 
  EXPECT_TRUE(check_finite(function, "x", x))
    << "check_finite should be true with finite x: " << x;
  x = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_finite(function, "x", x), std::domain_error)
    << "check_finite should throw exception on Inf: " << x;
  x = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_finite(function, "x", x), std::domain_error) 
    << "check_finite should throw exception on -Inf: " << x;

  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(check_finite(function, "x", x), std::domain_error)
    << "check_finite should throw exception on NaN: " << x;
}

// ---------- check_finite: vector tests ----------
TEST(ErrorHandlingScalar,CheckFinite_Vector) {
  const char* function = "check_finite";
  std::vector<double> x;
  
  x.clear();
  x.push_back (-1);
  x.push_back (0);
  x.push_back (1);
  ASSERT_TRUE(check_finite(function, "x", x)) 
    << "check_finite should be true with finite x";

  x.clear();
  x.push_back(-1);
  x.push_back(0);
  x.push_back(std::numeric_limits<double>::infinity());
  EXPECT_THROW(check_finite(function, "x", x), std::domain_error) 
    << "check_finite should throw exception on Inf";

  x.clear();
  x.push_back(-1);
  x.push_back(0);
  x.push_back(-std::numeric_limits<double>::infinity());
  EXPECT_THROW(check_finite(function, "x", x), std::domain_error)
    << "check_finite should throw exception on -Inf";
  
  x.clear();
  x.push_back(-1);
  x.push_back(0);
  x.push_back(std::numeric_limits<double>::quiet_NaN());
  EXPECT_THROW(check_finite(function, "x", x), std::domain_error)
    << "check_finite should throw exception on NaN";
}

// ---------- check_finite: matrix tests ----------
TEST(ErrorHandlingScalar,CheckFinite_Matrix) {
  const char* function = "check_finite";
  Eigen::Matrix<double,Eigen::Dynamic,1> x;
  
  x.resize(3);
  x << -1, 0, 1;
  ASSERT_TRUE(check_finite(function, "x", x))
    << "check_finite should be true with finite x";

  x.resize(3);
  x << -1, 0, std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_finite(function, "x", x), std::domain_error)
    << "check_finite should throw exception on Inf";

  x.resize(3);
  x << -1, 0, -std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_finite(function, "x", x), std::domain_error)
    << "check_finite should throw exception on -Inf";
  
  x.resize(3);
  x << -1, 0, std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(check_finite(function, "x", x), std::domain_error) 
    << "check_finite should throw exception on NaN";
}


TEST(ErrorHandlingScalar,CheckFinite_Matrix_one_indexed_message) {
  const char* function = "check_finite";
  Eigen::Matrix<double,Eigen::Dynamic,1> x;
  std::string message;

  x.resize(3);
  x << -1, 0, std::numeric_limits<double>::infinity();
  try {
    check_finite(function, "x", x);
    FAIL() << "should have thrown";
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    FAIL() << "threw the wrong error";
  }

  EXPECT_NE(std::string::npos, message.find("[3]"))
    << message;
}

TEST(ErrorHandlingScalar,CheckFinite_nan) {
  const char* function = "check_finite";
  double nan = std::numeric_limits<double>::quiet_NaN();

  EXPECT_THROW(check_finite(function, "x", nan), std::domain_error);

  std::vector<double> x;
  x.push_back (nan);
  x.push_back (0);
  x.push_back (1);
  EXPECT_THROW(check_finite(function, "x", x), std::domain_error);

  x[0] = 1.0;
  x[1] = nan;
  EXPECT_THROW(check_finite(function, "x", x), std::domain_error);

  x[1] = 0.0;
  x[2] = nan;
  EXPECT_THROW(check_finite(function, "x", x), std::domain_error);

  Eigen::Matrix<double,Eigen::Dynamic,1> x_mat(3);
  x_mat << nan, 0, 1;
  EXPECT_THROW(check_finite(function, "x_mat", x_mat),
               std::domain_error);

  x_mat << 1, nan, 1;
  EXPECT_THROW(check_finite(function, "x_mat", x_mat),
               std::domain_error);

  x_mat << 1, 0, nan;
  EXPECT_THROW(check_finite(function, "x_mat", x_mat),
               std::domain_error);
}

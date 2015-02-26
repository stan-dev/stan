#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/arr/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/meta/value_type.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/length.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <gtest/gtest.h>

using stan::math::check_positive_finite;

TEST(ErrorHandlingScalar,CheckPositiveFinite) {
  const char* function = "check_positive_finite";
  double x = 1;
 
  EXPECT_TRUE(check_positive_finite(function, "x", x))
    << "check_positive_finite should be true with finite x: " << x;
  x = -1;
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error)
    << "check_positive_finite should throw exception on x= " << x;
  x = 0;
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error)
    << "check_positive_finite should throw exception on x= " << x;
  x = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error)
    << "check_positive_finite should throw exception on Inf: " << x;
  x = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error) 
    << "check_positive_finite should throw exception on -Inf: " << x;

  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error)
    << "check_positive_finite should throw exception on NaN: " << x;
}

// ---------- check_positive_finite: vector tests ----------
TEST(ErrorHandlingScalar,CheckPositiveFinite_Vector) {
  const char* function = "check_positive_finite";
  std::vector<double> x;
  
  x.clear();
  x.push_back (1.5);
  x.push_back (0.1);
  x.push_back (1);
  ASSERT_TRUE(check_positive_finite(function, "x", x)) 
    << "check_positive_finite should be true with finite x";

  x.clear();
  x.push_back(1);
  x.push_back(2);
  x.push_back(std::numeric_limits<double>::infinity());
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error) 
    << "check_positive_finite should throw exception on Inf";

  x.clear();
  x.push_back(-1);
  x.push_back(2);
  x.push_back(std::numeric_limits<double>::infinity());
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error) 
    << "check_positive_finite should throw exception on negative x";

  x.clear();
  x.push_back(0);
  x.push_back(2);
  x.push_back(std::numeric_limits<double>::infinity());
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error) 
    << "check_positive_finite should throw exception on x=0";

  x.clear();
  x.push_back(1);
  x.push_back(2);
  x.push_back(-std::numeric_limits<double>::infinity());
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error)
    << "check_positive_finite should throw exception on -Inf";
  
  x.clear();
  x.push_back(1);
  x.push_back(2);
  x.push_back(std::numeric_limits<double>::quiet_NaN());
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error)
    << "check_positive_finite should throw exception on NaN";
}

// ---------- check_positive_finite: matrix tests ----------
TEST(ErrorHandlingScalar,CheckPositiveFinite_Matrix) {
  const char* function = "check_positive_finite";
  Eigen::Matrix<double,Eigen::Dynamic,1> x;
  
  x.resize(3);
  x << 3, 2, 1;
  ASSERT_TRUE(check_positive_finite(function, "x", x))
    << "check_positive_finite should be true with finite x";

  x.resize(3);
  x << 2, 1, std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error)
    << "check_positive_finite should throw exception on Inf";

  x.resize(3);
  x << 0, 1, std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error)
    << "check_positive_finite should throw exception on x=0";

  x.resize(3);
  x << -1, 1, std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error)
    << "check_positive_finite should throw exception on x=-1";

  x.resize(3);
  x << 2, 1, -std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error)
    << "check_positive_finite should throw exception on -Inf";
  
  x.resize(3);
  x << 1, 2, std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(check_positive_finite(function, "x", x), std::domain_error) 
    << "check_positive_finite should throw exception on NaN";
}


TEST(ErrorHandlingScalar,CheckPositiveFinite_Matrix_one_indexed_message) {
  const char* function = "check_positive_finite";
  Eigen::Matrix<double,Eigen::Dynamic,1> x;
  std::string message;

  x.resize(3);
  x << 1, 2, std::numeric_limits<double>::infinity();
  try {
    check_positive_finite(function, "x", x);
    FAIL() << "should have thrown";
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    FAIL() << "threw the wrong error";
  }

  EXPECT_NE(std::string::npos, message.find("[3]"))
    << message;
}
TEST(ErrorHandlingScalar,CheckPositiveFinite_Matrix_one_indexed_message_2) {
  const char* function = "check_positive_finite";
  Eigen::Matrix<double,Eigen::Dynamic,1> x;
  std::string message;

  x.resize(3);
  x << -1, 2, std::numeric_limits<double>::infinity();
  try {
    check_positive_finite(function, "x", x);
    FAIL() << "should have thrown";
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    FAIL() << "threw the wrong error";
  }

  EXPECT_NE(std::string::npos, message.find("[1]"))
    << message;
}

TEST(ErrorHandlingScalar,CheckPositiveFinite_Matrix_one_indexed_message_3) {
  const char* function = "check_positive_finite";
  Eigen::Matrix<double,Eigen::Dynamic,1> x;
  std::string message;

  x.resize(3);
  x << 1, 0, std::numeric_limits<double>::infinity();
  try {
    check_positive_finite(function, "x", x);
    FAIL() << "should have thrown";
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    FAIL() << "threw the wrong error";
  }

  EXPECT_NE(std::string::npos, message.find("[2]"))
    << message;
}

TEST(ErrorHandlingScalar,CheckPositiveFinite_nan) {
  const char* function = "check_positive_finite";
  double nan = std::numeric_limits<double>::quiet_NaN();

  EXPECT_THROW(check_positive_finite(function, "x", nan),
               std::domain_error);

  std::vector<double> x;
  x.push_back(1.0);
  x.push_back(2.0);
  x.push_back(3.0);

  for (size_t i = 0; i < x.size(); i++) {
    x[i] = nan;
    EXPECT_THROW(check_positive_finite(function, "x", x),
                 std::domain_error);
    x[i] = i;
  }

  Eigen::Matrix<double,Eigen::Dynamic,1> x_mat(3);
  x_mat   << 1, 2, 3;
  for (int i = 0; i < x_mat.size(); i++) {
    x_mat(i) = nan;
    EXPECT_THROW(check_positive_finite(function, "x", x_mat), 
                 std::domain_error);
    x_mat(i) = i;
  }
}

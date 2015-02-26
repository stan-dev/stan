#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/prim/mat/err/check_ordered.hpp>
#include <gtest/gtest.h>

using stan::math::check_ordered;

TEST(ErrorHandlingMatrix, checkOrdered) {
  Eigen::Matrix<double, Eigen::Dynamic, 1> y;
  y.resize(3);

  y << 0, 1, 2;
  EXPECT_TRUE(check_ordered("check_ordered", "y", y));

  y << 0, 10, std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_ordered("check_ordered", "y", y));

  y << -10, 10, std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_ordered("check_ordered", "y", y));

  y << -std::numeric_limits<double>::infinity(), 10, std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_ordered("check_ordered", "y", y));

  y << 0, 0, 0;
  EXPECT_THROW(check_ordered("check_ordered", "y", y),
               std::domain_error);

  y << 0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_ordered("check_ordered", "y", y),
               std::domain_error);


  y << -1, 3, 2;
  EXPECT_THROW(check_ordered("check_ordered", "y", y),
               std::domain_error);

  std::vector<double> y_;
  y_.push_back(0.0);
  y_.push_back(1.0);
  y_.push_back(2.0);
  EXPECT_TRUE(check_ordered("check_ordered", "y", y_));

  y_[2] = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_ordered("check_ordered", "y", y_));

  y_[0] = -10.0;
  EXPECT_TRUE(check_ordered("check_ordered", "y", y_));

  y_[0] = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(check_ordered("check_ordered", "y", y_));

  y_[0] = 0.0;
  y_[1] = 0.0;
  y_[2] = 0.0;
  EXPECT_THROW(check_ordered("check_ordered", "y", y_),
               std::domain_error);

  y_[1] = std::numeric_limits<double>::infinity();
  y_[2] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(check_ordered("check_ordered", "y", y_),
               std::domain_error);
}

TEST(ErrorHandlingMatrix, checkOrdered_one_indexed_message) {
  std::string message;
  Eigen::Matrix<double, Eigen::Dynamic, 1> y;
  y.resize(3);
  
  y << 0, 5, 1;
  try {
    check_ordered("check_ordered", "y", y);
    FAIL() << "should have thrown";
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    FAIL() << "threw the wrong error";
  }

  EXPECT_NE(std::string::npos, message.find("element at 3"))
    << message;
}

TEST(ErrorHandlingMatrix, checkOrdered_nan) {
  Eigen::Matrix<double, Eigen::Dynamic, 1> y;
  std::vector<double> y_;
  double nan = std::numeric_limits<double>::quiet_NaN();
  y.resize(3);

  y << 0, 1, 2;
  y_.push_back(0.0);
  y_.push_back(1.0);
  y_.push_back(2.0);
  for (int i = 0; i < y.size(); i++) {
    y[i] = nan;
    y_[i] = nan;
    EXPECT_THROW(check_ordered("check_ordered", "y", y),
                 std::domain_error);
    EXPECT_THROW(check_ordered("check_ordered", "y", y_),
                 std::domain_error);
    y[i] = i;
    y_[i] = i;
  }
  for (int i = 0; i < y.size(); i++) {
    y << 0, 10, std::numeric_limits<double>::infinity();
    y_[0] = 0.0;
    y_[1] = 10.0;
    y_[2] = std::numeric_limits<double>::infinity();
    y[i] = nan;
    y_[i] = nan;
    EXPECT_THROW(check_ordered("check_ordered", "y", y),
                 std::domain_error);
    EXPECT_THROW(check_ordered("check_ordered", "y", y_),
                 std::domain_error);
  }
}

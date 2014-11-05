#include <stan/error_handling/matrix/check_simplex.hpp>
#include <gtest/gtest.h>

TEST(ErrorHandlingMatrix, checkSimplex) {
  Eigen::Matrix<double,Eigen::Dynamic,1> y(2);
  y.setZero();
  y << 0.5, 0.5;
  
  EXPECT_TRUE(stan::error_handling::check_simplex("checkSimplex",
                                                  "y", y));
                  
  y[1] = 0.55;
  EXPECT_THROW(stan::error_handling::check_simplex("checkSimplex", 
                                                   "y", y), 
               std::domain_error);
}

TEST(ErrorHandlingMatrix, checkSimplex_message_negative_value) {
  Eigen::Matrix<double,Eigen::Dynamic,1> y(100);
  y.setZero();
  std::string message;


  y[0] = -0.1;
  y[1] = 1.1;
  try {
    stan::error_handling::check_simplex("checkSimplex",
                                        "y", y);
    FAIL() << "should have thrown";
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    FAIL() << "threw the wrong error";
  }

  EXPECT_TRUE(std::string::npos != message.find(" y is not a valid simplex"))
    << message;

  EXPECT_TRUE(std::string::npos != message.find("y[1] = -0.1"))
    << message;



  y.setZero();
  y[0] = 0.1;
  y[1] = -0.1;
  y[2] = 1.0;
  try {
    stan::error_handling::check_simplex("checkSimplex",
                                        "y", y);
    FAIL() << "should have thrown";
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    FAIL() << "threw the wrong error";
  }

  EXPECT_TRUE(std::string::npos != message.find(" y is not a valid simplex"))
    << message;

  EXPECT_TRUE(std::string::npos != message.find(" y[2] = -0.1"))
    << message;
}

TEST(ErrorHandlingMatrix, checkSimplex_message_sum) {
  Eigen::Matrix<double,Eigen::Dynamic,1> y(100);
  y.setZero();
  std::string message;

  y[13] = 0.9;

  try {
    stan::error_handling::check_simplex("checkSimplex",
                                        "y", y);
    FAIL() << "should have thrown";
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    FAIL() << "threw the wrong error";
  }

  EXPECT_TRUE(std::string::npos != message.find(" y is not a valid simplex"))
    << message;

  EXPECT_TRUE(std::string::npos != message.find("sum(y) = 0.9"))
    << message;
}


TEST(ErrorHandlingMatrix, checkSimplex_message_length) {
  Eigen::Matrix<double,Eigen::Dynamic,1> y;
  std::string message;
  y.resize(0);

  try {
    stan::error_handling::check_simplex("checkSimplex",
                                        "y", y);
    FAIL() << "should have thrown";
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    FAIL() << "threw the wrong error";
  }

  EXPECT_TRUE(std::string::npos != message.find(" y is not a valid simplex"))
    << message;

  EXPECT_TRUE(std::string::npos != message.find("length(y) = 0"))
    << message;
}

TEST(ErrorHandlingMatrix, checkSimplex_nan) {
  Eigen::Matrix<double,Eigen::Dynamic,1> y(2);
  y.setZero();
  double nan = std::numeric_limits<double>::quiet_NaN();
  y << nan, 0.5;
  
  EXPECT_THROW(stan::error_handling::check_simplex("checkSimplex",
                                                   "y", y),
               std::domain_error);
                  
  y[1] = 0.55;
  EXPECT_THROW(stan::error_handling::check_simplex("checkSimplex", 
                                                   "y", y), 
               std::domain_error);

  y[0] = 0.5;
  y[1] = nan;
  EXPECT_THROW(stan::error_handling::check_simplex("checkSimplex", 
                                                   "y", y), 
               std::domain_error);

  y[0] = nan;
  EXPECT_THROW(stan::error_handling::check_simplex("checkSimplex", 
                                                   "y", y), 
               std::domain_error);
}

#include <stan/error_handling/matrix/check_matching_sizes.hpp>
#include <gtest/gtest.h>

TEST(ErrorHandlingMatrix, checkMatchingSizesMatrix) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> x;
  
  y.resize(3,3);
  x.resize(3,3);
  EXPECT_TRUE(stan::error_handling::check_matching_sizes("checkMatchingSizes", "x", x,
                                                         "y", y));
  x.resize(0,0);
  y.resize(0,0);
  EXPECT_TRUE(stan::error_handling::check_matching_sizes("checkMatchingSizes", "x", x,
                                                         "y", y));

  y.resize(1,2);


  EXPECT_THROW(stan::error_handling::check_matching_sizes("checkMatchingSizes", "x", x,
                                                          "y", y), 
               std::domain_error);

  x.resize(2,1);
  EXPECT_TRUE(stan::error_handling::check_matching_sizes("checkMatchingSizes", "x", x,
                                                         "y", y));

  std::vector<double> a;
  std::vector<double> b;
  x.resize(0,0);

  EXPECT_TRUE(stan::error_handling::check_matching_sizes("checkMatchingSizes", "a", a,
                                                         "b", b));
  EXPECT_TRUE(stan::error_handling::check_matching_sizes("checkMatchingSizes", "x", x,
                                                         "b", b));
  EXPECT_TRUE(stan::error_handling::check_matching_sizes("checkMatchingSizes", "a", a,
                                                         "x", x));
  EXPECT_THROW(stan::error_handling::check_matching_sizes("checkMatchingSizes", "a", a,
                                                          "y", y),
               std::domain_error);


  a.push_back(3.0);
  a.push_back(3.0);

  EXPECT_THROW(stan::error_handling::check_matching_sizes("checkMatchingSizes", "a", a,
                                                          "b", b),
               std::domain_error);

  b.push_back(3.0);
  b.push_back(3.0);
  x.resize(2,1);

  EXPECT_TRUE(stan::error_handling::check_matching_sizes("checkMatchingSizes", "a", a,
                                                         "b", b));
  EXPECT_TRUE(stan::error_handling::check_matching_sizes("checkMatchingSizes", "x", x,
                                                         "b", b));
}

TEST(ErrorHandlingMatrix, checkMatchingSizesMatrix_nan) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> x;
  double nan = std::numeric_limits<double>::quiet_NaN();

  y.resize(3,3);
  x.resize(3,3);
  y << nan, nan, nan,nan, nan, nan,nan, nan, nan;
  x << nan, nan, nan,nan, nan, nan,nan, nan, nan;
  EXPECT_TRUE(stan::error_handling::check_matching_sizes("checkMatchingSizes", "x", x,
                                                         "y", y));
  x.resize(0,0);
  y.resize(0,0);
  EXPECT_TRUE(stan::error_handling::check_matching_sizes("checkMatchingSizes", "x", x,
                                                         "y", y));

  y.resize(1,2);
  y << nan, nan;
  EXPECT_THROW(stan::error_handling::check_matching_sizes("checkMatchingSizes", "x", x,
                                                          "y", y), 
               std::domain_error);

  x.resize(2,1);
  x << nan, nan;
  EXPECT_TRUE(stan::error_handling::check_matching_sizes("checkMatchingSizes", "x", x,
                                                         "y", y));

  std::vector<double> a;
  std::vector<double> b;
  x.resize(0,0);
  EXPECT_TRUE(stan::error_handling::check_matching_sizes("checkMatchingSizes", "a", a,
                                                         "b", b));
  EXPECT_TRUE(stan::error_handling::check_matching_sizes("checkMatchingSizes", "x", x,
                                                         "b", b));
  EXPECT_TRUE(stan::error_handling::check_matching_sizes("checkMatchingSizes", "a", a,
                                                         "x", x));
  EXPECT_THROW(stan::error_handling::check_matching_sizes("checkMatchingSizes", "a", a,
                                                          "y", y),
               std::domain_error);


  a.push_back(nan);
  a.push_back(nan);
  EXPECT_THROW(stan::error_handling::check_matching_sizes("checkMatchingSizes", "a", a,
                                                          "b", b),
               std::domain_error);

  b.push_back(nan);
  b.push_back(nan);
  x.resize(2,1);
  x << nan, nan;
  EXPECT_TRUE(stan::error_handling::check_matching_sizes("checkMatchingSizes", "a", a,
                                                         "b", b));
  EXPECT_TRUE(stan::error_handling::check_matching_sizes("checkMatchingSizes", "x", x,
                                                         "b", b));
}


#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/prim/mat/err/check_matching_dims.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>

TEST(ErrorHandlingMatrix, checkMatchingDimsMatrix) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> x;
  
  y.resize(3,3);
  x.resize(3,3);
  EXPECT_TRUE(stan::math::check_matching_dims("checkMatchingDims", "x", x,
                                                        "y", y));
  x.resize(0,0);
  y.resize(0,0);
  EXPECT_TRUE(stan::math::check_matching_dims("checkMatchingDims", "x", x,
                                                        "y", y));

  y.resize(1,2);
  EXPECT_THROW(stan::math::check_matching_dims("checkMatchingDims", "x", x,
                                                         "y", y), 
               std::invalid_argument);

  x.resize(2,1);
  EXPECT_THROW(stan::math::check_matching_dims("checkMatchingDims", "x", x,
                                                         "y", y), 
               std::invalid_argument);
}

TEST(ErrorHandlingMatrix, checkMatchingDimsMatrix_nan) {
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> x;
  double nan = std::numeric_limits<double>::quiet_NaN();

  y.resize(3,3);
  x.resize(3,3);
  y << nan, nan, nan,nan, nan, nan,nan, nan, nan;
  x << nan, nan, nan,nan, nan, nan,nan, nan, nan;
  EXPECT_TRUE(stan::math::check_matching_dims("checkMatchingDims", "x", x,
                                                        "y", y));
  x.resize(0,0);
  y.resize(0,0);
  EXPECT_TRUE(stan::math::check_matching_dims("checkMatchingDims", "x", x,
                                                        "y", y));

  y.resize(1,2);
  y << nan, nan;
  EXPECT_THROW(stan::math::check_matching_dims("checkMatchingDims", "x", x,
                                                         "y", y), 
               std::invalid_argument);

  x.resize(2,1);
  x << nan, nan;
  EXPECT_THROW(stan::math::check_matching_dims("checkMatchingDims", "x", x,
                                                         "y", y), 
               std::invalid_argument);
}

TEST(ErrorHandlingMatrix, checkMatchingDims_compile_time_sizes) {
  using stan::math::check_matching_dims;

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m_dynamic;
  Eigen::Matrix<double, 2, 2> m_2x2;
  Eigen::Matrix<double, Eigen::Dynamic, 1> vector(4);
  Eigen::Matrix<double, 1, Eigen::Dynamic> rowvector(4);


  m_dynamic.resize(2,2);
  EXPECT_TRUE(check_matching_dims("check_matching_dims", "dynamic", m_dynamic,
                                  "2x2", m_2x2));

  m_dynamic.resize(3,3);
  EXPECT_THROW_MSG(check_matching_dims("check_matching_dims", "dynamic", m_dynamic,
                                       "2x2", m_2x2),
                   std::invalid_argument,
                   "check_matching_dims: Rows of dynamic (3) and rows of 2x2 (2) must match in size");

  m_dynamic.resize(4,1);  
  EXPECT_TRUE(check_matching_dims("check_matching_dims", "dynamic", m_dynamic,
                                  "vector", vector));

  m_dynamic.resize(3,1);  
  EXPECT_THROW_MSG(check_matching_dims("check_matching_dims", "dynamic", m_dynamic,
                                       "vector", vector),
                   std::invalid_argument,
                   "check_matching_dims: Rows of dynamic (3) and rows of vector (4) must match in size");


  m_dynamic.resize(1,4);  
  EXPECT_TRUE(check_matching_dims("check_matching_dims", "dynamic", m_dynamic,
                                  "rowvector", rowvector));

  m_dynamic.resize(1,3);  
  EXPECT_THROW_MSG(check_matching_dims("check_matching_dims", "dynamic", m_dynamic,
                                       "rowvector", rowvector),
                   std::invalid_argument,
                   "check_matching_dims: Columns of dynamic (3) and columns of rowvector (4) must match in size");

  
                   
}

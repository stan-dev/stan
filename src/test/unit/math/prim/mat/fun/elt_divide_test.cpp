#include <stan/math/prim/mat/fun/elt_divide.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix,eltDivideVec) {
  stan::math::vector_d v1(2);
  stan::math::vector_d v2(2);
  v1 << 1, 2;
  v2 << 10, 100;
  stan::math::vector_d v = stan::math::elt_divide(v1,v2);
  EXPECT_FLOAT_EQ(0.1, v(0));
  EXPECT_FLOAT_EQ(0.02, v(1));
}
TEST(MathMatrix,eltDivideVecException) {
  stan::math::vector_d v1(2);
  stan::math::vector_d v2(3);
  v1 << 1, 2;
  v2 << 10, 100, 1000;
  EXPECT_THROW(stan::math::elt_divide(v1,v2), std::invalid_argument);
}
TEST(MathMatrix,eltDivideVecByScalar) {
  stan::math::vector_d v1(2);
  v1 << 1, 2;
  stan::math::vector_d v = stan::math::elt_divide(v1,10);
  EXPECT_FLOAT_EQ(0.1, v(0));
  EXPECT_FLOAT_EQ(0.2, v(1));
}
TEST(MathMatrix,eltDivideScalarByVec) {
  stan::math::vector_d v1(2);
  v1 << 1, 2;
  stan::math::vector_d v = stan::math::elt_divide(10,v1);
  EXPECT_FLOAT_EQ(10, v(0));
  EXPECT_FLOAT_EQ(5, v(1));
}
TEST(MathMatrix,eltDivideRowVec) {
  stan::math::row_vector_d v1(2);
  stan::math::row_vector_d v2(2);
  v1 << 1, 2;
  v2 << 10, 100;
  stan::math::row_vector_d v = stan::math::elt_divide(v1,v2);
  EXPECT_FLOAT_EQ(0.1, v(0));
  EXPECT_FLOAT_EQ(0.02, v(1));
}
TEST(MathMatrix,eltDivideRowVecException) {
  stan::math::row_vector_d v1(2);
  stan::math::row_vector_d v2(3);
  v1 << 1, 2;
  v2 << 10, 100, 1000;
  EXPECT_THROW(stan::math::elt_divide(v1,v2), std::invalid_argument);
}
TEST(MathMatrix,eltDivideRowVecByScalar) {
  stan::math::row_vector_d v1(2);
  v1 << 1, 2;
  stan::math::row_vector_d v = stan::math::elt_divide(v1,10);
  EXPECT_FLOAT_EQ(0.1, v(0));
  EXPECT_FLOAT_EQ(0.2, v(1));
}
TEST(MathMatrix,eltDivideScalarByRowVec) {
  stan::math::row_vector_d v1(2);
  v1 << 1, 2;
  stan::math::row_vector_d v = stan::math::elt_divide(10,v1);
  EXPECT_FLOAT_EQ(10, v(0));
  EXPECT_FLOAT_EQ(5, v(1));
}
TEST(MathMatrix,eltDivideMatrix) {
  stan::math::matrix_d m1(2,3);
  stan::math::matrix_d m2(2,3);
  m1 << 1, 2, 3, 4, 5, 6;
  m2 << 10, 100, 1000, 10000, 100000, 1000000;
  stan::math::matrix_d m = stan::math::elt_divide(m1,m2);
  
  EXPECT_EQ(2,m.rows());
  EXPECT_EQ(3,m.cols());
  EXPECT_FLOAT_EQ(0.1, m(0,0));
  EXPECT_FLOAT_EQ(0.02, m(0,1));
  EXPECT_FLOAT_EQ(0.003, m(0,2));
  EXPECT_FLOAT_EQ(0.0004, m(1,0));
  EXPECT_FLOAT_EQ(0.00005, m(1,1));
  EXPECT_FLOAT_EQ(0.000006, m(1,2));
}
TEST(MathMatrix,eltDivideMatrixException) {
  stan::math::matrix_d m1(2,3);
  stan::math::matrix_d m2(2,4);
  stan::math::matrix_d m3(4,3);
  m1 << 1, 2, 3, 4, 5, 6;
  m2 << -1, -2, -3, -4, -5, -6, -7, -8;
  m3 << 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24;
  EXPECT_THROW(stan::math::elt_divide(m1,m2),std::invalid_argument);
  EXPECT_THROW(stan::math::elt_divide(m1,m3),std::invalid_argument);
}
TEST(MathMatrix,eltDivideMatrixByScalar) {
  stan::math::matrix_d m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;
  stan::math::matrix_d m = stan::math::elt_divide(m1,10);
  
  EXPECT_EQ(2,m.rows());
  EXPECT_EQ(3,m.cols());
  EXPECT_FLOAT_EQ(0.1, m(0,0));
  EXPECT_FLOAT_EQ(0.2, m(0,1));
  EXPECT_FLOAT_EQ(0.3, m(0,2));
  EXPECT_FLOAT_EQ(0.4, m(1,0));
  EXPECT_FLOAT_EQ(0.5, m(1,1));
  EXPECT_FLOAT_EQ(0.6, m(1,2));
}
TEST(MathMatrix,eltDivideScalarByMatrix) {
  stan::math::matrix_d m1(2,3);
  m1 << 10, 100, 1000, 10000, 100000, 1000000;
  stan::math::matrix_d m = stan::math::elt_divide(10,m1);
  
  EXPECT_EQ(2,m.rows());
  EXPECT_EQ(3,m.cols());
  EXPECT_FLOAT_EQ(1, m(0,0));
  EXPECT_FLOAT_EQ(0.1, m(0,1));
  EXPECT_FLOAT_EQ(0.01, m(0,2));
  EXPECT_FLOAT_EQ(0.001, m(1,0));
  EXPECT_FLOAT_EQ(0.0001, m(1,1));
  EXPECT_FLOAT_EQ(0.00001, m(1,2));
}

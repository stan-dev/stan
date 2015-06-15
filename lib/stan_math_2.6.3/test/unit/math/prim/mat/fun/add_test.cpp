#include <stan/math/prim/mat/fun/add.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix,add_v_exception) {
  stan::math::vector_d d1, d2;

  d1.resize(3);
  d2.resize(3);
  EXPECT_NO_THROW(stan::math::add(d1, d2));

  d1.resize(0);
  d2.resize(0);
  EXPECT_NO_THROW(stan::math::add(d1, d2));

  d1.resize(2);
  d2.resize(3);
  EXPECT_THROW(stan::math::add(d1, d2), std::invalid_argument);
}
TEST(MathMatrix,add_rv_exception) {
  stan::math::row_vector_d d1, d2;

  d1.resize(3);
  d2.resize(3);
  EXPECT_NO_THROW(stan::math::add(d1, d2));

  d1.resize(0);
  d2.resize(0);
  EXPECT_NO_THROW(stan::math::add(d1, d2));

  d1.resize(2);
  d2.resize(3);
  EXPECT_THROW(stan::math::add(d1, d2), std::invalid_argument);
}
TEST(MathMatrix,add_m_exception) {
  stan::math::matrix_d d1, d2;

  d1.resize(2,3);
  d2.resize(2,3);
  EXPECT_NO_THROW(stan::math::add(d1, d2));

  d1.resize(0,0);
  d2.resize(0,0);
  EXPECT_NO_THROW(stan::math::add(d1, d2));

  d1.resize(2,3);
  d2.resize(3,3);
  EXPECT_THROW(stan::math::add(d1, d2), std::invalid_argument);
}


TEST(MathMatrix,add_c_m) {
  stan::math::matrix_d v(2,2);
  v << 1, 2, 3, 4;
  stan::math::matrix_d result;

  result = stan::math::add(2.0,v);
  EXPECT_FLOAT_EQ(3.0,result(0,0));
  EXPECT_FLOAT_EQ(4.0,result(0,1));
  EXPECT_FLOAT_EQ(5.0,result(1,0));
  EXPECT_FLOAT_EQ(6.0,result(1,1));

  result = stan::math::add(v,2.0);
  EXPECT_FLOAT_EQ(3.0,result(0,0));
  EXPECT_FLOAT_EQ(4.0,result(0,1));
  EXPECT_FLOAT_EQ(5.0,result(1,0));
  EXPECT_FLOAT_EQ(6.0,result(1,1));
}

TEST(MathMatrix,add_c_rv) {
  stan::math::row_vector_d v(3);
  v << 1, 2, 3;
  stan::math::row_vector_d result;

  result = stan::math::add(2.0,v);
  EXPECT_FLOAT_EQ(3.0,result(0));
  EXPECT_FLOAT_EQ(4.0,result(1));
  EXPECT_FLOAT_EQ(5.0,result(2));

  result = stan::math::add(v,2.0);
  EXPECT_FLOAT_EQ(3.0,result(0));
  EXPECT_FLOAT_EQ(4.0,result(1));
  EXPECT_FLOAT_EQ(5.0,result(2));
}


TEST(MathMatrix,add_c_v) {
  stan::math::vector_d v(3);
  v << 1, 2, 3;
  stan::math::vector_d result;

  result = stan::math::add(2.0,v);
  EXPECT_FLOAT_EQ(3.0,result(0));
  EXPECT_FLOAT_EQ(4.0,result(1));
  EXPECT_FLOAT_EQ(5.0,result(2));

  result = stan::math::add(v,2.0);
  EXPECT_FLOAT_EQ(3.0,result(0));
  EXPECT_FLOAT_EQ(4.0,result(1));
  EXPECT_FLOAT_EQ(5.0,result(2));
}

TEST(MathMatrix, add) {
  stan::math::vector_d v1(2);
  v1 << 1, 2;
  stan::math::vector_d v2(3);
  v2 << 10, 100, 1000;

  stan::math::row_vector_d rv1(2);
  v1 << 1, 2;
  stan::math::row_vector_d rv2(3);
  v2 << 10, 100, 1000;

  stan::math::matrix_d m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;
  stan::math::matrix_d m2(3,2);
  m2 << 10, 100, 1000, 0, -10, -12;

  using stan::math::add;
  EXPECT_THROW(add(v1,v2),std::invalid_argument);
  EXPECT_THROW(add(rv1,rv2),std::invalid_argument);
  EXPECT_THROW(add(m1,m2),std::invalid_argument);
}

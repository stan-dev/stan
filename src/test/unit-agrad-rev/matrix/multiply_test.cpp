#include <stan/agrad/rev/matrix/multiply.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/math/matrix/multiply.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>

TEST(AgradRevMatrix, multiply_scalar_scalar) {
  using stan::agrad::multiply;
  double d1, d2;
  AVAR   v1, v2;

  d1 = 10;
  v1 = 10;
  d2 = -2;
  v2 = -2;
  
  EXPECT_FLOAT_EQ(-20.0, multiply(d1,d2));
  EXPECT_FLOAT_EQ(-20.0, multiply(d1, v2).val());
  EXPECT_FLOAT_EQ(-20.0, multiply(v1, d2).val());
  EXPECT_FLOAT_EQ(-20.0, multiply(v1, v2).val());

  EXPECT_FLOAT_EQ(6.0, multiply(AVAR(3),AVAR(2)).val());
  EXPECT_FLOAT_EQ(6.0, multiply(3.0,AVAR(2)).val());
  EXPECT_FLOAT_EQ(6.0, multiply(AVAR(3),2.0).val());

  
}
TEST(AgradRevMatrix, multiply_vector_scalar) {
  using stan::math::vector_d;
  using stan::agrad::vector_v;

  vector_d d1(3);
  vector_v v1(3);
  double d2;
  AVAR v2;
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
  d2 = -2;
  v2 = -2;
  
  vector_v output;
  output = multiply(d1, v2);
  EXPECT_FLOAT_EQ(-200, output(0).val());
  EXPECT_FLOAT_EQ(   0, output(1).val());
  EXPECT_FLOAT_EQ(   6, output(2).val());

  output = multiply(v1, d2);
  EXPECT_FLOAT_EQ(-200, output(0).val());
  EXPECT_FLOAT_EQ(   0, output(1).val());
  EXPECT_FLOAT_EQ(   6, output(2).val());

  output = multiply(v1, v2);
  EXPECT_FLOAT_EQ(-200, output(0).val());
  EXPECT_FLOAT_EQ(   0, output(1).val());
  EXPECT_FLOAT_EQ(   6, output(2).val());
}
TEST(AgradRevMatrix, multiply_rowvector_scalar) {
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_v;

  row_vector_d d1(3);
  row_vector_v v1(3);
  double d2;
  AVAR v2;
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
  d2 = -2;
  v2 = -2;
  
  row_vector_v output;
  output = multiply(d1, v2);
  EXPECT_FLOAT_EQ(-200, output(0).val());
  EXPECT_FLOAT_EQ(   0, output(1).val());
  EXPECT_FLOAT_EQ(   6, output(2).val());

  output = multiply(v1, d2);
  EXPECT_FLOAT_EQ(-200, output(0).val());
  EXPECT_FLOAT_EQ(   0, output(1).val());
  EXPECT_FLOAT_EQ(   6, output(2).val());

  output = multiply(v1, v2);
  EXPECT_FLOAT_EQ(-200, output(0).val());
  EXPECT_FLOAT_EQ(   0, output(1).val());
  EXPECT_FLOAT_EQ(   6, output(2).val());
}
TEST(AgradRevMatrix, multiply_matrix_scalar) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;
  
  matrix_d d1(2,2);
  matrix_v v1(2,2);
  double d2;
  AVAR v2;
  
  d1 << 100, 0, -3, 4;
  v1 << 100, 0, -3, 4;
  d2 = -2;
  v2 = -2;
  
  matrix_v output;
  output = multiply(d1, v2);
  EXPECT_FLOAT_EQ(-200, output(0,0).val());
  EXPECT_FLOAT_EQ(   0, output(0,1).val());
  EXPECT_FLOAT_EQ(   6, output(1,0).val());
  EXPECT_FLOAT_EQ(  -8, output(1,1).val());

  output = multiply(v1, d2);
  EXPECT_FLOAT_EQ(-200, output(0,0).val());
  EXPECT_FLOAT_EQ(   0, output(0,1).val());
  EXPECT_FLOAT_EQ(   6, output(1,0).val());
  EXPECT_FLOAT_EQ(  -8, output(1,1).val());
 
  output = multiply(v1, v2);
  EXPECT_FLOAT_EQ(-200, output(0,0).val());
  EXPECT_FLOAT_EQ(   0, output(0,1).val());
  EXPECT_FLOAT_EQ(   6, output(1,0).val());
  EXPECT_FLOAT_EQ(  -8, output(1,1).val());
}
TEST(AgradRevMatrix, multiply_rowvector_vector) {
  using stan::math::vector_d;
  using stan::agrad::vector_v;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_v;

  row_vector_d d1(3);
  row_vector_v v1(3);
  vector_d d2(3);
  vector_v v2(3);
  
  d1 << 1, 3, -5;
  v1 << 1, 3, -5;
  d2 << 4, -2, -1;
  v2 << 4, -2, -1;

  EXPECT_FLOAT_EQ(3, multiply(v1, v2).val());
  EXPECT_FLOAT_EQ(3, multiply(v1, d2).val());
  EXPECT_FLOAT_EQ(3, multiply(d1, v2).val());
  
  d1.resize(1);
  v1.resize(1);
  EXPECT_THROW(multiply(v1, v2), std::domain_error);
  EXPECT_THROW(multiply(v1, d2), std::domain_error);
  EXPECT_THROW(multiply(d1, v2), std::domain_error);
}
TEST(AgradRevMatrix, multiply_vector_rowvector) {
  using stan::agrad::matrix_v;
  using stan::math::vector_d;
  using stan::agrad::vector_v;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_v;

  vector_d d1(3);
  vector_v v1(3);
  row_vector_d d2(3);
  row_vector_v v2(3);
  
  d1 << 1, 3, -5;
  v1 << 1, 3, -5;
  d2 << 4, -2, -1;
  v2 << 4, -2, -1;

  matrix_v output = multiply(v1, v2);
  EXPECT_EQ(3, output.rows());
  EXPECT_EQ(3, output.cols());
  EXPECT_FLOAT_EQ(  4, output(0,0).val());
  EXPECT_FLOAT_EQ( -2, output(0,1).val());
  EXPECT_FLOAT_EQ( -1, output(0,2).val());
  EXPECT_FLOAT_EQ( 12, output(1,0).val());
  EXPECT_FLOAT_EQ( -6, output(1,1).val());
  EXPECT_FLOAT_EQ( -3, output(1,2).val());
  EXPECT_FLOAT_EQ(-20, output(2,0).val());
  EXPECT_FLOAT_EQ( 10, output(2,1).val());
  EXPECT_FLOAT_EQ(  5, output(2,2).val());
  
  output = multiply(v1, d2);
  EXPECT_EQ(3, output.rows());
  EXPECT_EQ(3, output.cols());
  EXPECT_FLOAT_EQ(  4, output(0,0).val());
  EXPECT_FLOAT_EQ( -2, output(0,1).val());
  EXPECT_FLOAT_EQ( -1, output(0,2).val());
  EXPECT_FLOAT_EQ( 12, output(1,0).val());
  EXPECT_FLOAT_EQ( -6, output(1,1).val());
  EXPECT_FLOAT_EQ( -3, output(1,2).val());
  EXPECT_FLOAT_EQ(-20, output(2,0).val());
  EXPECT_FLOAT_EQ( 10, output(2,1).val());
  EXPECT_FLOAT_EQ(  5, output(2,2).val());
  
  output = multiply(d1, v2);
  EXPECT_EQ(3, output.rows());
  EXPECT_EQ(3, output.cols());
  EXPECT_FLOAT_EQ(  4, output(0,0).val());
  EXPECT_FLOAT_EQ( -2, output(0,1).val());
  EXPECT_FLOAT_EQ( -1, output(0,2).val());
  EXPECT_FLOAT_EQ( 12, output(1,0).val());
  EXPECT_FLOAT_EQ( -6, output(1,1).val());
  EXPECT_FLOAT_EQ( -3, output(1,2).val());
  EXPECT_FLOAT_EQ(-20, output(2,0).val());
  EXPECT_FLOAT_EQ( 10, output(2,1).val());
  EXPECT_FLOAT_EQ(  5, output(2,2).val());
}
TEST(AgradRevMatrix, multiply_matrix_vector) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;
  using stan::math::vector_d;
  using stan::agrad::vector_v;

  matrix_d d1(3,2);
  matrix_v v1(3,2);
  vector_d d2(2);
  vector_v v2(2);
  
  d1 << 1, 3, -5, 4, -2, -1;
  v1 << 1, 3, -5, 4, -2, -1;
  d2 << -2, 4;
  v2 << -2, 4;

  vector_v output = multiply(v1, v2);
  EXPECT_EQ(3, output.size());
  EXPECT_FLOAT_EQ(10, output(0).val());
  EXPECT_FLOAT_EQ(26, output(1).val());
  EXPECT_FLOAT_EQ( 0, output(2).val());

  
  output = multiply(v1, d2);
  EXPECT_EQ(3, output.size());
  EXPECT_FLOAT_EQ(10, output(0).val());
  EXPECT_FLOAT_EQ(26, output(1).val());
  EXPECT_FLOAT_EQ( 0, output(2).val());
  
  output = multiply(d1, v2);
  EXPECT_EQ(3, output.size());
  EXPECT_FLOAT_EQ(10, output(0).val());
  EXPECT_FLOAT_EQ(26, output(1).val());
  EXPECT_FLOAT_EQ( 0, output(2).val());
}
TEST(AgradRevMatrix, multiply_matrix_vector_exception) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;
  using stan::math::vector_d;
  using stan::agrad::vector_v;

  matrix_d d1(3,2);
  matrix_v v1(3,2);
  vector_d d2(4);
  vector_v v2(4);
  EXPECT_THROW(multiply(v1, v2), std::domain_error);
  EXPECT_THROW(multiply(v1, d2), std::domain_error);
  EXPECT_THROW(multiply(d1, v2), std::domain_error);
}
TEST(AgradRevMatrix, multiply_rowvector_matrix) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;
  using stan::agrad::vector_v;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_v;

  row_vector_d d1(3);
  row_vector_v v1(3);
  matrix_d d2(3,2);
  matrix_v v2(3,2);
  
  d1 << -2, 4, 1;
  v1 << -2, 4, 1;
  d2 << 1, 3, -5, 4, -2, -1;
  v2 << 1, 3, -5, 4, -2, -1;

  vector_v output = multiply(v1, v2);
  EXPECT_EQ(2, output.size());
  EXPECT_FLOAT_EQ(-24, output(0).val());
  EXPECT_FLOAT_EQ(  9, output(1).val());

  output = multiply(v1, d2);
  EXPECT_EQ(2, output.size());
  EXPECT_FLOAT_EQ(-24, output(0).val());
  EXPECT_FLOAT_EQ(  9, output(1).val());
  
  output = multiply(d1, v2);
  EXPECT_EQ(2, output.size());
  EXPECT_FLOAT_EQ(-24, output(0).val());
  EXPECT_FLOAT_EQ(  9, output(1).val());
}
TEST(AgradRevMatrix, multiply_rowvector_matrix_exception) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_v;

  row_vector_d d1(4);
  row_vector_v v1(4);
  matrix_d d2(3,2);
  matrix_v v2(3,2);
  EXPECT_THROW(multiply(v1, v2), std::domain_error);
  EXPECT_THROW(multiply(v1, d2), std::domain_error);
  EXPECT_THROW(multiply(d1, v2), std::domain_error);
}
TEST(AgradRevMatrix, multiply_matrix_matrix) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;

  matrix_d d1(2,3);
  matrix_v v1(2,3);
  matrix_d d2(3,2);
  matrix_v v2(3,2);
  
  d1 << 9, 24, 3, 46, -9, -33;
  v1 << 9, 24, 3, 46, -9, -33;
  d2 << 1, 3, -5, 4, -2, -1;
  v2 << 1, 3, -5, 4, -2, -1;

  matrix_v output = multiply(v1, v2);
  EXPECT_EQ(2, output.rows());
  EXPECT_EQ(2, output.cols());
  EXPECT_FLOAT_EQ(-117, output(0,0).val());
  EXPECT_FLOAT_EQ( 120, output(0,1).val());
  EXPECT_FLOAT_EQ( 157, output(1,0).val());
  EXPECT_FLOAT_EQ( 135, output(1,1).val());

  output = multiply(v1, d2);
  EXPECT_EQ(2, output.rows());
  EXPECT_EQ(2, output.cols());
  EXPECT_FLOAT_EQ(-117, output(0,0).val());
  EXPECT_FLOAT_EQ( 120, output(0,1).val());
  EXPECT_FLOAT_EQ( 157, output(1,0).val());
  EXPECT_FLOAT_EQ( 135, output(1,1).val());
  
  output = multiply(d1, v2);
  EXPECT_EQ(2, output.rows());
  EXPECT_EQ(2, output.cols());
  EXPECT_FLOAT_EQ(-117, output(0,0).val());
  EXPECT_FLOAT_EQ( 120, output(0,1).val());
  EXPECT_FLOAT_EQ( 157, output(1,0).val());
  EXPECT_FLOAT_EQ( 135, output(1,1).val());
}
TEST(AgradRevMatrix, multiply_matrix_matrix_exception) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;

  matrix_d d1(2,2);
  matrix_v v1(2,2);
  matrix_d d2(3,2);
  matrix_v v2(3,2);

  EXPECT_THROW(multiply(v1, v2), std::domain_error);
  EXPECT_THROW(multiply(v1, d2), std::domain_error);
  EXPECT_THROW(multiply(d1, v2), std::domain_error);
}
TEST(AgradRevMatrix,multiply_scalar_vector_cv) {
  using stan::agrad::multiply;
  using stan::agrad::vector_v;

  vector_v x(3);
  x << 1, 2, 3;
  AVEC x_ind = createAVEC(x(0),x(1),x(2));
  vector_v y = multiply(2.0,x);
  EXPECT_FLOAT_EQ(2.0,y(0).val());
  EXPECT_FLOAT_EQ(4.0,y(1).val());
  EXPECT_FLOAT_EQ(6.0,y(2).val());

  VEC g = cgradvec(y(0),x_ind);
  EXPECT_FLOAT_EQ(2.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
  EXPECT_FLOAT_EQ(0.0,g[2]);
}
TEST(AgradRevMatrix,multiply_scalar_vector_vv) {
  using stan::agrad::multiply;
  using stan::agrad::vector_v;

  vector_v x(3);
  x << 1, 4, 9;
  AVAR two = 2.0;
  AVEC x_ind = createAVEC(x(0),x(1),x(2),two);
  vector_v y = multiply(two,x);
  EXPECT_FLOAT_EQ(2.0,y(0).val());
  EXPECT_FLOAT_EQ(8.0,y(1).val());
  EXPECT_FLOAT_EQ(18.0,y(2).val());

  VEC g = cgradvec(y(1),x_ind);
  EXPECT_FLOAT_EQ(0.0,g[0]);
  EXPECT_FLOAT_EQ(2.0,g[1]);
  EXPECT_FLOAT_EQ(0.0,g[2]);
  EXPECT_FLOAT_EQ(4.0,g[3]);
}
TEST(AgradRevMatrix,multiply_scalar_vector_vc) {
  using stan::agrad::multiply;
  using stan::agrad::vector_v;

  vector_v x(3);
  x << 1, 2, 3;
  AVAR two = 2.0;
  AVEC x_ind = createAVEC(two);
  vector_v y = multiply(two,x);
  EXPECT_FLOAT_EQ(2.0,y(0).val());
  EXPECT_FLOAT_EQ(4.0,y(1).val());
  EXPECT_FLOAT_EQ(6.0,y(2).val());

  VEC g = cgradvec(y(2),x_ind);
  EXPECT_FLOAT_EQ(3.0,g[0]);
}

TEST(AgradRevMatrix,multiply_scalar_row_vector_cv) {
  using stan::agrad::multiply;
  using stan::agrad::row_vector_v;

  row_vector_v x(3);
  x << 1, 2, 3;
  AVEC x_ind = createAVEC(x(0),x(1),x(2));
  row_vector_v y = multiply(2.0,x);
  EXPECT_FLOAT_EQ(2.0,y(0).val());
  EXPECT_FLOAT_EQ(4.0,y(1).val());
  EXPECT_FLOAT_EQ(6.0,y(2).val());

  VEC g = cgradvec(y(0),x_ind);
  EXPECT_FLOAT_EQ(2.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
  EXPECT_FLOAT_EQ(0.0,g[2]);
}
TEST(AgradRevMatrix,multiply_scalar_row_vector_vv) {
  using stan::agrad::multiply;
  using stan::agrad::row_vector_v;

  row_vector_v x(3);
  x << 1, 4, 9;
  AVAR two = 2.0;
  AVEC x_ind = createAVEC(x(0),x(1),x(2),two);
  row_vector_v y = multiply(two,x);
  EXPECT_FLOAT_EQ(2.0,y(0).val());
  EXPECT_FLOAT_EQ(8.0,y(1).val());
  EXPECT_FLOAT_EQ(18.0,y(2).val());

  VEC g = cgradvec(y(1),x_ind);
  EXPECT_FLOAT_EQ(0.0,g[0]);
  EXPECT_FLOAT_EQ(2.0,g[1]);
  EXPECT_FLOAT_EQ(0.0,g[2]);
  EXPECT_FLOAT_EQ(4.0,g[3]);
}
TEST(AgradRevMatrix,multiply_scalar_row_vector_vc) {
  using stan::agrad::multiply;
  using stan::agrad::row_vector_v;

  row_vector_v x(3);
  x << 1, 2, 3;
  AVAR two = 2.0;
  AVEC x_ind = createAVEC(two);
  row_vector_v y = multiply(two,x);
  EXPECT_FLOAT_EQ(2.0,y(0).val());
  EXPECT_FLOAT_EQ(4.0,y(1).val());
  EXPECT_FLOAT_EQ(6.0,y(2).val());

  VEC g = cgradvec(y(2),x_ind);
  EXPECT_FLOAT_EQ(3.0,g[0]);
}

TEST(AgradRevMatrix,multiply_scalar_matrix_cv) {
  using stan::agrad::multiply;
  using stan::agrad::matrix_v;

  matrix_v x(2,3);
  x << 1, 2, 3, 4, 5, 6;
  AVEC x_ind = createAVEC(x(0,0),x(0,1),x(0,2),x(1,0));
  matrix_v y = multiply(2.0,x);
  EXPECT_FLOAT_EQ(2.0,y(0,0).val());
  EXPECT_FLOAT_EQ(4.0,y(0,1).val());
  EXPECT_FLOAT_EQ(6.0,y(0,2).val());

  VEC g = cgradvec(y(0,0),x_ind);
  EXPECT_FLOAT_EQ(2.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
  EXPECT_FLOAT_EQ(0.0,g[2]);
  EXPECT_FLOAT_EQ(0.0,g[3]);
}

TEST(AgradRevMatrix,multiply_scalar_matrix_vc) {
  using stan::agrad::multiply;
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;

  matrix_d x(2,3);
  x << 1, 2, 3, 4, 5, 6;
  AVAR two = 2.0;
  AVEC x_ind = createAVEC(two);

  matrix_v y = multiply(two,x);
  EXPECT_FLOAT_EQ(2.0,y(0,0).val());
  EXPECT_FLOAT_EQ(4.0,y(0,1).val());
  EXPECT_FLOAT_EQ(6.0,y(0,2).val());

  VEC g = cgradvec(y(1,0),x_ind);
  EXPECT_FLOAT_EQ(4.0,g[0]);
}

TEST(AgradRevMatrix,multiply_vector_int) {
  using stan::agrad::multiply; // test namespace resolution
  using stan::math::multiply;
  using stan::math::vector_d;
  using stan::agrad::vector_v;

  vector_d dvec(3);
  dvec << 1, 2, 3;
  int a = 2;
  vector_d prod_vec = multiply(dvec,a);
  EXPECT_EQ(3,prod_vec.size());
  EXPECT_EQ(2.0, prod_vec[0]);
  EXPECT_EQ(4.0, prod_vec[1]);
  EXPECT_EQ(6.0, prod_vec[2]);
}


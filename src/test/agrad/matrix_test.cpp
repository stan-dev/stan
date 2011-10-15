#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <stdexcept>
#include <stan/agrad/matrix.hpp>

using stan::agrad::var;

using stan::maths::matrix_d;
using stan::agrad::matrix_v;
using stan::maths::vector_d;
using stan::agrad::vector_v;
using stan::maths::row_vector_d;
using stan::agrad::row_vector_v;

// to_var tests
TEST(agrad_matrix,to_var__scalar) {
  double d = 5.0;
  var v = 5.0;
  stan::agrad::var var_x = stan::agrad::to_var(d);
  EXPECT_FLOAT_EQ (5.0, var_x.val());
  
  var_x = stan::agrad::to_var(v);
  EXPECT_FLOAT_EQ (5.0, var_x.val());
}
TEST(agrad_matrix,to_var__matrix) {
  matrix_d m_d(2,3);
  m_d << 0, 1, 2, 3, 4, 5;
  matrix_v m_v = stan::agrad::to_var(m_d);
  
  EXPECT_EQ (2, m_v.rows());
  EXPECT_EQ (3, m_v.cols());
  for (int ii = 0; ii < 2; ii++) 
    for (int jj = 0; jj < 3; jj++)
      EXPECT_FLOAT_EQ (ii*3 + jj, m_v(ii, jj).val());
}
TEST(agrad_matrix,to_var_ref__matrix) {
  matrix_d m_d(2,3);
  m_d << 0, 1, 2, 3, 4, 5;

  matrix_v m_v(5,5);
  EXPECT_EQ(5, m_v.rows());
  EXPECT_EQ(5, m_v.cols());

  stan::agrad::to_var(m_d, m_v);  
  EXPECT_EQ (2, m_v.rows());
  EXPECT_EQ (3, m_v.cols());
  EXPECT_FLOAT_EQ (0, m_v(0, 0).val());
  EXPECT_FLOAT_EQ (1, m_v(0, 1).val());
  EXPECT_FLOAT_EQ (2, m_v(0, 2).val());
  EXPECT_FLOAT_EQ (3, m_v(1, 0).val());
  EXPECT_FLOAT_EQ (4, m_v(1, 1).val());
  EXPECT_FLOAT_EQ (5, m_v(1, 2).val());
}
TEST(agrad_matrix,to_var__vector) {
  vector_d d(5);
  vector_v v(5);
  
  d << 1, 2, 3, 4, 5;
  v << 1, 2, 3, 4, 5;
  
  vector_v out = stan::agrad::to_var(d);
  EXPECT_FLOAT_EQ (1, out(0).val());
  EXPECT_FLOAT_EQ (2, out(1).val());
  EXPECT_FLOAT_EQ (3, out(2).val());
  EXPECT_FLOAT_EQ (4, out(3).val());
  EXPECT_FLOAT_EQ (5, out(4).val());

  out = stan::agrad::to_var(v);
  EXPECT_FLOAT_EQ (1, out(0).val());
  EXPECT_FLOAT_EQ (2, out(1).val());
  EXPECT_FLOAT_EQ (3, out(2).val());
  EXPECT_FLOAT_EQ (4, out(3).val());
  EXPECT_FLOAT_EQ (5, out(4).val());
}
TEST(agrad_matrix,to_var_ref__vector) {
  vector_d d(5);
  vector_v v(5);
  
  d << 1, 2, 3, 4, 5;
  v << 1, 2, 3, 4, 5;
  
  vector_v output;
  stan::agrad::to_var(d, output);
  EXPECT_FLOAT_EQ (1, output(0).val());
  EXPECT_FLOAT_EQ (2, output(1).val());
  EXPECT_FLOAT_EQ (3, output(2).val());
  EXPECT_FLOAT_EQ (4, output(3).val());
  EXPECT_FLOAT_EQ (5, output(4).val());

  stan::agrad::to_var(v, output);
  EXPECT_FLOAT_EQ (1, output(0).val());
  EXPECT_FLOAT_EQ (2, output(1).val());
  EXPECT_FLOAT_EQ (3, output(2).val());
  EXPECT_FLOAT_EQ (4, output(3).val());
  EXPECT_FLOAT_EQ (5, output(4).val());
}
TEST(agrad_matrix,to_var__rowvector) {
  row_vector_d d(5);
  row_vector_v v(5);
  
  d << 1, 2, 3, 4, 5;
  v << 1, 2, 3, 4, 5;
  
  row_vector_v output = stan::agrad::to_var(d);
  EXPECT_FLOAT_EQ (1, output(0).val());
  EXPECT_FLOAT_EQ (2, output(1).val());
  EXPECT_FLOAT_EQ (3, output(2).val());
  EXPECT_FLOAT_EQ (4, output(3).val());
  EXPECT_FLOAT_EQ (5, output(4).val());

  output.resize(0);
  output = stan::agrad::to_var(v);
  EXPECT_FLOAT_EQ (1, output(0).val());
  EXPECT_FLOAT_EQ (2, output(1).val());
  EXPECT_FLOAT_EQ (3, output(2).val());
  EXPECT_FLOAT_EQ (4, output(3).val());
  EXPECT_FLOAT_EQ (5, output(4).val());
}
TEST(agrad_matrix,to_var_ref__rowvector) {
  row_vector_d d(5);
  row_vector_v v(5);

  d << 1, 2, 3, 4, 5;
  v << 1, 2, 3, 4, 5;
 
  row_vector_v output;
  stan::agrad::to_var(d, output);
  EXPECT_FLOAT_EQ (1, output(0).val());
  EXPECT_FLOAT_EQ (2, output(1).val());
  EXPECT_FLOAT_EQ (3, output(2).val());
  EXPECT_FLOAT_EQ (4, output(3).val());
  EXPECT_FLOAT_EQ (5, output(4).val());

  output.resize(0);
  stan::agrad::to_var(d, output);
  EXPECT_FLOAT_EQ (1, output(0).val());
  EXPECT_FLOAT_EQ (2, output(1).val());
  EXPECT_FLOAT_EQ (3, output(2).val());
  EXPECT_FLOAT_EQ (4, output(3).val());
  EXPECT_FLOAT_EQ (5, output(4).val());
}
// end to_var tests

// rows tests
TEST(agrad_matrix,rows__vector) {
  vector_v v(5);
  v << 0, 1, 2, 3, 4;
  EXPECT_EQ (5U, stan::agrad::rows(v));
  
  v.resize(0);
  EXPECT_EQ (0U, stan::agrad::rows(v));
}
TEST(agrad_matrix,rows__rowvector) {
  row_vector_v rv(5);
  rv << 0, 1, 2, 3, 4;
  EXPECT_EQ (1U, stan::agrad::rows(rv));

  rv.resize(0);
  EXPECT_EQ (1, stan::agrad::rows(rv));
}
TEST(agrad_matrix,rows__matrix) {
  matrix_v m(2,3);
  m << 0, 1, 2, 3, 4, 5;
  EXPECT_EQ (2, stan::agrad::rows(m));
  
  m.resize(0,2);
  EXPECT_EQ (0, stan::agrad::rows(m));
}
// end rows tests

// cols tests
TEST(agrad_matrix,cols__vector) {
  vector_v v(5);
  v << 0, 1, 2, 3, 4;
  EXPECT_EQ (1U, stan::agrad::cols(v));

  v.resize(0);
  EXPECT_EQ (1U, stan::agrad::cols(v));
}
TEST(agrad_matrix,cols__rowvector) {
  row_vector_v rv(5);
  rv << 0, 1, 2, 3, 4;
  EXPECT_EQ (5U, stan::agrad::cols(rv));
  
  rv.resize(0);
  EXPECT_EQ (0, stan::agrad::cols(rv));
}
TEST(agrad_matrix,cols__matrix) {
  matrix_v m(2,3);
  m << 0, 1, 2, 3, 4, 5;
  EXPECT_EQ (3, stan::agrad::cols(m));
  
  m.resize(5, 0);
  EXPECT_EQ (0, stan::agrad::cols(m));
}
// end cols_tests

// determinant tests
TEST(agrad_matrix,determinant) {
  matrix_v v(2,2);
  matrix_d d(2,2);
  v << 0, 1, 2, 3;
  d << 0, 1, 2, 3;

  var det;
  det = stan::agrad::determinant (v);
  EXPECT_FLOAT_EQ (-2, det.val());
  det = stan::agrad::determinant (d);
  EXPECT_FLOAT_EQ (-2, det.val());
}
TEST(agrad_matrix,deteriminant__exception) {
  matrix_v v(2,3);
  matrix_d d(2,3);

  var det;
  EXPECT_THROW (det = stan::agrad::determinant(d), std::domain_error);
  EXPECT_THROW (det = stan::agrad::determinant(v), std::domain_error);
}
// end determinant tests

// dot_product tests
TEST(agrad_matrix, dot_product__vector_vector) {
  vector_d vd_1(3), vd_2(3);
  vector_v vv_1(3), vv_2(3);
  
  vd_1 << 1, 3, -5;
  vv_1 << 1, 3, -5;
  vd_2 << 4, -2, -1;
  vv_2 << 4, -2, -1;

  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(vd_1, vd_2).val());
  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(vv_1, vd_2).val());
  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(vd_1, vv_2).val());
  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(vv_1, vv_2).val());
}
TEST(agrad_matrix, dot_product__vector_vector__exception) {
  vector_d d1(3);
  vector_v v1(3);
  vector_d d2(2);
  vector_v v2(4);

  EXPECT_THROW (stan::agrad::dot_product(d1, d2), std::invalid_argument);
  EXPECT_THROW (stan::agrad::dot_product(v1, d2), std::invalid_argument);
  EXPECT_THROW (stan::agrad::dot_product(d1, v2), std::invalid_argument);
  EXPECT_THROW (stan::agrad::dot_product(v1, v2), std::invalid_argument);
}
TEST(agrad_matrix, dot_product__rowvector_vector) {
  row_vector_d d1(3);
  row_vector_v v1(3);
  vector_d d2(3);
  vector_v v2(3);
  
  d1 << 1, 3, -5;
  v1 << 1, 3, -5;
  d2 << 4, -2, -1;
  v2 << 4, -2, -1;

  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(d1, d2).val());
  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(v1, d2).val());
  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(d1, v2).val());
  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(v1, v2).val());
}
TEST(agrad_matrix, dot_product__rowvector_vector__exception) {
  row_vector_d d1(3);
  row_vector_v v1(3);
  vector_d d2(2);
  vector_v v2(4);

  EXPECT_THROW (stan::agrad::dot_product(d1, d2), std::invalid_argument);
  EXPECT_THROW (stan::agrad::dot_product(v1, d2), std::invalid_argument);
  EXPECT_THROW (stan::agrad::dot_product(d1, v2), std::invalid_argument);
  EXPECT_THROW (stan::agrad::dot_product(v1, v2), std::invalid_argument);
}
TEST(agrad_matrix, dot_product__vector_rowvector) {
  vector_d d1(3);
  vector_v v1(3);
  row_vector_d d2(3);
  row_vector_v v2(3);
  
  d1 << 1, 3, -5;
  v1 << 1, 3, -5;
  d2 << 4, -2, -1;
  v2 << 4, -2, -1;
  
  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(d1, d2).val());
  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(v1, d2).val());
  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(d1, v2).val());
  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(v1, v2).val());
}
TEST(agrad_matrix, dot_product__vector_rowvector__exception) {
  vector_d d1(3);
  vector_v v1(3);
  row_vector_d d2(2);
  row_vector_v v2(4);

  EXPECT_THROW (stan::agrad::dot_product(d1, d2), std::invalid_argument);
  EXPECT_THROW (stan::agrad::dot_product(v1, d2), std::invalid_argument);
  EXPECT_THROW (stan::agrad::dot_product(d1, v2), std::invalid_argument);
  EXPECT_THROW (stan::agrad::dot_product(v1, v2), std::invalid_argument);
}
TEST(agrad_matrix, dot_product__rowvector_rowvector) {
  row_vector_d d1(3), d2(3);
  row_vector_v v1(3), v2(3);
  
  d1 << 1, 3, -5;
  v1 << 1, 3, -5;
  d2 << 4, -2, -1;
  v2 << 4, -2, -1;

  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(d1, d2).val());
  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(v1, d2).val());
  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(d1, v2).val());
  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(v1, v2).val());
}
TEST(agrad_matrix, dot_product__rowvector_rowvector__exception) {
  row_vector_d d1(3), d2(2);
  row_vector_v v1(3), v2(4);

  EXPECT_THROW (stan::agrad::dot_product(d1, d2), std::invalid_argument);
  EXPECT_THROW (stan::agrad::dot_product(v1, d2), std::invalid_argument);
  EXPECT_THROW (stan::agrad::dot_product(d1, v2), std::invalid_argument);
  EXPECT_THROW (stan::agrad::dot_product(v1, v2), std::invalid_argument);
}
// end dot_product tests

// add tests
TEST(agrad_matrix, add__vector_vector) {
  vector_v expected_output(5), output;
  vector_d vd_1(5), vd_2(5);
  vector_v vv_1(5), vv_2(5);
  
  vd_1 << 1, 2, 3, 4, 5;
  vv_1 << 1, 2, 3, 4, 5;
  vd_2 << 2, 3, 4, 5, 6;
  vv_2 << 2, 3, 4, 5, 6;
  
  expected_output << 3, 5, 7, 9, 11;
  
  output = stan::agrad::add(vd_1, vd_2);
  EXPECT_FLOAT_EQ (expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ (expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ (expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ (expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ (expected_output(4).val(), output(4).val());  

  output = stan::agrad::add(vv_1, vd_2);
  EXPECT_FLOAT_EQ (expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ (expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ (expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ (expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ (expected_output(4).val(), output(4).val());  

  output = stan::agrad::add(vd_1, vv_2);
  EXPECT_FLOAT_EQ (expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ (expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ (expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ (expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ (expected_output(4).val(), output(4).val());  

  output = stan::agrad::add(vv_1, vv_2);
  EXPECT_FLOAT_EQ (expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ (expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ (expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ (expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ (expected_output(4).val(), output(4).val());  
}
TEST(agrad_matrix, add__vector_vector__exception) {
  vector_d d1(5), d2(1);
  vector_v v1(5), v2(1);
  
  vector_v output;
  EXPECT_THROW(output = stan::agrad::add(d1, d2), std::invalid_argument);
  EXPECT_THROW(output = stan::agrad::add(v1, d2), std::invalid_argument);
  EXPECT_THROW(output = stan::agrad::add(d1, v2), std::invalid_argument);
  EXPECT_THROW(output = stan::agrad::add(v1, v2), std::invalid_argument);
}
TEST(agrad_matrix, add__rowvector_rowvector) {
  row_vector_v expected_output(5), output;
  row_vector_d rvd_1(5), rvd_2(5);
  row_vector_v rvv_1(5), rvv_2(5);

  rvd_1 << 1, 2, 3, 4, 5;
  rvv_1 << 1, 2, 3, 4, 5;
  rvd_2 << 2, 3, 4, 5, 6;
  rvv_2 << 2, 3, 4, 5, 6;
  
  expected_output << 3, 5, 7, 9, 11;
  
  output = stan::agrad::add(rvd_1, rvd_2);
  EXPECT_FLOAT_EQ (expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ (expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ (expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ (expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ (expected_output(4).val(), output(4).val());  

  output = stan::agrad::add(rvv_1, rvd_2);
  EXPECT_FLOAT_EQ (expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ (expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ (expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ (expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ (expected_output(4).val(), output(4).val());  

  output = stan::agrad::add(rvd_1, rvv_2);
  EXPECT_FLOAT_EQ (expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ (expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ (expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ (expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ (expected_output(4).val(), output(4).val());  

  output = stan::agrad::add(rvv_1, rvv_2);
  EXPECT_FLOAT_EQ (expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ (expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ (expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ (expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ (expected_output(4).val(), output(4).val());  
}
TEST(agrad_matrix, add__rowvector_rowvector__exception) {
  row_vector_d d1(5), d2(2);
  row_vector_v v1(5), v2(2);

  row_vector_v output;
  EXPECT_THROW(output = stan::agrad::add(d1, d2), std::invalid_argument);
  EXPECT_THROW(output = stan::agrad::add(d1, v2), std::invalid_argument);
  EXPECT_THROW(output = stan::agrad::add(v1, d2), std::invalid_argument);
  EXPECT_THROW(output = stan::agrad::add(v1, v2), std::invalid_argument);
}
TEST(agrad_matrix, add__matrix_matrix) {
  matrix_v expected_output(2,2), output;
  matrix_d md_1(2,2), md_2(2,2);
  matrix_v mv_1(2,2), mv_2(2,2);

  md_1 << -10, 1, 10, 0;
  mv_1 << -10, 1, 10, 0;
  md_2 << 10, -10, 1, 2;
  mv_2 << 10, -10, 1, 2;
  
  expected_output << 0, -9, 11, 2;
  
  output = stan::agrad::add(md_1, md_2);
  EXPECT_FLOAT_EQ (expected_output(0,0).val(), output(0,0).val());
  EXPECT_FLOAT_EQ (expected_output(0,1).val(), output(0,1).val());
  EXPECT_FLOAT_EQ (expected_output(1,0).val(), output(1,0).val());
  EXPECT_FLOAT_EQ (expected_output(1,1).val(), output(1,1).val());

  output = stan::agrad::add(mv_1, md_2);
  EXPECT_FLOAT_EQ (expected_output(0,0).val(), output(0,0).val());
  EXPECT_FLOAT_EQ (expected_output(0,1).val(), output(0,1).val());
  EXPECT_FLOAT_EQ (expected_output(1,0).val(), output(1,0).val());
  EXPECT_FLOAT_EQ (expected_output(1,1).val(), output(1,1).val());

  output = stan::agrad::add(md_1, mv_2);
  EXPECT_FLOAT_EQ (expected_output(0,0).val(), output(0,0).val());
  EXPECT_FLOAT_EQ (expected_output(0,1).val(), output(0,1).val());
  EXPECT_FLOAT_EQ (expected_output(1,0).val(), output(1,0).val());
  EXPECT_FLOAT_EQ (expected_output(1,1).val(), output(1,1).val());

  output = stan::agrad::add(mv_1, mv_2);
  EXPECT_FLOAT_EQ (expected_output(0,0).val(), output(0,0).val());
  EXPECT_FLOAT_EQ (expected_output(0,1).val(), output(0,1).val());
  EXPECT_FLOAT_EQ (expected_output(1,0).val(), output(1,0).val());
  EXPECT_FLOAT_EQ (expected_output(1,1).val(), output(1,1).val());
}
TEST(agrad_matrix, add__matrix_matrix__exception) {
  matrix_d d1(2,2), d2(1,2);
  matrix_v v1(2,2), v2(1,2);

  matrix_v output;
  EXPECT_THROW(output = stan::agrad::add(d1, d2), std::invalid_argument);
  EXPECT_THROW(output = stan::agrad::add(d1, v2), std::invalid_argument);
  EXPECT_THROW(output = stan::agrad::add(v1, d2), std::invalid_argument);
  EXPECT_THROW(output = stan::agrad::add(v1, v2), std::invalid_argument);
}
// end add tests

// subtract tests
TEST(agrad_matrix, subtract__vector_vector) {
  vector_v expected_output(5), output;
  vector_d vd_1(5), vd_2(5);
  vector_v vv_1(5), vv_2(5);

  vd_1 << 0, 2, -6, 10, 6;
  vv_1 << 0, 2, -6, 10, 6;
  vd_2 << 2, 3, 4, 5, 6;
  vv_2 << 2, 3, 4, 5, 6;
  
  expected_output << -2, -1, -10, 5, 0;
  
  output = stan::agrad::subtract(vd_1, vd_2);
  EXPECT_FLOAT_EQ (expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ (expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ (expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ (expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ (expected_output(4).val(), output(4).val());  

  output = stan::agrad::subtract(vv_1, vd_2);
  EXPECT_FLOAT_EQ (expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ (expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ (expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ (expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ (expected_output(4).val(), output(4).val());  

  output = stan::agrad::subtract(vd_1, vv_2);
  EXPECT_FLOAT_EQ (expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ (expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ (expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ (expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ (expected_output(4).val(), output(4).val());  

  output = stan::agrad::subtract(vv_1, vv_2);
  EXPECT_FLOAT_EQ (expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ (expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ (expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ (expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ (expected_output(4).val(), output(4).val());  
}
TEST(agrad_matrix, subtract__vector_vector__exception) {
  vector_d d1(5), d2(1);
  vector_v v1(5), v2(1);
  
  vector_v output;
  EXPECT_THROW(output = stan::agrad::subtract(d1, d2), std::invalid_argument);
  EXPECT_THROW(output = stan::agrad::subtract(v1, d2), std::invalid_argument);
  EXPECT_THROW(output = stan::agrad::subtract(d1, v2), std::invalid_argument);
  EXPECT_THROW(output = stan::agrad::subtract(v1, v2), std::invalid_argument);
}
TEST(agrad_matrix, subtract__rowvector_rowvector) {
  row_vector_v expected_output(5), output;
  row_vector_d rvd_1(5), rvd_2(5);
  row_vector_v rvv_1(5), rvv_2(5);

  rvd_1 << 0, 2, -6, 10, 6;
  rvv_1 << 0, 2, -6, 10, 6;
  rvd_2 << 2, 3, 4, 5, 6;
  rvv_2 << 2, 3, 4, 5, 6;
  
  expected_output << -2, -1, -10, 5, 0;
  
  output = stan::agrad::subtract(rvd_1, rvd_2);
  EXPECT_FLOAT_EQ (expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ (expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ (expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ (expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ (expected_output(4).val(), output(4).val());  

  output = stan::agrad::subtract(rvv_1, rvd_2);
  EXPECT_FLOAT_EQ (expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ (expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ (expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ (expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ (expected_output(4).val(), output(4).val());  

  output = stan::agrad::subtract(rvd_1, rvv_2);
  EXPECT_FLOAT_EQ (expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ (expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ (expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ (expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ (expected_output(4).val(), output(4).val());  

  output = stan::agrad::subtract(rvv_1, rvv_2);
  EXPECT_FLOAT_EQ (expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ (expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ (expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ (expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ (expected_output(4).val(), output(4).val());  
}
TEST(agrad_matrix, subtract__rowvector_rowvector__exception) {
  row_vector_d d1(5), d2(2);
  row_vector_v v1(5), v2(2);

  row_vector_v output;
  EXPECT_THROW(output = stan::agrad::subtract(d1, d2), std::invalid_argument);
  EXPECT_THROW(output = stan::agrad::subtract(d1, v2), std::invalid_argument);
  EXPECT_THROW(output = stan::agrad::subtract(v1, d2), std::invalid_argument);
  EXPECT_THROW(output = stan::agrad::subtract(v1, v2), std::invalid_argument);
}
TEST(agrad_matrix, subtract__matrix_matrix) {
  matrix_v expected_output(2,2), output;
  matrix_d md_1(2,2), md_2(2,2);
  matrix_v mv_1(2,2), mv_2(2,2);
  matrix_d md_mis (2, 3);
  matrix_v mv_mis (1, 1);

  md_1 << -10, 1, 10, 0;
  mv_1 << -10, 1, 10, 0;
  md_2 << 10, -10, 1, 2;
  mv_2 << 10, -10, 1, 2;
  
  expected_output << -20, 11, 9, -2;
  
  output = stan::agrad::subtract(md_1, md_2);
  EXPECT_FLOAT_EQ (expected_output(0,0).val(), output(0,0).val());
  EXPECT_FLOAT_EQ (expected_output(0,1).val(), output(0,1).val());
  EXPECT_FLOAT_EQ (expected_output(1,0).val(), output(1,0).val());
  EXPECT_FLOAT_EQ (expected_output(1,1).val(), output(1,1).val());

  output = stan::agrad::subtract(mv_1, md_2);
  EXPECT_FLOAT_EQ (expected_output(0,0).val(), output(0,0).val());
  EXPECT_FLOAT_EQ (expected_output(0,1).val(), output(0,1).val());
  EXPECT_FLOAT_EQ (expected_output(1,0).val(), output(1,0).val());
  EXPECT_FLOAT_EQ (expected_output(1,1).val(), output(1,1).val());

  output = stan::agrad::subtract(md_1, mv_2);
  EXPECT_FLOAT_EQ (expected_output(0,0).val(), output(0,0).val());
  EXPECT_FLOAT_EQ (expected_output(0,1).val(), output(0,1).val());
  EXPECT_FLOAT_EQ (expected_output(1,0).val(), output(1,0).val());
  EXPECT_FLOAT_EQ (expected_output(1,1).val(), output(1,1).val());

  output = stan::agrad::subtract(mv_1, mv_2);
  EXPECT_FLOAT_EQ (expected_output(0,0).val(), output(0,0).val());
  EXPECT_FLOAT_EQ (expected_output(0,1).val(), output(0,1).val());
  EXPECT_FLOAT_EQ (expected_output(1,0).val(), output(1,0).val());
  EXPECT_FLOAT_EQ (expected_output(1,1).val(), output(1,1).val());
}
TEST(agrad_matrix, subtract__matrix_matrix__exception) {
  matrix_d d1(2,2), d2(1,2);
  matrix_v v1(2,2), v2(1,2);

  matrix_v output;
  EXPECT_THROW(output = stan::agrad::subtract(d1, d2), std::invalid_argument);
  EXPECT_THROW(output = stan::agrad::subtract(d1, v2), std::invalid_argument);
  EXPECT_THROW(output = stan::agrad::subtract(v1, d2), std::invalid_argument);
  EXPECT_THROW(output = stan::agrad::subtract(v1, v2), std::invalid_argument);
}
// end subtract tests

// minus tests
TEST(agrad_matrix, minus__scalar) {
  double x = 10;
  var v = 11;
  
  EXPECT_FLOAT_EQ (-10, stan::agrad::minus(x).val());
  EXPECT_FLOAT_EQ (-11, stan::agrad::minus(v).val());
}
TEST(agrad_matrix, minus__vector) {
  vector_d d(3);
  vector_v v(3);

  d << -100, 0, 1;
  v << -100, 0, 1;
  
  vector_v output;
  output = stan::agrad::minus (d);
  EXPECT_FLOAT_EQ (100, output[0].val());
  EXPECT_FLOAT_EQ (0, output[1].val());
  EXPECT_FLOAT_EQ (-1, output[2].val());

  output = stan::agrad::minus (v);
  EXPECT_FLOAT_EQ (100, output[0].val());
  EXPECT_FLOAT_EQ (0, output[1].val());
  EXPECT_FLOAT_EQ (-1, output[2].val());
}
TEST(agrad_matrix, minus__rowvector) {
  row_vector_d d(3);
  row_vector_v v(3);

  d << -100, 0, 1;
  v << -100, 0, 1;
  
  row_vector_v output;
  output = stan::agrad::minus (d);
  EXPECT_FLOAT_EQ (100, output[0].val());
  EXPECT_FLOAT_EQ (0, output[1].val());
  EXPECT_FLOAT_EQ (-1, output[2].val());

  output = stan::agrad::minus (v);
  EXPECT_FLOAT_EQ (100, output[0].val());
  EXPECT_FLOAT_EQ (0, output[1].val());
  EXPECT_FLOAT_EQ (-1, output[2].val());
}
TEST(agrad_matrix, minus__matrix) {
  matrix_d d(2, 3);
  matrix_v v(2, 3);

  d << -100, 0, 1, 20, -40, 2;
  v << -100, 0, 1, 20, -40, 2;

  matrix_v output;
  output = stan::agrad::minus (d);
  EXPECT_FLOAT_EQ (100, output(0,0).val());
  EXPECT_FLOAT_EQ (  0, output(0,1).val());
  EXPECT_FLOAT_EQ ( -1, output(0,2).val());
  EXPECT_FLOAT_EQ (-20, output(1,0).val());
  EXPECT_FLOAT_EQ ( 40, output(1,1).val());
  EXPECT_FLOAT_EQ ( -2, output(1,2).val());

  output = stan::agrad::minus (v);
  EXPECT_FLOAT_EQ (100, output(0,0).val());
  EXPECT_FLOAT_EQ (  0, output(0,1).val());
  EXPECT_FLOAT_EQ ( -1, output(0,2).val());
  EXPECT_FLOAT_EQ (-20, output(1,0).val());
  EXPECT_FLOAT_EQ ( 40, output(1,1).val());
  EXPECT_FLOAT_EQ ( -2, output(1,2).val());
}
// end minus tests

// divide tests
TEST(agrad_matrix, divide__scalar) {
  double d1, d2;
  var    v1, v2;

  d1 = 10;
  v1 = 10;
  d2 = -2;
  v2 = -2;
  
  EXPECT_FLOAT_EQ (-5, stan::agrad::divide(d1, d2).val());
  EXPECT_FLOAT_EQ (-5, stan::agrad::divide(d1, v2).val());
  EXPECT_FLOAT_EQ (-5, stan::agrad::divide(v1, d2).val());
  EXPECT_FLOAT_EQ (-5, stan::agrad::divide(v1, v2).val());

  d2 = 0;
  v2 = 0;

  EXPECT_FLOAT_EQ (std::numeric_limits<double>::infinity(), stan::agrad::divide(d1, d2).val());
  EXPECT_FLOAT_EQ (std::numeric_limits<double>::infinity(), stan::agrad::divide(d1, v2).val());
  EXPECT_FLOAT_EQ (std::numeric_limits<double>::infinity(), stan::agrad::divide(v1, d2).val());
  EXPECT_FLOAT_EQ (std::numeric_limits<double>::infinity(), stan::agrad::divide(v1, v2).val());

  d1 = 0;
  v1 = 0;
  EXPECT_TRUE (std::isnan(stan::agrad::divide(d1, d2).val()));
  EXPECT_TRUE (std::isnan(stan::agrad::divide(d1, v2).val()));
  EXPECT_TRUE (std::isnan(stan::agrad::divide(v1, d2).val()));
  EXPECT_TRUE (std::isnan(stan::agrad::divide(v1, v2).val()));
}
TEST(agrad_matrix, divide__vector) {
  vector_d d1(3);
  vector_v v1(3);
  double d2;
  var v2;
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
  d2 = -2;
  v2 = -2;
  
  vector_v output;
  output = stan::agrad::divide(d1, d2);
  EXPECT_FLOAT_EQ (-50, output(0).val());
  EXPECT_FLOAT_EQ (  0, output(1).val());
  EXPECT_FLOAT_EQ (1.5, output(2).val());

  output = stan::agrad::divide(d1, v2);
  EXPECT_FLOAT_EQ (-50, output(0).val());
  EXPECT_FLOAT_EQ (  0, output(1).val());
  EXPECT_FLOAT_EQ (1.5, output(2).val());

  output = stan::agrad::divide(v1, d2);
  EXPECT_FLOAT_EQ (-50, output(0).val());
  EXPECT_FLOAT_EQ (  0, output(1).val());
  EXPECT_FLOAT_EQ (1.5, output(2).val());

  output = stan::agrad::divide(v1, v2);
  EXPECT_FLOAT_EQ (-50, output(0).val());
  EXPECT_FLOAT_EQ (  0, output(1).val());
  EXPECT_FLOAT_EQ (1.5, output(2).val());


  d2 = 0;
  v2 = 0;
  output = stan::agrad::divide(d1, d2);
  EXPECT_FLOAT_EQ (std::numeric_limits<double>::infinity(), output(0).val());
  EXPECT_TRUE (std::isnan(output(1).val()));
  EXPECT_FLOAT_EQ (-std::numeric_limits<double>::infinity(), output(2).val());

  output = stan::agrad::divide(d1, v2);
  EXPECT_FLOAT_EQ (std::numeric_limits<double>::infinity(), output(0).val());
  EXPECT_TRUE (std::isnan(output(1).val()));
  EXPECT_FLOAT_EQ (-std::numeric_limits<double>::infinity(), output(2).val());

  output = stan::agrad::divide(v1, d2);
  EXPECT_FLOAT_EQ (std::numeric_limits<double>::infinity(), output(0).val());
  EXPECT_TRUE (std::isnan(output(1).val()));
  EXPECT_FLOAT_EQ (-std::numeric_limits<double>::infinity(), output(2).val());

  output = stan::agrad::divide(v1, v2);
  EXPECT_FLOAT_EQ (std::numeric_limits<double>::infinity(), output(0).val());
  EXPECT_TRUE (std::isnan(output(1).val()));
  EXPECT_FLOAT_EQ (-std::numeric_limits<double>::infinity(), output(2).val());
}
TEST(agrad_matrix, divide__rowvector) {
  row_vector_d d1(3);
  row_vector_v v1(3);
  double d2;
  var v2;
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
  d2 = -2;
  v2 = -2;
  
  row_vector_v output;
  output = stan::agrad::divide(d1, d2);
  EXPECT_FLOAT_EQ (-50, output(0).val());
  EXPECT_FLOAT_EQ (  0, output(1).val());
  EXPECT_FLOAT_EQ (1.5, output(2).val());

  output = stan::agrad::divide(d1, v2);
  EXPECT_FLOAT_EQ (-50, output(0).val());
  EXPECT_FLOAT_EQ (  0, output(1).val());
  EXPECT_FLOAT_EQ (1.5, output(2).val());

  output = stan::agrad::divide(v1, d2);
  EXPECT_FLOAT_EQ (-50, output(0).val());
  EXPECT_FLOAT_EQ (  0, output(1).val());
  EXPECT_FLOAT_EQ (1.5, output(2).val());

  output = stan::agrad::divide(v1, v2);
  EXPECT_FLOAT_EQ (-50, output(0).val());
  EXPECT_FLOAT_EQ (  0, output(1).val());
  EXPECT_FLOAT_EQ (1.5, output(2).val());


  d2 = 0;
  v2 = 0;
  output = stan::agrad::divide(d1, d2);
  EXPECT_FLOAT_EQ (std::numeric_limits<double>::infinity(), output(0).val());
  EXPECT_TRUE (std::isnan(output(1).val()));
  EXPECT_FLOAT_EQ (-std::numeric_limits<double>::infinity(), output(2).val());

  output = stan::agrad::divide(d1, v2);
  EXPECT_FLOAT_EQ (std::numeric_limits<double>::infinity(), output(0).val());
  EXPECT_TRUE (std::isnan(output(1).val()));
  EXPECT_FLOAT_EQ (-std::numeric_limits<double>::infinity(), output(2).val());

  output = stan::agrad::divide(v1, d2);
  EXPECT_FLOAT_EQ (std::numeric_limits<double>::infinity(), output(0).val());
  EXPECT_TRUE (std::isnan(output(1).val()));
  EXPECT_FLOAT_EQ (-std::numeric_limits<double>::infinity(), output(2).val());

  output = stan::agrad::divide(v1, v2);
  EXPECT_FLOAT_EQ (std::numeric_limits<double>::infinity(), output(0).val());
  EXPECT_TRUE (std::isnan(output(1).val()));
  EXPECT_FLOAT_EQ (-std::numeric_limits<double>::infinity(), output(2).val());
}
TEST(agrad_matrix, divide__matrix) {
  matrix_d d1(2,2);
  matrix_v v1(2,2);
  double d2;
  var v2;
  
  d1 << 100, 0, -3, 4;
  v1 << 100, 0, -3, 4;
  d2 = -2;
  v2 = -2;
  
  matrix_v output;
  output = stan::agrad::divide(d1, d2);
  EXPECT_FLOAT_EQ (-50, output(0,0).val());
  EXPECT_FLOAT_EQ (  0, output(0,1).val());
  EXPECT_FLOAT_EQ (1.5, output(1,0).val());
  EXPECT_FLOAT_EQ ( -2, output(1,1).val());

  output = stan::agrad::divide(d1, v2);
  EXPECT_FLOAT_EQ (-50, output(0,0).val());
  EXPECT_FLOAT_EQ (  0, output(0,1).val());
  EXPECT_FLOAT_EQ (1.5, output(1,0).val());
  EXPECT_FLOAT_EQ ( -2, output(1,1).val());
  
  output = stan::agrad::divide(v1, d2);
  EXPECT_FLOAT_EQ (-50, output(0,0).val());
  EXPECT_FLOAT_EQ (  0, output(0,1).val());
  EXPECT_FLOAT_EQ (1.5, output(1,0).val());
  EXPECT_FLOAT_EQ ( -2, output(1,1).val());
  
  output = stan::agrad::divide(v1, v2);
  EXPECT_FLOAT_EQ (-50, output(0,0).val());
  EXPECT_FLOAT_EQ (  0, output(0,1).val());
  EXPECT_FLOAT_EQ (1.5, output(1,0).val());
  EXPECT_FLOAT_EQ ( -2, output(1,1).val());

  d2 = 0;
  v2 = 0;
  output = stan::agrad::divide(d1, d2);
  EXPECT_FLOAT_EQ (std::numeric_limits<double>::infinity(), output(0,0).val());
  EXPECT_TRUE (std::isnan(output(0,1).val()));
  EXPECT_FLOAT_EQ (-std::numeric_limits<double>::infinity(), output(1,0).val());
  EXPECT_FLOAT_EQ (std::numeric_limits<double>::infinity(), output(1,1).val());

  output = stan::agrad::divide(d1, v2);
  EXPECT_FLOAT_EQ (std::numeric_limits<double>::infinity(), output(0,0).val());
  EXPECT_TRUE (std::isnan(output(0,1).val()));
  EXPECT_FLOAT_EQ (-std::numeric_limits<double>::infinity(), output(1,0).val());
  EXPECT_FLOAT_EQ (std::numeric_limits<double>::infinity(), output(1,1).val());

  output = stan::agrad::divide(v1, d2);
  EXPECT_FLOAT_EQ (std::numeric_limits<double>::infinity(), output(0,0).val());
  EXPECT_TRUE (std::isnan(output(0,1).val()));
  EXPECT_FLOAT_EQ (-std::numeric_limits<double>::infinity(), output(1,0).val());
  EXPECT_FLOAT_EQ (std::numeric_limits<double>::infinity(), output(1,1).val());

  output = stan::agrad::divide(v1, v2);
  EXPECT_FLOAT_EQ (std::numeric_limits<double>::infinity(), output(0,0).val());
  EXPECT_TRUE (std::isnan(output(0,1).val()));
  EXPECT_FLOAT_EQ (-std::numeric_limits<double>::infinity(), output(1,0).val());
  EXPECT_FLOAT_EQ (std::numeric_limits<double>::infinity(), output(1,1).val());
}
// end divide tests

// min tests
TEST (agrad_matrix, min__vector) {
  vector_d d1(3);
  vector_v v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
  
  var output;
  output = stan::agrad::min(d1);
  EXPECT_FLOAT_EQ (-3, output.val());
		   
  output = stan::agrad::min(v1);
  EXPECT_FLOAT_EQ (-3, output.val());
}
TEST (agrad_matrix, min__vector__exception) {
  vector_d d;
  vector_v v;
  d.resize(0);
  v.resize(0);
  EXPECT_THROW (stan::agrad::min(d), std::domain_error);
  EXPECT_THROW (stan::agrad::min(v), std::domain_error);
}
TEST (agrad_matrix, min__rowvector) {
  row_vector_d d1(3);
  row_vector_v v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
  
  var output;
  output = stan::agrad::min(d1);
  EXPECT_FLOAT_EQ (-3, output.val());
		   
  output = stan::agrad::min(v1);
  EXPECT_FLOAT_EQ (-3, output.val());
}
TEST (agrad_matrix, min__rowvector__exception) {
  row_vector_d d;
  row_vector_v v;
  EXPECT_THROW (stan::agrad::min(d), std::domain_error);
  EXPECT_THROW (stan::agrad::min(v), std::domain_error);
}
TEST (agrad_matrix, min__matrix) {
  matrix_d d1(3,1);
  matrix_v v1(1,3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
  
  var output;
  output = stan::agrad::min(d1);
  EXPECT_FLOAT_EQ (-3, output.val());
		   
  output = stan::agrad::min(v1);
  EXPECT_FLOAT_EQ (-3, output.val());
}
TEST (agrad_matrix, min__matrix__exception) {
  matrix_d d;
  matrix_v v;
  EXPECT_THROW (stan::agrad::min(d), std::domain_error);
  EXPECT_THROW (stan::agrad::min(v), std::domain_error);
}
// end min tests

// max tests
TEST (agrad_matrix, max__vector) {
  vector_d d1(3);
  vector_v v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
  
  var output;
  output = stan::agrad::max(d1);
  EXPECT_FLOAT_EQ (100, output.val());
		   
  output = stan::agrad::max(v1);
  EXPECT_FLOAT_EQ (100, output.val());
}
TEST (agrad_matrix, max__vector__exception) {
  vector_d d;
  vector_v v;
  EXPECT_THROW (stan::agrad::max(d), std::domain_error);
  EXPECT_THROW (stan::agrad::max(v), std::domain_error);
}
TEST (agrad_matrix, max__rowvector) {
  row_vector_d d1(3);
  row_vector_v v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
  
  var output;
  output = stan::agrad::max(d1);
  EXPECT_FLOAT_EQ (100, output.val());
		   
  output = stan::agrad::max(v1);
  EXPECT_FLOAT_EQ (100, output.val());
}
TEST(agrad_matrix, max__rowvector__exception) {
  row_vector_d d;
  row_vector_v v;
  EXPECT_THROW (stan::agrad::max(d), std::domain_error);
  EXPECT_THROW (stan::agrad::max(v), std::domain_error);
}
TEST (agrad_matrix, max__matrix) {
  matrix_d d1(3,1);
  matrix_v v1(1,3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
  
  var output;
  output = stan::agrad::max(d1);
  EXPECT_FLOAT_EQ (100, output.val());
		   
  output = stan::agrad::max(v1);
  EXPECT_FLOAT_EQ (100, output.val());
}
TEST (agrad_matrix, max__matrix__exception) {
  matrix_d d;
  matrix_v v;
  EXPECT_THROW (stan::agrad::max(d), std::domain_error);
  EXPECT_THROW (stan::agrad::max(v), std::domain_error);
}
// end max tests

// mean tests
TEST (agrad_matrix, mean__vector) {
  vector_d d1(3);
  vector_v v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
  
  var output;
  output = stan::agrad::mean(d1);
  EXPECT_FLOAT_EQ (97.0/3.0, output.val());
		   
  output = stan::agrad::mean(v1);
  EXPECT_FLOAT_EQ (97.0/3.0, output.val());
}
TEST (agrad_matrix, mean__vector__exception) {
  vector_d d;
  vector_v v;
  EXPECT_THROW (stan::agrad::mean(d), std::domain_error);
  EXPECT_THROW (stan::agrad::mean(v), std::domain_error);
}
TEST (agrad_matrix, mean__rowvector) {
  row_vector_d d1(3);
  row_vector_v v1(3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
  
  var output;
  output = stan::agrad::mean(d1);
  EXPECT_FLOAT_EQ (97.0/3.0, output.val());
		   
  output = stan::agrad::mean(v1);
  EXPECT_FLOAT_EQ (97.0/3.0, output.val());
}
TEST (agrad_matrix, mean__rowvector__exception) {
  row_vector_d d;
  row_vector_v v;
  EXPECT_THROW (stan::agrad::mean(d), std::domain_error);
  EXPECT_THROW (stan::agrad::mean(v), std::domain_error);
}
TEST (agrad_matrix, mean__matrix) {
  matrix_d d1(3,1);
  matrix_v v1(1,3);
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
  
  var output;
  output = stan::agrad::mean(d1);
  EXPECT_FLOAT_EQ (97.0/3.0, output.val());
		   
  output = stan::agrad::mean(v1);
  EXPECT_FLOAT_EQ (97.0/3.0, output.val());
}
TEST (agrad_matrix, mean__matrix__exception) {
  matrix_d d;
  matrix_v v;
  EXPECT_THROW (stan::agrad::mean(d), std::domain_error);
  EXPECT_THROW (stan::agrad::mean(v), std::domain_error);
}
// end mean tests

// variance tests
TEST (agrad_matrix, variance__vector) {
  vector_d d1(6);
  vector_v v1(6);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
  
  var output;
  output = stan::agrad::variance(d1);
  EXPECT_FLOAT_EQ (17.5/5.0, output.val());
		   
  output = stan::agrad::variance(v1);
  EXPECT_FLOAT_EQ (17.5/5.0, output.val());

  d1.resize(1);
  v1.resize(1);
  output = stan::agrad::variance(d1);
  EXPECT_FLOAT_EQ (0.0, output.val());
  output = stan::agrad::variance(v1);
  EXPECT_FLOAT_EQ (0.0, output.val());  
}
TEST (agrad_matrix, variance__vector__exception) {
  vector_d d1;
  vector_v v1;
  EXPECT_THROW (stan::agrad::variance(d1), std::domain_error);
  EXPECT_THROW (stan::agrad::variance(v1), std::domain_error);
}
TEST (agrad_matrix, variance__rowvector) {
  row_vector_d d1(6);
  row_vector_v v1(6);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
  
  var output;
  output = stan::agrad::variance(d1);
  EXPECT_FLOAT_EQ (17.5/5.0, output.val());
		   
  output = stan::agrad::variance(v1);
  EXPECT_FLOAT_EQ (17.5/5.0, output.val());

  d1.resize(1);
  v1.resize(1);
  output = stan::agrad::variance(d1);
  EXPECT_FLOAT_EQ (0.0, output.val());
  output = stan::agrad::variance(v1);
  EXPECT_FLOAT_EQ (0.0, output.val());  
}
TEST (agrad_matrix, variance__rowvector__exception) {
  row_vector_d d1;
  row_vector_v v1;
  EXPECT_THROW (stan::agrad::variance(d1), std::domain_error);
  EXPECT_THROW (stan::agrad::variance(v1), std::domain_error);
}
TEST (agrad_matrix, variance__matrix) {
  matrix_d d1(2, 3);
  matrix_v v1(2, 3);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
  
  var output;
  output = stan::agrad::variance(d1);
  EXPECT_FLOAT_EQ (17.5/5.0, output.val());
		   
  output = stan::agrad::variance(v1);
  EXPECT_FLOAT_EQ (17.5/5.0, output.val());

  d1.resize(1,1);
  v1.resize(1,1);
  output = stan::agrad::variance(d1);
  EXPECT_FLOAT_EQ (0.0, output.val());
  output = stan::agrad::variance(v1);
  EXPECT_FLOAT_EQ (0.0, output.val());  
}
TEST (agrad_matrix, variance__matrix__exception) {
  matrix_d d1;
  matrix_v v1;
  EXPECT_THROW (stan::agrad::variance(d1), std::domain_error);
  EXPECT_THROW (stan::agrad::variance(v1), std::domain_error);

  d1.resize(0,1);
  v1.resize(0,1);
  EXPECT_THROW (stan::agrad::variance(d1), std::domain_error);
  EXPECT_THROW (stan::agrad::variance(v1), std::domain_error);

  d1.resize(1,0);
  v1.resize(1,0);
  EXPECT_THROW (stan::agrad::variance(d1), std::domain_error);
  EXPECT_THROW (stan::agrad::variance(v1), std::domain_error);
}
// end variance tests

// sd tests
TEST (agrad_matrix, sd__vector) {
  vector_d d1(6);
  vector_v v1(6);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
  
  var output;
  output = stan::agrad::sd(d1);
  EXPECT_FLOAT_EQ (std::sqrt(17.5/5.0), output.val());
		   
  output = stan::agrad::sd(v1);
  EXPECT_FLOAT_EQ (std::sqrt(17.5/5.0), output.val());
  
  d1.resize(1);
  v1.resize(1);
  output = stan::agrad::sd(d1);
  EXPECT_FLOAT_EQ (0.0, output.val());
  output = stan::agrad::sd(v1);
  EXPECT_FLOAT_EQ (0.0, output.val());
}
TEST (agrad_matrix, sd__vector__exception) {
  vector_d d1;
  vector_v v1;
  EXPECT_THROW (stan::agrad::sd(d1), std::domain_error);
  EXPECT_THROW (stan::agrad::sd(v1), std::domain_error);
}
TEST (agrad_matrix, sd__rowvector) {
  row_vector_d d1(6);
  row_vector_v v1(6);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
  
  var output;
  output = stan::agrad::sd(d1);
  EXPECT_FLOAT_EQ (std::sqrt(17.5/5.0), output.val());
		   
  output = stan::agrad::sd(v1);
  EXPECT_FLOAT_EQ (std::sqrt(17.5/5.0), output.val());

  d1.resize(1);
  v1.resize(1);
  output = stan::agrad::sd(d1);
  EXPECT_FLOAT_EQ (0.0, output.val());
  output = stan::agrad::sd(v1);
  EXPECT_FLOAT_EQ (0.0, output.val());
}
TEST (agrad_matrix, sd__rowvector__exception) {
  row_vector_d d;
  row_vector_v v;
  
  EXPECT_THROW (stan::agrad::sd(d), std::domain_error);
  EXPECT_THROW (stan::agrad::sd(v), std::domain_error);
}
TEST (agrad_matrix, sd__matrix) {
  matrix_d d1(2, 3);
  matrix_v v1(2, 3);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
  
  var output;
  output = stan::agrad::sd(d1);
  EXPECT_FLOAT_EQ (std::sqrt(17.5/5.0), output.val());
		   
  output = stan::agrad::sd(v1);
  EXPECT_FLOAT_EQ (std::sqrt(17.5/5.0), output.val());

  d1.resize(1, 1);
  v1.resize(1, 1);
  output = stan::agrad::sd(d1);
  EXPECT_FLOAT_EQ (0.0, output.val());
  output = stan::agrad::sd(v1);
  EXPECT_FLOAT_EQ (0.0, output.val());
}
TEST (agrad_matrix, sd__matrix__exception) {
  matrix_d d;
  matrix_v v;

  EXPECT_THROW (stan::agrad::sd(d), std::domain_error);
  EXPECT_THROW (stan::agrad::sd(v), std::domain_error);

  d.resize(1, 0);
  v.resize(1, 0);
  EXPECT_THROW (stan::agrad::sd(d), std::domain_error);
  EXPECT_THROW (stan::agrad::sd(v), std::domain_error);

  d.resize(0, 1);
  v.resize(0, 1);
  EXPECT_THROW (stan::agrad::sd(d), std::domain_error);
  EXPECT_THROW (stan::agrad::sd(v), std::domain_error);
}
// end sd tests

// sum tests
TEST (agrad_matrix, sum__vector) {
  vector_d d(6);
  vector_v v(6);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
  
  var output;
  output = stan::agrad::sum(d);
  EXPECT_FLOAT_EQ (21.0, output.val());
		   
  output = stan::agrad::sum(v);
  EXPECT_FLOAT_EQ (21.0, output.val());

  d.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ (0.0, stan::agrad::sum(d).val());
  EXPECT_FLOAT_EQ (0.0, stan::agrad::sum(v).val());
}
TEST (agrad_matrix, sum__rowvector) {
  row_vector_d d(6);
  row_vector_v v(6);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
  
  var output;
  output = stan::agrad::sum(d);
  EXPECT_FLOAT_EQ (21.0, output.val());
		   
  output = stan::agrad::sum(v);
  EXPECT_FLOAT_EQ (21.0, output.val());

  d.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ (0.0, stan::agrad::sum(d).val());
  EXPECT_FLOAT_EQ (0.0, stan::agrad::sum(v).val());
}
TEST (agrad_matrix, sum__matrix) {
  matrix_d d(2, 3);
  matrix_v v(2, 3);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
  
  var output;
  output = stan::agrad::sum(d);
  EXPECT_FLOAT_EQ (21.0, output.val());
		   
  output = stan::agrad::sum(v);
  EXPECT_FLOAT_EQ (21.0, output.val());

  d.resize(0, 0);
  v.resize(0, 0);
  EXPECT_FLOAT_EQ (0.0, stan::agrad::sum(d).val());
  EXPECT_FLOAT_EQ (0.0, stan::agrad::sum(v).val());
}
// end sum tests

// multiply tests
TEST(agrad_matrix, multiply__scalar_scalar) {
  double d1, d2;
  var    v1, v2;

  d1 = 10;
  v1 = 10;
  d2 = -2;
  v2 = -2;
  
  EXPECT_FLOAT_EQ (-20.0, stan::agrad::multiply(d1, d2).val());
  EXPECT_FLOAT_EQ (-20.0, stan::agrad::multiply(d1, v2).val());
  EXPECT_FLOAT_EQ (-20.0, stan::agrad::multiply(v1, d2).val());
  EXPECT_FLOAT_EQ (-20.0, stan::agrad::multiply(v1, v2).val());
}
TEST(agrad_matrix, multiply__vector_scalar) {
  vector_d d1(3);
  vector_v v1(3);
  double d2;
  var v2;
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
  d2 = -2;
  v2 = -2;
  
  vector_v output;
  output = stan::agrad::multiply(d1, d2);
  EXPECT_FLOAT_EQ (-200, output(0).val());
  EXPECT_FLOAT_EQ (   0, output(1).val());
  EXPECT_FLOAT_EQ (   6, output(2).val());

  output = stan::agrad::multiply(d1, v2);
  EXPECT_FLOAT_EQ (-200, output(0).val());
  EXPECT_FLOAT_EQ (   0, output(1).val());
  EXPECT_FLOAT_EQ (   6, output(2).val());

  output = stan::agrad::multiply(v1, d2);
  EXPECT_FLOAT_EQ (-200, output(0).val());
  EXPECT_FLOAT_EQ (   0, output(1).val());
  EXPECT_FLOAT_EQ (   6, output(2).val());

  output = stan::agrad::multiply(v1, v2);
  EXPECT_FLOAT_EQ (-200, output(0).val());
  EXPECT_FLOAT_EQ (   0, output(1).val());
  EXPECT_FLOAT_EQ (   6, output(2).val());
}
TEST(agrad_matrix, multiply__rowvector_scalar) {
  row_vector_d d1(3);
  row_vector_v v1(3);
  double d2;
  var v2;
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
  d2 = -2;
  v2 = -2;
  
  row_vector_v output;
  output = stan::agrad::multiply(d1, d2);
  EXPECT_FLOAT_EQ (-200, output(0).val());
  EXPECT_FLOAT_EQ (   0, output(1).val());
  EXPECT_FLOAT_EQ (   6, output(2).val());

  output = stan::agrad::multiply(d1, v2);
  EXPECT_FLOAT_EQ (-200, output(0).val());
  EXPECT_FLOAT_EQ (   0, output(1).val());
  EXPECT_FLOAT_EQ (   6, output(2).val());

  output = stan::agrad::multiply(v1, d2);
  EXPECT_FLOAT_EQ (-200, output(0).val());
  EXPECT_FLOAT_EQ (   0, output(1).val());
  EXPECT_FLOAT_EQ (   6, output(2).val());

  output = stan::agrad::multiply(v1, v2);
  EXPECT_FLOAT_EQ (-200, output(0).val());
  EXPECT_FLOAT_EQ (   0, output(1).val());
  EXPECT_FLOAT_EQ (   6, output(2).val());
}
TEST(agrad_matrix, multiply__matrix_scalar) {
  matrix_d d1(2,2);
  matrix_v v1(2,2);
  double d2;
  var v2;
  
  d1 << 100, 0, -3, 4;
  v1 << 100, 0, -3, 4;
  d2 = -2;
  v2 = -2;
  
  matrix_v output;
  output = stan::agrad::multiply(d1, d2);
  EXPECT_FLOAT_EQ (-200, output(0,0).val());
  EXPECT_FLOAT_EQ (   0, output(0,1).val());
  EXPECT_FLOAT_EQ (   6, output(1,0).val());
  EXPECT_FLOAT_EQ (  -8, output(1,1).val());

  output = stan::agrad::multiply(d1, v2);
  EXPECT_FLOAT_EQ (-200, output(0,0).val());
  EXPECT_FLOAT_EQ (   0, output(0,1).val());
  EXPECT_FLOAT_EQ (   6, output(1,0).val());
  EXPECT_FLOAT_EQ (  -8, output(1,1).val());

  output = stan::agrad::multiply(v1, d2);
  EXPECT_FLOAT_EQ (-200, output(0,0).val());
  EXPECT_FLOAT_EQ (   0, output(0,1).val());
  EXPECT_FLOAT_EQ (   6, output(1,0).val());
  EXPECT_FLOAT_EQ (  -8, output(1,1).val());
 
  output = stan::agrad::multiply(v1, v2);
  EXPECT_FLOAT_EQ (-200, output(0,0).val());
  EXPECT_FLOAT_EQ (   0, output(0,1).val());
  EXPECT_FLOAT_EQ (   6, output(1,0).val());
  EXPECT_FLOAT_EQ (  -8, output(1,1).val());
}
TEST(agrad_matrix, multiply__rowvector_vector) {
  row_vector_d d1(3);
  row_vector_v v1(3);
  vector_d d2(3);
  vector_v v2(3);
  
  d1 << 1, 3, -5;
  v1 << 1, 3, -5;
  d2 << 4, -2, -1;
  v2 << 4, -2, -1;

  EXPECT_FLOAT_EQ (3, stan::agrad::multiply(v1, v2).val());
  EXPECT_FLOAT_EQ (3, stan::agrad::multiply(v1, d2).val());
  EXPECT_FLOAT_EQ (3, stan::agrad::multiply(d1, v2).val());
  EXPECT_FLOAT_EQ (3, stan::agrad::multiply(d1, d2).val());
  
  d1.resize(1);
  v1.resize(1);
  EXPECT_THROW(stan::agrad::multiply(v1, v2), std::invalid_argument);
  EXPECT_THROW(stan::agrad::multiply(v1, d2), std::invalid_argument);
  EXPECT_THROW(stan::agrad::multiply(d1, v2), std::invalid_argument);
  EXPECT_THROW(stan::agrad::multiply(d1, d2), std::invalid_argument);
}
TEST(agrad_matrix, multiply__vector_rowvector) {
  vector_d d1(3);
  vector_v v1(3);
  row_vector_d d2(3);
  row_vector_v v2(3);
  
  d1 << 1, 3, -5;
  v1 << 1, 3, -5;
  d2 << 4, -2, -1;
  v2 << 4, -2, -1;

  matrix_v output = stan::agrad::multiply(v1, v2);
  EXPECT_EQ (3, output.rows());
  EXPECT_EQ (3, output.cols());
  EXPECT_FLOAT_EQ (  4, output(0,0).val());
  EXPECT_FLOAT_EQ ( -2, output(0,1).val());
  EXPECT_FLOAT_EQ ( -1, output(0,2).val());
  EXPECT_FLOAT_EQ ( 12, output(1,0).val());
  EXPECT_FLOAT_EQ ( -6, output(1,1).val());
  EXPECT_FLOAT_EQ ( -3, output(1,2).val());
  EXPECT_FLOAT_EQ (-20, output(2,0).val());
  EXPECT_FLOAT_EQ ( 10, output(2,1).val());
  EXPECT_FLOAT_EQ (  5, output(2,2).val());
  
  output = stan::agrad::multiply(v1, d2);
  EXPECT_EQ (3, output.rows());
  EXPECT_EQ (3, output.cols());
  EXPECT_FLOAT_EQ (  4, output(0,0).val());
  EXPECT_FLOAT_EQ ( -2, output(0,1).val());
  EXPECT_FLOAT_EQ ( -1, output(0,2).val());
  EXPECT_FLOAT_EQ ( 12, output(1,0).val());
  EXPECT_FLOAT_EQ ( -6, output(1,1).val());
  EXPECT_FLOAT_EQ ( -3, output(1,2).val());
  EXPECT_FLOAT_EQ (-20, output(2,0).val());
  EXPECT_FLOAT_EQ ( 10, output(2,1).val());
  EXPECT_FLOAT_EQ (  5, output(2,2).val());
  
  output = stan::agrad::multiply(d1, v2);
  EXPECT_EQ (3, output.rows());
  EXPECT_EQ (3, output.cols());
  EXPECT_FLOAT_EQ (  4, output(0,0).val());
  EXPECT_FLOAT_EQ ( -2, output(0,1).val());
  EXPECT_FLOAT_EQ ( -1, output(0,2).val());
  EXPECT_FLOAT_EQ ( 12, output(1,0).val());
  EXPECT_FLOAT_EQ ( -6, output(1,1).val());
  EXPECT_FLOAT_EQ ( -3, output(1,2).val());
  EXPECT_FLOAT_EQ (-20, output(2,0).val());
  EXPECT_FLOAT_EQ ( 10, output(2,1).val());
  EXPECT_FLOAT_EQ (  5, output(2,2).val());
  
  output = stan::agrad::multiply(d1, d2);
  EXPECT_EQ (3, output.rows());
  EXPECT_EQ (3, output.cols());
  EXPECT_FLOAT_EQ (  4, output(0,0).val());
  EXPECT_FLOAT_EQ ( -2, output(0,1).val());
  EXPECT_FLOAT_EQ ( -1, output(0,2).val());
  EXPECT_FLOAT_EQ ( 12, output(1,0).val());
  EXPECT_FLOAT_EQ ( -6, output(1,1).val());
  EXPECT_FLOAT_EQ ( -3, output(1,2).val());
  EXPECT_FLOAT_EQ (-20, output(2,0).val());
  EXPECT_FLOAT_EQ ( 10, output(2,1).val());
  EXPECT_FLOAT_EQ (  5, output(2,2).val());
}
TEST(agrad_matrix, multiply__matrix_vector) {
  matrix_d d1(3,2);
  matrix_v v1(3,2);
  vector_d d2(2);
  vector_v v2(2);
  
  d1 << 1, 3, -5, 4, -2, -1;
  v1 << 1, 3, -5, 4, -2, -1;
  d2 << -2, 4;
  v2 << -2, 4;

  vector_v output = stan::agrad::multiply(v1, v2);
  EXPECT_EQ (3, output.size());
  EXPECT_FLOAT_EQ (10, output(0).val());
  EXPECT_FLOAT_EQ (26, output(1).val());
  EXPECT_FLOAT_EQ ( 0, output(2).val());

  
  output = stan::agrad::multiply(v1, d2);
  EXPECT_EQ (3, output.size());
  EXPECT_FLOAT_EQ (10, output(0).val());
  EXPECT_FLOAT_EQ (26, output(1).val());
  EXPECT_FLOAT_EQ ( 0, output(2).val());
  
  output = stan::agrad::multiply(d1, v2);
  EXPECT_EQ (3, output.size());
  EXPECT_FLOAT_EQ (10, output(0).val());
  EXPECT_FLOAT_EQ (26, output(1).val());
  EXPECT_FLOAT_EQ ( 0, output(2).val());
  
  output = stan::agrad::multiply(d1, d2);
  EXPECT_EQ (3, output.size());
  EXPECT_FLOAT_EQ (10, output(0).val());
  EXPECT_FLOAT_EQ (26, output(1).val());
  EXPECT_FLOAT_EQ ( 0, output(2).val());
}
TEST(agrad_matrix, multiply__matrix_vector__exception) {
  matrix_d d1(3,2);
  matrix_v v1(3,2);
  vector_d d2(4);
  vector_v v2(4);
  EXPECT_THROW(stan::agrad::multiply(v1, v2), std::invalid_argument);
  EXPECT_THROW(stan::agrad::multiply(v1, d2), std::invalid_argument);
  EXPECT_THROW(stan::agrad::multiply(d1, v2), std::invalid_argument);
  EXPECT_THROW(stan::agrad::multiply(d1, d2), std::invalid_argument);
}
TEST(agrad_matrix, multiply__rowvector_matrix) {
  row_vector_d d1(3);
  row_vector_v v1(3);
  matrix_d d2(3,2);
  matrix_v v2(3,2);
  
  d1 << -2, 4, 1;
  v1 << -2, 4, 1;
  d2 << 1, 3, -5, 4, -2, -1;
  v2 << 1, 3, -5, 4, -2, -1;

  vector_v output = stan::agrad::multiply(v1, v2);
  EXPECT_EQ (2, output.size());
  EXPECT_FLOAT_EQ (-24, output(0).val());
  EXPECT_FLOAT_EQ (  9, output(1).val());

  output = stan::agrad::multiply(v1, d2);
  EXPECT_EQ (2, output.size());
  EXPECT_FLOAT_EQ (-24, output(0).val());
  EXPECT_FLOAT_EQ (  9, output(1).val());
  
  output = stan::agrad::multiply(d1, v2);
  EXPECT_EQ (2, output.size());
  EXPECT_FLOAT_EQ (-24, output(0).val());
  EXPECT_FLOAT_EQ (  9, output(1).val());
  
  output = stan::agrad::multiply(d1, d2);
  EXPECT_EQ (2, output.size());
  EXPECT_FLOAT_EQ (-24, output(0).val());
  EXPECT_FLOAT_EQ (  9, output(1).val());
}
TEST(agrad_matrix, multiply__rowvector_matrix__exception) {
  row_vector_d d1(4);
  row_vector_v v1(4);
  matrix_d d2(3,2);
  matrix_v v2(3,2);
  EXPECT_THROW(stan::agrad::multiply(v1, v2), std::invalid_argument);
  EXPECT_THROW(stan::agrad::multiply(v1, d2), std::invalid_argument);
  EXPECT_THROW(stan::agrad::multiply(d1, v2), std::invalid_argument);
  EXPECT_THROW(stan::agrad::multiply(d1, d2), std::invalid_argument);
}
TEST(agrad_matrix, multiply__matrix_matrix) {
  matrix_d d1(2,3);
  matrix_v v1(2,3);
  matrix_d d2(3,2);
  matrix_v v2(3,2);
  
  d1 << 9, 24, 3, 46, -9, -33;
  v1 << 9, 24, 3, 46, -9, -33;
  d2 << 1, 3, -5, 4, -2, -1;
  v2 << 1, 3, -5, 4, -2, -1;

  matrix_v output = stan::agrad::multiply(v1, v2);
  EXPECT_EQ (2, output.rows());
  EXPECT_EQ (2, output.cols());
  EXPECT_FLOAT_EQ (-117, output(0,0).val());
  EXPECT_FLOAT_EQ ( 120, output(0,1).val());
  EXPECT_FLOAT_EQ ( 157, output(1,0).val());
  EXPECT_FLOAT_EQ ( 135, output(1,1).val());

  output = stan::agrad::multiply(v1, d2);
  EXPECT_EQ (2, output.rows());
  EXPECT_EQ (2, output.cols());
  EXPECT_FLOAT_EQ (-117, output(0,0).val());
  EXPECT_FLOAT_EQ ( 120, output(0,1).val());
  EXPECT_FLOAT_EQ ( 157, output(1,0).val());
  EXPECT_FLOAT_EQ ( 135, output(1,1).val());
  
  output = stan::agrad::multiply(d1, v2);
  EXPECT_EQ (2, output.rows());
  EXPECT_EQ (2, output.cols());
  EXPECT_FLOAT_EQ (-117, output(0,0).val());
  EXPECT_FLOAT_EQ ( 120, output(0,1).val());
  EXPECT_FLOAT_EQ ( 157, output(1,0).val());
  EXPECT_FLOAT_EQ ( 135, output(1,1).val());
  
  output = stan::agrad::multiply(d1, d2);
  EXPECT_EQ (2, output.rows());
  EXPECT_EQ (2, output.cols());
  EXPECT_FLOAT_EQ (-117, output(0,0).val());
  EXPECT_FLOAT_EQ ( 120, output(0,1).val());
  EXPECT_FLOAT_EQ ( 157, output(1,0).val());
  EXPECT_FLOAT_EQ ( 135, output(1,1).val());
}
TEST(agrad_matrix, multiply__matrix_matrix__exception) {
  matrix_d d1(2,2);
  matrix_v v1(2,2);
  matrix_d d2(3,2);
  matrix_v v2(3,2);

  EXPECT_THROW(stan::agrad::multiply(v1, v2), std::invalid_argument);
  EXPECT_THROW(stan::agrad::multiply(v1, d2), std::invalid_argument);
  EXPECT_THROW(stan::agrad::multiply(d1, v2), std::invalid_argument);
  EXPECT_THROW(stan::agrad::multiply(d1, d2), std::invalid_argument);
}
// end multiply tests

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <stan/agrad/matrix.hpp>

using stan::agrad::var;

using stan::maths::matrix_d;
using stan::agrad::matrix_v;
using stan::maths::vector_d;
using stan::agrad::vector_v;
using stan::maths::row_vector_d;
using stan::agrad::row_vector_v;

// to_var tests
TEST(agrad_matrix,to_var) {
  double x = 5.0;
  stan::agrad::var var_x = stan::agrad::to_var(x);
  EXPECT_FLOAT_EQ (5.0, var_x.val());
}
TEST(agrad_matrix,m_to_var) {
  matrix_d m_d(2,3);
  m_d << 0, 1, 2, 3, 4, 5;
  matrix_v m_v = stan::agrad::to_var(m_d);
  
  EXPECT_EQ (2, m_v.rows());
  EXPECT_EQ (3, m_v.cols());
  for (int ii = 0; ii < 2; ii++) 
    for (int jj = 0; jj < 3; jj++)
      EXPECT_FLOAT_EQ (ii*3 + jj, m_v(ii, jj).val());
}
TEST(agrad_matrix,m_to_var_ref) {
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
TEST(agrad_matrix,v_to_var) {
  vector_d v_d(5);
  v_d << 1, 2, 3, 4, 5;
  
  vector_v v_v = stan::agrad::to_var(v_d);
  EXPECT_FLOAT_EQ (1, v_v(0).val());
  EXPECT_FLOAT_EQ (2, v_v(1).val());
  EXPECT_FLOAT_EQ (3, v_v(2).val());
  EXPECT_FLOAT_EQ (4, v_v(3).val());
  EXPECT_FLOAT_EQ (5, v_v(4).val());
}
TEST(agrad_matrix,v_to_var_ref) {
  vector_d v_d(5);
  v_d << 1, 2, 3, 4, 5;
  
  vector_v v_v;
  EXPECT_EQ(0, v_v.size());

  stan::agrad::to_var(v_d, v_v);
  EXPECT_FLOAT_EQ (1, v_v(0).val());
  EXPECT_FLOAT_EQ (2, v_v(1).val());
  EXPECT_FLOAT_EQ (3, v_v(2).val());
  EXPECT_FLOAT_EQ (4, v_v(3).val());
  EXPECT_FLOAT_EQ (5, v_v(4).val());
}
TEST(agrad_matrix,rv_to_var) {
  row_vector_d rv_d(5);
  rv_d << 1, 2, 3, 4, 5;
  
  row_vector_v rv_v = stan::agrad::to_var(rv_d);
  EXPECT_FLOAT_EQ (1, rv_v(0).val());
  EXPECT_FLOAT_EQ (2, rv_v(1).val());
  EXPECT_FLOAT_EQ (3, rv_v(2).val());
  EXPECT_FLOAT_EQ (4, rv_v(3).val());
  EXPECT_FLOAT_EQ (5, rv_v(4).val());
}
TEST(agrad_matrix,rv_to_var_asref) {
  row_vector_d rv_d(5);
  rv_d << 1, 2, 3, 4, 5;
 
  row_vector_v rv_v;
  EXPECT_EQ(0, rv_v.size());

  stan::agrad::to_var(rv_d, rv_v);
  EXPECT_FLOAT_EQ (1, rv_v(0).val());
  EXPECT_FLOAT_EQ (2, rv_v(1).val());
  EXPECT_FLOAT_EQ (3, rv_v(2).val());
  EXPECT_FLOAT_EQ (4, rv_v(3).val());
  EXPECT_FLOAT_EQ (5, rv_v(4).val());
}
// end to_var tests

// rows tests
TEST(agrad_matrix,rows_v) {
  vector_v v(5);
  v << 0, 1, 2, 3, 4;
  EXPECT_EQ (5U, stan::agrad::rows(v));
}
TEST(agrad_matrix,rows_rv) {
  row_vector_v rv(5);
  rv << 0, 1, 2, 3, 4;
  EXPECT_EQ (1U, stan::agrad::rows(rv));
}
TEST(agrad_matrix,rows_m) {
  matrix_v m(2,3);
  m << 0, 1, 2, 3, 4, 5;
  EXPECT_EQ (2, stan::agrad::rows(m));
}
// end rows tests

// cols tests
TEST(agrad_matrix,cols_v) {
  vector_v v(5);
  v << 0, 1, 2, 3, 4;
  EXPECT_EQ (1U, stan::agrad::cols(v));
}
TEST(agrad_matrix,cols_rv) {
  row_vector_v rv(5);
  rv << 0, 1, 2, 3, 4;
  EXPECT_EQ (5U, stan::agrad::cols(rv));
}
TEST(agrad_matrix,cols_m) {
  matrix_v m(2,3);
  m << 0, 1, 2, 3, 4, 5;
  EXPECT_EQ (3, stan::agrad::cols(m));
}
// end cols_tests

// determinant tests
TEST(agrad_matrix,determinant) {
  matrix_v m(2,2);
  m << 0, 1, 2, 3;
  var det = stan::agrad::determinant (m);
  
  EXPECT_FLOAT_EQ (-2, det.val());
}
// end determinant tests

// dot_product tests
TEST(agrad_matrix,dot_product_vv_vv) {
  vector_v v1(3);
  vector_v v2(3);
  v1 << 1, 3, -5;
  v2 << 4, -2, -1;
  
  var v = stan::agrad::dot_product(v1, v2);
  EXPECT_FLOAT_EQ (3, v.val());
}

TEST(agrad_matrix,dot_product_vv_vd) {
  vector_v v_v(3);
  vector_d v_d(3);
  v_v << 1, 3, -5;
  v_d << 4, -2, -1;
  
  var v = stan::agrad::dot_product(v_v, v_d);
  EXPECT_FLOAT_EQ (3, v.val());
}

TEST(agrad_matrix,dot_product_vd_vv) {
  vector_d v_d(3);
  vector_v v_v(3);
  v_d << 1, 3, -5;
  v_v << 4, -2, -1;
  
  var v = stan::agrad::dot_product(v_d, v_v);
  EXPECT_FLOAT_EQ (3, v.val());
}

TEST(agrad_matrix,dot_product_vv_rv) {
  vector_v v_v(3);
  row_vector_v r_v(3);
  v_v << 1, 3, -5;
  r_v << 4, -2, -1;
  
  var v = stan::agrad::dot_product(v_v, r_v);
  EXPECT_FLOAT_EQ (3, v.val());
}

TEST(agrad_matrix,determinant_vv_rd) {
  vector_v v_v(3);
  row_vector_d r_d(3);
  v_v << 1, 3, -5;
  r_d << 4, -2, -1;
  
  var v = stan::agrad::dot_product(v_v, r_d);
  EXPECT_FLOAT_EQ (3, v.val());
}

TEST(agrad_matrix,dot_product_vd_rv) {
  vector_d v_d(3);
  row_vector_v r_v(3);
  v_d << 1, 3, -5;
  r_v << 4, -2, -1;
  
  var v = stan::agrad::dot_product(v_d, r_v);
  EXPECT_FLOAT_EQ (3, v.val());
}
TEST(agrad_matrix,determinant_rv_vv) {
  row_vector_v r_v(3);
  vector_v v_v(3);
  r_v << 1, 3, -5;
  v_v << 4, -2, -1;
  
  var v = stan::agrad::dot_product(r_v, v_v);
  EXPECT_FLOAT_EQ (3, v.val());
}
TEST(agrad_matrix,determinant_rv_vd) {
  row_vector_v r_v(3);
  vector_d v_d(3);
  r_v << 1, 3, -5;
  v_d << 4, -2, -1;
  
  var v = stan::agrad::dot_product(r_v, v_d);
  EXPECT_FLOAT_EQ (3, v.val());
}
TEST(agrad_matrix,determinant_rd_vv) {
  row_vector_d r_d(3);
  vector_v v_v(3);
  r_d << 1, 3, -5;
  v_v << 4, -2, -1;
  
  var v = stan::agrad::dot_product(r_d, v_v);
  EXPECT_FLOAT_EQ (3, v.val());
}
TEST(agrad_matrix,determinant_rv_rv) {
  row_vector_v r_v1(3);
  row_vector_v r_v2(3);
  r_v1 << 1, 3, -5;
  r_v2 << 4, -2, -1;
  
  var v = stan::agrad::dot_product(r_v1, r_v2);
  EXPECT_FLOAT_EQ (3, v.val());

}
TEST(agrad_matrix,determinant_rv_rd) {
  row_vector_v r_v(3);
  row_vector_d r_d(3);
  r_v << 1, 3, -5;
  r_d << 4, -2, -1;
  
  var v = stan::agrad::dot_product(r_v, r_d);
  EXPECT_FLOAT_EQ (3, v.val());
}
TEST(agrad_matrix,determinant_rd_rv) {
  row_vector_d r_d(3);
  row_vector_v r_v(3);
  r_d << 1, 3, -5;
  r_v << 4, -2, -1;
  
  var v = stan::agrad::dot_product(r_d, r_v);
  EXPECT_FLOAT_EQ (3, v.val());
}
// end dot_product tests

// add tests
TEST(agrad_matrix, add_vector) {
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
TEST(agrad_matrix, add_row_vector) {
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
// end add tests

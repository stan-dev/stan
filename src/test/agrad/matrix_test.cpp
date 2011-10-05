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

  m.resize(2,3);
  EXPECT_DEATH (det = stan::agrad::determinant(m), "[[:print:]]*determinant");
}
// end determinant tests

// dot_product tests
TEST(agrad_matrix, dot_product_vector_vector) {
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

  vector_d vd_mis(2);
  vector_v vv_mis(4);
  EXPECT_DEATH (stan::agrad::dot_product(vd_1,   vd_mis), "[[:print:]]*dot_product");
  EXPECT_DEATH (stan::agrad::dot_product(vd_mis, vd_2),   "[[:print:]]*dot_product");
  EXPECT_DEATH (stan::agrad::dot_product(vv_1,   vd_mis), "[[:print:]]*dot_product");
  EXPECT_DEATH (stan::agrad::dot_product(vv_mis, vd_2),   "[[:print:]]*dot_product");
  EXPECT_DEATH (stan::agrad::dot_product(vd_1,   vv_mis), "[[:print:]]*dot_product");
  EXPECT_DEATH (stan::agrad::dot_product(vd_mis, vv_2),   "[[:print:]]*dot_product");
  EXPECT_DEATH (stan::agrad::dot_product(vv_1,   vv_mis), "[[:print:]]*dot_product");
  EXPECT_DEATH (stan::agrad::dot_product(vv_mis, vv_2),   "[[:print:]]*dot_product");
}
TEST(agrad_matrix, dot_product_rowvector_vector) {
  row_vector_d rvd_1(3);
  row_vector_v rvv_1(3);
  vector_d vd_2(3);
  vector_v vv_2(3);
  
  rvd_1 << 1, 3, -5;
  rvv_1 << 1, 3, -5;
  vd_2 << 4, -2, -1;
  vv_2 << 4, -2, -1;

  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(rvd_1, vd_2).val());
  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(rvv_1, vd_2).val());
  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(rvd_1, vv_2).val());
  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(rvv_1, vv_2).val());

  row_vector_d rvd_mis(2);
  row_vector_v rvv_mis(4);
  EXPECT_DEATH (stan::agrad::dot_product(rvd_mis, vd_2),   "[[:print:]]*dot_product");
  EXPECT_DEATH (stan::agrad::dot_product(rvv_mis, vd_2),   "[[:print:]]*dot_product");
  EXPECT_DEATH (stan::agrad::dot_product(rvd_mis, vv_2),   "[[:print:]]*dot_product");
  EXPECT_DEATH (stan::agrad::dot_product(rvv_mis, vv_2),   "[[:print:]]*dot_product");
}
TEST(agrad_matrix, dot_product_vector_rowvector) {
  vector_d vd_1(3);
  vector_v vv_1(3);
  row_vector_d rvd_2(3);
  row_vector_v rvv_2(3);
  
  vd_1 << 1, 3, -5;
  vv_1 << 1, 3, -5;
  rvd_2 << 4, -2, -1;
  rvv_2 << 4, -2, -1;

  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(vd_1, rvd_2).val());
  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(vv_1, rvd_2).val());
  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(vd_1, rvv_2).val());
  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(vv_1, rvv_2).val());

  row_vector_d rvd_mis(2);
  row_vector_v rvv_mis(4);
  EXPECT_DEATH (stan::agrad::dot_product(vd_1, rvd_mis),   "[[:print:]]*dot_product");
  EXPECT_DEATH (stan::agrad::dot_product(vd_1, rvv_mis),   "[[:print:]]*dot_product");
  EXPECT_DEATH (stan::agrad::dot_product(vv_1, rvd_mis),   "[[:print:]]*dot_product");
  EXPECT_DEATH (stan::agrad::dot_product(vv_1, rvv_mis),   "[[:print:]]*dot_product");
}
TEST(agrad_matrix, dot_product_rowvector_rowvector) {
  row_vector_d rvd_1(3), rvd_2(3);
  row_vector_v rvv_1(3), rvv_2(3);
  
  rvd_1 << 1, 3, -5;
  rvv_1 << 1, 3, -5;
  rvd_2 << 4, -2, -1;
  rvv_2 << 4, -2, -1;

  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(rvd_1, rvd_2).val());
  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(rvv_1, rvd_2).val());
  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(rvd_1, rvv_2).val());
  EXPECT_FLOAT_EQ (3, stan::agrad::dot_product(rvv_1, rvv_2).val());

  vector_d rvd_mis(2);
  vector_v rvv_mis(4);
  EXPECT_DEATH (stan::agrad::dot_product(rvd_1,   rvd_mis), "[[:print:]]*dot_product");
  EXPECT_DEATH (stan::agrad::dot_product(rvd_mis, rvd_2),   "[[:print:]]*dot_product");
  EXPECT_DEATH (stan::agrad::dot_product(rvv_1,   rvd_mis), "[[:print:]]*dot_product");
  EXPECT_DEATH (stan::agrad::dot_product(rvv_mis, rvd_2),   "[[:print:]]*dot_product");
  EXPECT_DEATH (stan::agrad::dot_product(rvd_1,   rvv_mis), "[[:print:]]*dot_product");
  EXPECT_DEATH (stan::agrad::dot_product(rvd_mis, rvv_2),   "[[:print:]]*dot_product");
  EXPECT_DEATH (stan::agrad::dot_product(rvv_1,   rvv_mis), "[[:print:]]*dot_product");
  EXPECT_DEATH (stan::agrad::dot_product(rvv_mis, rvv_2),   "[[:print:]]*dot_product");
}
// end dot_product tests

// add tests
TEST(agrad_matrix, add_vector) {
  vector_v expected_output(5), output;
  vector_d vd_1(5), vd_2(5);
  vector_v vv_1(5), vv_2(5);
  vector_d vd_mis(4);
  vector_v vv_mis(3);
  
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

  EXPECT_DEATH(output = stan::agrad::add(vv_mis, vv_2), "");
  EXPECT_DEATH(output = stan::agrad::add(vv_1, vv_mis), "");
  EXPECT_DEATH(output = stan::agrad::add(vv_mis, vd_2), "");
  EXPECT_DEATH(output = stan::agrad::add(vv_1, vd_mis), "");
  EXPECT_DEATH(output = stan::agrad::add(vd_mis, vv_2), "");
  EXPECT_DEATH(output = stan::agrad::add(vd_1, vv_mis), "");
  EXPECT_DEATH(output = stan::agrad::add(vd_mis, vd_2), "");
  EXPECT_DEATH(output = stan::agrad::add(vd_1, vd_mis), "");
}
TEST(agrad_matrix, add_row_vector) {
  row_vector_v expected_output(5), output;
  row_vector_d rvd_1(5), rvd_2(5);
  row_vector_v rvv_1(5), rvv_2(5);
  row_vector_d rvd_mis(10);
  row_vector_v rvv_mis(2);

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

  EXPECT_DEATH(output = stan::agrad::add(rvv_mis, rvv_2), "");
  EXPECT_DEATH(output = stan::agrad::add(rvv_1, rvv_mis), "");
  EXPECT_DEATH(output = stan::agrad::add(rvv_mis, rvd_2), "");
  EXPECT_DEATH(output = stan::agrad::add(rvv_1, rvd_mis), "");
  EXPECT_DEATH(output = stan::agrad::add(rvd_mis, rvv_2), "");
  EXPECT_DEATH(output = stan::agrad::add(rvd_1, rvv_mis), "");
  EXPECT_DEATH(output = stan::agrad::add(rvd_mis, rvd_2), "");
  EXPECT_DEATH(output = stan::agrad::add(rvd_1, rvd_mis), "");
}
TEST(agrad_matrix, add_matrix) {
  matrix_v expected_output(2,2), output;
  matrix_d md_1(2,2), md_2(2,2);
  matrix_v mv_1(2,2), mv_2(2,2);
  matrix_d md_mis (2, 3);
  matrix_v mv_mis (1, 1);

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

  EXPECT_DEATH(output = stan::agrad::add(mv_mis, mv_2), "");
  EXPECT_DEATH(output = stan::agrad::add(mv_1,   mv_mis), "");
  EXPECT_DEATH(output = stan::agrad::add(mv_mis, md_2), "");
  EXPECT_DEATH(output = stan::agrad::add(mv_1,   md_mis), "");
  EXPECT_DEATH(output = stan::agrad::add(md_mis, mv_2), "");
  EXPECT_DEATH(output = stan::agrad::add(md_1,   mv_mis), "");
  EXPECT_DEATH(output = stan::agrad::add(md_mis, md_2), "");
  EXPECT_DEATH(output = stan::agrad::add(md_1,   md_mis), "");
}
// end add tests

// subtract tests
TEST(agrad_matrix, subtract_vector) {
  vector_v expected_output(5), output;
  vector_d vd_1(5), vd_2(5);
  vector_v vv_1(5), vv_2(5);
  vector_d vd_mis(4);
  vector_v vv_mis(3);

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

  EXPECT_DEATH(output = stan::agrad::subtract(vv_mis, vv_2), "");
  EXPECT_DEATH(output = stan::agrad::subtract(vv_1,   vv_mis), "");
  EXPECT_DEATH(output = stan::agrad::subtract(vv_mis, vd_2), "");
  EXPECT_DEATH(output = stan::agrad::subtract(vv_1,   vd_mis), "");
  EXPECT_DEATH(output = stan::agrad::subtract(vd_mis, vv_2), "");
  EXPECT_DEATH(output = stan::agrad::subtract(vd_1,   vv_mis), "");
  EXPECT_DEATH(output = stan::agrad::subtract(vd_mis, vd_2), "");
  EXPECT_DEATH(output = stan::agrad::subtract(vd_1,   vd_mis), "");
}
TEST(agrad_matrix, subtract_row_vector) {
  row_vector_v expected_output(5), output;
  row_vector_d rvd_1(5), rvd_2(5);
  row_vector_v rvv_1(5), rvv_2(5);
  row_vector_d rvd_mis(10);
  row_vector_v rvv_mis(2);

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

  EXPECT_DEATH(output = stan::agrad::subtract(rvv_mis, rvv_2), "");
  EXPECT_DEATH(output = stan::agrad::subtract(rvv_1,   rvv_mis), "");
  EXPECT_DEATH(output = stan::agrad::subtract(rvv_mis, rvd_2), "");
  EXPECT_DEATH(output = stan::agrad::subtract(rvv_1,   rvd_mis), "");
  EXPECT_DEATH(output = stan::agrad::subtract(rvd_mis, rvv_2), "");
  EXPECT_DEATH(output = stan::agrad::subtract(rvd_1,   rvv_mis), "");
  EXPECT_DEATH(output = stan::agrad::subtract(rvd_mis, rvd_2), "");
  EXPECT_DEATH(output = stan::agrad::subtract(rvd_1,   rvd_mis), "");
}
TEST(agrad_matrix, subtract_matrix) {
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

  EXPECT_DEATH(output = stan::agrad::subtract(mv_mis, mv_2), "");
  EXPECT_DEATH(output = stan::agrad::subtract(mv_1,   mv_mis), "");
  EXPECT_DEATH(output = stan::agrad::subtract(mv_mis, md_2), "");
  EXPECT_DEATH(output = stan::agrad::subtract(mv_1,   md_mis), "");
  EXPECT_DEATH(output = stan::agrad::subtract(md_mis, mv_2), "");
  EXPECT_DEATH(output = stan::agrad::subtract(md_1,   mv_mis), "");
  EXPECT_DEATH(output = stan::agrad::subtract(md_mis, md_2), "");
  EXPECT_DEATH(output = stan::agrad::subtract(md_1,   md_mis), "");
}
// end subtract tests

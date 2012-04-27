#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <complex>

#include <stan/agrad/matrix.hpp>

using stan::agrad::var;

using stan::math::matrix_d;
using stan::agrad::matrix_v;
using stan::math::vector_d;
using stan::agrad::vector_v;
using stan::math::row_vector_d;
using stan::agrad::row_vector_v;

typedef stan::agrad::var AVAR;
typedef std::vector<AVAR> AVEC;
typedef std::vector<double> VEC;


AVEC createAVEC(AVAR x) {
  AVEC v;
  v.push_back(x);
  return v;
}
AVEC createAVEC(AVAR x1, AVAR x2) {
  AVEC v = createAVEC(x1);
  v.push_back(x2);
  return v;
}
AVEC createAVEC(AVAR x1, AVAR x2, AVAR x3) {
  AVEC v = createAVEC(x1,x2);
  v.push_back(x3);
  return v;
}
AVEC createAVEC(AVAR x1, AVAR x2, AVAR x3, AVAR x4) {
  AVEC v = createAVEC(x1,x2,x3);
  v.push_back(x4);
  return v;
}

VEC cgrad(AVAR f, AVAR x1) {
  AVEC x = createAVEC(x1);
  VEC g;
  f.grad(x,g);
  return g;
}

VEC cgrad(AVAR f, AVAR x1, AVAR x2) {
  AVEC x = createAVEC(x1,x2);
  VEC g;
  f.grad(x,g);
  return g;
}

VEC cgrad(AVAR f, AVAR x1, AVAR x2, AVAR x3) {
  AVEC x = createAVEC(x1,x2,x3);
  VEC g;
  f.grad(x,g);
  return g;
}

VEC cgrad(AVAR f, AVAR x1, AVAR x2, AVAR x3, AVAR x4) {
  AVEC x = createAVEC(x1,x2,x3,x4);
  VEC g;
  f.grad(x,g);
  return g;
}

VEC cgradvec(AVAR f, AVEC x) {
  VEC g;
  f.grad(x,g);
  return g;
}


// to_var tests
TEST(agrad_matrix,to_var__scalar) {
  double d = 5.0;
  var v = 5.0;
  stan::agrad::var var_x = stan::agrad::to_var(d);
  EXPECT_FLOAT_EQ(5.0, var_x.val());
  
  var_x = stan::agrad::to_var(v);
  EXPECT_FLOAT_EQ(5.0, var_x.val());
}
TEST(agrad_matrix,to_var__matrix) {
  matrix_d m_d(2,3);
  m_d << 0, 1, 2, 3, 4, 5;
  matrix_v m_v = stan::agrad::to_var(m_d);
  
  EXPECT_EQ(2, m_v.rows());
  EXPECT_EQ(3, m_v.cols());
  for (int ii = 0; ii < 2; ii++) 
    for (int jj = 0; jj < 3; jj++)
      EXPECT_FLOAT_EQ(ii*3 + jj, m_v(ii, jj).val());
}
TEST(agrad_matrix,to_var_ref__matrix) {
  matrix_d m_d(2,3);
  m_d << 0, 1, 2, 3, 4, 5;

  matrix_v m_v(5,5);
  EXPECT_EQ(5, m_v.rows());
  EXPECT_EQ(5, m_v.cols());

  stan::agrad::to_var(m_d, m_v);  
  EXPECT_EQ(2, m_v.rows());
  EXPECT_EQ(3, m_v.cols());
  EXPECT_FLOAT_EQ(0, m_v(0, 0).val());
  EXPECT_FLOAT_EQ(1, m_v(0, 1).val());
  EXPECT_FLOAT_EQ(2, m_v(0, 2).val());
  EXPECT_FLOAT_EQ(3, m_v(1, 0).val());
  EXPECT_FLOAT_EQ(4, m_v(1, 1).val());
  EXPECT_FLOAT_EQ(5, m_v(1, 2).val());
}
TEST(agrad_matrix,to_var__vector) {
  vector_d d(5);
  vector_v v(5);
  
  d << 1, 2, 3, 4, 5;
  v << 1, 2, 3, 4, 5;
  
  vector_v out = stan::agrad::to_var(d);
  EXPECT_FLOAT_EQ(1, out(0).val());
  EXPECT_FLOAT_EQ(2, out(1).val());
  EXPECT_FLOAT_EQ(3, out(2).val());
  EXPECT_FLOAT_EQ(4, out(3).val());
  EXPECT_FLOAT_EQ(5, out(4).val());

  out = stan::agrad::to_var(v);
  EXPECT_FLOAT_EQ(1, out(0).val());
  EXPECT_FLOAT_EQ(2, out(1).val());
  EXPECT_FLOAT_EQ(3, out(2).val());
  EXPECT_FLOAT_EQ(4, out(3).val());
  EXPECT_FLOAT_EQ(5, out(4).val());
}
TEST(agrad_matrix,to_var_ref__vector) {
  vector_d d(5);
  vector_v v(5);
  
  d << 1, 2, 3, 4, 5;
  v << 1, 2, 3, 4, 5;
  
  vector_v output;
  stan::agrad::to_var(d, output);
  EXPECT_FLOAT_EQ(1, output(0).val());
  EXPECT_FLOAT_EQ(2, output(1).val());
  EXPECT_FLOAT_EQ(3, output(2).val());
  EXPECT_FLOAT_EQ(4, output(3).val());
  EXPECT_FLOAT_EQ(5, output(4).val());

  stan::agrad::to_var(v, output);
  EXPECT_FLOAT_EQ(1, output(0).val());
  EXPECT_FLOAT_EQ(2, output(1).val());
  EXPECT_FLOAT_EQ(3, output(2).val());
  EXPECT_FLOAT_EQ(4, output(3).val());
  EXPECT_FLOAT_EQ(5, output(4).val());
}
TEST(agrad_matrix,to_var__rowvector) {
  row_vector_d d(5);
  row_vector_v v(5);
  
  d << 1, 2, 3, 4, 5;
  v << 1, 2, 3, 4, 5;
  
  row_vector_v output = stan::agrad::to_var(d);
  EXPECT_FLOAT_EQ(1, output(0).val());
  EXPECT_FLOAT_EQ(2, output(1).val());
  EXPECT_FLOAT_EQ(3, output(2).val());
  EXPECT_FLOAT_EQ(4, output(3).val());
  EXPECT_FLOAT_EQ(5, output(4).val());

  output.resize(0);
  output = stan::agrad::to_var(v);
  EXPECT_FLOAT_EQ(1, output(0).val());
  EXPECT_FLOAT_EQ(2, output(1).val());
  EXPECT_FLOAT_EQ(3, output(2).val());
  EXPECT_FLOAT_EQ(4, output(3).val());
  EXPECT_FLOAT_EQ(5, output(4).val());
}
TEST(agrad_matrix,to_var_ref__rowvector) {
  row_vector_d d(5);
  row_vector_v v(5);

  d << 1, 2, 3, 4, 5;
  v << 1, 2, 3, 4, 5;
 
  row_vector_v output;
  stan::agrad::to_var(d, output);
  EXPECT_FLOAT_EQ(1, output(0).val());
  EXPECT_FLOAT_EQ(2, output(1).val());
  EXPECT_FLOAT_EQ(3, output(2).val());
  EXPECT_FLOAT_EQ(4, output(3).val());
  EXPECT_FLOAT_EQ(5, output(4).val());

  output.resize(0);
  stan::agrad::to_var(d, output);
  EXPECT_FLOAT_EQ(1, output(0).val());
  EXPECT_FLOAT_EQ(2, output(1).val());
  EXPECT_FLOAT_EQ(3, output(2).val());
  EXPECT_FLOAT_EQ(4, output(3).val());
  EXPECT_FLOAT_EQ(5, output(4).val());
}
// end to_var tests

// rows tests
TEST(agrad_matrix,rows__vector) {
  vector_v v(5);
  v << 0, 1, 2, 3, 4;
  EXPECT_EQ(5U, stan::agrad::rows(v));
  
  v.resize(0);
  EXPECT_EQ(0U, stan::agrad::rows(v));
}
TEST(agrad_matrix,rows__rowvector) {
  row_vector_v rv(5);
  rv << 0, 1, 2, 3, 4;
  EXPECT_EQ(1U, stan::agrad::rows(rv));

  rv.resize(0);
  EXPECT_EQ(1U, stan::agrad::rows(rv));
}
TEST(agrad_matrix,rows__matrix) {
  matrix_v m(2,3);
  m << 0, 1, 2, 3, 4, 5;
  EXPECT_EQ(2U, stan::agrad::rows(m));
  
  m.resize(0,2);
  EXPECT_EQ(0U, stan::agrad::rows(m));
}
// end rows tests

// cols tests
TEST(agrad_matrix,cols__vector) {
  vector_v v(5);
  v << 0, 1, 2, 3, 4;
  EXPECT_EQ(1U, stan::agrad::cols(v));

  v.resize(0);
  EXPECT_EQ(1U, stan::agrad::cols(v));
}
TEST(agrad_matrix,cols__rowvector) {
  row_vector_v rv(5);
  rv << 0, 1, 2, 3, 4;
  EXPECT_EQ(5U, stan::agrad::cols(rv));
  
  rv.resize(0);
  EXPECT_EQ(0U, stan::agrad::cols(rv));
}
TEST(agrad_matrix,cols__matrix) {
  matrix_v m(2,3);
  m << 0, 1, 2, 3, 4, 5;
  EXPECT_EQ(3U, stan::agrad::cols(m));
  
  m.resize(5, 0);
  EXPECT_EQ(0U, stan::agrad::cols(m));
}
// end cols_tests

// determinant tests
TEST(agrad_matrix,determinant) {
  matrix_v v(2,2);
  v << 0, 1, 2, 3;

  var det;
  det = stan::agrad::determinant(v);
  EXPECT_FLOAT_EQ(-2, det.val());
}
TEST(agrad_matrix,deteriminant__exception) {
  matrix_v v(2,3);

  var det;
  EXPECT_THROW (det = stan::agrad::determinant(v), std::domain_error);
}
TEST(agrad_matrix,determinant_grad) {
  matrix_v X(2,2);
  AVAR a = 2.0;
  AVAR b = 3.0;
  AVAR c = 5.0;
  AVAR d = 7.0;
  X << a, b, c, d;

  AVEC x = createAVEC(a,b,c,d);

  AVAR f = X.determinant();

  // det = ad - bc
  EXPECT_FLOAT_EQ(-1.0,f.val());

  std::vector<double> g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(7.0,g[0]);
  EXPECT_FLOAT_EQ(-5.0,g[1]);
  EXPECT_FLOAT_EQ(-3.0,g[2]);
  EXPECT_FLOAT_EQ(2.0,g[3]);
}
TEST(agrad_matrix,determinant3by3) {
  // just test it can handle it
  matrix_v Z(9,9);
  for (int i = 0; i < 9; ++i)
    for (int j = 0; j < 9; ++j)
      Z(i,j) = i * j + 1;
  AVAR h = Z.determinant();
  h = h; // supresses set but not used warning
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

  EXPECT_FLOAT_EQ(3, stan::agrad::dot_product(vv_1, vd_2).val());
  EXPECT_FLOAT_EQ(3, stan::agrad::dot_product(vd_1, vv_2).val());
  EXPECT_FLOAT_EQ(3, stan::agrad::dot_product(vv_1, vv_2).val());
}
TEST(agrad_matrix, dot_product__vector_vector__exception) {
  vector_d d1(3);
  vector_v v1(3);
  vector_d d2(2);
  vector_v v2(4);

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

  EXPECT_FLOAT_EQ(3, stan::agrad::dot_product(v1, d2).val());
  EXPECT_FLOAT_EQ(3, stan::agrad::dot_product(d1, v2).val());
  EXPECT_FLOAT_EQ(3, stan::agrad::dot_product(v1, v2).val());
}
TEST(agrad_matrix, dot_product__rowvector_vector__exception) {
  row_vector_d d1(3);
  row_vector_v v1(3);
  vector_d d2(2);
  vector_v v2(4);

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
  
  EXPECT_FLOAT_EQ(3, stan::agrad::dot_product(v1, d2).val());
  EXPECT_FLOAT_EQ(3, stan::agrad::dot_product(d1, v2).val());
  EXPECT_FLOAT_EQ(3, stan::agrad::dot_product(v1, v2).val());
}
TEST(agrad_matrix, dot_product__vector_rowvector__exception) {
  vector_d d1(3);
  vector_v v1(3);
  row_vector_d d2(2);
  row_vector_v v2(4);

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

  EXPECT_FLOAT_EQ(3, stan::agrad::dot_product(v1, d2).val());
  EXPECT_FLOAT_EQ(3, stan::agrad::dot_product(d1, v2).val());
  EXPECT_FLOAT_EQ(3, stan::agrad::dot_product(v1, v2).val());
}
TEST(agrad_matrix, dot_product__rowvector_rowvector__exception) {
  row_vector_d d1(3), d2(2);
  row_vector_v v1(3), v2(4);

  EXPECT_THROW (stan::agrad::dot_product(v1, d2), std::invalid_argument);
  EXPECT_THROW (stan::agrad::dot_product(d1, v2), std::invalid_argument);
  EXPECT_THROW (stan::agrad::dot_product(v1, v2), std::invalid_argument);
}
// end dot_product tests

// exp tests
TEST(agrad_matrix, exp__matrix) {
  matrix_d expected_output(2,2);
  matrix_v mv(2,2), output;
  int i,j;

  mv << 1, 2, 3, 4;
  expected_output << std::exp(1), std::exp(2), std::exp(3), std::exp(4);
  output = stan::agrad::exp(mv);

  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(expected_output(i,j), output(i,j).val());
}

// log tests
TEST(agrad_matrix, log__matrix) {
  matrix_d expected_output(2,2);
  matrix_v mv(2,2), output;
  int i,j;

  mv << 1, 2, 3, 4;
  expected_output << std::log(1), std::log(2), std::log(3), std::log(4);
  output = stan::agrad::log(mv);

  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(expected_output(i,j), output(i,j).val());
}

// scalar add/subtract tests
TEST(agrad_matrix,add__scalar) {
  matrix_v v(2,2);
  v << 1, 2, 3, 4;
  matrix_v result;

  result = stan::agrad::add(2.0,v);
  EXPECT_FLOAT_EQ(3.0,result(0,0).val());
  EXPECT_FLOAT_EQ(4.0,result(0,1).val());
  EXPECT_FLOAT_EQ(5.0,result(1,0).val());
  EXPECT_FLOAT_EQ(6.0,result(1,1).val());

  result = stan::agrad::add(v,2.0);
  EXPECT_FLOAT_EQ(3.0,result(0,0).val());
  EXPECT_FLOAT_EQ(4.0,result(0,1).val());
  EXPECT_FLOAT_EQ(5.0,result(1,0).val());
  EXPECT_FLOAT_EQ(6.0,result(1,1).val());
}

TEST(agrad_matrix,subtract__scalar) {
  matrix_v v(2,2);
  v << 1, 2, 3, 4;
  matrix_v result;

  result = stan::agrad::subtract(2.0,v);
  EXPECT_FLOAT_EQ(1.0,result(0,0).val());
  EXPECT_FLOAT_EQ(0.0,result(0,1).val());
  EXPECT_FLOAT_EQ(-1.0,result(1,0).val());
  EXPECT_FLOAT_EQ(-2.0,result(1,1).val());

  result = stan::agrad::subtract(v,2.0);
  EXPECT_FLOAT_EQ(-1.0,result(0,0).val());
  EXPECT_FLOAT_EQ(0.0,result(0,1).val());
  EXPECT_FLOAT_EQ(1.0,result(1,0).val());
  EXPECT_FLOAT_EQ(2.0,result(1,1).val());
}

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
  EXPECT_FLOAT_EQ(expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ(expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ(expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ(expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ(expected_output(4).val(), output(4).val());  

  output = stan::agrad::add(vv_1, vd_2);
  EXPECT_FLOAT_EQ(expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ(expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ(expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ(expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ(expected_output(4).val(), output(4).val());  

  output = stan::agrad::add(vd_1, vv_2);
  EXPECT_FLOAT_EQ(expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ(expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ(expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ(expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ(expected_output(4).val(), output(4).val());  

  output = stan::agrad::add(vv_1, vv_2);
  EXPECT_FLOAT_EQ(expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ(expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ(expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ(expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ(expected_output(4).val(), output(4).val());  
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
  EXPECT_FLOAT_EQ(expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ(expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ(expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ(expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ(expected_output(4).val(), output(4).val());  

  output = stan::agrad::add(rvv_1, rvd_2);
  EXPECT_FLOAT_EQ(expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ(expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ(expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ(expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ(expected_output(4).val(), output(4).val());  

  output = stan::agrad::add(rvd_1, rvv_2);
  EXPECT_FLOAT_EQ(expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ(expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ(expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ(expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ(expected_output(4).val(), output(4).val());  

  output = stan::agrad::add(rvv_1, rvv_2);
  EXPECT_FLOAT_EQ(expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ(expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ(expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ(expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ(expected_output(4).val(), output(4).val());  
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
  EXPECT_FLOAT_EQ(expected_output(0,0).val(), output(0,0).val());
  EXPECT_FLOAT_EQ(expected_output(0,1).val(), output(0,1).val());
  EXPECT_FLOAT_EQ(expected_output(1,0).val(), output(1,0).val());
  EXPECT_FLOAT_EQ(expected_output(1,1).val(), output(1,1).val());

  output = stan::agrad::add(mv_1, md_2);
  EXPECT_FLOAT_EQ(expected_output(0,0).val(), output(0,0).val());
  EXPECT_FLOAT_EQ(expected_output(0,1).val(), output(0,1).val());
  EXPECT_FLOAT_EQ(expected_output(1,0).val(), output(1,0).val());
  EXPECT_FLOAT_EQ(expected_output(1,1).val(), output(1,1).val());

  output = stan::agrad::add(md_1, mv_2);
  EXPECT_FLOAT_EQ(expected_output(0,0).val(), output(0,0).val());
  EXPECT_FLOAT_EQ(expected_output(0,1).val(), output(0,1).val());
  EXPECT_FLOAT_EQ(expected_output(1,0).val(), output(1,0).val());
  EXPECT_FLOAT_EQ(expected_output(1,1).val(), output(1,1).val());

  output = stan::agrad::add(mv_1, mv_2);
  EXPECT_FLOAT_EQ(expected_output(0,0).val(), output(0,0).val());
  EXPECT_FLOAT_EQ(expected_output(0,1).val(), output(0,1).val());
  EXPECT_FLOAT_EQ(expected_output(1,0).val(), output(1,0).val());
  EXPECT_FLOAT_EQ(expected_output(1,1).val(), output(1,1).val());
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
  EXPECT_FLOAT_EQ(expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ(expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ(expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ(expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ(expected_output(4).val(), output(4).val());  

  output = stan::agrad::subtract(vv_1, vd_2);
  EXPECT_FLOAT_EQ(expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ(expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ(expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ(expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ(expected_output(4).val(), output(4).val());  

  output = stan::agrad::subtract(vd_1, vv_2);
  EXPECT_FLOAT_EQ(expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ(expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ(expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ(expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ(expected_output(4).val(), output(4).val());  

  output = stan::agrad::subtract(vv_1, vv_2);
  EXPECT_FLOAT_EQ(expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ(expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ(expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ(expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ(expected_output(4).val(), output(4).val());  
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
  EXPECT_FLOAT_EQ(expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ(expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ(expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ(expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ(expected_output(4).val(), output(4).val());  

  output = stan::agrad::subtract(rvv_1, rvd_2);
  EXPECT_FLOAT_EQ(expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ(expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ(expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ(expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ(expected_output(4).val(), output(4).val());  

  output = stan::agrad::subtract(rvd_1, rvv_2);
  EXPECT_FLOAT_EQ(expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ(expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ(expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ(expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ(expected_output(4).val(), output(4).val());  

  output = stan::agrad::subtract(rvv_1, rvv_2);
  EXPECT_FLOAT_EQ(expected_output(0).val(), output(0).val());
  EXPECT_FLOAT_EQ(expected_output(1).val(), output(1).val());
  EXPECT_FLOAT_EQ(expected_output(2).val(), output(2).val());
  EXPECT_FLOAT_EQ(expected_output(3).val(), output(3).val());
  EXPECT_FLOAT_EQ(expected_output(4).val(), output(4).val());  
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
  EXPECT_FLOAT_EQ(expected_output(0,0).val(), output(0,0).val());
  EXPECT_FLOAT_EQ(expected_output(0,1).val(), output(0,1).val());
  EXPECT_FLOAT_EQ(expected_output(1,0).val(), output(1,0).val());
  EXPECT_FLOAT_EQ(expected_output(1,1).val(), output(1,1).val());

  output = stan::agrad::subtract(mv_1, md_2);
  EXPECT_FLOAT_EQ(expected_output(0,0).val(), output(0,0).val());
  EXPECT_FLOAT_EQ(expected_output(0,1).val(), output(0,1).val());
  EXPECT_FLOAT_EQ(expected_output(1,0).val(), output(1,0).val());
  EXPECT_FLOAT_EQ(expected_output(1,1).val(), output(1,1).val());

  output = stan::agrad::subtract(md_1, mv_2);
  EXPECT_FLOAT_EQ(expected_output(0,0).val(), output(0,0).val());
  EXPECT_FLOAT_EQ(expected_output(0,1).val(), output(0,1).val());
  EXPECT_FLOAT_EQ(expected_output(1,0).val(), output(1,0).val());
  EXPECT_FLOAT_EQ(expected_output(1,1).val(), output(1,1).val());

  output = stan::agrad::subtract(mv_1, mv_2);
  EXPECT_FLOAT_EQ(expected_output(0,0).val(), output(0,0).val());
  EXPECT_FLOAT_EQ(expected_output(0,1).val(), output(0,1).val());
  EXPECT_FLOAT_EQ(expected_output(1,0).val(), output(1,0).val());
  EXPECT_FLOAT_EQ(expected_output(1,1).val(), output(1,1).val());
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
  
  EXPECT_FLOAT_EQ(-10, stan::agrad::minus(x).val());
  EXPECT_FLOAT_EQ(-11, stan::agrad::minus(v).val());
}
TEST(agrad_matrix, minus__vector) {
  vector_d d(3);
  vector_v v(3);

  d << -100, 0, 1;
  v << -100, 0, 1;
  
  vector_v output;
  output = stan::agrad::minus (d);
  EXPECT_FLOAT_EQ(100, output[0].val());
  EXPECT_FLOAT_EQ(0, output[1].val());
  EXPECT_FLOAT_EQ(-1, output[2].val());

  output = stan::agrad::minus (v);
  EXPECT_FLOAT_EQ(100, output[0].val());
  EXPECT_FLOAT_EQ(0, output[1].val());
  EXPECT_FLOAT_EQ(-1, output[2].val());
}
TEST(agrad_matrix, minus__rowvector) {
  row_vector_d d(3);
  row_vector_v v(3);

  d << -100, 0, 1;
  v << -100, 0, 1;
  
  row_vector_v output;
  output = stan::agrad::minus (d);
  EXPECT_FLOAT_EQ(100, output[0].val());
  EXPECT_FLOAT_EQ(0, output[1].val());
  EXPECT_FLOAT_EQ(-1, output[2].val());

  output = stan::agrad::minus (v);
  EXPECT_FLOAT_EQ(100, output[0].val());
  EXPECT_FLOAT_EQ(0, output[1].val());
  EXPECT_FLOAT_EQ(-1, output[2].val());
}
TEST(agrad_matrix, minus__matrix) {
  matrix_d d(2, 3);
  matrix_v v(2, 3);

  d << -100, 0, 1, 20, -40, 2;
  v << -100, 0, 1, 20, -40, 2;

  matrix_v output;
  output = stan::agrad::minus (d);
  EXPECT_FLOAT_EQ(100, output(0,0).val());
  EXPECT_FLOAT_EQ(  0, output(0,1).val());
  EXPECT_FLOAT_EQ( -1, output(0,2).val());
  EXPECT_FLOAT_EQ(-20, output(1,0).val());
  EXPECT_FLOAT_EQ( 40, output(1,1).val());
  EXPECT_FLOAT_EQ( -2, output(1,2).val());

  output = stan::agrad::minus (v);
  EXPECT_FLOAT_EQ(100, output(0,0).val());
  EXPECT_FLOAT_EQ(  0, output(0,1).val());
  EXPECT_FLOAT_EQ( -1, output(0,2).val());
  EXPECT_FLOAT_EQ(-20, output(1,0).val());
  EXPECT_FLOAT_EQ( 40, output(1,1).val());
  EXPECT_FLOAT_EQ( -2, output(1,2).val());
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
  
  EXPECT_FLOAT_EQ(-5, stan::agrad::divide(d1, d2).val());
  EXPECT_FLOAT_EQ(-5, stan::agrad::divide(d1, v2).val());
  EXPECT_FLOAT_EQ(-5, stan::agrad::divide(v1, d2).val());
  EXPECT_FLOAT_EQ(-5, stan::agrad::divide(v1, v2).val());

  d2 = 0;
  v2 = 0;

  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), stan::agrad::divide(d1, d2).val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), stan::agrad::divide(d1, v2).val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), stan::agrad::divide(v1, d2).val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), stan::agrad::divide(v1, v2).val());

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
  EXPECT_FLOAT_EQ(-50, output(0).val());
  EXPECT_FLOAT_EQ(  0, output(1).val());
  EXPECT_FLOAT_EQ(1.5, output(2).val());

  output = stan::agrad::divide(d1, v2);
  EXPECT_FLOAT_EQ(-50, output(0).val());
  EXPECT_FLOAT_EQ(  0, output(1).val());
  EXPECT_FLOAT_EQ(1.5, output(2).val());

  output = stan::agrad::divide(v1, d2);
  EXPECT_FLOAT_EQ(-50, output(0).val());
  EXPECT_FLOAT_EQ(  0, output(1).val());
  EXPECT_FLOAT_EQ(1.5, output(2).val());

  output = stan::agrad::divide(v1, v2);
  EXPECT_FLOAT_EQ(-50, output(0).val());
  EXPECT_FLOAT_EQ(  0, output(1).val());
  EXPECT_FLOAT_EQ(1.5, output(2).val());


  d2 = 0;
  v2 = 0;
  output = stan::agrad::divide(d1, d2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0).val());
  EXPECT_TRUE (std::isnan(output(1).val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output(2).val());

  output = stan::agrad::divide(d1, v2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0).val());
  EXPECT_TRUE (std::isnan(output(1).val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output(2).val());

  output = stan::agrad::divide(v1, d2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0).val());
  EXPECT_TRUE (std::isnan(output(1).val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output(2).val());

  output = stan::agrad::divide(v1, v2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0).val());
  EXPECT_TRUE (std::isnan(output(1).val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output(2).val());
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
  EXPECT_FLOAT_EQ(-50, output(0).val());
  EXPECT_FLOAT_EQ(  0, output(1).val());
  EXPECT_FLOAT_EQ(1.5, output(2).val());

  output = stan::agrad::divide(d1, v2);
  EXPECT_FLOAT_EQ(-50, output(0).val());
  EXPECT_FLOAT_EQ(  0, output(1).val());
  EXPECT_FLOAT_EQ(1.5, output(2).val());

  output = stan::agrad::divide(v1, d2);
  EXPECT_FLOAT_EQ(-50, output(0).val());
  EXPECT_FLOAT_EQ(  0, output(1).val());
  EXPECT_FLOAT_EQ(1.5, output(2).val());

  output = stan::agrad::divide(v1, v2);
  EXPECT_FLOAT_EQ(-50, output(0).val());
  EXPECT_FLOAT_EQ(  0, output(1).val());
  EXPECT_FLOAT_EQ(1.5, output(2).val());


  d2 = 0;
  v2 = 0;
  output = stan::agrad::divide(d1, d2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0).val());
  EXPECT_TRUE (std::isnan(output(1).val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output(2).val());

  output = stan::agrad::divide(d1, v2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0).val());
  EXPECT_TRUE (std::isnan(output(1).val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output(2).val());

  output = stan::agrad::divide(v1, d2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0).val());
  EXPECT_TRUE (std::isnan(output(1).val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output(2).val());

  output = stan::agrad::divide(v1, v2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0).val());
  EXPECT_TRUE (std::isnan(output(1).val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output(2).val());
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
  EXPECT_FLOAT_EQ(-50, output(0,0).val());
  EXPECT_FLOAT_EQ(  0, output(0,1).val());
  EXPECT_FLOAT_EQ(1.5, output(1,0).val());
  EXPECT_FLOAT_EQ( -2, output(1,1).val());

  output = stan::agrad::divide(d1, v2);
  EXPECT_FLOAT_EQ(-50, output(0,0).val());
  EXPECT_FLOAT_EQ(  0, output(0,1).val());
  EXPECT_FLOAT_EQ(1.5, output(1,0).val());
  EXPECT_FLOAT_EQ( -2, output(1,1).val());
  
  output = stan::agrad::divide(v1, d2);
  EXPECT_FLOAT_EQ(-50, output(0,0).val());
  EXPECT_FLOAT_EQ(  0, output(0,1).val());
  EXPECT_FLOAT_EQ(1.5, output(1,0).val());
  EXPECT_FLOAT_EQ( -2, output(1,1).val());
  
  output = stan::agrad::divide(v1, v2);
  EXPECT_FLOAT_EQ(-50, output(0,0).val());
  EXPECT_FLOAT_EQ(  0, output(0,1).val());
  EXPECT_FLOAT_EQ(1.5, output(1,0).val());
  EXPECT_FLOAT_EQ( -2, output(1,1).val());

  d2 = 0;
  v2 = 0;
  output = stan::agrad::divide(d1, d2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0,0).val());
  EXPECT_TRUE (std::isnan(output(0,1).val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output(1,0).val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(1,1).val());

  output = stan::agrad::divide(d1, v2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0,0).val());
  EXPECT_TRUE (std::isnan(output(0,1).val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output(1,0).val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(1,1).val());

  output = stan::agrad::divide(v1, d2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0,0).val());
  EXPECT_TRUE (std::isnan(output(0,1).val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output(1,0).val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(1,1).val());

  output = stan::agrad::divide(v1, v2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0,0).val());
  EXPECT_TRUE (std::isnan(output(0,1).val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output(1,0).val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(1,1).val());
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
  EXPECT_FLOAT_EQ(-3, output.val());
                   
  output = stan::agrad::min(v1);
  EXPECT_FLOAT_EQ(-3, output.val());
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
  EXPECT_FLOAT_EQ(-3, output.val());
                   
  output = stan::agrad::min(v1);
  EXPECT_FLOAT_EQ(-3, output.val());
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
  EXPECT_FLOAT_EQ(-3, output.val());
                   
  output = stan::agrad::min(v1);
  EXPECT_FLOAT_EQ(-3, output.val());
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
  EXPECT_FLOAT_EQ(100, output.val());
                   
  output = stan::agrad::max(v1);
  EXPECT_FLOAT_EQ(100, output.val());
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
  EXPECT_FLOAT_EQ(100, output.val());
                   
  output = stan::agrad::max(v1);
  EXPECT_FLOAT_EQ(100, output.val());
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
  EXPECT_FLOAT_EQ(100, output.val());
                   
  output = stan::agrad::max(v1);
  EXPECT_FLOAT_EQ(100, output.val());
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
  EXPECT_FLOAT_EQ(97.0/3.0, output.val());
                   
  output = stan::agrad::mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val());
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
  EXPECT_FLOAT_EQ(97.0/3.0, output.val());
                   
  output = stan::agrad::mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val());
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
  EXPECT_FLOAT_EQ(97.0/3.0, output.val());
                   
  output = stan::agrad::mean(v1);
  EXPECT_FLOAT_EQ(97.0/3.0, output.val());
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
  EXPECT_FLOAT_EQ(17.5/5.0, output.val());
                   
  output = stan::agrad::variance(v1);
  EXPECT_FLOAT_EQ(17.5/5.0, output.val());

  d1.resize(1);
  v1.resize(1);
  output = stan::agrad::variance(d1);
  EXPECT_FLOAT_EQ(0.0, output.val());
  output = stan::agrad::variance(v1);
  EXPECT_FLOAT_EQ(0.0, output.val());  
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
  EXPECT_FLOAT_EQ(17.5/5.0, output.val());
                   
  output = stan::agrad::variance(v1);
  EXPECT_FLOAT_EQ(17.5/5.0, output.val());

  d1.resize(1);
  v1.resize(1);
  output = stan::agrad::variance(d1);
  EXPECT_FLOAT_EQ(0.0, output.val());
  output = stan::agrad::variance(v1);
  EXPECT_FLOAT_EQ(0.0, output.val());  
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
  EXPECT_FLOAT_EQ(17.5/5.0, output.val());
                   
  output = stan::agrad::variance(v1);
  EXPECT_FLOAT_EQ(17.5/5.0, output.val());

  d1.resize(1,1);
  v1.resize(1,1);
  output = stan::agrad::variance(d1);
  EXPECT_FLOAT_EQ(0.0, output.val());
  output = stan::agrad::variance(v1);
  EXPECT_FLOAT_EQ(0.0, output.val());  
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
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), output.val());
                   
  output = stan::agrad::sd(v1);
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), output.val());
  
  d1.resize(1);
  v1.resize(1);
  output = stan::agrad::sd(d1);
  EXPECT_FLOAT_EQ(0.0, output.val());
  output = stan::agrad::sd(v1);
  EXPECT_FLOAT_EQ(0.0, output.val());
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
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), output.val());
                   
  output = stan::agrad::sd(v1);
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), output.val());

  d1.resize(1);
  v1.resize(1);
  output = stan::agrad::sd(d1);
  EXPECT_FLOAT_EQ(0.0, output.val());
  output = stan::agrad::sd(v1);
  EXPECT_FLOAT_EQ(0.0, output.val());
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
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), output.val());
                   
  output = stan::agrad::sd(v1);
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), output.val());

  d1.resize(1, 1);
  v1.resize(1, 1);
  output = stan::agrad::sd(d1);
  EXPECT_FLOAT_EQ(0.0, output.val());
  output = stan::agrad::sd(v1);
  EXPECT_FLOAT_EQ(0.0, output.val());
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
  EXPECT_FLOAT_EQ(21.0, output.val());
                   
  output = stan::agrad::sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val());

  d.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ(0.0, stan::agrad::sum(d).val());
  EXPECT_FLOAT_EQ(0.0, stan::agrad::sum(v).val());
}
TEST (agrad_matrix, sum__rowvector) {
  row_vector_d d(6);
  row_vector_v v(6);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
  
  var output;
  output = stan::agrad::sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val());
                   
  output = stan::agrad::sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val());

  d.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ(0.0, stan::agrad::sum(d).val());
  EXPECT_FLOAT_EQ(0.0, stan::agrad::sum(v).val());
}
TEST (agrad_matrix, sum__matrix) {
  matrix_d d(2, 3);
  matrix_v v(2, 3);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
  
  var output;
  output = stan::agrad::sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val());
                   
  output = stan::agrad::sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val());

  d.resize(0, 0);
  v.resize(0, 0);
  EXPECT_FLOAT_EQ(0.0, stan::agrad::sum(d).val());
  EXPECT_FLOAT_EQ(0.0, stan::agrad::sum(v).val());
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
  
  EXPECT_FLOAT_EQ(-20.0, stan::agrad::multiply(d1, v2).val());
  EXPECT_FLOAT_EQ(-20.0, stan::agrad::multiply(v1, d2).val());
  EXPECT_FLOAT_EQ(-20.0, stan::agrad::multiply(v1, v2).val());
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
  output = stan::agrad::multiply(d1, v2);
  EXPECT_FLOAT_EQ(-200, output(0).val());
  EXPECT_FLOAT_EQ(   0, output(1).val());
  EXPECT_FLOAT_EQ(   6, output(2).val());

  output = stan::agrad::multiply(v1, d2);
  EXPECT_FLOAT_EQ(-200, output(0).val());
  EXPECT_FLOAT_EQ(   0, output(1).val());
  EXPECT_FLOAT_EQ(   6, output(2).val());

  output = stan::agrad::multiply(v1, v2);
  EXPECT_FLOAT_EQ(-200, output(0).val());
  EXPECT_FLOAT_EQ(   0, output(1).val());
  EXPECT_FLOAT_EQ(   6, output(2).val());
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
  output = stan::agrad::multiply(d1, v2);
  EXPECT_FLOAT_EQ(-200, output(0).val());
  EXPECT_FLOAT_EQ(   0, output(1).val());
  EXPECT_FLOAT_EQ(   6, output(2).val());

  output = stan::agrad::multiply(v1, d2);
  EXPECT_FLOAT_EQ(-200, output(0).val());
  EXPECT_FLOAT_EQ(   0, output(1).val());
  EXPECT_FLOAT_EQ(   6, output(2).val());

  output = stan::agrad::multiply(v1, v2);
  EXPECT_FLOAT_EQ(-200, output(0).val());
  EXPECT_FLOAT_EQ(   0, output(1).val());
  EXPECT_FLOAT_EQ(   6, output(2).val());
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
  output = stan::agrad::multiply(d1, v2);
  EXPECT_FLOAT_EQ(-200, output(0,0).val());
  EXPECT_FLOAT_EQ(   0, output(0,1).val());
  EXPECT_FLOAT_EQ(   6, output(1,0).val());
  EXPECT_FLOAT_EQ(  -8, output(1,1).val());

  output = stan::agrad::multiply(v1, d2);
  EXPECT_FLOAT_EQ(-200, output(0,0).val());
  EXPECT_FLOAT_EQ(   0, output(0,1).val());
  EXPECT_FLOAT_EQ(   6, output(1,0).val());
  EXPECT_FLOAT_EQ(  -8, output(1,1).val());
 
  output = stan::agrad::multiply(v1, v2);
  EXPECT_FLOAT_EQ(-200, output(0,0).val());
  EXPECT_FLOAT_EQ(   0, output(0,1).val());
  EXPECT_FLOAT_EQ(   6, output(1,0).val());
  EXPECT_FLOAT_EQ(  -8, output(1,1).val());
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

  EXPECT_FLOAT_EQ(3, stan::agrad::multiply(v1, v2).val());
  EXPECT_FLOAT_EQ(3, stan::agrad::multiply(v1, d2).val());
  EXPECT_FLOAT_EQ(3, stan::agrad::multiply(d1, v2).val());
  
  d1.resize(1);
  v1.resize(1);
  EXPECT_THROW(stan::agrad::multiply(v1, v2), std::invalid_argument);
  EXPECT_THROW(stan::agrad::multiply(v1, d2), std::invalid_argument);
  EXPECT_THROW(stan::agrad::multiply(d1, v2), std::invalid_argument);
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
  
  output = stan::agrad::multiply(v1, d2);
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
  
  output = stan::agrad::multiply(d1, v2);
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
  EXPECT_EQ(3, output.size());
  EXPECT_FLOAT_EQ(10, output(0).val());
  EXPECT_FLOAT_EQ(26, output(1).val());
  EXPECT_FLOAT_EQ( 0, output(2).val());

  
  output = stan::agrad::multiply(v1, d2);
  EXPECT_EQ(3, output.size());
  EXPECT_FLOAT_EQ(10, output(0).val());
  EXPECT_FLOAT_EQ(26, output(1).val());
  EXPECT_FLOAT_EQ( 0, output(2).val());
  
  output = stan::agrad::multiply(d1, v2);
  EXPECT_EQ(3, output.size());
  EXPECT_FLOAT_EQ(10, output(0).val());
  EXPECT_FLOAT_EQ(26, output(1).val());
  EXPECT_FLOAT_EQ( 0, output(2).val());
}
TEST(agrad_matrix, multiply__matrix_vector__exception) {
  matrix_d d1(3,2);
  matrix_v v1(3,2);
  vector_d d2(4);
  vector_v v2(4);
  EXPECT_THROW(stan::agrad::multiply(v1, v2), std::invalid_argument);
  EXPECT_THROW(stan::agrad::multiply(v1, d2), std::invalid_argument);
  EXPECT_THROW(stan::agrad::multiply(d1, v2), std::invalid_argument);
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
  EXPECT_EQ(2, output.size());
  EXPECT_FLOAT_EQ(-24, output(0).val());
  EXPECT_FLOAT_EQ(  9, output(1).val());

  output = stan::agrad::multiply(v1, d2);
  EXPECT_EQ(2, output.size());
  EXPECT_FLOAT_EQ(-24, output(0).val());
  EXPECT_FLOAT_EQ(  9, output(1).val());
  
  output = stan::agrad::multiply(d1, v2);
  EXPECT_EQ(2, output.size());
  EXPECT_FLOAT_EQ(-24, output(0).val());
  EXPECT_FLOAT_EQ(  9, output(1).val());
}
TEST(agrad_matrix, multiply__rowvector_matrix__exception) {
  row_vector_d d1(4);
  row_vector_v v1(4);
  matrix_d d2(3,2);
  matrix_v v2(3,2);
  EXPECT_THROW(stan::agrad::multiply(v1, v2), std::invalid_argument);
  EXPECT_THROW(stan::agrad::multiply(v1, d2), std::invalid_argument);
  EXPECT_THROW(stan::agrad::multiply(d1, v2), std::invalid_argument);
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
  EXPECT_EQ(2, output.rows());
  EXPECT_EQ(2, output.cols());
  EXPECT_FLOAT_EQ(-117, output(0,0).val());
  EXPECT_FLOAT_EQ( 120, output(0,1).val());
  EXPECT_FLOAT_EQ( 157, output(1,0).val());
  EXPECT_FLOAT_EQ( 135, output(1,1).val());

  output = stan::agrad::multiply(v1, d2);
  EXPECT_EQ(2, output.rows());
  EXPECT_EQ(2, output.cols());
  EXPECT_FLOAT_EQ(-117, output(0,0).val());
  EXPECT_FLOAT_EQ( 120, output(0,1).val());
  EXPECT_FLOAT_EQ( 157, output(1,0).val());
  EXPECT_FLOAT_EQ( 135, output(1,1).val());
  
  output = stan::agrad::multiply(d1, v2);
  EXPECT_EQ(2, output.rows());
  EXPECT_EQ(2, output.cols());
  EXPECT_FLOAT_EQ(-117, output(0,0).val());
  EXPECT_FLOAT_EQ( 120, output(0,1).val());
  EXPECT_FLOAT_EQ( 157, output(1,0).val());
  EXPECT_FLOAT_EQ( 135, output(1,1).val());
}
TEST(agrad_matrix, multiply__matrix_matrix__exception) {
  matrix_d d1(2,2);
  matrix_v v1(2,2);
  matrix_d d2(3,2);
  matrix_v v2(3,2);

  EXPECT_THROW(stan::agrad::multiply(v1, v2), std::invalid_argument);
  EXPECT_THROW(stan::agrad::multiply(v1, d2), std::invalid_argument);
  EXPECT_THROW(stan::agrad::multiply(d1, v2), std::invalid_argument);
}
// end multiply tests



TEST(agrad_matrix,transpose_matrix) {
  matrix_v a(2,3);
  a << -1.0, 2.0, -3.0, 
        5.0, 10.0, 100.0;
  
  AVEC x = createAVEC(a(0,0), a(0,2), a(1,1));
  
  matrix_v c = transpose(a);
  EXPECT_FLOAT_EQ(-1.0,c(0,0).val());
  EXPECT_FLOAT_EQ(10.0,c(1,1).val());
  EXPECT_FLOAT_EQ(-3.0,c(2,0).val());
  EXPECT_EQ(3,c.rows());
  EXPECT_EQ(2,c.cols());

  VEC g = cgradvec(c(2,0),x);
  EXPECT_FLOAT_EQ(0.0,g[0]);
  EXPECT_FLOAT_EQ(1.0,g[1]);
  EXPECT_FLOAT_EQ(0.0,g[2]);
}
TEST(agrad_matrix,transpose_vector) {
  vector_v a(3);
  a << 1.0, 2.0, 3.0;
  
  AVEC x = createAVEC(a(0),a(1),a(2));

  row_vector_v a_tr = transpose(a);
  EXPECT_EQ(a.size(),a_tr.size());
  for (size_t i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(a(i).val(),a_tr(i).val());

  VEC g = cgradvec(a_tr(1),x);
  EXPECT_FLOAT_EQ(0.0,g[0]);
  EXPECT_FLOAT_EQ(1.0,g[1]);
  EXPECT_FLOAT_EQ(0.0,g[2]);
}
TEST(agrad_matrix,transpose_row_vector) {
  row_vector_v a(3);
  a << 1.0, 2.0, 3.0;
  
  AVEC x = createAVEC(a(0),a(1),a(2));

  vector_v a_tr = transpose(a);
  EXPECT_EQ(a.size(),a_tr.size());
  for (size_t i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(a(i).val(),a_tr(i).val());

  VEC g = cgradvec(a_tr(1),x);
  EXPECT_FLOAT_EQ(0.0,g[0]);
  EXPECT_FLOAT_EQ(1.0,g[1]);
  EXPECT_FLOAT_EQ(0.0,g[2]);
}


TEST(agrad_matrix,mv_trace) {
  matrix_v a(2,2);
  a << -1.0, 2.0, 
       5.0, 10.0;
  
  AVEC x = createAVEC(a(0,0), a(0,1), a(1,0), a(1,1));

  AVAR s = trace(a);
  EXPECT_FLOAT_EQ(9.0,s.val());
  
  VEC g = cgradvec(s,x);
  EXPECT_FLOAT_EQ(1.0, g[0]);
  EXPECT_FLOAT_EQ(0.0, g[1]);
  EXPECT_FLOAT_EQ(0.0, g[2]);
  EXPECT_FLOAT_EQ(1.0, g[3]);
}  


TEST(agrad_matrix,mdivide_left_val) {
  matrix_v Av(2,2);
  matrix_d Ad(2,2);
  matrix_v I;

  Av << 2.0, 3.0, 
        5.0, 7.0;
  Ad << 2.0, 3.0, 
        5.0, 7.0;

  I = mdivide_left(Av,Av);
  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);

  I = mdivide_left(Av,Ad);
  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);

  I = mdivide_left(Ad,Av);
  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);
}

TEST(agrad_matrix,mdivide_right_val) {
  matrix_v Av(2,2);
  matrix_d Ad(2,2);
  matrix_v I;

  Av << 2.0, 3.0, 
        5.0, 7.0;
  Ad << 2.0, 3.0, 
        5.0, 7.0;

  I = mdivide_right(Av,Av);
  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);

  I = mdivide_right(Av,Ad);
  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);

  I = mdivide_right(Ad,Av);
  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);
}

TEST(agrad_matrix,inverse_val) {
  using stan::math::inverse;
  matrix_v a(2,2);
  a << 2.0, 3.0, 
       5.0, 7.0;

  matrix_v a_inv = inverse(a);

  matrix_v I = multiply(a,a_inv);

  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);
}
TEST(agrad_matrix,inverse_grad) {
  using stan::math::inverse;
  
  for (size_t k = 0; k < 2; ++k) {
    for (size_t l = 0; l < 2; ++l) {

      matrix_v ad(2,2);
      ad << 2.0, 3.0, 
        5.0, 7.0;

      AVEC x = createAVEC(ad(0,0),ad(0,1),ad(1,0),ad(1,1));

      matrix_v ad_inv = inverse(ad);

      // int k = 0;
      // int l = 1;
      std::vector<double> g;
      ad_inv(k,l).grad(x,g);

      int idx = 0;
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
          EXPECT_FLOAT_EQ(-ad_inv(k,i).val() * ad_inv(j,l).val(), g[idx]);
          ++idx;
        }
      }
    }
  }
}
TEST(agrad_matrix,inverse_inverse_sum) {
  using stan::math::sum;
  using stan::math::inverse;

  matrix_v a(4,4);
  a << 2.0, 3.0, 4.0, 5.0, 
    9.0, -1.0, 2.0, 2.0,
    4.0, 3.0, 7.0, -1.0,
    0.0, 1.0, 19.0, 112.0;

  AVEC x;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      x.push_back(a(i,j));

  AVAR a_inv_inv_sum = sum(inverse(inverse(a)));

  VEC g = cgradvec(a_inv_inv_sum,x);

  for (size_t k = 0; k < x.size(); ++k)
    EXPECT_FLOAT_EQ(1.0,g[k]);
}



TEST(agrad_matrix,eigenval_sum) {
  using stan::math::eigenvalues;
  using stan::math::sum;

  matrix_v a(3,3);
  a << 1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 13.0, 11.0, 19.0;
  AVEC x = createAVEC(a(0,0), a(1,1), a(2,2), a(1,2));
  x.push_back(a(0,1));
  x.push_back(a(2,0));

  // grad sum eig = I
  vector_v a_eigenvalues = eigenvalues(a);
  
  AVAR sum_a_eigenvalues = sum(a_eigenvalues);
  
  VEC g = cgradvec(sum_a_eigenvalues,x);

  EXPECT_NEAR(1.0,g[0],1.0E-11);
  EXPECT_NEAR(1.0,g[1],1.0E-11);
  EXPECT_NEAR(1.0,g[2],1.0E-11);

  EXPECT_NEAR(0.0,g[3],1.0E-10);
  EXPECT_NEAR(0.0,g[4],1.0E-10);
  EXPECT_NEAR(0.0,g[5],1.0E-10);
}

TEST(agrad_matrix,mat_cholesky) {
  // symmetric
  matrix_v X(2,2);
  AVAR a = 3.0;
  AVAR b = -1.0;
  AVAR c = -1.0;
  AVAR d = 1.0;
  X << a, b, 
       c, d;
  
  matrix_v L = cholesky_decompose(X);

  matrix_v LL_trans = multiply(L,transpose(L));
  EXPECT_FLOAT_EQ(a.val(),LL_trans(0,0).val());
  EXPECT_FLOAT_EQ(b.val(),LL_trans(0,1).val());
  EXPECT_FLOAT_EQ(c.val(),LL_trans(1,0).val());
  EXPECT_FLOAT_EQ(d.val(),LL_trans(1,1).val());
}


// norm tests for raw calls; move into promotion lib
TEST(agrad_matrix,mv_squaredNorm) {
  matrix_v a(2,2);
  a << -1.0, 2.0, 
       5.0, 10.0;
  
  AVEC x = createAVEC(a(0,0), a(0,1), a(1,0), a(1,1));

  AVAR s = a.squaredNorm();
  EXPECT_FLOAT_EQ(130.0,s.val());
  
  VEC g = cgradvec(s,x);
  EXPECT_FLOAT_EQ(-2.0, g[0]);
  EXPECT_FLOAT_EQ(4.0, g[1]);
  EXPECT_FLOAT_EQ(10.0, g[2]);
  EXPECT_FLOAT_EQ(20.0, g[3]);
}  
TEST(agrad_matrix,mv_norm) {
  matrix_v a(2,1);
  a << -3.0, 4.0;
  
  AVEC x = createAVEC(a(0,0), a(1,0));

  AVAR s = a.norm();
  EXPECT_FLOAT_EQ(5.0,s.val());

  // (see hypot in special_functions_test) 
  VEC g = cgradvec(s,x);
  EXPECT_FLOAT_EQ(-3.0/5.0, g[0]);
  EXPECT_FLOAT_EQ(4.0/5.0, g[1]);
}  
TEST(agrad_matrix,mv_lp_norm) {
  matrix_v a(2,2);
  a << -1.0, 2.0, 
    5.0, 0.0;
  
  AVEC x = createAVEC(a(0,0), a(0,1), a(1,0), a(1,1));

  AVAR s = a.lpNorm<1>();
  EXPECT_FLOAT_EQ(8.0,s.val());
  
  VEC g = cgradvec(s,x);
  EXPECT_FLOAT_EQ(-1.0,g[0]);
  EXPECT_FLOAT_EQ(1.0,g[1]);
  EXPECT_FLOAT_EQ(1.0,g[2]);
  EXPECT_FLOAT_EQ(0.0,g[3]); // ? depends on impl here, could be -1 or 1
}  
TEST(agrad_matrix,mv_lp_norm_inf) {
  matrix_v a(2,2);
  a << -1.0, 2.0, 
    -5.0, 0.0;
  
  AVEC x = createAVEC(a(0,0), a(0,1), a(1,0), a(1,1));

  AVAR s = a.lpNorm<Eigen::Infinity>();
  EXPECT_FLOAT_EQ(5.0,s.val());
  
  VEC g = cgradvec(s,x);
  EXPECT_FLOAT_EQ(0.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
  EXPECT_FLOAT_EQ(-1.0,g[2]);
  EXPECT_FLOAT_EQ(0.0,g[3]); 
}  

TEST(agradMatrix,multiply_scalar_vector_cv) {
  using stan::agrad::multiply;
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
TEST(agradMatrix,multiply_scalar_vector_vv) {
  using stan::agrad::multiply;
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
TEST(agradMatrix,multiply_scalar_vector_vc) {
  using stan::agrad::multiply;
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

TEST(agradMatrix,multiply_scalar_row_vector_cv) {
  using stan::agrad::multiply;
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
TEST(agradMatrix,multiply_scalar_row_vector_vv) {
  using stan::agrad::multiply;
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
TEST(agradMatrix,multiply_scalar_row_vector_vc) {
  using stan::agrad::multiply;
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

TEST(agradMatrix,multiply_scalar_matrix_cv) {
  using stan::agrad::multiply;
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

TEST(agradMatrix,multiply_scalar_matrix_vc) {
  using stan::agrad::multiply;
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

TEST(agradMatrix,elt_multiply_vec_vv) {
  using stan::agrad::elt_multiply;
  vector_v x(2);
  x << 2, 5;
  vector_v y(2);
  y << 10, 100;
  AVEC x_ind = createAVEC(x(0),x(1),y(0),y(1));
  vector_v z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0).val());
  EXPECT_FLOAT_EQ(500.0,z(1).val());

  VEC g = cgradvec(z(0),x_ind);
  EXPECT_FLOAT_EQ(10.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
  EXPECT_FLOAT_EQ(2.0,g[2]);
  EXPECT_FLOAT_EQ(0.0,g[3]);
}
TEST(agradMatrix,elt_multiply_vec_vd) {
  using stan::agrad::elt_multiply;
  vector_v x(2);
  x << 2, 5;
  vector_d y(2);
  y << 10, 100;
  AVEC x_ind = createAVEC(x(0),x(1));
  vector_v z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0).val());
  EXPECT_FLOAT_EQ(500.0,z(1).val());

  VEC g = cgradvec(z(0),x_ind);
  EXPECT_FLOAT_EQ(10.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
}
TEST(agradMatrix,elt_multiply_vec_dv) {
  using stan::agrad::elt_multiply;
  vector_d x(2);
  x << 2, 5;
  vector_v y(2);
  y << 10, 100;
  AVEC x_ind = createAVEC(y(0),y(1));
  vector_v z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0).val());
  EXPECT_FLOAT_EQ(500.0,z(1).val());

  VEC g = cgradvec(z(0),x_ind);
  EXPECT_FLOAT_EQ(2.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
}

TEST(agradMatrix,elt_multiply_row_vec_vv) {
  using stan::agrad::elt_multiply;
  row_vector_v x(2);
  x << 2, 5;
  row_vector_v y(2);
  y << 10, 100;
  AVEC x_ind = createAVEC(x(0),x(1),y(0),y(1));
  row_vector_v z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0).val());
  EXPECT_FLOAT_EQ(500.0,z(1).val());

  VEC g = cgradvec(z(0),x_ind);
  EXPECT_FLOAT_EQ(10.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
  EXPECT_FLOAT_EQ(2.0,g[2]);
  EXPECT_FLOAT_EQ(0.0,g[3]);
}
TEST(agradMatrix,elt_multiply_row_vec_vd) {
  using stan::agrad::elt_multiply;
  row_vector_v x(2);
  x << 2, 5;
  row_vector_d y(2);
  y << 10, 100;
  AVEC x_ind = createAVEC(x(0),x(1));
  row_vector_v z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0).val());
  EXPECT_FLOAT_EQ(500.0,z(1).val());

  VEC g = cgradvec(z(0),x_ind);
  EXPECT_FLOAT_EQ(10.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
}
TEST(agradMatrix,elt_multiply_row_vec_dv) {
  using stan::agrad::elt_multiply;
  row_vector_d x(2);
  x << 2, 5;
  row_vector_v y(2);
  y << 10, 100;
  AVEC x_ind = createAVEC(y(0),y(1));
  row_vector_v z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0).val());
  EXPECT_FLOAT_EQ(500.0,z(1).val());

  VEC g = cgradvec(z(0),x_ind);
  EXPECT_FLOAT_EQ(2.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
}


TEST(agradMatrix,elt_multiply_matrix_vv) {
  using stan::agrad::elt_multiply;
  matrix_v x(2,3);
  x << 2, 5, 6, 9, 13, 29;
  matrix_v y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;
  AVEC x_ind = createAVEC(x(0,0),x(0,1),x(0,2),y(0,0));
  matrix_v z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0,0).val());
  EXPECT_FLOAT_EQ(500.0,z(0,1).val());
  EXPECT_FLOAT_EQ(29000000.0,z(1,2).val());

  VEC g = cgradvec(z(0,0),x_ind);
  EXPECT_FLOAT_EQ(10.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
  EXPECT_FLOAT_EQ(0.0,g[2]);
  EXPECT_FLOAT_EQ(2.0,g[3]);
}
TEST(agradMatrix,elt_multiply_matrix_vd) {
  using stan::agrad::elt_multiply;
  matrix_v x(2,3);
  x << 2, 5, 6, 9, 13, 29;
  matrix_d y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;
  AVEC x_ind = createAVEC(x(0,0),x(0,1),x(0,2),x(1,0));
  matrix_v z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0,0).val());
  EXPECT_FLOAT_EQ(500.0,z(0,1).val());
  EXPECT_FLOAT_EQ(29000000.0,z(1,2).val());

  VEC g = cgradvec(z(0,0),x_ind);
  EXPECT_FLOAT_EQ(10.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
  EXPECT_FLOAT_EQ(0.0,g[2]);
  EXPECT_FLOAT_EQ(0.0,g[3]);
}
TEST(agradMatrix,elt_multiply_matrix_dv) {
  using stan::agrad::elt_multiply;
  matrix_d x(2,3);
  x << 2, 5, 6, 9, 13, 29;
  matrix_v y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;
  AVEC x_ind = createAVEC(y(0,0),y(0,1));
  matrix_v z = elt_multiply(x,y);
  EXPECT_FLOAT_EQ(20.0,z(0,0).val());
  EXPECT_FLOAT_EQ(500.0,z(0,1).val());
  EXPECT_FLOAT_EQ(29000000.0,z(1,2).val());

  VEC g = cgradvec(z(0,0),x_ind);
  EXPECT_FLOAT_EQ(2.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
}

TEST(agradMatrix,elt_divide_vec_vv) {
  using stan::agrad::elt_divide;
  vector_v x(2);
  x << 2, 5;
  vector_v y(2);
  y << 10, 100;
  AVEC x_ind = createAVEC(x(0),x(1),y(0),y(1));
  vector_v z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val());
  EXPECT_FLOAT_EQ(0.05,z(1).val());

  VEC g = cgradvec(z(0),x_ind);
  EXPECT_FLOAT_EQ(1.0/10.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
  EXPECT_FLOAT_EQ(2.0 / (- 10.0 * 10.0), g[2]);
  EXPECT_FLOAT_EQ(0.0,g[3]);
}
TEST(agradMatrix,elt_divide_vec_vd) {
  using stan::agrad::elt_divide;
  vector_v x(2);
  x << 2, 5;
  vector_d y(2);
  y << 10, 100;
  AVEC x_ind = createAVEC(x(0),x(1));
  vector_v z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val());
  EXPECT_FLOAT_EQ(0.05,z(1).val());

  VEC g = cgradvec(z(0),x_ind);
  EXPECT_FLOAT_EQ(1.0/10.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
}
TEST(agradMatrix,elt_divide_vec_dv) {
  using stan::agrad::elt_divide;
  vector_d x(2);
  x << 2, 5;
  vector_v y(2);
  y << 10, 100;
  AVEC x_ind = createAVEC(y(0),y(1));
  vector_v z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val());
  EXPECT_FLOAT_EQ(0.05,z(1).val());

  VEC g = cgradvec(z(0),x_ind);
  EXPECT_FLOAT_EQ(2.0 / (- 10.0 * 10.0), g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
}

TEST(agradMatrix,elt_divide_rowvec_vv) {
  using stan::agrad::elt_divide;
  row_vector_v x(2);
  x << 2, 5;
  row_vector_v y(2);
  y << 10, 100;
  AVEC x_ind = createAVEC(x(0),x(1),y(0),y(1));
  row_vector_v z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val());
  EXPECT_FLOAT_EQ(0.05,z(1).val());

  VEC g = cgradvec(z(0),x_ind);
  EXPECT_FLOAT_EQ(1.0/10.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
  EXPECT_FLOAT_EQ(2.0 / (- 10.0 * 10.0), g[2]);
  EXPECT_FLOAT_EQ(0.0,g[3]);
}
TEST(agradMatrix,elt_divide_rowvec_vd) {
  using stan::agrad::elt_divide;
  row_vector_v x(2);
  x << 2, 5;
  row_vector_d y(2);
  y << 10, 100;
  AVEC x_ind = createAVEC(x(0),x(1));
  row_vector_v z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val());
  EXPECT_FLOAT_EQ(0.05,z(1).val());

  VEC g = cgradvec(z(0),x_ind);
  EXPECT_FLOAT_EQ(1.0/10.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
}
TEST(agradMatrix,elt_divide_rowvec_dv) {
  using stan::agrad::elt_divide;
  row_vector_d x(2);
  x << 2, 5;
  row_vector_v y(2);
  y << 10, 100;
  AVEC x_ind = createAVEC(y(0),y(1));
  row_vector_v z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0).val());
  EXPECT_FLOAT_EQ(0.05,z(1).val());

  VEC g = cgradvec(z(0),x_ind);
  EXPECT_FLOAT_EQ(2.0 / (- 10.0 * 10.0), g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
}


TEST(agradMatrix,elt_divide_mat_vv) {
  using stan::agrad::elt_divide;
  matrix_v x(2,3);
  x << 2, 5, 7, 13, 29, 112;
  matrix_v y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;
  AVEC x_ind = createAVEC(x(0,0),x(0,1),y(0,0),y(0,1));
  matrix_v z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val());
  EXPECT_FLOAT_EQ(0.05,z(0,1).val());
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val());

  VEC g = cgradvec(z(0),x_ind);
  EXPECT_FLOAT_EQ(1.0/10.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
  EXPECT_FLOAT_EQ(2.0 / (- 10.0 * 10.0), g[2]);
  EXPECT_FLOAT_EQ(0.0,g[3]);
}
TEST(agradMatrix,elt_divide_mat_vd) {
  using stan::agrad::elt_divide;
  matrix_v x(2,3);
  x << 2, 5, 7, 13, 29, 112;
  matrix_d y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;
  AVEC x_ind = createAVEC(x(0,0),x(0,1));
  matrix_v z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val());
  EXPECT_FLOAT_EQ(0.05,z(0,1).val());
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val());

  VEC g = cgradvec(z(0),x_ind);
  EXPECT_FLOAT_EQ(1.0/10.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
}
TEST(agradMatrix,elt_divide_mat_dv) {
  using stan::agrad::elt_divide;
  matrix_d x(2,3);
  x << 2, 5, 7, 13, 29, 112;
  matrix_v y(2,3);
  y << 10, 100, 1000, 10000, 100000, 1000000;
  AVEC x_ind = createAVEC(y(0,0),y(0,1));
  matrix_v z = elt_divide(x,y);
  EXPECT_FLOAT_EQ(0.2,z(0,0).val());
  EXPECT_FLOAT_EQ(0.05,z(0,1).val());
  EXPECT_FLOAT_EQ(112.0/1000000.0,z(1,2).val());

  VEC g = cgradvec(z(0),x_ind);
  EXPECT_FLOAT_EQ(2.0 / (- 10.0 * 10.0), g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
}
TEST(agradMatrix,col_v) {
  using stan::agrad::col;
  matrix_v y(2,3);
  y << 1, 2, 3, 4, 5, 6;
  vector_v z = col(y,1);
  EXPECT_EQ(2,z.size());
  EXPECT_FLOAT_EQ(1.0,z[0].val());
  EXPECT_FLOAT_EQ(4.0,z[1].val());

  vector_v w = col(y,2);
  EXPECT_EQ(2,w.size());
  EXPECT_EQ(2.0,w[0].val());
  EXPECT_EQ(5.0,w[1].val());
}
TEST(agradMatrix,col_v_exc0) {
  using stan::agrad::col;
  matrix_v y(2,3);
  y << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(col(y,0),std::invalid_argument);
}
TEST(agradMatrix,col_v_excHigh) {
  using stan::agrad::col;
  matrix_v y(2,3);
  y << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(col(y,5),std::invalid_argument);
}
TEST(agradMatrix,row_v) {
  using stan::agrad::row;
  matrix_v y(2,3);
  y << 1, 2, 3, 4, 5, 6;
  vector_v z = row(y,1);
  EXPECT_EQ(3,z.size());
  EXPECT_FLOAT_EQ(1.0,z[0].val());
  EXPECT_FLOAT_EQ(2.0,z[1].val());
  EXPECT_FLOAT_EQ(3.0,z[2].val());

  vector_v w = row(y,2);
  EXPECT_EQ(3,w.size());
  EXPECT_EQ(4.0,w[0].val());
  EXPECT_EQ(5.0,w[1].val());
  EXPECT_EQ(6.0,w[2].val());
}
TEST(agradMatrix,row_v_exc0) {
  using stan::agrad::row;
  matrix_v y(2,3);
  y << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(row(y,0),std::invalid_argument);
}
TEST(agradMatrix,row_v_excHigh) {
  using stan::agrad::row;
  matrix_v y(2,3);
  y << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(row(y,5),std::invalid_argument);
}
TEST(agradMatrix, dot_product_vv) {
  std::vector<var> a, b;
  var c;
  for (int i = -1; i < 2; i++) { // a = (-1, 0, 1), b = (1, 2, 3)
    a.push_back(i);
    b.push_back(i + 2);
  }
  c = dot_product(&a[0], &b[0], 3);
  EXPECT_EQ(2, c);
  std::vector<var> ab;
  std::vector<double> grad;
  for (size_t i = 0; i < 3; i++) {
    ab.push_back(a[i]);
    ab.push_back(b[i]);
  }
  c.grad(ab, grad);
  EXPECT_EQ(grad[0], 1);
  EXPECT_EQ(grad[1], -1);
  EXPECT_EQ(grad[2], 2);
  EXPECT_EQ(grad[3], 0);
  EXPECT_EQ(grad[4], 3);
  EXPECT_EQ(grad[5], 1);
}
TEST(agradMatrix, dot_product_dv) {
  std::vector<double> a;
  std::vector<var> b;
  var c;
  for (int i = -1; i < 2; i++) { // a = (-1, 0, 1), b = (1, 2, 3)
    a.push_back(i);
    b.push_back(i + 2);
  }
  c = dot_product(&a[0], &b[0], 3);
  EXPECT_EQ(2, c);
  std::vector<double> grad;
  c.grad(b, grad);
  EXPECT_EQ(grad[0], -1);
  EXPECT_EQ(grad[1], 0);
  EXPECT_EQ(grad[2], 1);
}
TEST(agradMatrix, dot_product_vd) {
  std::vector<var> a;
  std::vector<double> b;
  var c;
  for (int i = -1; i < 2; i++) { // a = (-1, 0, 1), b = (1, 2, 3)
    a.push_back(i);
    b.push_back(i + 2);
  }
  c = dot_product(&a[0], &b[0], 3);
  EXPECT_EQ(2, c);
  std::vector<double> grad;
  c.grad(a, grad);
  EXPECT_EQ(grad[0], 1);
  EXPECT_EQ(grad[1], 2);
  EXPECT_EQ(grad[2], 3);
}
TEST(agradMatrix, dot_product_vv_vec) {
  std::vector<var> a, b;
  var c;
  for (int i = -1; i < 2; i++) { // a = (-1, 0, 1), b = (1, 2, 3)
    a.push_back(i);
    b.push_back(i + 2);
  }
  c = dot_product(a, b);
  EXPECT_EQ(2, c);
  std::vector<var> ab;
  std::vector<double> grad;
  for (size_t i = 0; i < 3; i++) {
    ab.push_back(a[i]);
    ab.push_back(b[i]);
  }
  c.grad(ab, grad);
  EXPECT_EQ(grad[0], 1);
  EXPECT_EQ(grad[1], -1);
  EXPECT_EQ(grad[2], 2);
  EXPECT_EQ(grad[3], 0);
  EXPECT_EQ(grad[4], 3);
  EXPECT_EQ(grad[5], 1);
}
TEST(agradMatrix, dot_product_dv_vec) {
  std::vector<double> a;
  std::vector<var> b;
  var c;
  for (int i = -1; i < 2; i++) { // a = (-1, 0, 1), b = (1, 2, 3)
    a.push_back(i);
    b.push_back(i + 2);
  }
  c = dot_product(a, b);
  EXPECT_EQ(2, c);
  std::vector<double> grad;
  c.grad(b, grad);
  EXPECT_EQ(grad[0], -1);
  EXPECT_EQ(grad[1], 0);
  EXPECT_EQ(grad[2], 1);
}
TEST(agradMatrix, dot_product_vd_vec) {
  std::vector<var> a;
  std::vector<double> b;
  var c;
  for (int i = -1; i < 2; i++) { // a = (-1, 0, 1), b = (1, 2, 3)
    a.push_back(i);
    b.push_back(i + 2);
  }
  c = dot_product(a, b);
  EXPECT_EQ(2, c);
  std::vector<double> grad;
  c.grad(a, grad);
  EXPECT_EQ(grad[0], 1);
  EXPECT_EQ(grad[1], 2);
  EXPECT_EQ(grad[2], 3);
}

template <int R, int C>
void assert_val_grad(Eigen::Matrix<stan::agrad::var,R,C>& v) {
  v << -1.0, 0.0, 3.0;
  AVEC x = createAVEC(v(0),v(1),v(2));
  AVAR f = dot_self(v);
  std::vector<double> g;
  f.grad(x,g);
  
  EXPECT_FLOAT_EQ(-2.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
  EXPECT_FLOAT_EQ(6.0,g[2]);
}  


TEST(agradMatrix, dot_self_vec) {
  using stan::agrad::var;
  using stan::math::dot_self;

  Eigen::Matrix<var,Eigen::Dynamic,1> v1(1);
  v1 << 2.0;
  EXPECT_NEAR(4.0,dot_self(v1).val(),1E-12);
  Eigen::Matrix<var,Eigen::Dynamic,1> v2(2);
  v2 << 2.0, 3.0;
  EXPECT_NEAR(13.0,dot_self(v2).val(),1E-12);
  Eigen::Matrix<var,Eigen::Dynamic,1> v3(3);
  v3 << 2.0, 3.0, 4.0;
  EXPECT_NEAR(29.0,dot_self(v3).val(),1E-12);  

  Eigen::Matrix<var,Eigen::Dynamic,1> v(3);
  assert_val_grad(v);

  Eigen::Matrix<var,1,Eigen::Dynamic> vv(3);
  assert_val_grad(vv);

  Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic> vvv(3,1);
  assert_val_grad(vvv);

  Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic> vvvv(1,3);
  assert_val_grad(vvvv);

  
}

TEST(agradMatrix,columns_dot_self) {
  using stan::agrad::var;
  using stan::math::columns_dot_self;

  Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic> m1(1,1);
  m1 << 2.0;
  EXPECT_NEAR(4.0,columns_dot_self(m1)(0,0).val(),1E-12);
  Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic> m2(1,2);
  m2 << 2.0, 3.0;
  Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic> x;
  x = columns_dot_self(m2);
  EXPECT_NEAR(4.0,x(0,0).val(),1E-12);
  EXPECT_NEAR(9.0,x(1,0).val(),1E-12);
  Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic> m3(2,2);
  m3 << 2.0, 3.0, 4.0, 5.0;
  x = columns_dot_self(m3);
  EXPECT_NEAR(20.0,x(0,0).val(),1E-12);
  EXPECT_NEAR(34.0,x(1,0).val(),1E-12);

  Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic> vvv(3,1);
  assert_val_grad(vvv);

  Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic> vvvv(1,3);
  assert_val_grad(vvvv);
}


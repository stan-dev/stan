#include <vector>
#include <stan/agrad.hpp>
#include <stan/agrad/rev/matrix.hpp>
#include <stan/agrad/rev/matrix/multiply.hpp>
#include <stan/math/matrix/quad_form_diag.hpp>
#include <test/unit/agrad/rev/matrix/expect_matrix_eq.hpp>
#include <gtest/gtest.h>

using Eigen::Matrix;
using Eigen::Dynamic;
using stan::agrad::var;
using stan::math::quad_form_diag;

TEST(MathMatrix,quadFormDiag2_vv) {
  Matrix<var,Dynamic,Dynamic> m(3,3);
  m << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  Matrix<var,Dynamic,1> v(3);
  v << 1, 2, 3;

  Matrix<var,Dynamic,Dynamic> v_m(3,3);
  v_m << 
    1, 0, 0,
    0, 2, 0,
    0, 0, 3;
  
  Matrix<var,Dynamic,Dynamic> v_m_times_m = v_m * m * v_m;

  expect_matrix_eq(v_m_times_m, quad_form_diag(m,v));

  Matrix<var,1,Dynamic> rv(3);
  rv << 1, 2, 3;

  expect_matrix_eq(v_m_times_m, quad_form_diag(m,rv));
}

TEST(MathMatrix,quadFormDiag2_vd) {
  Matrix<var,Dynamic,Dynamic> m1(3,3);
  m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  Matrix<double,Dynamic,1> v(3);
  v << 1, 2, 3;

  Matrix<var,Dynamic,Dynamic> v_m(3,3);
  v_m << 
    1, 0, 0,
    0, 2, 0,
    0, 0, 3;
  
  Matrix<var,Dynamic,Dynamic> v_m_times_m1 = v_m * m1 * v_m;

  Matrix<var,Dynamic,Dynamic> m2(3,3);
  m2 << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  expect_matrix_eq(v_m_times_m1, quad_form_diag(m2,v));
}

TEST(MathMatrix,quadFormDiag2_dv) {
  Matrix<double,Dynamic,Dynamic> m1(3,3);
  m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  Matrix<var,Dynamic,1> v(3);
  v << 1, 2, 3;


  Matrix<var,Dynamic,Dynamic> m2(3,3);
  m2 << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  Matrix<var,Dynamic,Dynamic> v_m(3,3);
  v_m << 
    1, 0, 0,
    0, 2, 0,
    0, 0, 3;

  Matrix<var,Dynamic,Dynamic> v_m_times_m2 = v_m * m2 * v_m;

  expect_matrix_eq(v_m_times_m2, quad_form_diag(m1,v));

}


TEST(MathMatrix,quadFormDiagGrad_vv) {
  Matrix<var,Dynamic,Dynamic> m1(3,3);
  m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  Matrix<var,Dynamic,1> v(3);
  v << 1, 2, 3;

  std::vector<var> xs1;
  for (int i = 0; i < 9; ++i)
    xs1.push_back(m1(i));
  for (int i = 0; i < 3; ++i)
    xs1.push_back(v(i));

  Matrix<var,Dynamic,Dynamic> v_post_multipy_m1 = quad_form_diag(m1, v);

  var norm1 = v_post_multipy_m1.norm();

  std::vector<double> g1;
  norm1.grad(xs1,g1);


  Matrix<var,Dynamic,Dynamic> m2(3,3);
  m2 << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  Matrix<var,Dynamic,Dynamic> v_m(3,3);
  v_m << 
    1, 0, 0,
    0, 2, 0,
    0, 0, 3;

  std::vector<var> xs2;
  for (int i = 0; i < 9; ++i)
    xs2.push_back(m2(i));
  for (int i = 0; i < 3; ++i)
    xs2.push_back(v_m(i,i));

  
  Matrix<var,Dynamic,Dynamic> m_times_vm = v_m * m2 * v_m;

  var norm2 = m_times_vm.norm();
  std::vector<double> g2;
  norm2.grad(xs2,g2);
  
  EXPECT_EQ(g1.size(), g2.size());
  for (size_t i = 0; i < g1.size(); ++i)
    EXPECT_FLOAT_EQ(g1[i], g2[i]);
}

TEST(MathMatrix,quadFormDiagGrad_vd) {
  Matrix<var,Dynamic,Dynamic> m1(3,3);
  m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  Matrix<double,Dynamic,1> v(3);
  v << 1, 2, 3;

  std::vector<var> xs1;
  for (int i = 0; i < 9; ++i)
    xs1.push_back(m1(i));

  Matrix<var,Dynamic,Dynamic> m1_post_multipy_v = quad_form_diag(m1, v);

  var norm1 = m1_post_multipy_v.norm();

  std::vector<double> g1;
  norm1.grad(xs1,g1);


  Matrix<var,Dynamic,Dynamic> m2(3,3);
  m2 << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  Matrix<var,Dynamic,Dynamic> v_m(3,3);  // OK to use var, just for comparison
  v_m << 
    1, 0, 0,
    0, 2, 0,
    0, 0, 3;

  std::vector<var> xs2;
  for (int i = 0; i < 9; ++i)
    xs2.push_back(m2(i));
  
  Matrix<var,Dynamic,Dynamic> v_m_times_v_m = v_m * m2 * v_m;

  var norm2 = v_m_times_v_m.norm();
  std::vector<double> g2;
  norm2.grad(xs2,g2);
  
  EXPECT_EQ(g1.size(), g2.size());
  for (size_t i = 0; i < g1.size(); ++i)
    EXPECT_FLOAT_EQ(g1[i], g2[i]);
}

TEST(MathMatrix,quadFormDiagGrad_dv) {
  Matrix<double,Dynamic,Dynamic> m1(3,3);
  m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  Matrix<var,Dynamic,1> v(3);
  v << 1, 2, 3;

  std::vector<var> xs1;
  for (int i = 0; i < 3; ++i)
    xs1.push_back(v(i));

  Matrix<var,Dynamic,Dynamic> m1_post_multipy_v = quad_form_diag(m1, v);

  var norm1 = m1_post_multipy_v.norm();

  std::vector<double> g1;
  norm1.grad(xs1,g1);


  Matrix<var,Dynamic,Dynamic> m2(3,3);
  m2 << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  Matrix<var,Dynamic,Dynamic> v_m(3,3);  // OK to use var, just for comparison
  v_m << 
    1, 0, 0,
    0, 2, 0,
    0, 0, 3;

  std::vector<var> xs2;
  for (int i = 0; i < 3; ++i)
    xs2.push_back(v_m(i,i));
  
  Matrix<var,Dynamic,Dynamic> v_m_times_v_m = v_m * m2 * v_m;

  var norm2 = v_m_times_v_m.norm();
  std::vector<double> g2;
  norm2.grad(xs2,g2);
  
  EXPECT_EQ(g1.size(), g2.size());
  for (size_t i = 0; i < g1.size(); ++i)
    EXPECT_FLOAT_EQ(g1[i], g2[i]);
}


TEST(MathMatrix,quadFormDiagException) {
  Matrix<var,Dynamic,Dynamic> m(2,2);
  m << 
    2, 3,
    4, 5;
  EXPECT_THROW(quad_form_diag(m,m), std::domain_error);

  Matrix<var,Dynamic,1> v(3);
  v << 1, 2, 3;
  EXPECT_THROW(quad_form_diag(m,v), std::domain_error);
}

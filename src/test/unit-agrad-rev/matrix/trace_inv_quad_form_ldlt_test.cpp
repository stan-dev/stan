#include <stan/agrad/rev/matrix.hpp>
#include <stan/agrad/rev/matrix/trace_inv_quad_form_ldlt.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/agrad/rev/matrix/sum.hpp>
#include <stan/math/matrix.hpp>
#include <stan/math/matrix/sum.hpp>
#include <stan/math/matrix/trace_inv_quad_form_ldlt.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/inverse.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>
#include <stan/agrad/rev/matrix/to_var.hpp>

TEST(AgradRevMatrix, trace_inv_quad_form_ldlt_mat) {
  using stan::agrad::matrix_v;
  using stan::math::matrix_d;
  
  matrix_v av(4,4);
  matrix_d ad(4,4);
  matrix_d bd(4,2);
  matrix_v bv(4,2);
  AVAR res;
  AVEC vars;
  VEC grad;
  
  
  bd << 100, 10,
          0,  1,
         -3, -3,
          5,  2;
  bv << 100, 10,
          0,  1,
         -3, -3,
          5,  2;
  ad << 9.0,  3.0, 3.0,   3.0, 
        3.0, 10.0, 2.0,   2.0,
        3.0,  2.0, 7.0,   1.0,
        3.0,  2.0, 1.0, 112.0;
  av << 9.0,  3.0, 3.0,   3.0, 
        3.0, 10.0, 2.0,   2.0,
        3.0,  2.0, 7.0,   1.0,
        3.0,  2.0, 1.0, 112.0;

  stan::math::LDLT_factor<double,-1,-1> ldlt_ad;
  stan::math::LDLT_factor<stan::agrad::var,-1,-1> ldlt_av;
  ldlt_av.compute(av);
  ASSERT_TRUE(ldlt_av.success());
  ldlt_ad.compute(ad);
  ASSERT_TRUE(ldlt_ad.success());
  
  // double-double
  res = trace_inv_quad_form_ldlt(ldlt_ad,bd);
  EXPECT_FLOAT_EQ(1439.1061766207, res.val());

  // var-double
  res = trace_inv_quad_form_ldlt(ldlt_av,bd);
  EXPECT_FLOAT_EQ(1439.1061766207, res.val());
  
  // double-var
  res = trace_inv_quad_form_ldlt(ldlt_ad,bv);
  EXPECT_FLOAT_EQ(1439.1061766207, res.val());
  
  // var-var
  res = trace_inv_quad_form_ldlt(ldlt_av,bv);
  EXPECT_FLOAT_EQ(1439.1061766207, res.val());
}

TEST(AgradRevMatrix, trace_quad_form_ldlt_mat_grad_vd) {
  using stan::math::sum;
  using stan::agrad::matrix_v;
  using stan::math::matrix_d;
  
  matrix_v av(4,4);
  matrix_d ad(4,4);
  matrix_d bd(4,2);
  AVAR res;
  AVEC vars;
  VEC grad;
  size_t i,j,pos;
  
  
  bd << 100, 10,
  0,  1,
  -3, -3,
  5,  2;
  
  ad << 9.0,  3.0, 3.0,   3.0, 
        3.0, 10.0, 2.0,   2.0,
        3.0,  2.0, 7.0,   1.0,
        3.0,  2.0, 1.0, 112.0;
  av << 9.0,  3.0, 3.0,   3.0, 
        3.0, 10.0, 2.0,   2.0,
        3.0,  2.0, 7.0,   1.0,
        3.0,  2.0, 1.0, 112.0;

  stan::math::LDLT_factor<stan::agrad::var,-1,-1> ldlt_av;
  ldlt_av.compute(av);
  ASSERT_TRUE(ldlt_av.success());
  
  matrix_d ainv(ad.inverse());
  matrix_d dqda(-ainv*bd*bd.transpose()*ainv);
  
  // var-var
  res = trace_inv_quad_form_ldlt(ldlt_av,bd);
  
  vars.clear();
  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++)
      vars.push_back(av(i,j));
  grad = cgradvec(res,vars);
  pos = 0;
  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++, pos++)
      EXPECT_FLOAT_EQ(grad[pos], dqda(i,j));
}

TEST(AgradRevMatrix, trace_quad_form_ldlt_mat_grad_dv) {
  using stan::math::sum;
  using stan::agrad::matrix_v;
  using stan::math::matrix_d;
  
  matrix_d ad(4,4);
  matrix_d bd(4,2);
  matrix_v bv(4,2);
  AVAR res;
  AVEC vars;
  VEC grad;
  size_t i,j,pos;
  
  
  bd << 100, 10,
  0,  1,
  -3, -3,
  5,  2;
  bv << 100, 10,
  0,  1,
  -3, -3,
  5,  2;
  ad << 9.0,  3.0, 3.0,   3.0, 
        3.0, 10.0, 2.0,   2.0,
        3.0,  2.0, 7.0,   1.0,
        3.0,  2.0, 1.0, 112.0;
  
  stan::math::LDLT_factor<double,-1,-1> ldlt_ad;
  ldlt_ad.compute(ad);
  ASSERT_TRUE(ldlt_ad.success());
  
  matrix_d ainv(ad.inverse());
  matrix_d dqdb(ainv*bd + ainv.transpose()*bd);
  
  // var-var
  res = trace_inv_quad_form_ldlt(ldlt_ad,bv);
  
  vars.clear();
  for (i = 0; i < 4; i++)
    for (j = 0; j < 2; j++)
      vars.push_back(bv(i,j));
  grad = cgradvec(res,vars);
  pos = 0;
  for (i = 0; i < 4; i++)
    for (j = 0; j < 2; j++, pos++)
      EXPECT_FLOAT_EQ(grad[pos], dqdb(i,j));
}

TEST(AgradRevMatrix, trace_quad_form_ldlt_mat_grad_vv) {
  using stan::math::sum;
  using stan::agrad::matrix_v;
  using stan::math::matrix_d;
  
  matrix_v av(4,4);
  matrix_d ad(4,4);
  matrix_d bd(4,2);
  matrix_v bv(4,2);
  AVAR res;
  AVEC vars;
  VEC grad;
  size_t i,j,pos;
  
  
  bd << 100, 10,
  0,  1,
  -3, -3,
  5,  2;
  bv << 100, 10,
  0,  1,
  -3, -3,
  5,  2;
  ad << 9.0,  3.0, 3.0,   3.0, 
        3.0, 10.0, 2.0,   2.0,
        3.0,  2.0, 7.0,   1.0,
        3.0,  2.0, 1.0, 112.0;
  av << 9.0,  3.0, 3.0,   3.0, 
        3.0, 10.0, 2.0,   2.0,
        3.0,  2.0, 7.0,   1.0,
        3.0,  2.0, 1.0, 112.0;

  stan::math::LDLT_factor<double,-1,-1> ldlt_ad;
  stan::math::LDLT_factor<stan::agrad::var,-1,-1> ldlt_av;
  ldlt_ad.compute(ad);
  ASSERT_TRUE(ldlt_ad.success());
  ldlt_av.compute(av);
  ASSERT_TRUE(ldlt_av.success());
  
  matrix_d ainv(ad.inverse());
  matrix_d dqdb(ainv*bd + ainv.transpose()*bd);
  matrix_d dqda(-ainv*bd*bd.transpose()*ainv);
  
  // var-var
  res = trace_inv_quad_form_ldlt(ldlt_av,bv);
  
  vars.clear();
  for (i = 0; i < 4; i++)
    for (j = 0; j < 2; j++)
      vars.push_back(bv(i,j));
  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++)
      vars.push_back(av(i,j));
  grad = cgradvec(res,vars);
  pos = 0;
  for (i = 0; i < 4; i++)
    for (j = 0; j < 2; j++, pos++)
      EXPECT_FLOAT_EQ(grad[pos], dqdb(i,j));
  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++, pos++)
      EXPECT_FLOAT_EQ(grad[pos], dqda(i,j));
}

TEST(AgradRevMatrix, trace_quad_form_ldlt_vv_basic) {
  using stan::agrad::matrix_v;
  using stan::math::matrix_d;
  using stan::math::LDLT_factor;
  using stan::agrad::var;
  using std::vector;
  using stan::math::trace;
  using stan::math::transpose;
  using stan::math::inverse;
  
  matrix_v A(4,4);
  matrix_v B(4,2);
  LDLT_factor<var,-1,-1> ldlt_A;
  var x, x_basic;
  double x_val, x_basic_val;
  vector<var> vars;
  vector<double> grad, grad_basic;
  
  // solve using trace_quad_form_ldlt
  A << 
    9.0,  3.0, 3.0,   3.0, 
    3.0, 10.0, 2.0,   2.0,
    3.0,  2.0, 7.0,   1.0,
    3.0,  2.0, 1.0, 112.0;
  B << 
    100, 10,
    0,  1,
    -3, -3,
    5,  2;
  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());
  x = trace_inv_quad_form_ldlt(ldlt_A,B);
  x_val = x.val();
  
  vars.clear();
  for (int n = 0; n < A.size(); n++) 
    vars.push_back(A(n));
  for (int n = 0; n < B.size(); n++) 
    vars.push_back(B(n));
  x.grad(vars, grad);

  // solve using basic math
  A << 
    9.0,  3.0, 3.0,   3.0, 
    3.0, 10.0, 2.0,   2.0,
    3.0,  2.0, 7.0,   1.0,
    3.0,  2.0, 1.0, 112.0;
  B << 
    100, 10,
    0,  1,
    -3, -3,
    5,  2;
  matrix_v tmp = transpose(B) * inverse(A) * B;
  x_basic = trace(tmp);
  x_basic_val = x_basic.val();
  
  vars.clear();
  for (int n = 0; n < A.size(); n++) 
    vars.push_back(A(n));
  for (int n = 0; n < B.size(); n++) 
    vars.push_back(B(n));
  x_basic.grad(vars, grad_basic);

  
  // check values
  EXPECT_FLOAT_EQ(x_basic_val, x_val);
  ASSERT_EQ(grad_basic.size(), grad.size());
  for (size_t n = 0; n < grad_basic.size(); n++)
  EXPECT_FLOAT_EQ(grad_basic[n], grad[n]);
}

TEST(AgradRevMatrix, trace_quad_form_ldlt_vd_basic) {
  using stan::agrad::matrix_v;
  using stan::math::matrix_d;
  using stan::math::LDLT_factor;
  using stan::agrad::var;
  using std::vector;
  using stan::math::trace;
  using stan::math::transpose;
  using stan::math::inverse;
  
  matrix_v A(4,4);
  matrix_d B(4,2);
  LDLT_factor<var,-1,-1> ldlt_A;
  var x, x_basic;
  double x_val, x_basic_val;
  vector<var> vars;
  vector<double> grad, grad_basic;
  
  // solve using trace_quad_form_ldlt
  A << 
    9.0,  3.0, 3.0,   3.0, 
    3.0, 10.0, 2.0,   2.0,
    3.0,  2.0, 7.0,   1.0,
    3.0,  2.0, 1.0, 112.0;
  B << 
    100, 10,
    0,  1,
    -3, -3,
    5,  2;
  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());
  x = trace_inv_quad_form_ldlt(ldlt_A,B);
  x_val = x.val();
  
  vars.clear();
  for (int n = 0; n < A.size(); n++) 
    vars.push_back(A(n));
  x.grad(vars, grad);

  // solve using basic math
  A << 
    9.0,  3.0, 3.0,   3.0, 
    3.0, 10.0, 2.0,   2.0,
    3.0,  2.0, 7.0,   1.0,
    3.0,  2.0, 1.0, 112.0;
  B << 
    100, 10,
    0,  1,
    -3, -3,
    5,  2;
  matrix_v tmp = stan::agrad::to_var(transpose(B)) * inverse(A) 
    * stan::agrad::to_var(B);
  x_basic = trace(tmp);
  x_basic_val = x_basic.val();
  
  vars.clear();
  for (int n = 0; n < A.size(); n++) 
    vars.push_back(A(n));
  x_basic.grad(vars, grad_basic);

  
  // check values
  EXPECT_FLOAT_EQ(x_basic_val, x_val);
  ASSERT_EQ(grad_basic.size(), grad.size());
  for (size_t n = 0; n < grad_basic.size(); n++)
    EXPECT_FLOAT_EQ(grad_basic[n], grad[n]);
}

TEST(AgradRevMatrix, trace_quad_form_ldlt_dv_basic) {
  using stan::agrad::matrix_v;
  using stan::math::matrix_d;
  using stan::math::LDLT_factor;
  using stan::agrad::var;
  using std::vector;
  using stan::math::trace;
  using stan::math::transpose;
  using stan::math::inverse;
  
  matrix_d A(4,4);
  matrix_v B(4,2);
  LDLT_factor<double,-1,-1> ldlt_A;
  var x, x_basic;
  double x_val, x_basic_val;
  vector<var> vars;
  vector<double> grad, grad_basic;
  
  // solve using trace_quad_form_ldlt
  A << 
    9.0,  3.0, 3.0,   3.0, 
    3.0, 10.0, 2.0,   2.0,
    3.0,  2.0, 7.0,   1.0,
    3.0,  2.0, 1.0, 112.0;
  B << 
    100, 10,
    0,  1,
    -3, -3,
    5,  2;
  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());
  x = trace_inv_quad_form_ldlt(ldlt_A,B);
  x_val = x.val();
  
  vars.clear();
  for (int n = 0; n < B.size(); n++) 
    vars.push_back(B(n));
  x.grad(vars, grad);

  // solve using basic math
  A << 
    9.0,  3.0, 3.0,   3.0, 
    3.0, 10.0, 2.0,   2.0,
    3.0,  2.0, 7.0,   1.0,
    3.0,  2.0, 1.0, 112.0;
  B << 
    100, 10,
    0,  1,
    -3, -3,
    5,  2;
  matrix_v tmp = transpose(B) * stan::agrad::to_var(A.inverse().eval()) * B;
  x_basic = trace(tmp);
  x_basic_val = x_basic.val();
  
  vars.clear();
  for (int n = 0; n < B.size(); n++) 
    vars.push_back(B(n));
  x_basic.grad(vars, grad_basic);

  
  // check values
  EXPECT_FLOAT_EQ(x_basic_val, x_val);
  ASSERT_EQ(grad_basic.size(), grad.size());
  for (size_t n = 0; n < grad_basic.size(); n++)
    EXPECT_FLOAT_EQ(grad_basic[n], grad[n]);
}


#include <stan/agrad/rev/matrix.hpp>
#include <stan/agrad/rev/matrix/trace_gen_inv_quad_form_ldlt.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/math/matrix/multiply.hpp>
#include <stan/math/matrix/trace_gen_inv_quad_form_ldlt.hpp>
#include <stan/math/matrix/inverse.hpp>
#include <stan/math/matrix/sum.hpp>
#include <stan/math/matrix/trace.hpp>
#include <stan/math/matrix/transpose.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev/matrix/multiply.hpp>
#include <stan/agrad/rev/matrix/sum.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>

TEST(AgradRevMatrix, trace_gen_inv_quad_form_ldlt) {
  using stan::agrad::matrix_v;
  using stan::math::matrix_d;
  
  matrix_v av(4,4);
  matrix_d ad(4,4);
  matrix_d bd(4,2);
  matrix_v bv(4,2);
  matrix_d cd(2,2);
  matrix_v cv(2,2);
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
  cd.setIdentity(2,2);
  cv.setIdentity(2,2);

  stan::math::LDLT_factor<double,-1,-1> ldlt_ad;
  stan::math::LDLT_factor<stan::agrad::var,-1,-1> ldlt_av;
  ldlt_av.compute(av);
  ASSERT_TRUE(ldlt_av.success());
  ldlt_ad.compute(ad);
  ASSERT_TRUE(ldlt_ad.success());
  
  // double-double-double
  res = trace_gen_inv_quad_form_ldlt(cd,ldlt_ad,bd);
  EXPECT_FLOAT_EQ(1439.1061766207, res.val());
  
  // double-var-double
  res = trace_gen_inv_quad_form_ldlt(cd,ldlt_av,bd);
  EXPECT_FLOAT_EQ(1439.1061766207, res.val());
  
  // double-double-var
  res = trace_gen_inv_quad_form_ldlt(cd,ldlt_ad,bv);
  EXPECT_FLOAT_EQ(1439.1061766207, res.val());
  
  // double-var-var
  res = trace_gen_inv_quad_form_ldlt(cd,ldlt_av,bv);
  EXPECT_FLOAT_EQ(1439.1061766207, res.val());

  // var-double-double
  res = trace_gen_inv_quad_form_ldlt(cv,ldlt_ad,bd);
  EXPECT_FLOAT_EQ(1439.1061766207, res.val());
  
  // var-var-double
  res = trace_gen_inv_quad_form_ldlt(cv,ldlt_av,bd);
  EXPECT_FLOAT_EQ(1439.1061766207, res.val());
  
  // var-double-var
  res = trace_gen_inv_quad_form_ldlt(cv,ldlt_ad,bv);
  EXPECT_FLOAT_EQ(1439.1061766207, res.val());
  
  // var-var-var
  res = trace_gen_inv_quad_form_ldlt(cv,ldlt_av,bv);
  EXPECT_FLOAT_EQ(1439.1061766207, res.val());
}

TEST(AgradRevMatrix, trace_gen_inv_quad_form_ldlt_grad_dvv) {
  using stan::math::sum;
  using stan::agrad::matrix_v;
  using stan::math::matrix_d;
  
  matrix_v av(4,4);
  matrix_d ad(4,4);
  matrix_d bd(4,2);
  matrix_v bv(4,2);
  matrix_d cd(2,2);
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
  cd.setIdentity(2,2);

  stan::math::LDLT_factor<stan::agrad::var,-1,-1> ldlt_av;
  ldlt_av.compute(av);
  ASSERT_TRUE(ldlt_av.success());

  matrix_d ainv(ad.inverse());
  matrix_d dqda(-ainv*bd*cd.transpose()*bd.transpose()*ainv);
  matrix_d dqdb(ainv*bd*cd.transpose() + ainv.transpose()*bd*cd);
  
  // var-var
  res = trace_gen_inv_quad_form_ldlt(cd,ldlt_av,bv);
  
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


TEST(AgradRevMatrix, trace_gen_inv_quad_form_ldlt_grad_vdv) {
  using stan::math::sum;
  using stan::agrad::matrix_v;
  using stan::math::matrix_d;
  
  matrix_d ad(4,4);
  matrix_d bd(4,2);
  matrix_v bv(4,2);
  matrix_d cd(2,2);
  matrix_v cv(2,2);
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
  cd.setIdentity(2,2);
  cv.setIdentity(2,2);

  stan::math::LDLT_factor<double,-1,-1> ldlt_ad;
  ldlt_ad.compute(ad);
  ASSERT_TRUE(ldlt_ad.success());

  matrix_d ainv(ad.inverse());
  matrix_d dqdb(ainv*bd*cd.transpose() + ainv.transpose()*bd*cd);
  matrix_d dqdc(bd.transpose()*ainv.transpose()*bd);
  
  // var-var
  res = trace_gen_inv_quad_form_ldlt(cv,ldlt_ad,bv);
  
  vars.clear();
  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      vars.push_back(cv(i,j));
  for (i = 0; i < 4; i++)
    for (j = 0; j < 2; j++)
      vars.push_back(bv(i,j));
  grad = cgradvec(res,vars);
  pos = 0;
  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++, pos++)
      EXPECT_FLOAT_EQ(grad[pos], dqdc(i,j));
  for (i = 0; i < 4; i++)
    for (j = 0; j < 2; j++, pos++)
      EXPECT_FLOAT_EQ(grad[pos], dqdb(i,j));
}

TEST(AgradRevMatrix, trace_gen_inv_quad_form_ldlt_grad_vvd) {
  using stan::math::sum;
  using stan::agrad::matrix_v;
  using stan::math::matrix_d;
  
  matrix_v av(4,4);
  matrix_d ad(4,4);
  matrix_d bd(4,2);
  matrix_d cd(2,2);
  matrix_v cv(2,2);
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
  cd.setIdentity(2,2);
  cv.setIdentity(2,2);

  stan::math::LDLT_factor<stan::agrad::var,-1,-1> ldlt_av;
  ldlt_av.compute(av);
  ASSERT_TRUE(ldlt_av.success());

  matrix_d ainv(ad.inverse());
  matrix_d dqda(-ainv*bd*cd.transpose()*bd.transpose()*ainv);
  matrix_d dqdc(bd.transpose()*ainv.transpose()*bd);
  
  // var-var
  res = trace_gen_inv_quad_form_ldlt(cv,ldlt_av,bd);
  
  vars.clear();
  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      vars.push_back(cv(i,j));
  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++)
      vars.push_back(av(i,j));
  grad = cgradvec(res,vars);
  pos = 0;
  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++, pos++)
      EXPECT_FLOAT_EQ(grad[pos], dqdc(i,j));
  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++, pos++)
      EXPECT_FLOAT_EQ(grad[pos], dqda(i,j));
}


TEST(AgradRevMatrix, trace_gen_inv_quad_form_ldlt_grad_vdd) {
  using stan::math::sum;
  using stan::agrad::matrix_v;
  using stan::math::matrix_d;
  
  matrix_d ad(4,4);
  matrix_d bd(4,2);
  matrix_d cd(2,2);
  matrix_v cv(2,2);
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
  cd.setIdentity(2,2);
  cv.setIdentity(2,2);

  stan::math::LDLT_factor<double,-1,-1> ldlt_ad;
  ldlt_ad.compute(ad);
  ASSERT_TRUE(ldlt_ad.success());

  matrix_d ainv(ad.inverse());
  matrix_d dqdc(bd.transpose()*ainv.transpose()*bd);
  
  // var-var
  res = trace_gen_inv_quad_form_ldlt(cv,ldlt_ad,bd);
  
  vars.clear();
  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      vars.push_back(cv(i,j));
  grad = cgradvec(res,vars);
  pos = 0;
  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++, pos++)
      EXPECT_FLOAT_EQ(grad[pos], dqdc(i,j));
}

TEST(AgradRevMatrix, trace_gen_inv_quad_form_ldlt_grad_dvd) {
  using stan::math::sum;
  using stan::agrad::matrix_v;
  using stan::math::matrix_d;
  
  matrix_v av(4,4);
  matrix_d ad(4,4);
  matrix_d bd(4,2);
  matrix_d cd(2,2);
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
  cd.setIdentity(2,2);

  stan::math::LDLT_factor<stan::agrad::var,-1,-1> ldlt_av;
  ldlt_av.compute(av);
  ASSERT_TRUE(ldlt_av.success());

  matrix_d ainv(ad.inverse());
  matrix_d dqda(-ainv*bd*cd.transpose()*bd.transpose()*ainv);
  
  // var-var
  res = trace_gen_inv_quad_form_ldlt(cd,ldlt_av,bd);
  
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

TEST(AgradRevMatrix, trace_gen_inv_quad_form_ldlt_grad_ddv) {
  using stan::math::sum;
  using stan::agrad::matrix_v;
  using stan::math::matrix_d;
  
  matrix_d ad(4,4);
  matrix_d bd(4,2);
  matrix_v bv(4,2);
  matrix_d cd(2,2);
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
  cd.setIdentity(2,2);

  stan::math::LDLT_factor<double,-1,-1> ldlt_ad;
  ldlt_ad.compute(ad);
  ASSERT_TRUE(ldlt_ad.success());

  matrix_d ainv(ad.inverse());
  matrix_d dqdb(ainv*bd*cd.transpose() + ainv.transpose()*bd*cd);
  
  // var-var
  res = trace_gen_inv_quad_form_ldlt(cd,ldlt_ad,bv);
  
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

TEST(AgradRevMatrix, trace_gen_inv_quad_form_ldlt_grad_vvv) {
  using stan::math::sum;
  using stan::agrad::matrix_v;
  using stan::math::matrix_d;
  
  matrix_v av(4,4);
  matrix_d ad(4,4);
  matrix_d bd(4,2);
  matrix_v bv(4,2);
  matrix_d cd(2,2);
  matrix_v cv(2,2);
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
  cd.setIdentity(2,2);
  cv.setIdentity(2,2);

  stan::math::LDLT_factor<stan::agrad::var,-1,-1> ldlt_av;
  ldlt_av.compute(av);
  ASSERT_TRUE(ldlt_av.success());

  matrix_d ainv(ad.inverse());
  matrix_d dqda(-ainv*bd*cd.transpose()*bd.transpose()*ainv);
  matrix_d dqdb(ainv*bd*cd.transpose() + ainv.transpose()*bd*cd);
  matrix_d dqdc(bd.transpose()*ainv.transpose()*bd);
  
  // var-var
  res = trace_gen_inv_quad_form_ldlt(cv,ldlt_av,bv);
  
  vars.clear();
  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      vars.push_back(cv(i,j));
  for (i = 0; i < 4; i++)
    for (j = 0; j < 2; j++)
      vars.push_back(bv(i,j));
  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++)
      vars.push_back(av(i,j));
  grad = cgradvec(res,vars);
  pos = 0;
  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++, pos++)
      EXPECT_FLOAT_EQ(grad[pos], dqdc(i,j));
  for (i = 0; i < 4; i++)
    for (j = 0; j < 2; j++, pos++)
      EXPECT_FLOAT_EQ(grad[pos], dqdb(i,j));
  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++, pos++)
      EXPECT_FLOAT_EQ(grad[pos], dqda(i,j));
}

TEST(AgradRevMatrix, trace_gen_inv_quad_form_ldlt_grad_dvv_basic) {
  using stan::math::sum;
  using stan::agrad::matrix_v;
  using stan::math::matrix_d;
  using stan::math::trace;
  using stan::math::multiply;
  using stan::math::inverse;
  
  matrix_d cd(2,2);
  matrix_v av(4,4);
  matrix_v bv(4,2);
  AVAR result, result_basic;
  double result_val, result_basic_val;
  AVEC vars;
  VEC grad, grad_basic;
  size_type i,j;
  stan::math::LDLT_factor<stan::agrad::var,-1,-1> ldlt_av;
  
  // calculate gradient using trace_gen_inv_quad_form_ldlt
  bv << 100, 10,
          0,  1,
         -3, -3,
          5,  2;
  av << 9.0,  3.0, 3.0,   3.0, 
        3.0, 10.0, 2.0,   2.0,
        3.0,  2.0, 7.0,   1.0,
        3.0,  2.0, 1.0, 112.0;
  cd << 1, 2, 3, 4;
  ldlt_av.compute(av);
  ASSERT_TRUE(ldlt_av.success());
  result = trace_gen_inv_quad_form_ldlt(cd,ldlt_av,bv);
  
  vars.clear();
  for (i = 0; i < bv.rows(); i++)
    for (j = 0; j < bv.cols(); j++)
      vars.push_back(bv(i,j));
  for (i = 0; i < av.rows(); i++)
    for (j = 0; j < av.cols(); j++)
      vars.push_back(av(i,j));
  grad = cgradvec(result,vars);
  result_val = result.val();
  

  // calculate gradient using basic math
  bv << 100, 10,
          0,  1,
         -3, -3,
          5,  2;
  av << 9.0,  3.0, 3.0,   3.0, 
        3.0, 10.0, 2.0,   2.0,
        3.0,  2.0, 7.0,   1.0,
        3.0,  2.0, 1.0, 112.0;
  cd << 1, 2, 3, 4;
  
  matrix_v tmp = bv.transpose() * inverse(av) * bv;
  matrix_v gen_inv_quad_form = multiply(cd, tmp);
  result_basic = trace(gen_inv_quad_form);
  vars.clear();
  for (i = 0; i < bv.rows(); i++)
    for (j = 0; j < bv.cols(); j++)
      vars.push_back(bv(i,j));
  for (i = 0; i < av.rows(); i++)
    for (j = 0; j < av.cols(); j++)
      vars.push_back(av(i,j));
  grad_basic = cgradvec(result_basic,vars);
  result_basic_val = result_basic.val();
  

  // check values;
  EXPECT_FLOAT_EQ(result_basic_val, result_val);
  ASSERT_EQ(grad_basic.size(), grad.size());
  for (size_t n = 0; n < grad_basic.size(); n++) {
    EXPECT_FLOAT_EQ(grad_basic[n], grad[n]);
  }
}

TEST(AgradRevMatrix, trace_gen_inv_quad_form_ldlt_grad_vdv_basic) {
  using stan::math::sum;
  using stan::agrad::matrix_v;
  using stan::math::matrix_d;
  using stan::math::inverse;
  using stan::math::multiply;
  using stan::math::trace;
  
  matrix_v cv(2,2);
  matrix_d ad(4,4);
  matrix_v bv(4,2);
  AVAR result, result_basic;
  double result_val, result_basic_val;
  AVEC vars;
  VEC grad, grad_basic;
  size_t i,j;
  stan::math::LDLT_factor<double,-1,-1> ldlt_ad;

  // calculate gradient using trace_gen_inv_quad_form_ldlt  
  bv << 100, 10,
        0,  1,
        -3, -3,
         5,  2;
  ad << 9.0,  3.0, 3.0,   3.0, 
        3.0, 10.0, 2.0,   2.0,
        3.0,  2.0, 7.0,   1.0,
        3.0,  2.0, 1.0, 112.0;
  cv << 1, 2, 3, 4;

  ldlt_ad.compute(ad);
  ASSERT_TRUE(ldlt_ad.success());
  result = trace_gen_inv_quad_form_ldlt(cv,ldlt_ad,bv);
  
  vars.clear();
  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      vars.push_back(cv(i,j));
  for (i = 0; i < 4; i++)
    for (j = 0; j < 2; j++)
      vars.push_back(bv(i,j));
  grad = cgradvec(result,vars);
  result_val = result.val();


  // calculate gradient using basic math
  bv << 100, 10,
        0,  1,
        -3, -3,
         5,  2;
  ad << 9.0,  3.0, 3.0,   3.0, 
        3.0, 10.0, 2.0,   2.0,
        3.0,  2.0, 7.0,   1.0,
        3.0,  2.0, 1.0, 112.0;
  cv << 1, 2, 3, 4;
  
  matrix_v tmp = bv.transpose();
  matrix_d tmp_d = ad.inverse();
  tmp = multiply(tmp, tmp_d);
  tmp = tmp * bv;
  matrix_v gen_inv_quad_form = multiply(cv, tmp);
  result_basic = trace(gen_inv_quad_form);
  vars.clear();
  vars.clear();
  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      vars.push_back(cv(i,j));
  for (i = 0; i < 4; i++)
    for (j = 0; j < 2; j++)
      vars.push_back(bv(i,j));
  grad_basic = cgradvec(result_basic,vars);
  result_basic_val = result_basic.val();

  // check values;
  EXPECT_FLOAT_EQ(result_basic_val, result_val);
  ASSERT_EQ(grad_basic.size(), grad.size());
  for (size_t n = 0; n < grad_basic.size(); n++) {
    EXPECT_FLOAT_EQ(grad_basic[n], grad[n]);
  }
}


TEST(AgradRevMatrix, trace_gen_inv_quad_form_ldlt_grad_vvd_basic) {
  using stan::math::sum;
  using stan::agrad::matrix_v;
  using stan::math::matrix_d;
  using stan::math::inverse;
  using stan::math::multiply;
  using stan::math::trace;
  
  matrix_v cv(2,2);
  matrix_v av(4,4);
  matrix_d bd(4,2);
  AVAR result, result_basic;
  double result_val, result_basic_val;
  AVEC vars;
  VEC grad, grad_basic;
  size_t i,j;
  stan::math::LDLT_factor<stan::agrad::var,-1,-1> ldlt_av;
  
  // calculate gradient using trace_gen_inv_quad_form_ldlt  
  bd << 100, 10,
          0,  1,
         -3, -3,
          5,  2;
  av << 9.0,  3.0, 3.0,   3.0, 
        3.0, 10.0, 2.0,   2.0,
        3.0,  2.0, 7.0,   1.0,
        3.0,  2.0, 1.0, 112.0;
  cv << 1, 2, 3, 4;

  ldlt_av.compute(av);
  ASSERT_TRUE(ldlt_av.success());
  result = trace_gen_inv_quad_form_ldlt(cv,ldlt_av,bd);
  
  vars.clear();
  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      vars.push_back(cv(i,j));
  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++)
      vars.push_back(av(i,j));
  grad = cgradvec(result,vars);
  result_val = result.val();

  // calculate gradient using basic math
  bd << 100, 10,
          0,  1,
         -3, -3,
          5,  2;
  av << 9.0,  3.0, 3.0,   3.0, 
        3.0, 10.0, 2.0,   2.0,
        3.0,  2.0, 7.0,   1.0,
        3.0,  2.0, 1.0, 112.0;
  cv << 1, 2, 3, 4;
  
  matrix_d tmp_d = bd.transpose();
  matrix_v tmp = inverse(av);
  tmp = multiply(tmp_d, tmp);
  tmp = multiply(tmp, bd);
  matrix_v gen_inv_quad_form = cv * tmp;

  result_basic = trace(gen_inv_quad_form);

  vars.clear();
  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      vars.push_back(cv(i,j));
  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++)
      vars.push_back(av(i,j));
  grad_basic = cgradvec(result_basic,vars);
  result_basic_val = result_basic.val();
  
  // check values;
  EXPECT_FLOAT_EQ(result_basic_val, result_val);
  ASSERT_EQ(grad_basic.size(), grad.size());
  for (size_t n = 0; n < grad_basic.size(); n++) {
    EXPECT_FLOAT_EQ(grad_basic[n], grad[n]);
  }
}


TEST(AgradRevMatrix, trace_gen_inv_quad_form_ldlt_grad_vdd_basic) {
  using stan::math::sum;
  using stan::agrad::matrix_v;
  using stan::math::matrix_d;
  using stan::math::inverse;
  using stan::math::multiply;
  using stan::math::trace;
  
  matrix_v cv(2,2);
  matrix_d ad(4,4);
  matrix_d bd(4,2);
  AVAR result, result_basic;
  double result_val, result_basic_val;
  AVEC vars;
  VEC grad, grad_basic;
  size_t i,j;
  stan::math::LDLT_factor<double,-1,-1> ldlt_ad;
  
  // calculate gradient using trace_gen_inv_quad_form_ldlt  
  bd << 100, 10,
          0,  1,
         -3, -3,
          5,  2;
  ad << 9.0,  3.0, 3.0,   3.0, 
        3.0, 10.0, 2.0,   2.0,
        3.0,  2.0, 7.0,   1.0,
        3.0,  2.0, 1.0, 112.0;
  cv << 1, 2, 3, 4;

  ldlt_ad.compute(ad);
  ASSERT_TRUE(ldlt_ad.success());
  result = trace_gen_inv_quad_form_ldlt(cv,ldlt_ad,bd);
  
  vars.clear();
  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      vars.push_back(cv(i,j));
  grad = cgradvec(result,vars);
  result_val = result.val();

  // calculate gradient using basic math
  bd << 100, 10,
          0,  1,
         -3, -3,
          5,  2;
  ad << 9.0,  3.0, 3.0,   3.0, 
        3.0, 10.0, 2.0,   2.0,
        3.0,  2.0, 7.0,   1.0,
        3.0,  2.0, 1.0, 112.0;
  cv << 1, 2, 3, 4;
  
  matrix_d tmp_d = bd.transpose() * ad.inverse()* bd;
  matrix_v gen_inv_quad_form = multiply(cv, tmp_d);

  result_basic = trace(gen_inv_quad_form);

  vars.clear();
  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      vars.push_back(cv(i,j));
  grad_basic = cgradvec(result_basic,vars);
  result_basic_val = result_basic.val();
  
  // check values;
  EXPECT_FLOAT_EQ(result_basic_val, result_val);
  ASSERT_EQ(grad_basic.size(), grad.size());
  for (size_t n = 0; n < grad_basic.size(); n++) {
    EXPECT_FLOAT_EQ(grad_basic[n], grad[n]);
  }
}

TEST(AgradRevMatrix, trace_gen_inv_quad_form_ldlt_grad_dvd_basic) {
  using stan::math::sum;
  using stan::agrad::matrix_v;
  using stan::math::matrix_d;
  using stan::math::inverse;
  using stan::math::multiply;
  using stan::math::trace;
  
  matrix_d cd(2,2);
  matrix_v av(4,4);
  matrix_d bd(4,2);
  AVAR result, result_basic;
  double result_val, result_basic_val;
  AVEC vars;
  VEC grad, grad_basic;
  size_t i,j;
  stan::math::LDLT_factor<stan::agrad::var,-1,-1> ldlt_av;
  
  // calculate gradient using trace_gen_inv_quad_form_ldlt  
  bd << 100, 10,
          0,  1,
         -3, -3,
          5,  2;
  av << 9.0,  3.0, 3.0,   3.0, 
        3.0, 10.0, 2.0,   2.0,
        3.0,  2.0, 7.0,   1.0,
        3.0,  2.0, 1.0, 112.0;
  cd << 1, 2, 3, 4;

  ldlt_av.compute(av);
  ASSERT_TRUE(ldlt_av.success());
  result = trace_gen_inv_quad_form_ldlt(cd,ldlt_av,bd);
  
  vars.clear();
  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++)
      vars.push_back(av(i,j));
  grad = cgradvec(result,vars);
  result_val = result.val();

  // calculate gradient using basic math
  bd << 100, 10,
          0,  1,
         -3, -3,
          5,  2;
  av << 9.0,  3.0, 3.0,   3.0, 
        3.0, 10.0, 2.0,   2.0,
        3.0,  2.0, 7.0,   1.0,
        3.0,  2.0, 1.0, 112.0;
  cd << 1, 2, 3, 4;
  
  matrix_d tmp_d = bd.transpose();
  matrix_v tmp = inverse(av);
  tmp = multiply(tmp_d, tmp);
  tmp = multiply(tmp, bd);
  matrix_v gen_inv_quad_form = multiply(cd, tmp);

  result_basic = trace(gen_inv_quad_form);

  vars.clear();
  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++)
      vars.push_back(av(i,j));
  grad_basic = cgradvec(result_basic,vars);
  result_basic_val = result_basic.val();
  
  // check values;
  EXPECT_FLOAT_EQ(result_basic_val, result_val);
  ASSERT_EQ(grad_basic.size(), grad.size());
  for (size_t n = 0; n < grad_basic.size(); n++) {
    EXPECT_FLOAT_EQ(grad_basic[n], grad[n]);
  }
}

TEST(AgradRevMatrix, trace_gen_inv_quad_form_ldlt_grad_ddv_basic) {
  using stan::math::sum;
  using stan::agrad::matrix_v;
  using stan::math::matrix_d;
  using stan::math::inverse;
  using stan::math::multiply;
  using stan::math::trace;
  
  matrix_d cd(2,2);
  matrix_d ad(4,4);
  matrix_v bv(4,2);
  stan::math::LDLT_factor<double,-1,-1> ldlt_ad;
  AVAR result, result_basic;
  double result_val, result_basic_val;
  AVEC vars;
  VEC grad, grad_basic;
  size_t i,j;
  
  // calculate gradient using trace_gen_inv_quad_form_ldlt  
  cd << 1, 2, 3, 4;
  ad << 9.0,  3.0, 3.0,   3.0, 
        3.0, 10.0, 2.0,   2.0,
        3.0,  2.0, 7.0,   1.0,
        3.0,  2.0, 1.0, 112.0;
  bv << 100, 10,
          0,  1,
         -3, -3,
          5,  2;
  ldlt_ad.compute(ad);
  ASSERT_TRUE(ldlt_ad.success());
  result = trace_gen_inv_quad_form_ldlt(cd,ldlt_ad,bv);
  
  vars.clear();
  for (i = 0; i < 4; i++)
    for (j = 0; j < 2; j++)
      vars.push_back(bv(i,j));
  grad = cgradvec(result,vars);
  result_val = result.val();

  // calculate gradient using basic math
  cd << 1, 2, 3, 4;
  ad << 9.0,  3.0, 3.0,   3.0, 
        3.0, 10.0, 2.0,   2.0,
        3.0,  2.0, 7.0,   1.0,
        3.0,  2.0, 1.0, 112.0;
  bv << 100, 10,
          0,  1,
         -3, -3,
          5,  2;

  matrix_v tmp = bv.transpose();
  matrix_d tmp_d = ad.inverse();
  tmp = multiply(tmp, tmp_d);
  tmp = multiply(tmp, bv);
  matrix_v gen_inv_quad_form = multiply(cd, tmp);
  result_basic = trace(gen_inv_quad_form);

  vars.clear();
  for (i = 0; i < 4; i++)
    for (j = 0; j < 2; j++)
      vars.push_back(bv(i,j));
  grad_basic = cgradvec(result_basic,vars);
  result_basic_val = result_basic.val();
  
  // check values;
  EXPECT_FLOAT_EQ(result_basic_val, result_val);
  ASSERT_EQ(grad_basic.size(), grad.size());
  for (size_t n = 0; n < grad_basic.size(); n++) {
    EXPECT_FLOAT_EQ(grad_basic[n], grad[n]);
  }
}

TEST(AgradRevMatrix, trace_gen_inv_quad_form_ldlt_grad_vvv_basic) {
  using stan::math::sum;
  using stan::agrad::matrix_v;
  using stan::math::matrix_d;
  using stan::math::inverse;
  using stan::math::multiply;
  using stan::math::trace;
  using stan::math::transpose;
  
  matrix_v cv(2,2);
  matrix_v av(4,4);
  matrix_v bv(4,2);
  AVAR result, result_basic;
  double result_val, result_basic_val;
  AVEC vars;
  VEC grad, grad_basic;
  size_t i,j;
  stan::math::LDLT_factor<stan::agrad::var,-1,-1> ldlt_av;
  
  // calculate gradient using trace_gen_inv_quad_form_ldlt  
  bv << 100, 10,
          0,  1,
         -3, -3,
          5,  2;
  av << 9.0,  3.0, 3.0,   3.0, 
        3.0, 10.0, 2.0,   2.0,
        3.0,  2.0, 7.0,   1.0,
        3.0,  2.0, 1.0, 112.0;
  cv << 1, 2, 3, 4;

  ldlt_av.compute(av);
  ASSERT_TRUE(ldlt_av.success());
  result = trace_gen_inv_quad_form_ldlt(cv,ldlt_av,bv);
  
  vars.clear();
  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      vars.push_back(cv(i,j));
  for (i = 0; i < 4; i++)
    for (j = 0; j < 2; j++)
      vars.push_back(bv(i,j));
  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++)
      vars.push_back(av(i,j));
  grad = cgradvec(result,vars);
  result_val = result.val();

  // calculate gradient using basic math
  bv << 100, 10,
          0,  1,
         -3, -3,
          5,  2;
  av << 9.0,  3.0, 3.0,   3.0, 
        3.0, 10.0, 2.0,   2.0,
        3.0,  2.0, 7.0,   1.0,
        3.0,  2.0, 1.0, 112.0;
  cv << 1, 2, 3, 4;

  matrix_v tmp = bv.transpose() * inverse(av) * bv;
  matrix_v gen_inv_quad_form = multiply(cv, tmp);
  result_basic = trace(gen_inv_quad_form);


  vars.clear();
  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      vars.push_back(cv(i,j));
  for (i = 0; i < 4; i++)
    for (j = 0; j < 2; j++)
      vars.push_back(bv(i,j));
  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++)
      vars.push_back(av(i,j));
  grad_basic = cgradvec(result_basic,vars);
  result_basic_val = result_basic.val();
  
  // check values;
  EXPECT_FLOAT_EQ(result_basic_val, result_val);
  ASSERT_EQ(grad_basic.size(), grad.size());
  for (size_t n = 0; n < grad_basic.size(); n++) {
    EXPECT_FLOAT_EQ(grad_basic[n], grad[n]);
  }
}

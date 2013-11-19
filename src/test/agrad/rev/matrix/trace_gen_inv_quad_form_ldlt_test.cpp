#include <stan/agrad/rev/matrix/ldlt.hpp>
#include <stan/agrad/rev/matrix/sum.hpp>
#include <gtest/gtest.h>
#include <test/agrad/util.hpp>
#include <stan/math/matrix/trace_gen_inv_quad_form_ldlt.hpp>
#include <stan/math/matrix/sum.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>

TEST(AgradRevMatrix, trace_gen_inv_quad_form_ldlt_mat) {
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
  ldlt_ad.compute(ad);
  
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

TEST(AgradRevMatrix, trace_gen_inv_quad_form_ldlt_mat_grad_dvv) {
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

TEST(AgradRevMatrix, trace_gen_inv_quad_form_ldlt_mat_grad_vdv) {
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

TEST(AgradRevMatrix, trace_gen_inv_quad_form_ldlt_mat_grad_vvd) {
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

TEST(AgradRevMatrix, trace_gen_inv_quad_form_ldlt_mat_grad_vdd) {
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

TEST(AgradRevMatrix, trace_gen_inv_quad_form_ldlt_mat_grad_dvd) {
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

TEST(AgradRevMatrix, trace_gen_inv_quad_form_ldlt_mat_grad_ddv) {
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

TEST(AgradRevMatrix, trace_gen_inv_quad_form_ldlt_mat_grad_vvv) {
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


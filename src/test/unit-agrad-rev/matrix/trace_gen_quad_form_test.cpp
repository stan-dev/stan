#include <stan/agrad/rev/matrix/trace_gen_quad_form.hpp>
#include <stan/agrad/rev/matrix/sum.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/math/matrix/trace_gen_quad_form.hpp>
#include <stan/math/matrix/sum.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>

TEST(AgradRevMatrix, trace_gen_quad_form_mat) {
  using stan::math::trace_gen_quad_form;
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
  ad << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  av << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  cd.setIdentity(2,2);
  cv.setIdentity(2,2);
  
  // double-double-double
  res = trace_gen_quad_form(cd,ad,bd);
  EXPECT_FLOAT_EQ(26758, res.val());
  
  // double-var-double
  res = trace_gen_quad_form(cd,av,bd);
  EXPECT_FLOAT_EQ(26758, res.val());
  
  // double-double-var
  res = trace_gen_quad_form(cd,ad,bv);
  EXPECT_FLOAT_EQ(26758, res.val());
  
  // double-var-var
  res = trace_gen_quad_form(cd,av,bv);
  EXPECT_FLOAT_EQ(26758, res.val());

  // var-double-double
  res = trace_gen_quad_form(cv,ad,bd);
  EXPECT_FLOAT_EQ(26758, res.val());
  
  // var-var-double
  res = trace_gen_quad_form(cv,av,bd);
  EXPECT_FLOAT_EQ(26758, res.val());
  
  // var-double-var
  res = trace_gen_quad_form(cv,ad,bv);
  EXPECT_FLOAT_EQ(26758, res.val());
  
  // var-var-var
  res = trace_gen_quad_form(cv,av,bv);
  EXPECT_FLOAT_EQ(26758, res.val());
}

TEST(AgradRevMatrix, trace_gen_quad_form_mat_grad_dvd) {
  using stan::math::trace_gen_quad_form;
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
  ad << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  av << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  cd.setIdentity(2,2);
  cv.setIdentity(2,2);
  
  matrix_d dqda(bd*cd.transpose()*bd.transpose());
  matrix_d dqdb(ad*bd*cd.transpose() + ad.transpose()*bd*cd);
  matrix_d dqdc(bd.transpose()*ad.transpose()*bd);
  
  // var-var
  res = trace_gen_quad_form(cd,av,bd);
  
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

TEST(AgradRevMatrix, trace_gen_quad_form_mat_grad_ddv) {
  using stan::math::trace_gen_quad_form;
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
  ad << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  av << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  cd.setIdentity(2,2);
  cv.setIdentity(2,2);
  
  matrix_d dqdb(ad*bd*cd.transpose() + ad.transpose()*bd*cd);
  
  // var-var
  res = trace_gen_quad_form(cd,ad,bv);
  
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

TEST(AgradRevMatrix, trace_gen_quad_form_mat_grad_vdd) {
  using stan::math::trace_gen_quad_form;
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
  ad << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  av << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  cd.setIdentity(2,2);
  cv.setIdentity(2,2);
  
  matrix_d dqdc(bd.transpose()*ad.transpose()*bd);
  
  res = trace_gen_quad_form(cv,ad,bd);
  
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

TEST(AgradRevMatrix, trace_gen_quad_form_mat_grad_vvd) {
  using stan::math::trace_gen_quad_form;
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
  ad << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  av << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  cd.setIdentity(2,2);
  cv.setIdentity(2,2);
  
  matrix_d dqda(bd*cd.transpose()*bd.transpose());
  matrix_d dqdc(bd.transpose()*ad.transpose()*bd);
  
  res = trace_gen_quad_form(cv,av,bd);
  
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

TEST(AgradRevMatrix, trace_gen_quad_form_mat_grad_vdv) {
  using stan::math::trace_gen_quad_form;
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
  ad << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  av << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  cd.setIdentity(2,2);
  cv.setIdentity(2,2);
  
  matrix_d dqdb(ad*bd*cd.transpose() + ad.transpose()*bd*cd);
  matrix_d dqdc(bd.transpose()*ad.transpose()*bd);
  
  res = trace_gen_quad_form(cv,ad,bv);
  
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

TEST(AgradRevMatrix, trace_gen_quad_form_mat_grad_dvv) {
  using stan::math::trace_gen_quad_form;
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
  ad << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  av << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  cd.setIdentity(2,2);
  cv.setIdentity(2,2);
  
  matrix_d dqda(bd*cd.transpose()*bd.transpose());
  matrix_d dqdb(ad*bd*cd.transpose() + ad.transpose()*bd*cd);
  
  res = trace_gen_quad_form(cd,av,bv);
  
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

TEST(AgradRevMatrix, trace_gen_quad_form_mat_grad_vvv) {
  using stan::math::trace_gen_quad_form;
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
  ad << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  av << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  cd.setIdentity(2,2);
  cv.setIdentity(2,2);
  
  matrix_d dqda(bd*cd.transpose()*bd.transpose());
  matrix_d dqdb(ad*bd*cd.transpose() + ad.transpose()*bd*cd);
  matrix_d dqdc(bd.transpose()*ad.transpose()*bd);
  
  // var-var
  res = trace_gen_quad_form(cv,av,bv);
  
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



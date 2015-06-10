#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/rev/mat/fun/quad_form.hpp>
#include <stan/math/rev/mat/fun/quad_form_sym.hpp>
#include <stan/math/prim/mat/fun/quad_form_sym.hpp>
#include <stan/math/rev/mat/fun/sum.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/prim/mat/fun/quad_form.hpp>
#include <stan/math/prim/mat/fun/sum.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>
#include <stan/math/rev/mat/fun/multiply.hpp>
#include <stan/math/rev/mat/fun/dot_product.hpp>
#include <stan/math/rev/core.hpp>

TEST(AgradRevMatrix, quad_form_mat) {
  using stan::math::quad_form;
  using stan::math::matrix_v;
  using stan::math::matrix_d;
  
  matrix_v av(4,4);
  matrix_d ad(4,4);
  matrix_d bd(4,2);
  matrix_v bv(4,2);
  matrix_v res;
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
  
  // double-double
  matrix_d resd = quad_form(ad,bd);
  EXPECT_FLOAT_EQ(26033, resd(0,0));
  EXPECT_FLOAT_EQ(3456, resd(0,1));
  EXPECT_FLOAT_EQ(3396, resd(1,0));
  EXPECT_FLOAT_EQ(725, resd(1,1));

  // var-double
  res = quad_form(av,bd);
  EXPECT_FLOAT_EQ(26033, res(0,0).val());
  EXPECT_FLOAT_EQ(3456, res(0,1).val());
  EXPECT_FLOAT_EQ(3396, res(1,0).val());
  EXPECT_FLOAT_EQ(725, res(1,1).val());
  
  // double-var
  res = quad_form(ad,bv);
  EXPECT_FLOAT_EQ(26033, res(0,0).val());
  EXPECT_FLOAT_EQ(3456, res(0,1).val());
  EXPECT_FLOAT_EQ(3396, res(1,0).val());
  EXPECT_FLOAT_EQ(725, res(1,1).val());
  
  // var-var
  res = quad_form(av,bv);
  EXPECT_FLOAT_EQ(26033, res(0,0).val());
  EXPECT_FLOAT_EQ(3456, res(0,1).val());
  EXPECT_FLOAT_EQ(3396, res(1,0).val());
  EXPECT_FLOAT_EQ(725, res(1,1).val());
}

TEST(AgradRevMatrix, quad_form_mat_grad_vd) {
  using stan::math::quad_form;
  using stan::math::sum;
  using stan::math::matrix_v;
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
  ad << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  av << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  
  matrix_d dqda(bd*matrix_d::Ones(2,2)*bd.transpose());
  
  // var-double
  res = sum(quad_form(av,bd));
  
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

TEST(AgradRevMatrix, quad_form_mat_grad_dv) {
  using stan::math::quad_form;
  using stan::math::sum;
  using stan::math::matrix_v;
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
  ad << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  
  matrix_d dqdb((ad*bd + ad.transpose()*bd)*matrix_d::Ones(2,2));
  
  // double-var
  res = sum(quad_form(ad,bv));
  
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

TEST(AgradRevMatrix, quad_form_mat_grad_vv) {
  using stan::math::quad_form;
  using stan::math::sum;
  using stan::math::matrix_v;
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
  ad << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  av << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  
  matrix_d dqda(bd*matrix_d::Ones(2,2)*bd.transpose());
  matrix_d dqdb((ad*bd + ad.transpose()*bd)*matrix_d::Ones(2,2));
  
  // var-var
  res = sum(quad_form(av,bv));
  
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

TEST(AgradRevMatrix, quad_form_sym_mat) {
  using stan::math::quad_form_sym;
  using stan::math::matrix_v;
  using stan::math::matrix_d;
  
  matrix_v av(4,4);
  matrix_d ad(4,4);
  matrix_d bd(4,2);
  matrix_v bv(4,2);
  matrix_v res;
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
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  av << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  
  // double-double
  matrix_d resd = quad_form_sym(ad,bd);
  EXPECT_FLOAT_EQ(25433, resd(0,0));
  EXPECT_FLOAT_EQ(3396, resd(0,1));
  EXPECT_FLOAT_EQ(3396, resd(1,0));
  EXPECT_FLOAT_EQ(725, resd(1,1));
  EXPECT_EQ(resd(1,0), resd(0,1));
  
  // var-double
  res = quad_form_sym(av,bd);
  EXPECT_FLOAT_EQ(25433, res(0,0).val());
  EXPECT_FLOAT_EQ(3396, res(0,1).val());
  EXPECT_FLOAT_EQ(3396, res(1,0).val());
  EXPECT_FLOAT_EQ(725, res(1,1).val());
  EXPECT_EQ(res(1,0).val(), res(0,1).val());
  
  // double-var
  res = quad_form_sym(ad,bv);
  EXPECT_FLOAT_EQ(25433, res(0,0).val());
  EXPECT_FLOAT_EQ(3396, res(0,1).val());
  EXPECT_FLOAT_EQ(3396, res(1,0).val());
  EXPECT_FLOAT_EQ(725, res(1,1).val());
  EXPECT_EQ(res(1,0).val(), res(0,1).val());
  
  // var-var
  res = quad_form_sym(av,bv);
  EXPECT_FLOAT_EQ(25433, res(0,0).val());
  EXPECT_FLOAT_EQ(3396, res(0,1).val());
  EXPECT_FLOAT_EQ(3396, res(1,0).val());
  EXPECT_FLOAT_EQ(725, res(1,1).val());
  EXPECT_EQ(res(1,0).val(), res(0,1).val());
}

TEST(AgradRevMatrix, quad_form_sym_mat_grad_vd) {
  using stan::math::quad_form_sym;
  using stan::math::sum;
  using stan::math::matrix_v;
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
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  av << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  
  matrix_d dqda(bd*matrix_d::Ones(2,2)*bd.transpose());
  
  // var-double
  res = sum(quad_form_sym(av,bd));
  
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

TEST(AgradRevMatrix, quad_form_sym_mat_grad_dv) {
  using stan::math::quad_form_sym;
  using stan::math::sum;
  using stan::math::matrix_v;
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
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  
  matrix_d dqdb((ad*bd + ad.transpose()*bd)*matrix_d::Ones(2,2));
  
  // double-var
  res = sum(quad_form_sym(ad,bv));
  
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

TEST(AgradRevMatrix, quad_form_sym_mat_grad_vv) {
  using stan::math::quad_form_sym;
  using stan::math::sum;
  using stan::math::matrix_v;
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
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  av << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  
  matrix_d dqda(bd*matrix_d::Ones(2,2)*bd.transpose());
  matrix_d dqdb((ad*bd + ad.transpose()*bd)*matrix_d::Ones(2,2));
  
  // var-var
  res = sum(quad_form_sym(av,bv));
  
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

TEST(AgradRevMatrix, quad_form_vec) {
  using stan::math::quad_form;
  using stan::math::matrix_v;
  using stan::math::vector_v;
  using stan::math::matrix_d;
  using stan::math::vector_d;

  matrix_v av(4,4);
  matrix_d ad(4,4);
  vector_d bd(4);
  vector_v bv(4);
  stan::math::var res;
  AVEC vars;
  VEC grad;

  
  bd << 100, 0, -3, 5;
  bv << 100, 0, -3, 5;
  ad << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  av << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  
  // double-double
  res = quad_form(ad,bd);
  EXPECT_FLOAT_EQ(26033, res.val());

  // var-double
  res = quad_form(av,bd);
  EXPECT_FLOAT_EQ(26033, res.val());

  // double-var
  res = quad_form(ad,bv);
  EXPECT_FLOAT_EQ(26033, res.val());
   
  // var-var
  res = quad_form(av,bv);
  EXPECT_FLOAT_EQ(26033, res.val());
}

TEST(AgradRevMatrix, quad_form_vec_grad_vd) {
  using stan::math::quad_form;
  using stan::math::matrix_v;
  using stan::math::vector_v;
  using stan::math::matrix_d;
  using stan::math::vector_d;
  
  matrix_v av(4,4);
  matrix_d ad(4,4);
  vector_d bd(4);
  vector_v bv(4);
  stan::math::var res;
  AVEC vars;
  VEC grad;
  size_t pos, i, j;
  
  
  bd << 100, 0, -3, 5;
  bv << 100, 0, -3, 5;
  ad << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  av << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  
  matrix_d dqda(bd*bd.transpose());
  
  // var-double
  res = quad_form(av,bd);
  
  vars.clear();
  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++)
      vars.push_back(av(i,j));
  grad = cgradvec(res,vars);
  for (i = 0, pos = 0; i < 4; i++)
    for (j = 0; j < 4; j++, pos++)
      EXPECT_FLOAT_EQ(grad[pos], dqda(i,j));
}

TEST(AgradRevMatrix, quad_form_vec_grad_dv) {
  using stan::math::quad_form;
  using stan::math::matrix_v;
  using stan::math::vector_v;
  using stan::math::matrix_d;
  using stan::math::vector_d;
  
  matrix_v av(4,4);
  matrix_d ad(4,4);
  vector_d bd(4);
  vector_v bv(4);
  stan::math::var res;
  AVEC vars;
  VEC grad;
  size_t pos, i;
  
  
  bd << 100, 0, -3, 5;
  bv << 100, 0, -3, 5;
  ad << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  av << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  
  vector_d dqdb(ad*bd + ad.transpose()*bd);
  
  // double-var
  res = quad_form(ad,bv);
  
  vars.clear();
  for (i = 0; i < 4; i++)
    vars.push_back(bv[i]);
  grad = cgradvec(res,vars);
  for (i = 0, pos = 0; i < 4; i++, pos++)
    EXPECT_FLOAT_EQ(grad[pos], dqdb[i]);
}

TEST(AgradRevMatrix, quad_form_vec_grad_vv) {
  using stan::math::quad_form;
  using stan::math::matrix_v;
  using stan::math::vector_v;
  using stan::math::matrix_d;
  using stan::math::vector_d;
  
  matrix_v av(4,4);
  matrix_d ad(4,4);
  vector_d bd(4);
  vector_v bv(4);
  stan::math::var res;
  AVEC vars;
  VEC grad;
  size_t pos, i, j;
  
  
  bd << 100, 0, -3, 5;
  bv << 100, 0, -3, 5;
  ad << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  av << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  
  matrix_d dqda(bd*bd.transpose());
  vector_d dqdb(ad*bd + ad.transpose()*bd);
  
  // var-var
  res = quad_form(av,bv);
  
  vars.clear();
  for (size_t i = 0; i < 4; i++)
    vars.push_back(bv[i]);
  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++)
      vars.push_back(av(i,j));
  grad = cgradvec(res,vars);
  for (i = 0, pos = 0; i < 4; i++, pos++)
    EXPECT_FLOAT_EQ(grad[pos], dqdb[i]);
  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++, pos++)
      EXPECT_FLOAT_EQ(grad[pos], dqda(i,j));
}

TEST(AgradRevMatrix, quad_form_sym_vec) {
  using stan::math::quad_form_sym;
  using stan::math::matrix_v;
  using stan::math::vector_v;
  using stan::math::matrix_d;
  using stan::math::vector_d;
  
  matrix_v av(4,4);
  matrix_d ad(4,4);
  vector_d bd(4);
  vector_v bv(4);
  stan::math::var res;
  AVEC vars;
  VEC grad;
  
  
  bd << 100, 0, -3, 5;
  bv << 100, 0, -3, 5;
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  av << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  
  // double-double
  res = quad_form_sym(ad,bd);
  EXPECT_FLOAT_EQ(25433, res.val());

  // var-double
  res = quad_form_sym(av,bd);
  EXPECT_FLOAT_EQ(25433, res.val());
  
  // double-var
  res = quad_form_sym(ad,bv);
  EXPECT_FLOAT_EQ(25433, res.val());
  
  // var-var
  res = quad_form_sym(av,bv);
  EXPECT_FLOAT_EQ(25433, res.val());
}

TEST(AgradRevMatrix, quad_form_sym_vec_grad_vd) {
  using stan::math::quad_form_sym;
  using stan::math::matrix_v;
  using stan::math::vector_v;
  using stan::math::matrix_d;
  using stan::math::vector_d;
  
  matrix_v av(4,4);
  matrix_d ad(4,4);
  vector_d bd(4);
  vector_v bv(4);
  stan::math::var res;
  AVEC vars;
  VEC grad;
  size_t pos, i, j;
  
  
  bd << 100, 0, -3, 5;
  bv << 100, 0, -3, 5;
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  av << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  
  matrix_d dqda(bd*bd.transpose());
  vector_d dqdb(ad*bd + ad.transpose()*bd);
  
  // var-double
  res = quad_form_sym(av,bd);
  
  vars.clear();
  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++)
      vars.push_back(av(i,j));
  grad = cgradvec(res,vars);
  for (i = 0, pos = 0; i < 4; i++)
    for (j = 0; j < 4; j++, pos++)
      EXPECT_FLOAT_EQ(grad[pos], dqda(i,j));
}

TEST(AgradRevMatrix, quad_form_sym_vec_grad_dv) {
  using stan::math::quad_form_sym;
  using stan::math::matrix_v;
  using stan::math::vector_v;
  using stan::math::matrix_d;
  using stan::math::vector_d;
  
  matrix_v av(4,4);
  matrix_d ad(4,4);
  vector_d bd(4);
  vector_v bv(4);
  stan::math::var res;
  AVEC vars;
  VEC grad;
  size_t pos, i;
  
  
  bd << 100, 0, -3, 5;
  bv << 100, 0, -3, 5;
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  av << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  
  matrix_d dqda(bd*bd.transpose());
  vector_d dqdb(ad*bd + ad.transpose()*bd);
  
  // double-var
  res = quad_form_sym(ad,bv);
  
  vars.clear();
  for (i = 0; i < 4; i++)
    vars.push_back(bv[i]);
  grad = cgradvec(res,vars);
  for (i = 0, pos = 0; i < 4; i++, pos++)
    EXPECT_FLOAT_EQ(grad[pos], dqdb[i]);
}

TEST(AgradRevMatrix, quad_form_sym_vec_grad_vv) {
  using stan::math::quad_form_sym;
  using stan::math::matrix_v;
  using stan::math::vector_v;
  using stan::math::matrix_d;
  using stan::math::vector_d;
  
  matrix_v av(4,4);
  matrix_d ad(4,4);
  vector_d bd(4);
  vector_v bv(4);
  stan::math::var res;
  AVEC vars;
  VEC grad;
  size_t pos, i, j;
  
  
  bd << 100, 0, -3, 5;
  bv << 100, 0, -3, 5;
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  av << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  
  matrix_d dqda(bd*bd.transpose());
  vector_d dqdb(ad*bd + ad.transpose()*bd);
  
  // var-var
  res = quad_form_sym(av,bv);
  
  vars.clear();
  for (size_t i = 0; i < 4; i++)
    vars.push_back(bv[i]);
  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++)
      vars.push_back(av(i,j));
  grad = cgradvec(res,vars);
  for (i = 0, pos = 0; i < 4; i++, pos++)
    EXPECT_FLOAT_EQ(grad[pos], dqdb[i]);
  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++, pos++)
      EXPECT_FLOAT_EQ(grad[pos], dqda(i,j));
}


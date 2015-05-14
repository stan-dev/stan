#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/rev/mat/fun/trace_quad_form.hpp>
#include <stan/math/rev/mat/fun/sum.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/prim/mat/fun/trace_quad_form.hpp>
#include <stan/math/prim/mat/fun/sum.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>

TEST(AgradRevMatrix, trace_quad_form_mat) {
  using stan::math::trace_quad_form;
  using stan::math::matrix_v;
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
  ad << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  av << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  
  // double-double
  res = trace_quad_form(ad,bd);
  EXPECT_FLOAT_EQ(26758, res.val());

  // var-double
  res = trace_quad_form(av,bd);
  EXPECT_FLOAT_EQ(26758, res.val());
  
  // double-var
  res = trace_quad_form(ad,bv);
  EXPECT_FLOAT_EQ(26758, res.val());
  
  // var-var
  res = trace_quad_form(av,bv);
  EXPECT_FLOAT_EQ(26758, res.val());
}

TEST(AgradRevMatrix, trace_quad_form_mat_grad_vd) {
  using stan::math::trace_quad_form;
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
  
  matrix_d dqda(bd*bd.transpose());
  
  // var-var
  res = trace_quad_form(av,bd);
  
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

TEST(AgradRevMatrix, trace_quad_form_mat_grad_dv) {
  using stan::math::trace_quad_form;
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
  
  matrix_d dqdb(ad*bd + ad.transpose()*bd);
  
  // var-var
  res = trace_quad_form(ad,bv);
  
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

TEST(AgradRevMatrix, trace_quad_form_mat_grad_vv) {
  using stan::math::trace_quad_form;
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
  
  matrix_d dqda(bd*bd.transpose());
  matrix_d dqdb(ad*bd + ad.transpose()*bd);
  
  // var-var
  res = trace_quad_form(av,bv);
  
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


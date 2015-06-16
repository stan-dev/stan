#include <stan/math/fwd/mat/fun/trace_quad_form.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>

using stan::math::fvar;
using stan::math::var;

TEST(AgradMixMatrixTraceQuadForm, mat_fv_1st_deriv) {
  using stan::math::trace_quad_form;
  using stan::math::matrix_fv;
  
  matrix_fv ad(4,4);
  matrix_fv bd(4,2);
  fvar<var> res;
  bd << 100, 10,
          0,  1,
         -3, -3,
          5,  2;
  ad << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(0,2).d_ = 1.0;
  ad(0,3).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  ad(1,2).d_ = 1.0;
  ad(1,3).d_ = 1.0;
  ad(2,0).d_ = 1.0;
  ad(2,1).d_ = 1.0;
  ad(2,2).d_ = 1.0;
  ad(2,3).d_ = 1.0;
  ad(3,0).d_ = 1.0;
  ad(3,1).d_ = 1.0;
  ad(3,2).d_ = 1.0;
  ad(3,3).d_ = 1.0;
  bd(0,0).d_ = 1.0;
  bd(0,1).d_ = 1.0;
  bd(1,0).d_ = 1.0;
  bd(1,1).d_ = 1.0;
  bd(2,0).d_ = 1.0;
  bd(2,1).d_ = 1.0;
  bd(3,0).d_ = 1.0;
  bd(3,1).d_ = 1.0;

  // fvar<var> - fvar<var>
  res = trace_quad_form(ad,bd);
  EXPECT_FLOAT_EQ(26758, res.val_.val());
  EXPECT_FLOAT_EQ(16126, res.d_.val());


  std::vector<var> z;
  z.push_back(ad(0,0).val_);
  z.push_back(ad(0,1).val_);
  z.push_back(ad(0,2).val_);
  z.push_back(ad(0,3).val_);
  z.push_back(ad(1,0).val_);
  z.push_back(ad(1,1).val_);
  z.push_back(ad(1,2).val_);
  z.push_back(ad(1,3).val_);
  z.push_back(ad(2,0).val_);
  z.push_back(ad(2,1).val_);
  z.push_back(ad(2,2).val_);
  z.push_back(ad(2,3).val_);
  z.push_back(ad(3,0).val_);
  z.push_back(ad(3,1).val_);
  z.push_back(ad(3,2).val_);
  z.push_back(ad(3,3).val_);
  z.push_back(bd(0,0).val_);
  z.push_back(bd(0,1).val_);
  z.push_back(bd(1,0).val_);
  z.push_back(bd(1,1).val_);
  z.push_back(bd(2,0).val_);
  z.push_back(bd(2,1).val_);
  z.push_back(bd(3,0).val_);
  z.push_back(bd(3,1).val_);

  VEC h;
  res.val_.grad(z,h);
  EXPECT_FLOAT_EQ(10100,h[0]);
  EXPECT_FLOAT_EQ(10,h[1]);
  EXPECT_FLOAT_EQ(-330,h[2]);
  EXPECT_FLOAT_EQ(520,h[3]);
  EXPECT_FLOAT_EQ(10,h[4]);
  EXPECT_FLOAT_EQ(1,h[5]);
  EXPECT_FLOAT_EQ(-3,h[6]);
  EXPECT_FLOAT_EQ(2,h[7]);
  EXPECT_FLOAT_EQ(-330,h[8]);
  EXPECT_FLOAT_EQ(-3,h[9]);
  EXPECT_FLOAT_EQ(18,h[10]);
  EXPECT_FLOAT_EQ(-21,h[11]);
  EXPECT_FLOAT_EQ(520,h[12]);
  EXPECT_FLOAT_EQ(2,h[13]);
  EXPECT_FLOAT_EQ(-21,h[14]);
  EXPECT_FLOAT_EQ(29,h[15]);
  EXPECT_FLOAT_EQ(432,h[16]);
  EXPECT_FLOAT_EQ(42,h[17]);
  EXPECT_FLOAT_EQ(908,h[18]);
  EXPECT_FLOAT_EQ(106,h[19]);
  EXPECT_FLOAT_EQ(1068,h[20]);
  EXPECT_FLOAT_EQ(76,h[21]);
  EXPECT_FLOAT_EQ(2414,h[22]);
  EXPECT_FLOAT_EQ(576,h[23]);
}


TEST(AgradMixMatrixTraceQuadForm, mat_fv_2nd_deriv) {
  using stan::math::trace_quad_form;
  using stan::math::matrix_fv;
  
  matrix_fv ad(4,4);
  matrix_fv bd(4,2);
  fvar<var> res;
  bd << 100, 10,
          0,  1,
         -3, -3,
          5,  2;
  ad << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(0,2).d_ = 1.0;
  ad(0,3).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  ad(1,2).d_ = 1.0;
  ad(1,3).d_ = 1.0;
  ad(2,0).d_ = 1.0;
  ad(2,1).d_ = 1.0;
  ad(2,2).d_ = 1.0;
  ad(2,3).d_ = 1.0;
  ad(3,0).d_ = 1.0;
  ad(3,1).d_ = 1.0;
  ad(3,2).d_ = 1.0;
  ad(3,3).d_ = 1.0;
  bd(0,0).d_ = 1.0;
  bd(0,1).d_ = 1.0;
  bd(1,0).d_ = 1.0;
  bd(1,1).d_ = 1.0;
  bd(2,0).d_ = 1.0;
  bd(2,1).d_ = 1.0;
  bd(3,0).d_ = 1.0;
  bd(3,1).d_ = 1.0;

  // fvar<var> - fvar<var>
  res = trace_quad_form(ad,bd);
  EXPECT_FLOAT_EQ(26758, res.val_.val());
  EXPECT_FLOAT_EQ(16126, res.d_.val());


  std::vector<var> z;
  z.push_back(ad(0,0).val_);
  z.push_back(ad(0,1).val_);
  z.push_back(ad(0,2).val_);
  z.push_back(ad(0,3).val_);
  z.push_back(ad(1,0).val_);
  z.push_back(ad(1,1).val_);
  z.push_back(ad(1,2).val_);
  z.push_back(ad(1,3).val_);
  z.push_back(ad(2,0).val_);
  z.push_back(ad(2,1).val_);
  z.push_back(ad(2,2).val_);
  z.push_back(ad(2,3).val_);
  z.push_back(ad(3,0).val_);
  z.push_back(ad(3,1).val_);
  z.push_back(ad(3,2).val_);
  z.push_back(ad(3,3).val_);
  z.push_back(bd(0,0).val_);
  z.push_back(bd(0,1).val_);
  z.push_back(bd(1,0).val_);
  z.push_back(bd(1,1).val_);
  z.push_back(bd(2,0).val_);
  z.push_back(bd(2,1).val_);
  z.push_back(bd(3,0).val_);
  z.push_back(bd(3,1).val_);

  VEC h;
  res.d_.grad(z,h);
  EXPECT_FLOAT_EQ(220,h[0]);
  EXPECT_FLOAT_EQ(111,h[1]);
  EXPECT_FLOAT_EQ(104,h[2]);
  EXPECT_FLOAT_EQ(117,h[3]);
  EXPECT_FLOAT_EQ(111,h[4]);
  EXPECT_FLOAT_EQ(2,h[5]);
  EXPECT_FLOAT_EQ(-5,h[6]);
  EXPECT_FLOAT_EQ(8,h[7]);
  EXPECT_FLOAT_EQ(104,h[8]);
  EXPECT_FLOAT_EQ(-5,h[9]);
  EXPECT_FLOAT_EQ(-12,h[10]);
  EXPECT_FLOAT_EQ(1,h[11]);
  EXPECT_FLOAT_EQ(117,h[12]);
  EXPECT_FLOAT_EQ(8,h[13]);
  EXPECT_FLOAT_EQ(1,h[14]);
  EXPECT_FLOAT_EQ(14,h[15]);
  EXPECT_FLOAT_EQ(241,h[16]);
  EXPECT_FLOAT_EQ(57,h[17]);
  EXPECT_FLOAT_EQ(241,h[18]);
  EXPECT_FLOAT_EQ(57,h[19]);
  EXPECT_FLOAT_EQ(235,h[20]);
  EXPECT_FLOAT_EQ(51,h[21]);
  EXPECT_FLOAT_EQ(447,h[22]);
  EXPECT_FLOAT_EQ(263,h[23]);
}

TEST(AgradMixMatrixTraceQuadForm, mat_ffv_1st_deriv) {
  using stan::math::trace_quad_form;
  using stan::math::matrix_ffv;
  
  matrix_ffv ad(4,4);
  matrix_ffv bd(4,2);
  fvar<fvar<var> > res;
  bd << 100, 10,
          0,  1,
         -3, -3,
          5,  2;
  ad << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(0,2).d_ = 1.0;
  ad(0,3).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  ad(1,2).d_ = 1.0;
  ad(1,3).d_ = 1.0;
  ad(2,0).d_ = 1.0;
  ad(2,1).d_ = 1.0;
  ad(2,2).d_ = 1.0;
  ad(2,3).d_ = 1.0;
  ad(3,0).d_ = 1.0;
  ad(3,1).d_ = 1.0;
  ad(3,2).d_ = 1.0;
  ad(3,3).d_ = 1.0;
  bd(0,0).d_ = 1.0;
  bd(0,1).d_ = 1.0;
  bd(1,0).d_ = 1.0;
  bd(1,1).d_ = 1.0;
  bd(2,0).d_ = 1.0;
  bd(2,1).d_ = 1.0;
  bd(3,0).d_ = 1.0;
  bd(3,1).d_ = 1.0;

  // fvar<fvar<var> > - fvar<fvar<var> >
  res = trace_quad_form(ad,bd);
  EXPECT_FLOAT_EQ(26758, res.val_.val_.val());
  EXPECT_FLOAT_EQ(16126, res.d_.val_.val());


  std::vector<var> z;
  z.push_back(ad(0,0).val_.val_);
  z.push_back(ad(0,1).val_.val_);
  z.push_back(ad(0,2).val_.val_);
  z.push_back(ad(0,3).val_.val_);
  z.push_back(ad(1,0).val_.val_);
  z.push_back(ad(1,1).val_.val_);
  z.push_back(ad(1,2).val_.val_);
  z.push_back(ad(1,3).val_.val_);
  z.push_back(ad(2,0).val_.val_);
  z.push_back(ad(2,1).val_.val_);
  z.push_back(ad(2,2).val_.val_);
  z.push_back(ad(2,3).val_.val_);
  z.push_back(ad(3,0).val_.val_);
  z.push_back(ad(3,1).val_.val_);
  z.push_back(ad(3,2).val_.val_);
  z.push_back(ad(3,3).val_.val_);
  z.push_back(bd(0,0).val_.val_);
  z.push_back(bd(0,1).val_.val_);
  z.push_back(bd(1,0).val_.val_);
  z.push_back(bd(1,1).val_.val_);
  z.push_back(bd(2,0).val_.val_);
  z.push_back(bd(2,1).val_.val_);
  z.push_back(bd(3,0).val_.val_);
  z.push_back(bd(3,1).val_.val_);

  VEC h;
  res.val_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(10100,h[0]);
  EXPECT_FLOAT_EQ(10,h[1]);
  EXPECT_FLOAT_EQ(-330,h[2]);
  EXPECT_FLOAT_EQ(520,h[3]);
  EXPECT_FLOAT_EQ(10,h[4]);
  EXPECT_FLOAT_EQ(1,h[5]);
  EXPECT_FLOAT_EQ(-3,h[6]);
  EXPECT_FLOAT_EQ(2,h[7]);
  EXPECT_FLOAT_EQ(-330,h[8]);
  EXPECT_FLOAT_EQ(-3,h[9]);
  EXPECT_FLOAT_EQ(18,h[10]);
  EXPECT_FLOAT_EQ(-21,h[11]);
  EXPECT_FLOAT_EQ(520,h[12]);
  EXPECT_FLOAT_EQ(2,h[13]);
  EXPECT_FLOAT_EQ(-21,h[14]);
  EXPECT_FLOAT_EQ(29,h[15]);
  EXPECT_FLOAT_EQ(432,h[16]);
  EXPECT_FLOAT_EQ(42,h[17]);
  EXPECT_FLOAT_EQ(908,h[18]);
  EXPECT_FLOAT_EQ(106,h[19]);
  EXPECT_FLOAT_EQ(1068,h[20]);
  EXPECT_FLOAT_EQ(76,h[21]);
  EXPECT_FLOAT_EQ(2414,h[22]);
  EXPECT_FLOAT_EQ(576,h[23]);
}


TEST(AgradMixMatrixTraceQuadForm, mat_ffv_2nd_deriv) {
  using stan::math::trace_quad_form;
  using stan::math::matrix_ffv;
  
  matrix_ffv ad(4,4);
  matrix_ffv bd(4,2);
  fvar<fvar<var> > res;
  bd << 100, 10,
          0,  1,
         -3, -3,
          5,  2;
  ad << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(0,2).d_ = 1.0;
  ad(0,3).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  ad(1,2).d_ = 1.0;
  ad(1,3).d_ = 1.0;
  ad(2,0).d_ = 1.0;
  ad(2,1).d_ = 1.0;
  ad(2,2).d_ = 1.0;
  ad(2,3).d_ = 1.0;
  ad(3,0).d_ = 1.0;
  ad(3,1).d_ = 1.0;
  ad(3,2).d_ = 1.0;
  ad(3,3).d_ = 1.0;
  bd(0,0).d_ = 1.0;
  bd(0,1).d_ = 1.0;
  bd(1,0).d_ = 1.0;
  bd(1,1).d_ = 1.0;
  bd(2,0).d_ = 1.0;
  bd(2,1).d_ = 1.0;
  bd(3,0).d_ = 1.0;
  bd(3,1).d_ = 1.0;

  // fvar<fvar<var> > - fvar<fvar<var> >
  res = trace_quad_form(ad,bd);
  EXPECT_FLOAT_EQ(26758, res.val_.val_.val());
  EXPECT_FLOAT_EQ(16126, res.d_.val_.val());


  std::vector<var> z;
  z.push_back(ad(0,0).val_.val_);
  z.push_back(ad(0,1).val_.val_);
  z.push_back(ad(0,2).val_.val_);
  z.push_back(ad(0,3).val_.val_);
  z.push_back(ad(1,0).val_.val_);
  z.push_back(ad(1,1).val_.val_);
  z.push_back(ad(1,2).val_.val_);
  z.push_back(ad(1,3).val_.val_);
  z.push_back(ad(2,0).val_.val_);
  z.push_back(ad(2,1).val_.val_);
  z.push_back(ad(2,2).val_.val_);
  z.push_back(ad(2,3).val_.val_);
  z.push_back(ad(3,0).val_.val_);
  z.push_back(ad(3,1).val_.val_);
  z.push_back(ad(3,2).val_.val_);
  z.push_back(ad(3,3).val_.val_);
  z.push_back(bd(0,0).val_.val_);
  z.push_back(bd(0,1).val_.val_);
  z.push_back(bd(1,0).val_.val_);
  z.push_back(bd(1,1).val_.val_);
  z.push_back(bd(2,0).val_.val_);
  z.push_back(bd(2,1).val_.val_);
  z.push_back(bd(3,0).val_.val_);
  z.push_back(bd(3,1).val_.val_);

  VEC h;
  res.d_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(220,h[0]);
  EXPECT_FLOAT_EQ(111,h[1]);
  EXPECT_FLOAT_EQ(104,h[2]);
  EXPECT_FLOAT_EQ(117,h[3]);
  EXPECT_FLOAT_EQ(111,h[4]);
  EXPECT_FLOAT_EQ(2,h[5]);
  EXPECT_FLOAT_EQ(-5,h[6]);
  EXPECT_FLOAT_EQ(8,h[7]);
  EXPECT_FLOAT_EQ(104,h[8]);
  EXPECT_FLOAT_EQ(-5,h[9]);
  EXPECT_FLOAT_EQ(-12,h[10]);
  EXPECT_FLOAT_EQ(1,h[11]);
  EXPECT_FLOAT_EQ(117,h[12]);
  EXPECT_FLOAT_EQ(8,h[13]);
  EXPECT_FLOAT_EQ(1,h[14]);
  EXPECT_FLOAT_EQ(14,h[15]);
  EXPECT_FLOAT_EQ(241,h[16]);
  EXPECT_FLOAT_EQ(57,h[17]);
  EXPECT_FLOAT_EQ(241,h[18]);
  EXPECT_FLOAT_EQ(57,h[19]);
  EXPECT_FLOAT_EQ(235,h[20]);
  EXPECT_FLOAT_EQ(51,h[21]);
  EXPECT_FLOAT_EQ(447,h[22]);
  EXPECT_FLOAT_EQ(263,h[23]);
}

TEST(AgradMixMatrixTraceQuadForm, mat_ffv_3rd_deriv) {
  using stan::math::trace_quad_form;
  using stan::math::matrix_ffv;
  
  matrix_ffv ad(4,4);
  matrix_ffv bd(4,2);
  fvar<fvar<var> > res;
  bd << 100, 10,
          0,  1,
         -3, -3,
          5,  2;
  ad << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(0,2).d_ = 1.0;
  ad(0,3).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  ad(1,2).d_ = 1.0;
  ad(1,3).d_ = 1.0;
  ad(2,0).d_ = 1.0;
  ad(2,1).d_ = 1.0;
  ad(2,2).d_ = 1.0;
  ad(2,3).d_ = 1.0;
  ad(3,0).d_ = 1.0;
  ad(3,1).d_ = 1.0;
  ad(3,2).d_ = 1.0;
  ad(3,3).d_ = 1.0;
  bd(0,0).d_ = 1.0;
  bd(0,1).d_ = 1.0;
  bd(1,0).d_ = 1.0;
  bd(1,1).d_ = 1.0;
  bd(2,0).d_ = 1.0;
  bd(2,1).d_ = 1.0;
  bd(3,0).d_ = 1.0;
  bd(3,1).d_ = 1.0; 
  ad(0,0).val_.d_ = 1.0;
  ad(0,1).val_.d_ = 1.0;
  ad(0,2).val_.d_ = 1.0;
  ad(0,3).val_.d_ = 1.0;
  ad(1,0).val_.d_ = 1.0;
  ad(1,1).val_.d_ = 1.0;
  ad(1,2).val_.d_ = 1.0;
  ad(1,3).val_.d_ = 1.0;
  ad(2,0).val_.d_ = 1.0;
  ad(2,1).val_.d_ = 1.0;
  ad(2,2).val_.d_ = 1.0;
  ad(2,3).val_.d_ = 1.0;
  ad(3,0).val_.d_ = 1.0;
  ad(3,1).val_.d_ = 1.0;
  ad(3,2).val_.d_ = 1.0;
  ad(3,3).val_.d_ = 1.0;
  bd(0,0).val_.d_ = 1.0;
  bd(0,1).val_.d_ = 1.0;
  bd(1,0).val_.d_ = 1.0;
  bd(1,1).val_.d_ = 1.0;
  bd(2,0).val_.d_ = 1.0;
  bd(2,1).val_.d_ = 1.0;
  bd(3,0).val_.d_ = 1.0;
  bd(3,1).val_.d_ = 1.0;

  // fvar<fvar<var> > - fvar<fvar<var> >
  res = trace_quad_form(ad,bd);
  EXPECT_FLOAT_EQ(26758, res.val_.val_.val());
  EXPECT_FLOAT_EQ(16126, res.d_.val_.val());


  std::vector<var> z;
  z.push_back(ad(0,0).val_.val_);
  z.push_back(ad(0,1).val_.val_);
  z.push_back(ad(0,2).val_.val_);
  z.push_back(ad(0,3).val_.val_);
  z.push_back(ad(1,0).val_.val_);
  z.push_back(ad(1,1).val_.val_);
  z.push_back(ad(1,2).val_.val_);
  z.push_back(ad(1,3).val_.val_);
  z.push_back(ad(2,0).val_.val_);
  z.push_back(ad(2,1).val_.val_);
  z.push_back(ad(2,2).val_.val_);
  z.push_back(ad(2,3).val_.val_);
  z.push_back(ad(3,0).val_.val_);
  z.push_back(ad(3,1).val_.val_);
  z.push_back(ad(3,2).val_.val_);
  z.push_back(ad(3,3).val_.val_);
  z.push_back(bd(0,0).val_.val_);
  z.push_back(bd(0,1).val_.val_);
  z.push_back(bd(1,0).val_.val_);
  z.push_back(bd(1,1).val_.val_);
  z.push_back(bd(2,0).val_.val_);
  z.push_back(bd(2,1).val_.val_);
  z.push_back(bd(3,0).val_.val_);
  z.push_back(bd(3,1).val_.val_);

  VEC h;
  res.d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(4,h[1]);
  EXPECT_FLOAT_EQ(4,h[2]);
  EXPECT_FLOAT_EQ(4,h[3]);
  EXPECT_FLOAT_EQ(4,h[4]);
  EXPECT_FLOAT_EQ(4,h[5]);
  EXPECT_FLOAT_EQ(4,h[6]);
  EXPECT_FLOAT_EQ(4,h[7]);
  EXPECT_FLOAT_EQ(4,h[8]);
  EXPECT_FLOAT_EQ(4,h[9]);
  EXPECT_FLOAT_EQ(4,h[10]);
  EXPECT_FLOAT_EQ(4,h[11]);
  EXPECT_FLOAT_EQ(4,h[12]);
  EXPECT_FLOAT_EQ(4,h[13]);
  EXPECT_FLOAT_EQ(4,h[14]);
  EXPECT_FLOAT_EQ(4,h[15]);
  EXPECT_FLOAT_EQ(16,h[16]);
  EXPECT_FLOAT_EQ(16,h[17]);
  EXPECT_FLOAT_EQ(16,h[18]);
  EXPECT_FLOAT_EQ(16,h[19]);
  EXPECT_FLOAT_EQ(16,h[20]);
  EXPECT_FLOAT_EQ(16,h[21]);
  EXPECT_FLOAT_EQ(16,h[22]);
  EXPECT_FLOAT_EQ(16,h[23]);
}

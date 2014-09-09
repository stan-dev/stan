#include <gtest/gtest.h>
#include <stan/math/matrix/quad_form.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/agrad/rev/matrix/multiply.hpp>
#include <stan/agrad/fwd/matrix/multiply.hpp>
#include <stan/agrad/fwd/matrix/dot_product.hpp>
#include <stan/agrad/rev/matrix/dot_product.hpp>
#include <stan/agrad/rev/operators.hpp>
#include <test/unit/agrad/util.hpp>

using stan::agrad::fvar;
using stan::agrad::var;

TEST(AgradFwdMatrixQuadForm, quad_form_mat_fd) {
  using stan::math::quad_form;
  using stan::agrad::matrix_fd;
  
  matrix_fd ad(4,4);
  matrix_fd bd(4,2);
  
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

  // fvar<double> - fvar<double> 
  matrix_fd resd = quad_form(ad,bd);
  EXPECT_FLOAT_EQ(26033, resd(0,0).val_);
  EXPECT_FLOAT_EQ(3456, resd(0,1).val_);
  EXPECT_FLOAT_EQ(3396, resd(1,0).val_);
  EXPECT_FLOAT_EQ(725, resd(1,1).val_);
  EXPECT_FLOAT_EQ(15226, resd(0,0).d_);
  EXPECT_FLOAT_EQ(3429, resd(0,1).d_);
  EXPECT_FLOAT_EQ(4233, resd(1,0).d_);
  EXPECT_FLOAT_EQ(900, resd(1,1).d_);
}

TEST(AgradFwdMatrixQuadForm, quad_form_sym_mat_fd) {
  using stan::math::quad_form_sym;
  using stan::agrad::matrix_fd;
  
  matrix_fd ad(4,4);
  matrix_fd bd(4,2);
  
  bd << 100, 10,
  0,  1,
  -3, -3,
  5,  2;
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  
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

  // fvar<double> - fvar<double>
  matrix_fd resd = quad_form_sym(ad,bd);
  EXPECT_FLOAT_EQ(25433, resd(0,0).val_);
  EXPECT_FLOAT_EQ(3396, resd(0,1).val_);
  EXPECT_FLOAT_EQ(3396, resd(1,0).val_);
  EXPECT_FLOAT_EQ(725, resd(1,1).val_);
  EXPECT_FLOAT_EQ(14320, resd(0,0).d_);
  EXPECT_FLOAT_EQ(3333, resd(0,1).d_);
  EXPECT_FLOAT_EQ(3333, resd(1,0).d_);
  EXPECT_FLOAT_EQ(810, resd(1,1).d_);
}

TEST(AgradFwdMatrixQuadForm, quad_form_vec_fd) {
  using stan::math::quad_form;
  using stan::agrad::matrix_fd;
  using stan::agrad::vector_fd;

  matrix_fd ad(4,4);
  vector_fd bd(4);
  fvar<double> res;

  
  bd << 100, 0, -3, 5;
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
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  bd(2).d_ = 1.0;
  bd(3).d_ = 1.0;

  // fvar<double> - fvar<double> 
  res = quad_form(ad,bd);
  EXPECT_FLOAT_EQ(26033, res.val_);
  EXPECT_FLOAT_EQ(15226, res.d_);
}

TEST(AgradFwdMatrixQuadForm, quad_form_sym_vec_fd) {
  using stan::math::quad_form_sym;
  using stan::agrad::matrix_fd;
  using stan::agrad::vector_fd;
  
  matrix_fd ad(4,4);
  vector_fd bd(4);
  fvar<double> res;
  
  bd << 100, 0, -3, 5;
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  
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
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  bd(2).d_ = 1.0;
  bd(3).d_ = 1.0;

  // fvar<double> - fvar<double> 
  res = quad_form_sym(ad,bd);
  EXPECT_FLOAT_EQ(25433, res.val_);
  EXPECT_FLOAT_EQ(14320, res.d_);
}

TEST(AgradFwdMatrixQuadForm, quad_form_sym_symmetry_fd) {
  using stan::math::quad_form_sym;
  using stan::agrad::matrix_fd;
  
  matrix_fd ad(4,4);
  matrix_fd bd(4,2);
  
  bd << 100, 10,
  0,  1,
  -3, -3,
  5,  2;
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0; 

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
  
  // fvar<double> - fvar<double>
  matrix_fd resd = quad_form_sym(ad,bd);
  EXPECT_EQ(resd(1,0).val_, resd(0,1).val_);
  EXPECT_EQ(resd(1,0).d_, resd(0,1).d_);
  
  bd.resize(4,3);
  bd << 100, 10, 11,
  0,  1, 12,
  -3, -3, 34,
  5,  2, 44;

  bd(0,0).d_ = 1.0;
  bd(0,1).d_ = 1.0;
  bd(0,2).d_ = 1.0;
  bd(1,0).d_ = 1.0;
  bd(1,1).d_ = 1.0;
  bd(1,2).d_ = 1.0;
  bd(2,0).d_ = 1.0;
  bd(2,1).d_ = 1.0;
  bd(2,2).d_ = 1.0;
  bd(3,0).d_ = 1.0;
  bd(3,1).d_ = 1.0;
  bd(3,2).d_ = 1.0;

  resd = quad_form_sym(ad,bd);
  EXPECT_EQ(resd(1,0).val_, resd(0,1).val_);  
  EXPECT_EQ(resd(2,0).val_, resd(0,2).val_);  
  EXPECT_EQ(resd(2,1).val_, resd(1,2).val_);  
  EXPECT_EQ(resd(1,0).d_, resd(0,1).d_);  
  EXPECT_EQ(resd(2,0).d_, resd(0,2).d_);  
  EXPECT_EQ(resd(2,1).d_, resd(1,2).d_);  
}

TEST(AgradFwdMatrixQuadForm, quad_form_sym_asymmetric_fd) {
  using stan::math::quad_form_sym;
  using stan::agrad::matrix_fd;
  
  matrix_fd ad(4,4);
  matrix_fd bd(4,2);
  
  
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
  
  // fvar<double>-fvar<double>
  EXPECT_THROW(quad_form_sym(ad,bd), std::domain_error);
}
TEST(AgradFwdMatrixQuadForm, quad_form_mat_ffd) {
  using stan::math::quad_form;
  using stan::agrad::matrix_ffd;
  
  matrix_ffd ad(4,4);
  matrix_ffd bd(4,2);
  
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

  // fvar<fvar<double> > - fvar<fvar<double> > 
  matrix_ffd resd = quad_form(ad,bd);
  EXPECT_FLOAT_EQ(26033, resd(0,0).val_.val_);
  EXPECT_FLOAT_EQ(3456, resd(0,1).val_.val_);
  EXPECT_FLOAT_EQ(3396, resd(1,0).val_.val_);
  EXPECT_FLOAT_EQ(725, resd(1,1).val_.val_);
  EXPECT_FLOAT_EQ(15226, resd(0,0).d_.val_);
  EXPECT_FLOAT_EQ(3429, resd(0,1).d_.val_);
  EXPECT_FLOAT_EQ(4233, resd(1,0).d_.val_);
  EXPECT_FLOAT_EQ(900, resd(1,1).d_.val_);
}

TEST(AgradFwdMatrixQuadForm, quad_form_sym_mat_ffd) {
  using stan::math::quad_form_sym;
  using stan::agrad::matrix_ffd;
  
  matrix_ffd ad(4,4);
  matrix_ffd bd(4,2);
  
  bd << 100, 10,
  0,  1,
  -3, -3,
  5,  2;
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  
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

  // fvar<fvar<double> > - fvar<fvar<double> >
  matrix_ffd resd = quad_form_sym(ad,bd);
  EXPECT_FLOAT_EQ(25433, resd(0,0).val_.val_);
  EXPECT_FLOAT_EQ(3396, resd(0,1).val_.val_);
  EXPECT_FLOAT_EQ(3396, resd(1,0).val_.val_);
  EXPECT_FLOAT_EQ(725, resd(1,1).val_.val_);
  EXPECT_FLOAT_EQ(14320, resd(0,0).d_.val_);
  EXPECT_FLOAT_EQ(3333, resd(0,1).d_.val_);
  EXPECT_FLOAT_EQ(3333, resd(1,0).d_.val_);
  EXPECT_FLOAT_EQ(810, resd(1,1).d_.val_);
}

TEST(AgradFwdMatrixQuadForm, quad_form_vec_ffd) {
  using stan::math::quad_form;
  using stan::agrad::matrix_ffd;
  using stan::agrad::vector_ffd;

  matrix_ffd ad(4,4);
  vector_ffd bd(4);
  fvar<fvar<double> > res;

  
  bd << 100, 0, -3, 5;
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
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  bd(2).d_ = 1.0;
  bd(3).d_ = 1.0;

  // fvar<fvar<double> > - fvar<fvar<double> > 
  res = quad_form(ad,bd);
  EXPECT_FLOAT_EQ(26033, res.val_.val_);
  EXPECT_FLOAT_EQ(15226, res.d_.val_);
}

TEST(AgradFwdMatrixQuadForm, quad_form_sym_vec_ffd) {
  using stan::math::quad_form_sym;
  using stan::agrad::matrix_ffd;
  using stan::agrad::vector_ffd;
  
  matrix_ffd ad(4,4);
  vector_ffd bd(4);
  fvar<fvar<double> > res;
  
  bd << 100, 0, -3, 5;
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  
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
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  bd(2).d_ = 1.0;
  bd(3).d_ = 1.0;

  // fvar<fvar<double> > - fvar<fvar<double> > 
  res = quad_form_sym(ad,bd);
  EXPECT_FLOAT_EQ(25433, res.val_.val_);
  EXPECT_FLOAT_EQ(14320, res.d_.val_);
}

TEST(AgradFwdMatrixQuadForm, quad_form_sym_symmetry_ffd) {
  using stan::math::quad_form_sym;
  using stan::agrad::matrix_ffd;
  
  matrix_ffd ad(4,4);
  matrix_ffd bd(4,2);
  
  bd << 100, 10,
  0,  1,
  -3, -3,
  5,  2;
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0; 

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
  
  // fvar<fvar<double> > - fvar<fvar<double> >
  matrix_ffd resd = quad_form_sym(ad,bd);
  EXPECT_EQ(resd(1,0).val_.val_, resd(0,1).val_.val_);
  EXPECT_EQ(resd(1,0).d_, resd(0,1).d_.val_);
  
  bd.resize(4,3);
  bd << 100, 10, 11,
  0,  1, 12,
  -3, -3, 34,
  5,  2, 44;

  bd(0,0).d_ = 1.0;
  bd(0,1).d_ = 1.0;
  bd(0,2).d_ = 1.0;
  bd(1,0).d_ = 1.0;
  bd(1,1).d_ = 1.0;
  bd(1,2).d_ = 1.0;
  bd(2,0).d_ = 1.0;
  bd(2,1).d_ = 1.0;
  bd(2,2).d_ = 1.0;
  bd(3,0).d_ = 1.0;
  bd(3,1).d_ = 1.0;
  bd(3,2).d_ = 1.0;

  resd = quad_form_sym(ad,bd);
  EXPECT_EQ(resd(1,0).val_.val_, resd(0,1).val_.val_);  
  EXPECT_EQ(resd(2,0).val_.val_, resd(0,2).val_.val_);  
  EXPECT_EQ(resd(2,1).val_.val_, resd(1,2).val_.val_);  
  EXPECT_EQ(resd(1,0).d_, resd(0,1).d_.val_);  
  EXPECT_EQ(resd(2,0).d_, resd(0,2).d_.val_);  
  EXPECT_EQ(resd(2,1).d_, resd(1,2).d_.val_);  
}

TEST(AgradFwdMatrixQuadForm, quad_form_sym_asymmetric_ffd) {
  using stan::math::quad_form_sym;
  using stan::agrad::matrix_ffd;
  
  matrix_ffd ad(4,4);
  matrix_ffd bd(4,2);
  
  
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
  
  // fvar<fvar<double> >-fvar<fvar<double> >
  EXPECT_THROW(quad_form_sym(ad,bd), std::domain_error);
}


TEST(AgradFwdMatrixQuadForm, quad_form_mat_fv_1st_deriv) {
  using stan::math::quad_form;
  using stan::agrad::matrix_fv;
  
  matrix_fv ad(4,4);
  matrix_fv bd(4,2);
  
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
  matrix_fv resd = quad_form(ad,bd);
  EXPECT_FLOAT_EQ(26033, resd(0,0).val_.val());
  EXPECT_FLOAT_EQ(3456, resd(0,1).val_.val());
  EXPECT_FLOAT_EQ(3396, resd(1,0).val_.val());
  EXPECT_FLOAT_EQ(725, resd(1,1).val_.val());
  EXPECT_FLOAT_EQ(15226, resd(0,0).d_.val());
  EXPECT_FLOAT_EQ(3429, resd(0,1).d_.val());
  EXPECT_FLOAT_EQ(4233, resd(1,0).d_.val());
  EXPECT_FLOAT_EQ(900, resd(1,1).d_.val());

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
  resd(0,0).val_.grad(z,h);
  EXPECT_FLOAT_EQ(10000,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(-300,h[2]);
  EXPECT_FLOAT_EQ(500,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
  EXPECT_FLOAT_EQ(0,h[6]);
  EXPECT_FLOAT_EQ(0.0,h[7]);
  EXPECT_FLOAT_EQ(-300,h[8]);
  EXPECT_FLOAT_EQ(0,h[9]);
  EXPECT_FLOAT_EQ(9,h[10]);
  EXPECT_FLOAT_EQ(-15,h[11]);
  EXPECT_FLOAT_EQ(500,h[12]);
  EXPECT_FLOAT_EQ(0.0,h[13]);
  EXPECT_FLOAT_EQ(-15,h[14]);
  EXPECT_FLOAT_EQ(25,h[15]);
  EXPECT_FLOAT_EQ(432,h[16]);
  EXPECT_FLOAT_EQ(0.0,h[17]);
  EXPECT_FLOAT_EQ(908,h[18]);
  EXPECT_FLOAT_EQ(0.0,h[19]);
  EXPECT_FLOAT_EQ(1068,h[20]);
  EXPECT_FLOAT_EQ(0.0,h[21]);
  EXPECT_FLOAT_EQ(2414,h[22]);
  EXPECT_FLOAT_EQ(0.0,h[23]);

}
TEST(AgradFwdMatrixQuadForm, quad_form_mat_fv_2nd_deriv) {
  using stan::math::quad_form;
  using stan::agrad::matrix_fv;
  
  matrix_fv ad(4,4);
  matrix_fv bd(4,2);
  
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
  matrix_fv resd = quad_form(ad,bd);

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
  resd(0,0).d_.grad(z,h);
  EXPECT_FLOAT_EQ(200,h[0]);
  EXPECT_FLOAT_EQ(100,h[1]);
  EXPECT_FLOAT_EQ(97,h[2]);
  EXPECT_FLOAT_EQ(105,h[3]);
  EXPECT_FLOAT_EQ(100,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
  EXPECT_FLOAT_EQ(-3,h[6]);
  EXPECT_FLOAT_EQ(5,h[7]);
  EXPECT_FLOAT_EQ(97,h[8]);
  EXPECT_FLOAT_EQ(-3,h[9]);
  EXPECT_FLOAT_EQ(-6,h[10]);
  EXPECT_FLOAT_EQ(2,h[11]);
  EXPECT_FLOAT_EQ(105,h[12]);
  EXPECT_FLOAT_EQ(5,h[13]);
  EXPECT_FLOAT_EQ(2,h[14]);
  EXPECT_FLOAT_EQ(10,h[15]);
  EXPECT_FLOAT_EQ(241,h[16]);
  EXPECT_FLOAT_EQ(0.0,h[17]);
  EXPECT_FLOAT_EQ(241,h[18]);
  EXPECT_FLOAT_EQ(0.0,h[19]);
  EXPECT_FLOAT_EQ(235,h[20]);
  EXPECT_FLOAT_EQ(0,h[21]);
  EXPECT_FLOAT_EQ(447,h[22]);
  EXPECT_FLOAT_EQ(0.0,h[23]);

}

TEST(AgradFwdMatrixQuadForm, quad_form_sym_mat_fv_1st_deriv) {
  using stan::math::quad_form_sym;
  using stan::agrad::matrix_fv;
  
  matrix_fv ad(4,4);
  matrix_fv bd(4,2);
  
  bd << 100, 10,
  0,  1,
  -3, -3,
  5,  2;
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  
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
  matrix_fv resd = quad_form_sym(ad,bd);
  EXPECT_FLOAT_EQ(25433, resd(0,0).val_.val());
  EXPECT_FLOAT_EQ(3396, resd(0,1).val_.val());
  EXPECT_FLOAT_EQ(3396, resd(1,0).val_.val());
  EXPECT_FLOAT_EQ(725, resd(1,1).val_.val());
  EXPECT_FLOAT_EQ(14320, resd(0,0).d_.val());
  EXPECT_FLOAT_EQ(3333, resd(0,1).d_.val());
  EXPECT_FLOAT_EQ(3333, resd(1,0).d_.val());
  EXPECT_FLOAT_EQ(810, resd(1,1).d_.val());


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
  resd(0,0).val_.grad(z,h);
  EXPECT_FLOAT_EQ(10000,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(-300,h[2]);
  EXPECT_FLOAT_EQ(500,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
  EXPECT_FLOAT_EQ(0,h[6]);
  EXPECT_FLOAT_EQ(0.0,h[7]);
  EXPECT_FLOAT_EQ(-300,h[8]);
  EXPECT_FLOAT_EQ(0,h[9]);
  EXPECT_FLOAT_EQ(9,h[10]);
  EXPECT_FLOAT_EQ(-15,h[11]);
  EXPECT_FLOAT_EQ(500,h[12]);
  EXPECT_FLOAT_EQ(0.0,h[13]);
  EXPECT_FLOAT_EQ(-15,h[14]);
  EXPECT_FLOAT_EQ(25,h[15]);
  EXPECT_FLOAT_EQ(426,h[16]);
  EXPECT_FLOAT_EQ(0.0,h[17]);
  EXPECT_FLOAT_EQ(608,h[18]);
  EXPECT_FLOAT_EQ(0.0,h[19]);
  EXPECT_FLOAT_EQ(768,h[20]);
  EXPECT_FLOAT_EQ(0.0,h[21]);
  EXPECT_FLOAT_EQ(2114,h[22]);
  EXPECT_FLOAT_EQ(0.0,h[23]);
}

TEST(AgradFwdMatrixQuadForm, quad_form_sym_mat_fv_2nd_deriv) {
  using stan::math::quad_form_sym;
  using stan::agrad::matrix_fv;
  
  matrix_fv ad(4,4);
  matrix_fv bd(4,2);
  
  bd << 100, 10,
  0,  1,
  -3, -3,
  5,  2;
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  
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
  matrix_fv resd = quad_form_sym(ad,bd);

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
  resd(0,0).d_.grad(z,h);
  EXPECT_FLOAT_EQ(200,h[0]);
  EXPECT_FLOAT_EQ(100,h[1]);
  EXPECT_FLOAT_EQ(97,h[2]);
  EXPECT_FLOAT_EQ(105,h[3]);
  EXPECT_FLOAT_EQ(100,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
  EXPECT_FLOAT_EQ(-3,h[6]);
  EXPECT_FLOAT_EQ(5,h[7]);
  EXPECT_FLOAT_EQ(97,h[8]);
  EXPECT_FLOAT_EQ(-3,h[9]);
  EXPECT_FLOAT_EQ(-6,h[10]);
  EXPECT_FLOAT_EQ(2,h[11]);
  EXPECT_FLOAT_EQ(105,h[12]);
  EXPECT_FLOAT_EQ(5,h[13]);
  EXPECT_FLOAT_EQ(2,h[14]);
  EXPECT_FLOAT_EQ(10,h[15]);
  EXPECT_FLOAT_EQ(232,h[16]);
  EXPECT_FLOAT_EQ(0.0,h[17]);
  EXPECT_FLOAT_EQ(238,h[18]);
  EXPECT_FLOAT_EQ(0.0,h[19]);
  EXPECT_FLOAT_EQ(232,h[20]);
  EXPECT_FLOAT_EQ(0.0,h[21]);
  EXPECT_FLOAT_EQ(444,h[22]);
  EXPECT_FLOAT_EQ(0.0,h[23]);
}
TEST(AgradFwdMatrixQuadForm, quad_form_vec_fv_1st_deriv) {
  using stan::math::quad_form;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;

  matrix_fv ad(4,4);
  vector_fv bd(4);
  fvar<var> res;

  
  bd << 100, 0, -3, 5;
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
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  bd(2).d_ = 1.0;
  bd(3).d_ = 1.0;

  // fvar<var> - fvar<var> 
  res = quad_form(ad,bd);
  EXPECT_FLOAT_EQ(26033, res.val_.val());
  EXPECT_FLOAT_EQ(15226, res.d_.val());


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
  z.push_back(bd(0).val_);
  z.push_back(bd(1).val_);
  z.push_back(bd(2).val_);
  z.push_back(bd(3).val_);

  VEC h;
  res.val_.grad(z,h);
  EXPECT_FLOAT_EQ(10000,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(-300,h[2]);
  EXPECT_FLOAT_EQ(500,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
  EXPECT_FLOAT_EQ(0,h[6]);
  EXPECT_FLOAT_EQ(0,h[7]);
  EXPECT_FLOAT_EQ(-300,h[8]);
  EXPECT_FLOAT_EQ(0,h[9]);
  EXPECT_FLOAT_EQ(9,h[10]);
  EXPECT_FLOAT_EQ(-15,h[11]);
  EXPECT_FLOAT_EQ(500,h[12]);
  EXPECT_FLOAT_EQ(0,h[13]);
  EXPECT_FLOAT_EQ(-15,h[14]);
  EXPECT_FLOAT_EQ(25,h[15]);
  EXPECT_FLOAT_EQ(432,h[16]);
  EXPECT_FLOAT_EQ(908,h[17]);
  EXPECT_FLOAT_EQ(1068,h[18]);
  EXPECT_FLOAT_EQ(2414,h[19]);
}

TEST(AgradFwdMatrixQuadForm, quad_form_vec_fv_2nd_deriv) {
  using stan::math::quad_form;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;

  matrix_fv ad(4,4);
  vector_fv bd(4);
  fvar<var> res;

  
  bd << 100, 0, -3, 5;
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
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  bd(2).d_ = 1.0;
  bd(3).d_ = 1.0;

  // fvar<var> - fvar<var> 
  res = quad_form(ad,bd);

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
  z.push_back(bd(0).val_);
  z.push_back(bd(1).val_);
  z.push_back(bd(2).val_);
  z.push_back(bd(3).val_);

  VEC h;
  res.d_.grad(z,h);
  EXPECT_FLOAT_EQ(200,h[0]);
  EXPECT_FLOAT_EQ(100,h[1]);
  EXPECT_FLOAT_EQ(97,h[2]);
  EXPECT_FLOAT_EQ(105,h[3]);
  EXPECT_FLOAT_EQ(100,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
  EXPECT_FLOAT_EQ(-3,h[6]);
  EXPECT_FLOAT_EQ(5,h[7]);
  EXPECT_FLOAT_EQ(97,h[8]);
  EXPECT_FLOAT_EQ(-3,h[9]);
  EXPECT_FLOAT_EQ(-6,h[10]);
  EXPECT_FLOAT_EQ(2,h[11]);
  EXPECT_FLOAT_EQ(105,h[12]);
  EXPECT_FLOAT_EQ(5,h[13]);
  EXPECT_FLOAT_EQ(2,h[14]);
  EXPECT_FLOAT_EQ(10,h[15]);
  EXPECT_FLOAT_EQ(241,h[16]);
  EXPECT_FLOAT_EQ(241,h[17]);
  EXPECT_FLOAT_EQ(235,h[18]);
  EXPECT_FLOAT_EQ(447,h[19]);
}
TEST(AgradFwdMatrixQuadForm, quad_form_sym_vec_fv_1st_deriv) {
  using stan::math::quad_form_sym;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;
  
  matrix_fv ad(4,4);
  vector_fv bd(4);
  fvar<var> res;
  
  bd << 100, 0, -3, 5;
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  
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
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  bd(2).d_ = 1.0;
  bd(3).d_ = 1.0;

  // fvar<var> - fvar<var> 
  res = quad_form_sym(ad,bd);
  EXPECT_FLOAT_EQ(25433, res.val_.val());
  EXPECT_FLOAT_EQ(14320, res.d_.val());

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
  z.push_back(bd(0).val_);
  z.push_back(bd(1).val_);
  z.push_back(bd(2).val_);
  z.push_back(bd(3).val_);

  VEC h;
  res.val_.grad(z,h);
  EXPECT_FLOAT_EQ(10000,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(-300,h[2]);
  EXPECT_FLOAT_EQ(500,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
  EXPECT_FLOAT_EQ(0,h[6]);
  EXPECT_FLOAT_EQ(0,h[7]);
  EXPECT_FLOAT_EQ(-300,h[8]);
  EXPECT_FLOAT_EQ(0,h[9]);
  EXPECT_FLOAT_EQ(9,h[10]);
  EXPECT_FLOAT_EQ(-15,h[11]);
  EXPECT_FLOAT_EQ(500,h[12]);
  EXPECT_FLOAT_EQ(0,h[13]);
  EXPECT_FLOAT_EQ(-15,h[14]);
  EXPECT_FLOAT_EQ(25,h[15]);
  EXPECT_FLOAT_EQ(426,h[16]);
  EXPECT_FLOAT_EQ(608,h[17]);
  EXPECT_FLOAT_EQ(768,h[18]);
  EXPECT_FLOAT_EQ(2114,h[19]);
}

TEST(AgradFwdMatrixQuadForm, quad_form_sym_vec_fv_2nd_deriv) {
  using stan::math::quad_form_sym;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;
  
  matrix_fv ad(4,4);
  vector_fv bd(4);
  fvar<var> res;
  
  bd << 100, 0, -3, 5;
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  
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
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  bd(2).d_ = 1.0;
  bd(3).d_ = 1.0;

  // fvar<var> - fvar<var> 
  res = quad_form_sym(ad,bd);

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
  z.push_back(bd(0).val_);
  z.push_back(bd(1).val_);
  z.push_back(bd(2).val_);
  z.push_back(bd(3).val_);

  VEC h;
  res.d_.grad(z,h);
  EXPECT_FLOAT_EQ(200,h[0]);
  EXPECT_FLOAT_EQ(100,h[1]);
  EXPECT_FLOAT_EQ(97,h[2]);
  EXPECT_FLOAT_EQ(105,h[3]);
  EXPECT_FLOAT_EQ(100,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
  EXPECT_FLOAT_EQ(-3,h[6]);
  EXPECT_FLOAT_EQ(5,h[7]);
  EXPECT_FLOAT_EQ(97,h[8]);
  EXPECT_FLOAT_EQ(-3,h[9]);
  EXPECT_FLOAT_EQ(-6,h[10]);
  EXPECT_FLOAT_EQ(2,h[11]);
  EXPECT_FLOAT_EQ(105,h[12]);
  EXPECT_FLOAT_EQ(5,h[13]);
  EXPECT_FLOAT_EQ(2,h[14]);
  EXPECT_FLOAT_EQ(10,h[15]);
  EXPECT_FLOAT_EQ(232,h[16]);
  EXPECT_FLOAT_EQ(238,h[17]);
  EXPECT_FLOAT_EQ(232,h[18]);
  EXPECT_FLOAT_EQ(444,h[19]);
}
TEST(AgradFwdMatrixQuadForm, quad_form_sym_symmetry_fv) {
  using stan::math::quad_form_sym;
  using stan::agrad::matrix_fv;
  
  matrix_fv ad(4,4);
  matrix_fv bd(4,2);
  
  bd << 100, 10,
  0,  1,
  -3, -3,
  5,  2;
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0; 

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
  matrix_fv resd = quad_form_sym(ad,bd);
  EXPECT_EQ(resd(1,0).val_.val(), resd(0,1).val_.val());
  EXPECT_EQ(resd(1,0).d_.val(), resd(0,1).d_.val());
  
  bd.resize(4,3);
  bd << 100, 10, 11,
  0,  1, 12,
  -3, -3, 34,
  5,  2, 44;

  bd(0,0).d_ = 1.0;
  bd(0,1).d_ = 1.0;
  bd(0,2).d_ = 1.0;
  bd(1,0).d_ = 1.0;
  bd(1,1).d_ = 1.0;
  bd(1,2).d_ = 1.0;
  bd(2,0).d_ = 1.0;
  bd(2,1).d_ = 1.0;
  bd(2,2).d_ = 1.0;
  bd(3,0).d_ = 1.0;
  bd(3,1).d_ = 1.0;
  bd(3,2).d_ = 1.0;

  resd = quad_form_sym(ad,bd);
  EXPECT_EQ(resd(1,0).val_.val(), resd(0,1).val_.val());  
  EXPECT_EQ(resd(2,0).val_.val(), resd(0,2).val_.val());  
  EXPECT_EQ(resd(2,1).val_.val(), resd(1,2).val_.val());  
  EXPECT_EQ(resd(1,0).d_.val(), resd(0,1).d_.val());  
  EXPECT_EQ(resd(2,0).d_.val(), resd(0,2).d_.val());  
  EXPECT_EQ(resd(2,1).d_.val(), resd(1,2).d_.val());  
}

TEST(AgradFwdMatrixQuadForm, quad_form_sym_asymmetric_fv) {
  using stan::math::quad_form_sym;
  using stan::agrad::matrix_fv;
  
  matrix_fv ad(4,4);
  matrix_fv bd(4,2);
  
  
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
  
  // fvar<var>-fvar<var>
  EXPECT_THROW(quad_form_sym(ad,bd), std::domain_error);
}


TEST(AgradFwdMatrixQuadForm, quad_form_mat_ffv_1st_deriv) {
  using stan::math::quad_form;
  using stan::agrad::matrix_ffv;
  
  matrix_ffv ad(4,4);
  matrix_ffv bd(4,2);
  
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
  matrix_ffv resd = quad_form(ad,bd);
  EXPECT_FLOAT_EQ(26033, resd(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(3456, resd(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(3396, resd(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(725, resd(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(15226, resd(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(3429, resd(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(4233, resd(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(900, resd(1,1).d_.val_.val());

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
  resd(0,0).val_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(10000,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(-300,h[2]);
  EXPECT_FLOAT_EQ(500,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
  EXPECT_FLOAT_EQ(0,h[6]);
  EXPECT_FLOAT_EQ(0.0,h[7]);
  EXPECT_FLOAT_EQ(-300,h[8]);
  EXPECT_FLOAT_EQ(0,h[9]);
  EXPECT_FLOAT_EQ(9,h[10]);
  EXPECT_FLOAT_EQ(-15,h[11]);
  EXPECT_FLOAT_EQ(500,h[12]);
  EXPECT_FLOAT_EQ(0.0,h[13]);
  EXPECT_FLOAT_EQ(-15,h[14]);
  EXPECT_FLOAT_EQ(25,h[15]);
  EXPECT_FLOAT_EQ(432,h[16]);
  EXPECT_FLOAT_EQ(0.0,h[17]);
  EXPECT_FLOAT_EQ(908,h[18]);
  EXPECT_FLOAT_EQ(0.0,h[19]);
  EXPECT_FLOAT_EQ(1068,h[20]);
  EXPECT_FLOAT_EQ(0.0,h[21]);
  EXPECT_FLOAT_EQ(2414,h[22]);
  EXPECT_FLOAT_EQ(0.0,h[23]);

}
TEST(AgradFwdMatrixQuadForm, quad_form_mat_ffv_2nd_deriv) {
  using stan::math::quad_form;
  using stan::agrad::matrix_ffv;
  
  matrix_ffv ad(4,4);
  matrix_ffv bd(4,2);
  
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
  matrix_ffv resd = quad_form(ad,bd);

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
  resd(0,0).d_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(200,h[0]);
  EXPECT_FLOAT_EQ(100,h[1]);
  EXPECT_FLOAT_EQ(97,h[2]);
  EXPECT_FLOAT_EQ(105,h[3]);
  EXPECT_FLOAT_EQ(100,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
  EXPECT_FLOAT_EQ(-3,h[6]);
  EXPECT_FLOAT_EQ(5,h[7]);
  EXPECT_FLOAT_EQ(97,h[8]);
  EXPECT_FLOAT_EQ(-3,h[9]);
  EXPECT_FLOAT_EQ(-6,h[10]);
  EXPECT_FLOAT_EQ(2,h[11]);
  EXPECT_FLOAT_EQ(105,h[12]);
  EXPECT_FLOAT_EQ(5,h[13]);
  EXPECT_FLOAT_EQ(2,h[14]);
  EXPECT_FLOAT_EQ(10,h[15]);
  EXPECT_FLOAT_EQ(241,h[16]);
  EXPECT_FLOAT_EQ(0.0,h[17]);
  EXPECT_FLOAT_EQ(241,h[18]);
  EXPECT_FLOAT_EQ(0.0,h[19]);
  EXPECT_FLOAT_EQ(235,h[20]);
  EXPECT_FLOAT_EQ(0,h[21]);
  EXPECT_FLOAT_EQ(447,h[22]);
  EXPECT_FLOAT_EQ(0.0,h[23]);

}

TEST(AgradFwdMatrixQuadForm, quad_form_mat_ffv_3rd_deriv) {
  using stan::math::quad_form;
  using stan::agrad::matrix_ffv;
  
  matrix_ffv ad(4,4);
  matrix_ffv bd(4,2);
  
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
  matrix_ffv resd = quad_form(ad,bd);

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
  resd(0,0).d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(2,h[0]);
  EXPECT_FLOAT_EQ(2,h[1]);
  EXPECT_FLOAT_EQ(2,h[2]);
  EXPECT_FLOAT_EQ(2,h[3]);
  EXPECT_FLOAT_EQ(2,h[4]);
  EXPECT_FLOAT_EQ(2,h[5]);
  EXPECT_FLOAT_EQ(2,h[6]);
  EXPECT_FLOAT_EQ(2,h[7]);
  EXPECT_FLOAT_EQ(2,h[8]);
  EXPECT_FLOAT_EQ(2,h[9]);
  EXPECT_FLOAT_EQ(2,h[10]);
  EXPECT_FLOAT_EQ(2,h[11]);
  EXPECT_FLOAT_EQ(2,h[12]);
  EXPECT_FLOAT_EQ(2,h[13]);
  EXPECT_FLOAT_EQ(2,h[14]);
  EXPECT_FLOAT_EQ(2,h[15]);
  EXPECT_FLOAT_EQ(16,h[16]);
  EXPECT_FLOAT_EQ(0.0,h[17]);
  EXPECT_FLOAT_EQ(16,h[18]);
  EXPECT_FLOAT_EQ(0.0,h[19]);
  EXPECT_FLOAT_EQ(16,h[20]);
  EXPECT_FLOAT_EQ(0,h[21]);
  EXPECT_FLOAT_EQ(16,h[22]);
  EXPECT_FLOAT_EQ(0.0,h[23]);

}
TEST(AgradFwdMatrixQuadForm, quad_form_sym_mat_ffv_1st_deriv) {
  using stan::math::quad_form_sym;
  using stan::agrad::matrix_ffv;
  
  matrix_ffv ad(4,4);
  matrix_ffv bd(4,2);
  
  bd << 100, 10,
  0,  1,
  -3, -3,
  5,  2;
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  
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
  matrix_ffv resd = quad_form_sym(ad,bd);
  EXPECT_FLOAT_EQ(25433, resd(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(3396, resd(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(3396, resd(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(725, resd(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(14320, resd(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(3333, resd(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(3333, resd(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(810, resd(1,1).d_.val_.val());


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
  resd(0,0).val_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(10000,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(-300,h[2]);
  EXPECT_FLOAT_EQ(500,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
  EXPECT_FLOAT_EQ(0,h[6]);
  EXPECT_FLOAT_EQ(0.0,h[7]);
  EXPECT_FLOAT_EQ(-300,h[8]);
  EXPECT_FLOAT_EQ(0,h[9]);
  EXPECT_FLOAT_EQ(9,h[10]);
  EXPECT_FLOAT_EQ(-15,h[11]);
  EXPECT_FLOAT_EQ(500,h[12]);
  EXPECT_FLOAT_EQ(0.0,h[13]);
  EXPECT_FLOAT_EQ(-15,h[14]);
  EXPECT_FLOAT_EQ(25,h[15]);
  EXPECT_FLOAT_EQ(426,h[16]);
  EXPECT_FLOAT_EQ(0.0,h[17]);
  EXPECT_FLOAT_EQ(608,h[18]);
  EXPECT_FLOAT_EQ(0.0,h[19]);
  EXPECT_FLOAT_EQ(768,h[20]);
  EXPECT_FLOAT_EQ(0.0,h[21]);
  EXPECT_FLOAT_EQ(2114,h[22]);
  EXPECT_FLOAT_EQ(0.0,h[23]);
}

TEST(AgradFwdMatrixQuadForm, quad_form_sym_mat_ffv_2nd_deriv) {
  using stan::math::quad_form_sym;
  using stan::agrad::matrix_ffv;
  
  matrix_ffv ad(4,4);
  matrix_ffv bd(4,2);
  
  bd << 100, 10,
  0,  1,
  -3, -3,
  5,  2;
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  
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
  matrix_ffv resd = quad_form_sym(ad,bd);

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
  resd(0,0).d_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(200,h[0]);
  EXPECT_FLOAT_EQ(100,h[1]);
  EXPECT_FLOAT_EQ(97,h[2]);
  EXPECT_FLOAT_EQ(105,h[3]);
  EXPECT_FLOAT_EQ(100,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
  EXPECT_FLOAT_EQ(-3,h[6]);
  EXPECT_FLOAT_EQ(5,h[7]);
  EXPECT_FLOAT_EQ(97,h[8]);
  EXPECT_FLOAT_EQ(-3,h[9]);
  EXPECT_FLOAT_EQ(-6,h[10]);
  EXPECT_FLOAT_EQ(2,h[11]);
  EXPECT_FLOAT_EQ(105,h[12]);
  EXPECT_FLOAT_EQ(5,h[13]);
  EXPECT_FLOAT_EQ(2,h[14]);
  EXPECT_FLOAT_EQ(10,h[15]);
  EXPECT_FLOAT_EQ(232,h[16]);
  EXPECT_FLOAT_EQ(0.0,h[17]);
  EXPECT_FLOAT_EQ(238,h[18]);
  EXPECT_FLOAT_EQ(0.0,h[19]);
  EXPECT_FLOAT_EQ(232,h[20]);
  EXPECT_FLOAT_EQ(0.0,h[21]);
  EXPECT_FLOAT_EQ(444,h[22]);
  EXPECT_FLOAT_EQ(0.0,h[23]);
}


TEST(AgradFwdMatrixQuadForm, quad_form_sym_mat_ffv_3rd_deriv) {
  using stan::math::quad_form_sym;
  using stan::agrad::matrix_ffv;
  
  matrix_ffv ad(4,4);
  matrix_ffv bd(4,2);
  
  bd << 100, 10,
  0,  1,
  -3, -3,
  5,  2;
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  
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
  matrix_ffv resd = quad_form_sym(ad,bd);

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
  resd(0,0).d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(2,h[0]);
  EXPECT_FLOAT_EQ(2,h[1]);
  EXPECT_FLOAT_EQ(2,h[2]);
  EXPECT_FLOAT_EQ(2,h[3]);
  EXPECT_FLOAT_EQ(2,h[4]);
  EXPECT_FLOAT_EQ(2,h[5]);
  EXPECT_FLOAT_EQ(2,h[6]);
  EXPECT_FLOAT_EQ(2,h[7]);
  EXPECT_FLOAT_EQ(2,h[8]);
  EXPECT_FLOAT_EQ(2,h[9]);
  EXPECT_FLOAT_EQ(2,h[10]);
  EXPECT_FLOAT_EQ(2,h[11]);
  EXPECT_FLOAT_EQ(2,h[12]);
  EXPECT_FLOAT_EQ(2,h[13]);
  EXPECT_FLOAT_EQ(2,h[14]);
  EXPECT_FLOAT_EQ(2,h[15]);
  EXPECT_FLOAT_EQ(16,h[16]);
  EXPECT_FLOAT_EQ(0.0,h[17]);
  EXPECT_FLOAT_EQ(16,h[18]);
  EXPECT_FLOAT_EQ(0.0,h[19]);
  EXPECT_FLOAT_EQ(16,h[20]);
  EXPECT_FLOAT_EQ(0.0,h[21]);
  EXPECT_FLOAT_EQ(16,h[22]);
  EXPECT_FLOAT_EQ(0.0,h[23]);
}
TEST(AgradFwdMatrixQuadForm, quad_form_vec_ffv_1st_deriv) {
  using stan::math::quad_form;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;

  matrix_ffv ad(4,4);
  vector_ffv bd(4);
  fvar<fvar<var> > res;

  
  bd << 100, 0, -3, 5;
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
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  bd(2).d_ = 1.0;
  bd(3).d_ = 1.0;

  // fvar<fvar<var> > - fvar<fvar<var> > 
  res = quad_form(ad,bd);
  EXPECT_FLOAT_EQ(26033, res.val_.val_.val());
  EXPECT_FLOAT_EQ(15226, res.d_.val_.val());


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
  z.push_back(bd(0).val_.val_);
  z.push_back(bd(1).val_.val_);
  z.push_back(bd(2).val_.val_);
  z.push_back(bd(3).val_.val_);

  VEC h;
  res.val_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(10000,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(-300,h[2]);
  EXPECT_FLOAT_EQ(500,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
  EXPECT_FLOAT_EQ(0,h[6]);
  EXPECT_FLOAT_EQ(0,h[7]);
  EXPECT_FLOAT_EQ(-300,h[8]);
  EXPECT_FLOAT_EQ(0,h[9]);
  EXPECT_FLOAT_EQ(9,h[10]);
  EXPECT_FLOAT_EQ(-15,h[11]);
  EXPECT_FLOAT_EQ(500,h[12]);
  EXPECT_FLOAT_EQ(0,h[13]);
  EXPECT_FLOAT_EQ(-15,h[14]);
  EXPECT_FLOAT_EQ(25,h[15]);
  EXPECT_FLOAT_EQ(432,h[16]);
  EXPECT_FLOAT_EQ(908,h[17]);
  EXPECT_FLOAT_EQ(1068,h[18]);
  EXPECT_FLOAT_EQ(2414,h[19]);
}

TEST(AgradFwdMatrixQuadForm, quad_form_vec_ffv_2nd_deriv) {
  using stan::math::quad_form;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;

  matrix_ffv ad(4,4);
  vector_ffv bd(4);
  fvar<fvar<var> > res;

  
  bd << 100, 0, -3, 5;
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
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  bd(2).d_ = 1.0;
  bd(3).d_ = 1.0;

  // fvar<fvar<var> > - fvar<fvar<var> > 
  res = quad_form(ad,bd);

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
  z.push_back(bd(0).val_.val_);
  z.push_back(bd(1).val_.val_);
  z.push_back(bd(2).val_.val_);
  z.push_back(bd(3).val_.val_);

  VEC h;
  res.d_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(200,h[0]);
  EXPECT_FLOAT_EQ(100,h[1]);
  EXPECT_FLOAT_EQ(97,h[2]);
  EXPECT_FLOAT_EQ(105,h[3]);
  EXPECT_FLOAT_EQ(100,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
  EXPECT_FLOAT_EQ(-3,h[6]);
  EXPECT_FLOAT_EQ(5,h[7]);
  EXPECT_FLOAT_EQ(97,h[8]);
  EXPECT_FLOAT_EQ(-3,h[9]);
  EXPECT_FLOAT_EQ(-6,h[10]);
  EXPECT_FLOAT_EQ(2,h[11]);
  EXPECT_FLOAT_EQ(105,h[12]);
  EXPECT_FLOAT_EQ(5,h[13]);
  EXPECT_FLOAT_EQ(2,h[14]);
  EXPECT_FLOAT_EQ(10,h[15]);
  EXPECT_FLOAT_EQ(241,h[16]);
  EXPECT_FLOAT_EQ(241,h[17]);
  EXPECT_FLOAT_EQ(235,h[18]);
  EXPECT_FLOAT_EQ(447,h[19]);
}
TEST(AgradFwdMatrixQuadForm, quad_form_vec_ffv_3rd_deriv) {
  using stan::math::quad_form;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;

  matrix_ffv ad(4,4);
  vector_ffv bd(4);
  fvar<fvar<var> > res;

  
  bd << 100, 0, -3, 5;
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
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  bd(2).d_ = 1.0;
  bd(3).d_ = 1.0;
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
  bd(0).val_.d_ = 1.0;
  bd(1).val_.d_ = 1.0;
  bd(2).val_.d_ = 1.0;
  bd(3).val_.d_ = 1.0;

  // fvar<fvar<var> > - fvar<fvar<var> > 
  res = quad_form(ad,bd);

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
  z.push_back(bd(0).val_.val_);
  z.push_back(bd(1).val_.val_);
  z.push_back(bd(2).val_.val_);
  z.push_back(bd(3).val_.val_);

  VEC h;
  res.d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(2,h[0]);
  EXPECT_FLOAT_EQ(2,h[1]);
  EXPECT_FLOAT_EQ(2,h[2]);
  EXPECT_FLOAT_EQ(2,h[3]);
  EXPECT_FLOAT_EQ(2,h[4]);
  EXPECT_FLOAT_EQ(2,h[5]);
  EXPECT_FLOAT_EQ(2,h[6]);
  EXPECT_FLOAT_EQ(2,h[7]);
  EXPECT_FLOAT_EQ(2,h[8]);
  EXPECT_FLOAT_EQ(2,h[9]);
  EXPECT_FLOAT_EQ(2,h[10]);
  EXPECT_FLOAT_EQ(2,h[11]);
  EXPECT_FLOAT_EQ(2,h[12]);
  EXPECT_FLOAT_EQ(2,h[13]);
  EXPECT_FLOAT_EQ(2,h[14]);
  EXPECT_FLOAT_EQ(2,h[15]);
  EXPECT_FLOAT_EQ(16,h[16]);
  EXPECT_FLOAT_EQ(16,h[17]);
  EXPECT_FLOAT_EQ(16,h[18]);
  EXPECT_FLOAT_EQ(16,h[19]);
}
TEST(AgradFwdMatrixQuadForm, quad_form_sym_vec_ffv_1st_deriv) {
  using stan::math::quad_form_sym;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  
  matrix_ffv ad(4,4);
  vector_ffv bd(4);
  fvar<fvar<var> > res;
  
  bd << 100, 0, -3, 5;
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  
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
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  bd(2).d_ = 1.0;
  bd(3).d_ = 1.0;

  // fvar<fvar<var> > - fvar<fvar<var> > 
  res = quad_form_sym(ad,bd);
  EXPECT_FLOAT_EQ(25433, res.val_.val_.val());
  EXPECT_FLOAT_EQ(14320, res.d_.val_.val());

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
  z.push_back(bd(0).val_.val_);
  z.push_back(bd(1).val_.val_);
  z.push_back(bd(2).val_.val_);
  z.push_back(bd(3).val_.val_);

  VEC h;
  res.val_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(10000,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(-300,h[2]);
  EXPECT_FLOAT_EQ(500,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
  EXPECT_FLOAT_EQ(0,h[6]);
  EXPECT_FLOAT_EQ(0,h[7]);
  EXPECT_FLOAT_EQ(-300,h[8]);
  EXPECT_FLOAT_EQ(0,h[9]);
  EXPECT_FLOAT_EQ(9,h[10]);
  EXPECT_FLOAT_EQ(-15,h[11]);
  EXPECT_FLOAT_EQ(500,h[12]);
  EXPECT_FLOAT_EQ(0,h[13]);
  EXPECT_FLOAT_EQ(-15,h[14]);
  EXPECT_FLOAT_EQ(25,h[15]);
  EXPECT_FLOAT_EQ(426,h[16]);
  EXPECT_FLOAT_EQ(608,h[17]);
  EXPECT_FLOAT_EQ(768,h[18]);
  EXPECT_FLOAT_EQ(2114,h[19]);
}

TEST(AgradFwdMatrixQuadForm, quad_form_sym_vec_ffv_2nd_deriv) {
  using stan::math::quad_form_sym;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  
  matrix_ffv ad(4,4);
  vector_ffv bd(4);
  fvar<fvar<var> > res;
  
  bd << 100, 0, -3, 5;
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  
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
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  bd(2).d_ = 1.0;
  bd(3).d_ = 1.0;

  // fvar<fvar<var> > - fvar<fvar<var> > 
  res = quad_form_sym(ad,bd);

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
  z.push_back(bd(0).val_.val_);
  z.push_back(bd(1).val_.val_);
  z.push_back(bd(2).val_.val_);
  z.push_back(bd(3).val_.val_);

  VEC h;
  res.d_.val_.grad(z,h);
  EXPECT_FLOAT_EQ(200,h[0]);
  EXPECT_FLOAT_EQ(100,h[1]);
  EXPECT_FLOAT_EQ(97,h[2]);
  EXPECT_FLOAT_EQ(105,h[3]);
  EXPECT_FLOAT_EQ(100,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
  EXPECT_FLOAT_EQ(-3,h[6]);
  EXPECT_FLOAT_EQ(5,h[7]);
  EXPECT_FLOAT_EQ(97,h[8]);
  EXPECT_FLOAT_EQ(-3,h[9]);
  EXPECT_FLOAT_EQ(-6,h[10]);
  EXPECT_FLOAT_EQ(2,h[11]);
  EXPECT_FLOAT_EQ(105,h[12]);
  EXPECT_FLOAT_EQ(5,h[13]);
  EXPECT_FLOAT_EQ(2,h[14]);
  EXPECT_FLOAT_EQ(10,h[15]);
  EXPECT_FLOAT_EQ(232,h[16]);
  EXPECT_FLOAT_EQ(238,h[17]);
  EXPECT_FLOAT_EQ(232,h[18]);
  EXPECT_FLOAT_EQ(444,h[19]);
}
TEST(AgradFwdMatrixQuadForm, quad_form_sym_vec_ffv_3rd_deriv) {
  using stan::math::quad_form_sym;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  
  matrix_ffv ad(4,4);
  vector_ffv bd(4);
  fvar<fvar<var> > res;
  
  bd << 100, 0, -3, 5;
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0;
  
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
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  bd(2).d_ = 1.0;
  bd(3).d_ = 1.0;
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
  bd(0).val_.d_ = 1.0;
  bd(1).val_.d_ = 1.0;
  bd(2).val_.d_ = 1.0;
  bd(3).val_.d_ = 1.0;

  // fvar<fvar<var> > - fvar<fvar<var> > 
  res = quad_form_sym(ad,bd);

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
  z.push_back(bd(0).val_.val_);
  z.push_back(bd(1).val_.val_);
  z.push_back(bd(2).val_.val_);
  z.push_back(bd(3).val_.val_);

  VEC h;
  res.d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(2,h[0]);
  EXPECT_FLOAT_EQ(2,h[1]);
  EXPECT_FLOAT_EQ(2,h[2]);
  EXPECT_FLOAT_EQ(2,h[3]);
  EXPECT_FLOAT_EQ(2,h[4]);
  EXPECT_FLOAT_EQ(2,h[5]);
  EXPECT_FLOAT_EQ(2,h[6]);
  EXPECT_FLOAT_EQ(2,h[7]);
  EXPECT_FLOAT_EQ(2,h[8]);
  EXPECT_FLOAT_EQ(2,h[9]);
  EXPECT_FLOAT_EQ(2,h[10]);
  EXPECT_FLOAT_EQ(2,h[11]);
  EXPECT_FLOAT_EQ(2,h[12]);
  EXPECT_FLOAT_EQ(2,h[13]);
  EXPECT_FLOAT_EQ(2,h[14]);
  EXPECT_FLOAT_EQ(2,h[15]);
  EXPECT_FLOAT_EQ(16,h[16]);
  EXPECT_FLOAT_EQ(16,h[17]);
  EXPECT_FLOAT_EQ(16,h[18]);
  EXPECT_FLOAT_EQ(16,h[19]);
}
TEST(AgradFwdMatrixQuadForm, quad_form_sym_symmetry_ffv) {
  using stan::math::quad_form_sym;
  using stan::agrad::matrix_ffv;
  
  matrix_ffv ad(4,4);
  matrix_ffv bd(4,2);
  
  bd << 100, 10,
  0,  1,
  -3, -3,
  5,  2;
  ad << 2.0,  3.0, 4.0,   5.0, 
  3.0, 10.0, 2.0,   2.0,
  4.0,  2.0, 7.0,   1.0,
  5.0,  2.0, 1.0, 112.0; 

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
  matrix_ffv resd = quad_form_sym(ad,bd);
  EXPECT_EQ(resd(1,0).val_.val_.val(), resd(0,1).val_.val_.val());
  EXPECT_EQ(resd(1,0).d_.val_.val(), resd(0,1).d_.val_.val());
  
  bd.resize(4,3);
  bd << 100, 10, 11,
  0,  1, 12,
  -3, -3, 34,
  5,  2, 44;

  bd(0,0).d_ = 1.0;
  bd(0,1).d_ = 1.0;
  bd(0,2).d_ = 1.0;
  bd(1,0).d_ = 1.0;
  bd(1,1).d_ = 1.0;
  bd(1,2).d_ = 1.0;
  bd(2,0).d_ = 1.0;
  bd(2,1).d_ = 1.0;
  bd(2,2).d_ = 1.0;
  bd(3,0).d_ = 1.0;
  bd(3,1).d_ = 1.0;
  bd(3,2).d_ = 1.0;

  resd = quad_form_sym(ad,bd);
  EXPECT_EQ(resd(1,0).val_.val_.val(), resd(0,1).val_.val_.val());  
  EXPECT_EQ(resd(2,0).val_.val_.val(), resd(0,2).val_.val_.val());  
  EXPECT_EQ(resd(2,1).val_.val_.val(), resd(1,2).val_.val_.val());  
  EXPECT_EQ(resd(1,0).d_.val_.val(), resd(0,1).d_.val_.val());  
  EXPECT_EQ(resd(2,0).d_.val_.val(), resd(0,2).d_.val_.val());  
  EXPECT_EQ(resd(2,1).d_.val_.val(), resd(1,2).d_.val_.val());  
}

TEST(AgradFwdMatrixQuadForm, quad_form_sym_asymmetric_ffv) {
  using stan::math::quad_form_sym;
  using stan::agrad::matrix_ffv;
  
  matrix_ffv ad(4,4);
  matrix_ffv bd(4,2);
  
  
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
  
  // fvar<fvar<var> >-fvar<fvar<var> >
  EXPECT_THROW(quad_form_sym(ad,bd), std::domain_error);
}

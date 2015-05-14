#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/quad_form_diag.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <stan/math/rev/mat/fun/multiply.hpp>
#include <stan/math/fwd/mat/fun/multiply.hpp>
#include <stan/math/fwd/mat/fun/dot_product.hpp>
#include <stan/math/rev/mat/fun/dot_product.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>

using stan::math::fvar;
using stan::math::var;

TEST(AgradMixMatrixQuadFormDiag, mat_fv_vector_fv_1) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_fv;
  using stan::math::vector_fv;
  
  matrix_fv ad(2,2);
  vector_fv bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;

  matrix_fv resd = quad_form_diag(ad,bd);
  EXPECT_FLOAT_EQ(20000, resd(0,0).val_.val());
  EXPECT_FLOAT_EQ(3000, resd(0,1).val_.val());
  EXPECT_FLOAT_EQ(4000, resd(1,0).val_.val());
  EXPECT_FLOAT_EQ(500, resd(1,1).val_.val());
  EXPECT_FLOAT_EQ(10400, resd(0,0).d_.val());
  EXPECT_FLOAT_EQ(1330, resd(0,1).d_.val());
  EXPECT_FLOAT_EQ(1440, resd(1,0).d_.val());
  EXPECT_FLOAT_EQ(200, resd(1,1).d_.val());

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(ad(0,0).val_);
  vars.push_back(ad(0,1).val_);
  vars.push_back(ad(1,0).val_);
  vars.push_back(ad(1,1).val_);
  vars.push_back(bd(0).val_);
  vars.push_back(bd(1).val_);

  resd(0,0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(10000, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(400, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
}

TEST(AgradMixMatrixQuadFormDiag, mat_fv_vector_fv_2) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_fv;
  using stan::math::vector_fv;
  
  matrix_fv ad(2,2);
  vector_fv bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;

  matrix_fv resd = quad_form_diag(ad,bd);

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(ad(0,0).val_);
  vars.push_back(ad(0,1).val_);
  vars.push_back(ad(1,0).val_);
  vars.push_back(ad(1,1).val_);
  vars.push_back(bd(0).val_);
  vars.push_back(bd(1).val_);

  resd(0,0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(200, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(204, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
}
TEST(AgradMixMatrixQuadFormDiag, mat_fv_row_vector_fv_1) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_fv;
  using stan::math::row_vector_fv;
  
  matrix_fv ad(2,2);
  row_vector_fv bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;

  matrix_fv resd = quad_form_diag(ad,bd);
  EXPECT_FLOAT_EQ(20000, resd(0,0).val_.val());
  EXPECT_FLOAT_EQ(3000, resd(0,1).val_.val());
  EXPECT_FLOAT_EQ(4000, resd(1,0).val_.val());
  EXPECT_FLOAT_EQ(500, resd(1,1).val_.val());
  EXPECT_FLOAT_EQ(10400, resd(0,0).d_.val());
  EXPECT_FLOAT_EQ(1330, resd(0,1).d_.val());
  EXPECT_FLOAT_EQ(1440, resd(1,0).d_.val());
  EXPECT_FLOAT_EQ(200, resd(1,1).d_.val());

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(ad(0,0).val_);
  vars.push_back(ad(0,1).val_);
  vars.push_back(ad(1,0).val_);
  vars.push_back(ad(1,1).val_);
  vars.push_back(bd(0).val_);
  vars.push_back(bd(1).val_);

  resd(0,0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(10000, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(400, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
}
TEST(AgradMixMatrixQuadFormDiag, mat_fv_row_vector_fv_2) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_fv;
  using stan::math::row_vector_fv;
  
  matrix_fv ad(2,2);
  row_vector_fv bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;

  matrix_fv resd = quad_form_diag(ad,bd);

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(ad(0,0).val_);
  vars.push_back(ad(0,1).val_);
  vars.push_back(ad(1,0).val_);
  vars.push_back(ad(1,1).val_);
  vars.push_back(bd(0).val_);
  vars.push_back(bd(1).val_);

  resd(0,0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(200, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(204, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
}

TEST(AgradMixMatrixQuadFormDiag, mat_fv_vector_fv_exception) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_fv;
  using stan::math::vector_fv;
  using stan::math::row_vector_fv;

  matrix_fv m1(2,2);
  matrix_fv m2(3,2);
  matrix_fv m3(2,3);
  vector_fv v1(3);
  vector_fv v2(4);
  row_vector_fv rv1(3);
  row_vector_fv rv2(4);

  EXPECT_THROW(quad_form_diag(m1, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, v1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m2, v2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m2, rv1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m2, rv2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m3, v1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m3, v2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m3, rv1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m3, rv2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m1, m1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m1, m2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m1, m3), std::invalid_argument);
}
TEST(AgradMixMatrixQuadFormDiag, mat_d_vector_fv_1) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::vector_fv;
  
  matrix_d ad(2,2);
  vector_fv bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;

  matrix_fv resd = quad_form_diag(ad,bd);
  EXPECT_FLOAT_EQ(20000, resd(0,0).val_.val());
  EXPECT_FLOAT_EQ(3000, resd(0,1).val_.val());
  EXPECT_FLOAT_EQ(4000, resd(1,0).val_.val());
  EXPECT_FLOAT_EQ(500, resd(1,1).val_.val());
  EXPECT_FLOAT_EQ(400, resd(0,0).d_.val());
  EXPECT_FLOAT_EQ(330, resd(0,1).d_.val());
  EXPECT_FLOAT_EQ(440, resd(1,0).d_.val());
  EXPECT_FLOAT_EQ(100, resd(1,1).d_.val());

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(bd(0).val_);
  vars.push_back(bd(1).val_);

  resd(0,0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(400, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}
TEST(AgradMixMatrixQuadFormDiag, mat_d_vector_fv_2) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::vector_fv;
  
  matrix_d ad(2,2);
  vector_fv bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;

  matrix_fv resd = quad_form_diag(ad,bd);

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(bd(0).val_);
  vars.push_back(bd(1).val_);

  resd(0,0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(4, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}

TEST(AgradMixMatrixQuadFormDiag, mat_d_row_vector_fv_1) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::row_vector_fv;
  
  matrix_d ad(2,2);
  row_vector_fv bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;

  matrix_fv resd = quad_form_diag(ad,bd);
  EXPECT_FLOAT_EQ(20000, resd(0,0).val_.val());
  EXPECT_FLOAT_EQ(3000, resd(0,1).val_.val());
  EXPECT_FLOAT_EQ(4000, resd(1,0).val_.val());
  EXPECT_FLOAT_EQ(500, resd(1,1).val_.val());
  EXPECT_FLOAT_EQ(400, resd(0,0).d_.val());
  EXPECT_FLOAT_EQ(330, resd(0,1).d_.val());
  EXPECT_FLOAT_EQ(440, resd(1,0).d_.val());
  EXPECT_FLOAT_EQ(100, resd(1,1).d_.val());

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(bd(0).val_);
  vars.push_back(bd(1).val_);

  resd(0,0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(400, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}

TEST(AgradMixMatrixQuadFormDiag, mat_d_row_vector_fv_2) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::row_vector_fv;
  
  matrix_d ad(2,2);
  row_vector_fv bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;

  matrix_fv resd = quad_form_diag(ad,bd);

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(bd(0).val_);
  vars.push_back(bd(1).val_);

  resd(0,0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(4, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}

TEST(AgradMixMatrixQuadFormDiag, mat_d_vector_fv_exception) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::math::vector_fv;
  using stan::math::row_vector_fv;

  matrix_d m1(2,2);
  matrix_d m2(3,2);
  matrix_d m3(2,3);
  vector_fv v1(3);
  vector_fv v2(4);
  row_vector_fv rv1(3);
  row_vector_fv rv2(4);

  EXPECT_THROW(quad_form_diag(m1, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, v1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m2, v2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m2, rv1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m2, rv2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m3, v1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m3, v2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m3, rv1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m3, rv2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m1, m1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m1, m2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m1, m3), std::invalid_argument);
}

TEST(AgradMixMatrixQuadFormDiag, mat_fv_vector_d_1) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_fv;
  using stan::math::vector_d;
  
  matrix_fv ad(2,2);
  vector_d bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;

  matrix_fv resd = quad_form_diag(ad,bd);
  EXPECT_FLOAT_EQ(20000, resd(0,0).val_.val());
  EXPECT_FLOAT_EQ(3000, resd(0,1).val_.val());
  EXPECT_FLOAT_EQ(4000, resd(1,0).val_.val());
  EXPECT_FLOAT_EQ(500, resd(1,1).val_.val());
  EXPECT_FLOAT_EQ(10000, resd(0,0).d_.val());
  EXPECT_FLOAT_EQ(1000, resd(0,1).d_.val());
  EXPECT_FLOAT_EQ(1000, resd(1,0).d_.val());
  EXPECT_FLOAT_EQ(100, resd(1,1).d_.val());

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(ad(0,0).val_);
  vars.push_back(ad(0,1).val_);
  vars.push_back(ad(1,0).val_);
  vars.push_back(ad(1,1).val_);

  resd(0,0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(10000, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradMixMatrixQuadFormDiag, mat_fv_vector_d_2) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_fv;
  using stan::math::vector_d;
  
  matrix_fv ad(2,2);
  vector_d bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;

  matrix_fv resd = quad_form_diag(ad,bd);

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(ad(0,0).val_);
  vars.push_back(ad(0,1).val_);
  vars.push_back(ad(1,0).val_);
  vars.push_back(ad(1,1).val_);

  resd(0,0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}

TEST(AgradMixMatrixQuadFormDiag, mat_fv_row_vector_d_1) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_fv;
  using stan::math::row_vector_d;
  
  matrix_fv ad(2,2);
  row_vector_d bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;

  matrix_fv resd = quad_form_diag(ad,bd);
  EXPECT_FLOAT_EQ(20000, resd(0,0).val_.val());
  EXPECT_FLOAT_EQ(3000, resd(0,1).val_.val());
  EXPECT_FLOAT_EQ(4000, resd(1,0).val_.val());
  EXPECT_FLOAT_EQ(500, resd(1,1).val_.val());
  EXPECT_FLOAT_EQ(10000, resd(0,0).d_.val());
  EXPECT_FLOAT_EQ(1000, resd(0,1).d_.val());
  EXPECT_FLOAT_EQ(1000, resd(1,0).d_.val());
  EXPECT_FLOAT_EQ(100, resd(1,1).d_.val());

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(ad(0,0).val_);
  vars.push_back(ad(0,1).val_);
  vars.push_back(ad(1,0).val_);
  vars.push_back(ad(1,1).val_);

  resd(0,0).val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(10000, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}

TEST(AgradMixMatrixQuadFormDiag, mat_fv_row_vector_d_2) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_fv;
  using stan::math::row_vector_d;
  
  matrix_fv ad(2,2);
  row_vector_d bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;

  matrix_fv resd = quad_form_diag(ad,bd);

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(ad(0,0).val_);
  vars.push_back(ad(0,1).val_);
  vars.push_back(ad(1,0).val_);
  vars.push_back(ad(1,1).val_);

  resd(0,0).d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}

TEST(AgradMixMatrixQuadFormDiag, mat_fv_vector_d_exception) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_fv;
  using stan::math::vector_d;
  using stan::math::row_vector_d;

  matrix_fv m1(2,2);
  matrix_fv m2(3,2);
  matrix_fv m3(2,3);
  vector_d v1(3);
  vector_d v2(4);
  row_vector_d rv1(3);
  row_vector_d rv2(4);

  EXPECT_THROW(quad_form_diag(m1, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, v1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m2, v2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m2, rv1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m2, rv2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m3, v1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m3, v2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m3, rv1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m3, rv2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m1, m1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m1, m2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m1, m3), std::invalid_argument);
}

TEST(AgradMixMatrixQuadFormDiag, mat_ffv_vector_ffv_1) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;
  
  matrix_ffv ad(2,2);
  vector_ffv bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  ad(0,0).val_.d_ = 1.0;
  ad(0,1).val_.d_ = 1.0;
  ad(1,0).val_.d_ = 1.0;
  ad(1,1).val_.d_ = 1.0;
  bd(0).val_.d_ = 1.0;
  bd(1).val_.d_ = 1.0;

  matrix_ffv resd = quad_form_diag(ad,bd);
  EXPECT_FLOAT_EQ(20000, resd(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(3000, resd(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(4000, resd(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(500, resd(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(10400, resd(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(1330, resd(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(1440, resd(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(200, resd(1,1).d_.val_.val());

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(ad(0,0).val_.val_);
  vars.push_back(ad(0,1).val_.val_);
  vars.push_back(ad(1,0).val_.val_);
  vars.push_back(ad(1,1).val_.val_);
  vars.push_back(bd(0).val_.val_);
  vars.push_back(bd(1).val_.val_);

  resd(0,0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(10000, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(400, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
}

TEST(AgradMixMatrixQuadFormDiag, mat_ffv_vector_ffv_2) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;
  
  matrix_ffv ad(2,2);
  vector_ffv bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  ad(0,0).val_.d_ = 1.0;
  ad(0,1).val_.d_ = 1.0;
  ad(1,0).val_.d_ = 1.0;
  ad(1,1).val_.d_ = 1.0;
  bd(0).val_.d_ = 1.0;
  bd(1).val_.d_ = 1.0;

  matrix_ffv resd = quad_form_diag(ad,bd);

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(ad(0,0).val_.val_);
  vars.push_back(ad(0,1).val_.val_);
  vars.push_back(ad(1,0).val_.val_);
  vars.push_back(ad(1,1).val_.val_);
  vars.push_back(bd(0).val_.val_);
  vars.push_back(bd(1).val_.val_);

  resd(0,0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(200, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(204, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
}
TEST(AgradMixMatrixQuadFormDiag, mat_ffv_vector_ffv_3) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;
  
  matrix_ffv ad(2,2);
  vector_ffv bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  ad(0,0).val_.d_ = 1.0;
  ad(0,1).val_.d_ = 1.0;
  ad(1,0).val_.d_ = 1.0;
  ad(1,1).val_.d_ = 1.0;
  bd(0).val_.d_ = 1.0;
  bd(1).val_.d_ = 1.0;

  matrix_ffv resd = quad_form_diag(ad,bd);

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(ad(0,0).val_.val_);
  vars.push_back(ad(0,1).val_.val_);
  vars.push_back(ad(1,0).val_.val_);
  vars.push_back(ad(1,1).val_.val_);
  vars.push_back(bd(0).val_.val_);
  vars.push_back(bd(1).val_.val_);

  resd(0,0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(200, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(204, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
}
TEST(AgradMixMatrixQuadFormDiag, mat_ffv_vector_ffv_4) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;
  
  matrix_ffv ad(2,2);
  vector_ffv bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  ad(0,0).val_.d_ = 1.0;
  ad(0,1).val_.d_ = 1.0;
  ad(1,0).val_.d_ = 1.0;
  ad(1,1).val_.d_ = 1.0;
  bd(0).val_.d_ = 1.0;
  bd(1).val_.d_ = 1.0;

  matrix_ffv resd = quad_form_diag(ad,bd);

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(ad(0,0).val_.val_);
  vars.push_back(ad(0,1).val_.val_);
  vars.push_back(ad(1,0).val_.val_);
  vars.push_back(ad(1,1).val_.val_);
  vars.push_back(bd(0).val_.val_);
  vars.push_back(bd(1).val_.val_);

  resd(0,0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(2, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(4, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
}
TEST(AgradMixMatrixQuadFormDiag, mat_ffv_row_vector_ffv_1) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;
  
  matrix_ffv ad(2,2);
  row_vector_ffv bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  ad(0,0).val_.d_ = 1.0;
  ad(0,1).val_.d_ = 1.0;
  ad(1,0).val_.d_ = 1.0;
  ad(1,1).val_.d_ = 1.0;
  bd(0).val_.d_ = 1.0;
  bd(1).val_.d_ = 1.0;

  matrix_ffv resd = quad_form_diag(ad,bd);
  EXPECT_FLOAT_EQ(20000, resd(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(3000, resd(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(4000, resd(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(500, resd(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(10400, resd(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(1330, resd(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(1440, resd(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(200, resd(1,1).d_.val_.val());

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(ad(0,0).val_.val_);
  vars.push_back(ad(0,1).val_.val_);
  vars.push_back(ad(1,0).val_.val_);
  vars.push_back(ad(1,1).val_.val_);
  vars.push_back(bd(0).val_.val_);
  vars.push_back(bd(1).val_.val_);

  resd(0,0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(10000, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(400, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
}
TEST(AgradMixMatrixQuadFormDiag, mat_ffv_row_vector_ffv_2) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;
  
  matrix_ffv ad(2,2);
  row_vector_ffv bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  ad(0,0).val_.d_ = 1.0;
  ad(0,1).val_.d_ = 1.0;
  ad(1,0).val_.d_ = 1.0;
  ad(1,1).val_.d_ = 1.0;
  bd(0).val_.d_ = 1.0;
  bd(1).val_.d_ = 1.0;

  matrix_ffv resd = quad_form_diag(ad,bd);

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(ad(0,0).val_.val_);
  vars.push_back(ad(0,1).val_.val_);
  vars.push_back(ad(1,0).val_.val_);
  vars.push_back(ad(1,1).val_.val_);
  vars.push_back(bd(0).val_.val_);
  vars.push_back(bd(1).val_.val_);

  resd(0,0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(200, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(204, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
}
TEST(AgradMixMatrixQuadFormDiag, mat_ffv_row_vector_ffv_3) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;
  
  matrix_ffv ad(2,2);
  row_vector_ffv bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  ad(0,0).val_.d_ = 1.0;
  ad(0,1).val_.d_ = 1.0;
  ad(1,0).val_.d_ = 1.0;
  ad(1,1).val_.d_ = 1.0;
  bd(0).val_.d_ = 1.0;
  bd(1).val_.d_ = 1.0;

  matrix_ffv resd = quad_form_diag(ad,bd);

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(ad(0,0).val_.val_);
  vars.push_back(ad(0,1).val_.val_);
  vars.push_back(ad(1,0).val_.val_);
  vars.push_back(ad(1,1).val_.val_);
  vars.push_back(bd(0).val_.val_);
  vars.push_back(bd(1).val_.val_);

  resd(0,0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(200, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(204, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
}
TEST(AgradMixMatrixQuadFormDiag, mat_ffv_row_vector_ffv_4) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;
  
  matrix_ffv ad(2,2);
  row_vector_ffv bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  ad(0,0).val_.d_ = 1.0;
  ad(0,1).val_.d_ = 1.0;
  ad(1,0).val_.d_ = 1.0;
  ad(1,1).val_.d_ = 1.0;
  bd(0).val_.d_ = 1.0;
  bd(1).val_.d_ = 1.0;

  matrix_ffv resd = quad_form_diag(ad,bd);

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(ad(0,0).val_.val_);
  vars.push_back(ad(0,1).val_.val_);
  vars.push_back(ad(1,0).val_.val_);
  vars.push_back(ad(1,1).val_.val_);
  vars.push_back(bd(0).val_.val_);
  vars.push_back(bd(1).val_.val_);

  resd(0,0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(2, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
  EXPECT_FLOAT_EQ(4, grads[4]);
  EXPECT_FLOAT_EQ(0, grads[5]);
}
TEST(AgradMixMatrixQuadFormDiag, mat_ffv_vector_ffv_exception) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;
  using stan::math::row_vector_ffv;

  matrix_ffv m1(2,2);
  matrix_ffv m2(3,2);
  matrix_ffv m3(2,3);
  vector_ffv v1(3);
  vector_ffv v2(4);
  row_vector_ffv rv1(3);
  row_vector_ffv rv2(4);

  EXPECT_THROW(quad_form_diag(m1, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, v1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m2, v2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m2, rv1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m2, rv2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m3, v1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m3, v2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m3, rv1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m3, rv2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m1, m1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m1, m2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m1, m3), std::invalid_argument);
}
TEST(AgradMixMatrixQuadFormDiag, mat_d_vector_ffv_1) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;
  
  matrix_d ad(2,2);
  vector_ffv bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  bd(0).val_.d_ = 1.0;
  bd(1).val_.d_ = 1.0;

  matrix_ffv resd = quad_form_diag(ad,bd);
  EXPECT_FLOAT_EQ(20000, resd(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(3000, resd(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(4000, resd(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(500, resd(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(400, resd(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(330, resd(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(440, resd(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(100, resd(1,1).d_.val_.val());

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(bd(0).val_.val_);
  vars.push_back(bd(1).val_.val_);

  resd(0,0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(400, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}
TEST(AgradMixMatrixQuadFormDiag, mat_d_vector_ffv_2) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;
  
  matrix_d ad(2,2);
  vector_ffv bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  bd(0).val_.d_ = 1.0;
  bd(1).val_.d_ = 1.0;

  matrix_ffv resd = quad_form_diag(ad,bd);

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(bd(0).val_.val_);
  vars.push_back(bd(1).val_.val_);

  resd(0,0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(4, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}
TEST(AgradMixMatrixQuadFormDiag, mat_d_vector_ffv_3) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;
  
  matrix_d ad(2,2);
  vector_ffv bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  bd(0).val_.d_ = 1.0;
  bd(1).val_.d_ = 1.0;

  matrix_ffv resd = quad_form_diag(ad,bd);

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(bd(0).val_.val_);
  vars.push_back(bd(1).val_.val_);

  resd(0,0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(4, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}
TEST(AgradMixMatrixQuadFormDiag, mat_d_vector_ffv_4) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;
  
  matrix_d ad(2,2);
  vector_ffv bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  bd(0).val_.d_ = 1.0;
  bd(1).val_.d_ = 1.0;

  matrix_ffv resd = quad_form_diag(ad,bd);

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(bd(0).val_.val_);
  vars.push_back(bd(1).val_.val_);

  resd(0,0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}

TEST(AgradMixMatrixQuadFormDiag, mat_d_row_vector_ffv_1) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;
  
  matrix_d ad(2,2);
  row_vector_ffv bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  bd(0).val_.d_ = 1.0;
  bd(1).val_.d_ = 1.0;

  matrix_ffv resd = quad_form_diag(ad,bd);
  EXPECT_FLOAT_EQ(20000, resd(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(3000, resd(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(4000, resd(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(500, resd(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(400, resd(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(330, resd(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(440, resd(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(100, resd(1,1).d_.val_.val());

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(bd(0).val_.val_);
  vars.push_back(bd(1).val_.val_);

  resd(0,0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(400, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}

TEST(AgradMixMatrixQuadFormDiag, mat_d_row_vector_ffv_2) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;
  
  matrix_d ad(2,2);
  row_vector_ffv bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  bd(0).val_.d_ = 1.0;
  bd(1).val_.d_ = 1.0;

  matrix_ffv resd = quad_form_diag(ad,bd);

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(bd(0).val_.val_);
  vars.push_back(bd(1).val_.val_);

  resd(0,0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(4, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}

TEST(AgradMixMatrixQuadFormDiag, mat_d_row_vector_ffv_3) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;
  
  matrix_d ad(2,2);
  row_vector_ffv bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  bd(0).val_.d_ = 1.0;
  bd(1).val_.d_ = 1.0;

  matrix_ffv resd = quad_form_diag(ad,bd);

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(bd(0).val_.val_);
  vars.push_back(bd(1).val_.val_);

  resd(0,0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(4, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}
TEST(AgradMixMatrixQuadFormDiag, mat_d_row_vector_ffv_4) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;
  
  matrix_d ad(2,2);
  row_vector_ffv bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;
  bd(0).val_.d_ = 1.0;
  bd(1).val_.d_ = 1.0;

  matrix_ffv resd = quad_form_diag(ad,bd);

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(bd(0).val_.val_);
  vars.push_back(bd(1).val_.val_);

  resd(0,0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
}
TEST(AgradMixMatrixQuadFormDiag, mat_d_vector_ffv_exception) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::math::vector_ffv;
  using stan::math::row_vector_ffv;

  matrix_d m1(2,2);
  matrix_d m2(3,2);
  matrix_d m3(2,3);
  vector_ffv v1(3);
  vector_ffv v2(4);
  row_vector_ffv rv1(3);
  row_vector_ffv rv2(4);

  EXPECT_THROW(quad_form_diag(m1, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, v1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m2, v2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m2, rv1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m2, rv2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m3, v1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m3, v2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m3, rv1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m3, rv2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m1, m1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m1, m2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m1, m3), std::invalid_argument);
}

TEST(AgradMixMatrixQuadFormDiag, mat_ffv_vector_d_1) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_ffv;
  using stan::math::vector_d;
  
  matrix_ffv ad(2,2);
  vector_d bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  ad(0,0).val_.d_ = 1.0;
  ad(0,1).val_.d_ = 1.0;
  ad(1,0).val_.d_ = 1.0;
  ad(1,1).val_.d_ = 1.0;

  matrix_ffv resd = quad_form_diag(ad,bd);
  EXPECT_FLOAT_EQ(20000, resd(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(3000, resd(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(4000, resd(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(500, resd(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(10000, resd(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(1000, resd(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(1000, resd(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(100, resd(1,1).d_.val_.val());

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(ad(0,0).val_.val_);
  vars.push_back(ad(0,1).val_.val_);
  vars.push_back(ad(1,0).val_.val_);
  vars.push_back(ad(1,1).val_.val_);

  resd(0,0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(10000, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradMixMatrixQuadFormDiag, mat_ffv_vector_d_2) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_ffv;
  using stan::math::vector_d;
  
  matrix_ffv ad(2,2);
  vector_d bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  ad(0,0).val_.d_ = 1.0;
  ad(0,1).val_.d_ = 1.0;
  ad(1,0).val_.d_ = 1.0;
  ad(1,1).val_.d_ = 1.0;

  matrix_ffv resd = quad_form_diag(ad,bd);

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(ad(0,0).val_.val_);
  vars.push_back(ad(0,1).val_.val_);
  vars.push_back(ad(1,0).val_.val_);
  vars.push_back(ad(1,1).val_.val_);

  resd(0,0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradMixMatrixQuadFormDiag, mat_ffv_vector_d_3) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_ffv;
  using stan::math::vector_d;
  
  matrix_ffv ad(2,2);
  vector_d bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  ad(0,0).val_.d_ = 1.0;
  ad(0,1).val_.d_ = 1.0;
  ad(1,0).val_.d_ = 1.0;
  ad(1,1).val_.d_ = 1.0;

  matrix_ffv resd = quad_form_diag(ad,bd);

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(ad(0,0).val_.val_);
  vars.push_back(ad(0,1).val_.val_);
  vars.push_back(ad(1,0).val_.val_);
  vars.push_back(ad(1,1).val_.val_);

  resd(0,0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradMixMatrixQuadFormDiag, mat_ffv_vector_d_4) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_ffv;
  using stan::math::vector_d;
  
  matrix_ffv ad(2,2);
  vector_d bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  ad(0,0).val_.d_ = 1.0;
  ad(0,1).val_.d_ = 1.0;
  ad(1,0).val_.d_ = 1.0;
  ad(1,1).val_.d_ = 1.0;

  matrix_ffv resd = quad_form_diag(ad,bd);

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(ad(0,0).val_.val_);
  vars.push_back(ad(0,1).val_.val_);
  vars.push_back(ad(1,0).val_.val_);
  vars.push_back(ad(1,1).val_.val_);

  resd(0,0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}

TEST(AgradMixMatrixQuadFormDiag, mat_ffv_row_vector_d_1) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_ffv;
  using stan::math::row_vector_d;
  
  matrix_ffv ad(2,2);
  row_vector_d bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  ad(0,0).val_.d_ = 1.0;
  ad(0,1).val_.d_ = 1.0;
  ad(1,0).val_.d_ = 1.0;
  ad(1,1).val_.d_ = 1.0;

  matrix_ffv resd = quad_form_diag(ad,bd);
  EXPECT_FLOAT_EQ(20000, resd(0,0).val_.val_.val());
  EXPECT_FLOAT_EQ(3000, resd(0,1).val_.val_.val());
  EXPECT_FLOAT_EQ(4000, resd(1,0).val_.val_.val());
  EXPECT_FLOAT_EQ(500, resd(1,1).val_.val_.val());
  EXPECT_FLOAT_EQ(10000, resd(0,0).d_.val_.val());
  EXPECT_FLOAT_EQ(1000, resd(0,1).d_.val_.val());
  EXPECT_FLOAT_EQ(1000, resd(1,0).d_.val_.val());
  EXPECT_FLOAT_EQ(100, resd(1,1).d_.val_.val());

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(ad(0,0).val_.val_);
  vars.push_back(ad(0,1).val_.val_);
  vars.push_back(ad(1,0).val_.val_);
  vars.push_back(ad(1,1).val_.val_);

  resd(0,0).val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(10000, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}

TEST(AgradMixMatrixQuadFormDiag, mat_ffv_row_vector_d_2) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_ffv;
  using stan::math::row_vector_d;
  
  matrix_ffv ad(2,2);
  row_vector_d bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  ad(0,0).val_.d_ = 1.0;
  ad(0,1).val_.d_ = 1.0;
  ad(1,0).val_.d_ = 1.0;
  ad(1,1).val_.d_ = 1.0;

  matrix_ffv resd = quad_form_diag(ad,bd);

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(ad(0,0).val_.val_);
  vars.push_back(ad(0,1).val_.val_);
  vars.push_back(ad(1,0).val_.val_);
  vars.push_back(ad(1,1).val_.val_);

  resd(0,0).val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradMixMatrixQuadFormDiag, mat_ffv_row_vector_d_3) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_ffv;
  using stan::math::row_vector_d;
  
  matrix_ffv ad(2,2);
  row_vector_d bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  ad(0,0).val_.d_ = 1.0;
  ad(0,1).val_.d_ = 1.0;
  ad(1,0).val_.d_ = 1.0;
  ad(1,1).val_.d_ = 1.0;

  matrix_ffv resd = quad_form_diag(ad,bd);

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(ad(0,0).val_.val_);
  vars.push_back(ad(0,1).val_.val_);
  vars.push_back(ad(1,0).val_.val_);
  vars.push_back(ad(1,1).val_.val_);

  resd(0,0).d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradMixMatrixQuadFormDiag, mat_ffv_row_vector_d_4) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_ffv;
  using stan::math::row_vector_d;
  
  matrix_ffv ad(2,2);
  row_vector_d bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  ad(0,0).val_.d_ = 1.0;
  ad(0,1).val_.d_ = 1.0;
  ad(1,0).val_.d_ = 1.0;
  ad(1,1).val_.d_ = 1.0;

  matrix_ffv resd = quad_form_diag(ad,bd);

  std::vector<var> vars;
  std::vector<double> grads;
  vars.push_back(ad(0,0).val_.val_);
  vars.push_back(ad(0,1).val_.val_);
  vars.push_back(ad(1,0).val_.val_);
  vars.push_back(ad(1,1).val_.val_);

  resd(0,0).d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(0, grads[3]);
}
TEST(AgradMixMatrixQuadFormDiag, mat_ffv_vector_d_exception) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_ffv;
  using stan::math::vector_d;
  using stan::math::row_vector_d;

  matrix_ffv m1(2,2);
  matrix_ffv m2(3,2);
  matrix_ffv m3(2,3);
  vector_d v1(3);
  vector_d v2(4);
  row_vector_d rv1(3);
  row_vector_d rv2(4);

  EXPECT_THROW(quad_form_diag(m1, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, v1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m2, v2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m2, rv1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m2, rv2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m3, v1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m3, v2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m3, rv1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m3, rv2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m1, m1), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m1, m2), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m1, m3), std::invalid_argument);
}

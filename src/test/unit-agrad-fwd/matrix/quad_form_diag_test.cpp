#include <gtest/gtest.h>
#include <stan/math/matrix/quad_form_diag.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev/matrix/multiply.hpp>
#include <stan/agrad/fwd/matrix/multiply.hpp>
#include <stan/agrad/fwd/matrix/dot_product.hpp>
#include <stan/agrad/rev/matrix/dot_product.hpp>
#include <stan/agrad/rev/operators.hpp>
#include <test/unit/agrad/util.hpp>

using stan::agrad::fvar;
using stan::agrad::var;

TEST(AgradFwdMatrixQuadFormDiag, mat_fd_vector_fd) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_fd;
  using stan::agrad::vector_fd;
  
  matrix_fd ad(2,2);
  vector_fd bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;

  matrix_fd resd = quad_form_diag(ad,bd);
  EXPECT_FLOAT_EQ(20000, resd(0,0).val_);
  EXPECT_FLOAT_EQ(3000, resd(0,1).val_);
  EXPECT_FLOAT_EQ(4000, resd(1,0).val_);
  EXPECT_FLOAT_EQ(500, resd(1,1).val_);
  EXPECT_FLOAT_EQ(10400, resd(0,0).d_);
  EXPECT_FLOAT_EQ(1330, resd(0,1).d_);
  EXPECT_FLOAT_EQ(1440, resd(1,0).d_);
  EXPECT_FLOAT_EQ(200, resd(1,1).d_);
}

TEST(AgradFwdMatrixQuadFormDiag, mat_fd_row_vector_fd) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_fd;
  using stan::agrad::row_vector_fd;
  
  matrix_fd ad(2,2);
  row_vector_fd bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;

  matrix_fd resd = quad_form_diag(ad,bd);
  EXPECT_FLOAT_EQ(20000, resd(0,0).val_);
  EXPECT_FLOAT_EQ(3000, resd(0,1).val_);
  EXPECT_FLOAT_EQ(4000, resd(1,0).val_);
  EXPECT_FLOAT_EQ(500, resd(1,1).val_);
  EXPECT_FLOAT_EQ(10400, resd(0,0).d_);
  EXPECT_FLOAT_EQ(1330, resd(0,1).d_);
  EXPECT_FLOAT_EQ(1440, resd(1,0).d_);
  EXPECT_FLOAT_EQ(200, resd(1,1).d_);
}

TEST(AgradFwdMatrixQuadFormDiag, mat_fd_vector_fd_exception) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_fd;
  using stan::agrad::vector_fd;
  using stan::agrad::row_vector_fd;

  matrix_fd m1(2,2);
  matrix_fd m2(3,2);
  matrix_fd m3(2,3);
  vector_fd v1(3);
  vector_fd v2(4);
  row_vector_fd rv1(3);
  row_vector_fd rv2(4);

  EXPECT_THROW(quad_form_diag(m1, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m3), std::domain_error);
}
TEST(AgradFwdMatrixQuadFormDiag, mat_d_vector_fd) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fd;
  using stan::agrad::vector_fd;
  
  matrix_d ad(2,2);
  vector_fd bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;

  matrix_fd resd = quad_form_diag(ad,bd);
  EXPECT_FLOAT_EQ(20000, resd(0,0).val_);
  EXPECT_FLOAT_EQ(3000, resd(0,1).val_);
  EXPECT_FLOAT_EQ(4000, resd(1,0).val_);
  EXPECT_FLOAT_EQ(500, resd(1,1).val_);
  EXPECT_FLOAT_EQ(400, resd(0,0).d_);
  EXPECT_FLOAT_EQ(330, resd(0,1).d_);
  EXPECT_FLOAT_EQ(440, resd(1,0).d_);
  EXPECT_FLOAT_EQ(100, resd(1,1).d_);
}

TEST(AgradFwdMatrixQuadFormDiag, mat_d_row_vector_fd) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fd;
  using stan::agrad::row_vector_fd;
  
  matrix_d ad(2,2);
  row_vector_fd bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;

  matrix_fd resd = quad_form_diag(ad,bd);
  EXPECT_FLOAT_EQ(20000, resd(0,0).val_);
  EXPECT_FLOAT_EQ(3000, resd(0,1).val_);
  EXPECT_FLOAT_EQ(4000, resd(1,0).val_);
  EXPECT_FLOAT_EQ(500, resd(1,1).val_);
  EXPECT_FLOAT_EQ(400, resd(0,0).d_);
  EXPECT_FLOAT_EQ(330, resd(0,1).d_);
  EXPECT_FLOAT_EQ(440, resd(1,0).d_);
  EXPECT_FLOAT_EQ(100, resd(1,1).d_);
}

TEST(AgradFwdMatrixQuadFormDiag, mat_d_vector_fd_exception) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::agrad::vector_fd;
  using stan::agrad::row_vector_fd;

  matrix_d m1(2,2);
  matrix_d m2(3,2);
  matrix_d m3(2,3);
  vector_fd v1(3);
  vector_fd v2(4);
  row_vector_fd rv1(3);
  row_vector_fd rv2(4);

  EXPECT_THROW(quad_form_diag(m1, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m3), std::domain_error);
}

TEST(AgradFwdMatrixQuadFormDiag, mat_fd_vector_d) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_fd;
  using stan::math::vector_d;
  
  matrix_fd ad(2,2);
  vector_d bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;

  matrix_fd resd = quad_form_diag(ad,bd);
  EXPECT_FLOAT_EQ(20000, resd(0,0).val_);
  EXPECT_FLOAT_EQ(3000, resd(0,1).val_);
  EXPECT_FLOAT_EQ(4000, resd(1,0).val_);
  EXPECT_FLOAT_EQ(500, resd(1,1).val_);
  EXPECT_FLOAT_EQ(10000, resd(0,0).d_);
  EXPECT_FLOAT_EQ(1000, resd(0,1).d_);
  EXPECT_FLOAT_EQ(1000, resd(1,0).d_);
  EXPECT_FLOAT_EQ(100, resd(1,1).d_);
}

TEST(AgradFwdMatrixQuadFormDiag, mat_fd_row_vector_d) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_fd;
  using stan::math::row_vector_d;
  
  matrix_fd ad(2,2);
  row_vector_d bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;

  matrix_fd resd = quad_form_diag(ad,bd);
  EXPECT_FLOAT_EQ(20000, resd(0,0).val_);
  EXPECT_FLOAT_EQ(3000, resd(0,1).val_);
  EXPECT_FLOAT_EQ(4000, resd(1,0).val_);
  EXPECT_FLOAT_EQ(500, resd(1,1).val_);
  EXPECT_FLOAT_EQ(10000, resd(0,0).d_);
  EXPECT_FLOAT_EQ(1000, resd(0,1).d_);
  EXPECT_FLOAT_EQ(1000, resd(1,0).d_);
  EXPECT_FLOAT_EQ(100, resd(1,1).d_);
}

TEST(AgradFwdMatrixQuadFormDiag, mat_fd_vector_d_exception) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_fd;
  using stan::math::vector_d;
  using stan::math::row_vector_d;

  matrix_fd m1(2,2);
  matrix_fd m2(3,2);
  matrix_fd m3(2,3);
  vector_d v1(3);
  vector_d v2(4);
  row_vector_d rv1(3);
  row_vector_d rv2(4);

  EXPECT_THROW(quad_form_diag(m1, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m3), std::domain_error);
}

TEST(AgradFwdMatrixQuadFormDiag, mat_ffd_vector_ffd) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_ffd;
  using stan::agrad::vector_ffd;
  
  matrix_ffd ad(2,2);
  vector_ffd bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;

  matrix_ffd resd = quad_form_diag(ad,bd);
  EXPECT_FLOAT_EQ(20000, resd(0,0).val_.val_);
  EXPECT_FLOAT_EQ(3000, resd(0,1).val_.val_);
  EXPECT_FLOAT_EQ(4000, resd(1,0).val_.val_);
  EXPECT_FLOAT_EQ(500, resd(1,1).val_.val_);
  EXPECT_FLOAT_EQ(10400, resd(0,0).d_.val_);
  EXPECT_FLOAT_EQ(1330, resd(0,1).d_.val_);
  EXPECT_FLOAT_EQ(1440, resd(1,0).d_.val_);
  EXPECT_FLOAT_EQ(200, resd(1,1).d_.val_);
}

TEST(AgradFwdMatrixQuadFormDiag, mat_ffd_row_vector_ffd) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_ffd;
  using stan::agrad::row_vector_ffd;
  
  matrix_ffd ad(2,2);
  row_vector_ffd bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;

  matrix_ffd resd = quad_form_diag(ad,bd);
  EXPECT_FLOAT_EQ(20000, resd(0,0).val_.val_);
  EXPECT_FLOAT_EQ(3000, resd(0,1).val_.val_);
  EXPECT_FLOAT_EQ(4000, resd(1,0).val_.val_);
  EXPECT_FLOAT_EQ(500, resd(1,1).val_.val_);
  EXPECT_FLOAT_EQ(10400, resd(0,0).d_.val_);
  EXPECT_FLOAT_EQ(1330, resd(0,1).d_.val_);
  EXPECT_FLOAT_EQ(1440, resd(1,0).d_.val_);
  EXPECT_FLOAT_EQ(200, resd(1,1).d_.val_);
}

TEST(AgradFwdMatrixQuadFormDiag, mat_ffd_vector_ffd_exception) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_ffd;
  using stan::agrad::vector_ffd;
  using stan::agrad::row_vector_ffd;

  matrix_ffd m1(2,2);
  matrix_ffd m2(3,2);
  matrix_ffd m3(2,3);
  vector_ffd v1(3);
  vector_ffd v2(4);
  row_vector_ffd rv1(3);
  row_vector_ffd rv2(4);

  EXPECT_THROW(quad_form_diag(m1, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m3), std::domain_error);
}
TEST(AgradFwdMatrixQuadFormDiag, mat_d_vector_ffd) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffd;
  using stan::agrad::vector_ffd;
  
  matrix_d ad(2,2);
  vector_ffd bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;

  matrix_ffd resd = quad_form_diag(ad,bd);
  EXPECT_FLOAT_EQ(20000, resd(0,0).val_.val_);
  EXPECT_FLOAT_EQ(3000, resd(0,1).val_.val_);
  EXPECT_FLOAT_EQ(4000, resd(1,0).val_.val_);
  EXPECT_FLOAT_EQ(500, resd(1,1).val_.val_);
  EXPECT_FLOAT_EQ(400, resd(0,0).d_.val_);
  EXPECT_FLOAT_EQ(330, resd(0,1).d_.val_);
  EXPECT_FLOAT_EQ(440, resd(1,0).d_.val_);
  EXPECT_FLOAT_EQ(100, resd(1,1).d_.val_);
}

TEST(AgradFwdMatrixQuadFormDiag, mat_d_row_vector_ffd) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffd;
  using stan::agrad::row_vector_ffd;
  
  matrix_d ad(2,2);
  row_vector_ffd bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  bd(0).d_ = 1.0;
  bd(1).d_ = 1.0;

  matrix_ffd resd = quad_form_diag(ad,bd);
  EXPECT_FLOAT_EQ(20000, resd(0,0).val_.val_);
  EXPECT_FLOAT_EQ(3000, resd(0,1).val_.val_);
  EXPECT_FLOAT_EQ(4000, resd(1,0).val_.val_);
  EXPECT_FLOAT_EQ(500, resd(1,1).val_.val_);
  EXPECT_FLOAT_EQ(400, resd(0,0).d_.val_);
  EXPECT_FLOAT_EQ(330, resd(0,1).d_.val_);
  EXPECT_FLOAT_EQ(440, resd(1,0).d_.val_);
  EXPECT_FLOAT_EQ(100, resd(1,1).d_.val_);
}

TEST(AgradFwdMatrixQuadFormDiag, mat_d_vector_ffd_exception) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::agrad::vector_ffd;
  using stan::agrad::row_vector_ffd;

  matrix_d m1(2,2);
  matrix_d m2(3,2);
  matrix_d m3(2,3);
  vector_ffd v1(3);
  vector_ffd v2(4);
  row_vector_ffd rv1(3);
  row_vector_ffd rv2(4);

  EXPECT_THROW(quad_form_diag(m1, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m3), std::domain_error);
}

TEST(AgradFwdMatrixQuadFormDiag, mat_ffd_vector_d) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_ffd;
  using stan::math::vector_d;
  
  matrix_ffd ad(2,2);
  vector_d bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;

  matrix_ffd resd = quad_form_diag(ad,bd);
  EXPECT_FLOAT_EQ(20000, resd(0,0).val_.val_);
  EXPECT_FLOAT_EQ(3000, resd(0,1).val_.val_);
  EXPECT_FLOAT_EQ(4000, resd(1,0).val_.val_);
  EXPECT_FLOAT_EQ(500, resd(1,1).val_.val_);
  EXPECT_FLOAT_EQ(10000, resd(0,0).d_.val_);
  EXPECT_FLOAT_EQ(1000, resd(0,1).d_.val_);
  EXPECT_FLOAT_EQ(1000, resd(1,0).d_.val_);
  EXPECT_FLOAT_EQ(100, resd(1,1).d_.val_);
}

TEST(AgradFwdMatrixQuadFormDiag, mat_ffd_row_vector_d) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_ffd;
  using stan::math::row_vector_d;
  
  matrix_ffd ad(2,2);
  row_vector_d bd(2);
  
  bd << 100, 10;
  ad << 2.0,  3.0, 4.0,   5.0;
  
  ad(0,0).d_ = 1.0;
  ad(0,1).d_ = 1.0;
  ad(1,0).d_ = 1.0;
  ad(1,1).d_ = 1.0;

  matrix_ffd resd = quad_form_diag(ad,bd);
  EXPECT_FLOAT_EQ(20000, resd(0,0).val_.val_);
  EXPECT_FLOAT_EQ(3000, resd(0,1).val_.val_);
  EXPECT_FLOAT_EQ(4000, resd(1,0).val_.val_);
  EXPECT_FLOAT_EQ(500, resd(1,1).val_.val_);
  EXPECT_FLOAT_EQ(10000, resd(0,0).d_.val_);
  EXPECT_FLOAT_EQ(1000, resd(0,1).d_.val_);
  EXPECT_FLOAT_EQ(1000, resd(1,0).d_.val_);
  EXPECT_FLOAT_EQ(100, resd(1,1).d_.val_);
}

TEST(AgradFwdMatrixQuadFormDiag, mat_ffd_vector_d_exception) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_ffd;
  using stan::math::vector_d;
  using stan::math::row_vector_d;

  matrix_ffd m1(2,2);
  matrix_ffd m2(3,2);
  matrix_ffd m3(2,3);
  vector_d v1(3);
  vector_d v2(4);
  row_vector_d rv1(3);
  row_vector_d rv2(4);

  EXPECT_THROW(quad_form_diag(m1, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m3), std::domain_error);
}

TEST(AgradFwdMatrixQuadFormDiag, mat_fv_vector_fv_1) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;
  
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

TEST(AgradFwdMatrixQuadFormDiag, mat_fv_vector_fv_2) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;
  
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
TEST(AgradFwdMatrixQuadFormDiag, mat_fv_row_vector_fv_1) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_fv;
  using stan::agrad::row_vector_fv;
  
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
TEST(AgradFwdMatrixQuadFormDiag, mat_fv_row_vector_fv_2) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_fv;
  using stan::agrad::row_vector_fv;
  
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

TEST(AgradFwdMatrixQuadFormDiag, mat_fv_vector_fv_exception) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;
  using stan::agrad::row_vector_fv;

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
  EXPECT_THROW(quad_form_diag(m2, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m3), std::domain_error);
}
TEST(AgradFwdMatrixQuadFormDiag, mat_d_vector_fv_1) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;
  
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
TEST(AgradFwdMatrixQuadFormDiag, mat_d_vector_fv_2) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;
  
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

TEST(AgradFwdMatrixQuadFormDiag, mat_d_row_vector_fv_1) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::row_vector_fv;
  
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

TEST(AgradFwdMatrixQuadFormDiag, mat_d_row_vector_fv_2) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::row_vector_fv;
  
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

TEST(AgradFwdMatrixQuadFormDiag, mat_d_vector_fv_exception) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::agrad::vector_fv;
  using stan::agrad::row_vector_fv;

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
  EXPECT_THROW(quad_form_diag(m2, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m3), std::domain_error);
}

TEST(AgradFwdMatrixQuadFormDiag, mat_fv_vector_d_1) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_fv;
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
TEST(AgradFwdMatrixQuadFormDiag, mat_fv_vector_d_2) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_fv;
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

TEST(AgradFwdMatrixQuadFormDiag, mat_fv_row_vector_d_1) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_fv;
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

TEST(AgradFwdMatrixQuadFormDiag, mat_fv_row_vector_d_2) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_fv;
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

TEST(AgradFwdMatrixQuadFormDiag, mat_fv_vector_d_exception) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_fv;
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
  EXPECT_THROW(quad_form_diag(m2, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m3), std::domain_error);
}

TEST(AgradFwdMatrixQuadFormDiag, mat_ffv_vector_ffv_1) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  
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

TEST(AgradFwdMatrixQuadFormDiag, mat_ffv_vector_ffv_2) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  
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
TEST(AgradFwdMatrixQuadFormDiag, mat_ffv_vector_ffv_3) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  
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
TEST(AgradFwdMatrixQuadFormDiag, mat_ffv_vector_ffv_4) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  
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
TEST(AgradFwdMatrixQuadFormDiag, mat_ffv_row_vector_ffv_1) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_ffv;
  using stan::agrad::row_vector_ffv;
  
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
TEST(AgradFwdMatrixQuadFormDiag, mat_ffv_row_vector_ffv_2) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_ffv;
  using stan::agrad::row_vector_ffv;
  
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
TEST(AgradFwdMatrixQuadFormDiag, mat_ffv_row_vector_ffv_3) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_ffv;
  using stan::agrad::row_vector_ffv;
  
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
TEST(AgradFwdMatrixQuadFormDiag, mat_ffv_row_vector_ffv_4) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_ffv;
  using stan::agrad::row_vector_ffv;
  
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
TEST(AgradFwdMatrixQuadFormDiag, mat_ffv_vector_ffv_exception) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::agrad::row_vector_ffv;

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
  EXPECT_THROW(quad_form_diag(m2, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m3), std::domain_error);
}
TEST(AgradFwdMatrixQuadFormDiag, mat_d_vector_ffv_1) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  
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
TEST(AgradFwdMatrixQuadFormDiag, mat_d_vector_ffv_2) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  
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
TEST(AgradFwdMatrixQuadFormDiag, mat_d_vector_ffv_3) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  
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
TEST(AgradFwdMatrixQuadFormDiag, mat_d_vector_ffv_4) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  
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

TEST(AgradFwdMatrixQuadFormDiag, mat_d_row_vector_ffv_1) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::row_vector_ffv;
  
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

TEST(AgradFwdMatrixQuadFormDiag, mat_d_row_vector_ffv_2) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::row_vector_ffv;
  
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

TEST(AgradFwdMatrixQuadFormDiag, mat_d_row_vector_ffv_3) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::row_vector_ffv;
  
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
TEST(AgradFwdMatrixQuadFormDiag, mat_d_row_vector_ffv_4) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::row_vector_ffv;
  
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
TEST(AgradFwdMatrixQuadFormDiag, mat_d_vector_ffv_exception) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::row_vector_ffv;

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
  EXPECT_THROW(quad_form_diag(m2, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m3), std::domain_error);
}

TEST(AgradFwdMatrixQuadFormDiag, mat_ffv_vector_d_1) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_ffv;
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
TEST(AgradFwdMatrixQuadFormDiag, mat_ffv_vector_d_2) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_ffv;
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
TEST(AgradFwdMatrixQuadFormDiag, mat_ffv_vector_d_3) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_ffv;
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
TEST(AgradFwdMatrixQuadFormDiag, mat_ffv_vector_d_4) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_ffv;
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

TEST(AgradFwdMatrixQuadFormDiag, mat_ffv_row_vector_d_1) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_ffv;
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

TEST(AgradFwdMatrixQuadFormDiag, mat_ffv_row_vector_d_2) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_ffv;
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
TEST(AgradFwdMatrixQuadFormDiag, mat_ffv_row_vector_d_3) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_ffv;
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
TEST(AgradFwdMatrixQuadFormDiag, mat_ffv_row_vector_d_4) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_ffv;
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
TEST(AgradFwdMatrixQuadFormDiag, mat_ffv_vector_d_exception) {
  using stan::math::quad_form_diag;
  using stan::agrad::matrix_ffv;
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
  EXPECT_THROW(quad_form_diag(m2, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, v1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, v2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, rv1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m3, rv2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m1), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m2), std::domain_error);
  EXPECT_THROW(quad_form_diag(m1, m3), std::domain_error);
}

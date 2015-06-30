#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/quad_form_diag.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/multiply.hpp>
#include <stan/math/fwd/mat/fun/dot_product.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>

using stan::math::fvar;

TEST(AgradFwdMatrixQuadFormDiag, mat_fd_vector_fd) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_fd;
  using stan::math::vector_fd;
  
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
  using stan::math::matrix_fd;
  using stan::math::row_vector_fd;
  
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
  using stan::math::matrix_fd;
  using stan::math::vector_fd;
  using stan::math::row_vector_fd;

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
TEST(AgradFwdMatrixQuadFormDiag, mat_d_vector_fd) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::math::matrix_fd;
  using stan::math::vector_fd;
  
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
  using stan::math::matrix_fd;
  using stan::math::row_vector_fd;
  
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
  using stan::math::vector_fd;
  using stan::math::row_vector_fd;

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

TEST(AgradFwdMatrixQuadFormDiag, mat_fd_vector_d) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_fd;
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
  using stan::math::matrix_fd;
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
  using stan::math::matrix_fd;
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

TEST(AgradFwdMatrixQuadFormDiag, mat_ffd_vector_ffd) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_ffd;
  using stan::math::vector_ffd;
  
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
  using stan::math::matrix_ffd;
  using stan::math::row_vector_ffd;
  
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
  using stan::math::matrix_ffd;
  using stan::math::vector_ffd;
  using stan::math::row_vector_ffd;

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
TEST(AgradFwdMatrixQuadFormDiag, mat_d_vector_ffd) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;
  using stan::math::vector_ffd;
  
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
  using stan::math::matrix_ffd;
  using stan::math::row_vector_ffd;
  
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
  using stan::math::vector_ffd;
  using stan::math::row_vector_ffd;

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

TEST(AgradFwdMatrixQuadFormDiag, mat_ffd_vector_d) {
  using stan::math::quad_form_diag;
  using stan::math::matrix_ffd;
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
  using stan::math::matrix_ffd;
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
  using stan::math::matrix_ffd;
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

#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/prod.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/fwd/core.hpp>

TEST(AgradFwdMatrixProd,fd_vector) {
  using stan::math::prod;
  using stan::math::vector_d;
  using stan::math::vector_fd;
  using stan::math::fvar;

  vector_d vd;
  vector_fd vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_);

  vd = vector_d(1);
  vv = vector_fd(1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_);
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_);

  vd = vector_d(2);
  vd << 2.0, 3.0;
  vv = vector_fd(2);
  vv << 2.0, 3.0;
   vv(0).d_ = 1.0;
   vv(1).d_ = 1.0;

  fvar<double> f = prod(vv);
  EXPECT_FLOAT_EQ(6.0,prod(vd));
  EXPECT_FLOAT_EQ(6.0,f.val_);
  EXPECT_FLOAT_EQ(5.0,f.d_);
}

TEST(AgradFwdMatrixProd,fd_rowvector) {
  using stan::math::prod;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;
  using stan::math::fvar;

  row_vector_d vd;
  row_vector_fd vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_);

  vd = row_vector_d(1);
  vv = row_vector_fd(1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_);
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_);

  vd = row_vector_d(2);
  vd << 2.0, 3.0;
  vv = row_vector_fd(2);
  vv << 2.0, 3.0;
   vv(0).d_ = 1.0;
   vv(1).d_ = 1.0;

  fvar<double> f = prod(vv);
  EXPECT_FLOAT_EQ(6.0,prod(vd));
  EXPECT_FLOAT_EQ(6.0,f.val_);
  EXPECT_FLOAT_EQ(5.0,f.d_);
}
TEST(AgradFwdMatrixProd,fd_matrix) {
  using stan::math::prod;
  using stan::math::matrix_d;
  using stan::math::matrix_fd;
  using stan::math::fvar;

  matrix_d vd;
  matrix_fd vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_);

  vd = matrix_d(1,1);
  vv = matrix_fd(1,1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_);
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_);

  vd = matrix_d(2,2);
  vd << 2.0, 3.0,2.0, 3.0;
  vv = matrix_fd(2,2);
  vv << 2.0, 3.0,2.0, 3.0;
   vv(0,0).d_ = 1.0;
   vv(0,1).d_ = 1.0;
   vv(1,0).d_ = 1.0;
   vv(1,1).d_ = 1.0;

  fvar<double> f = prod(vv);
  EXPECT_FLOAT_EQ(36.0,prod(vd));
  EXPECT_FLOAT_EQ(36.0,f.val_);
  EXPECT_FLOAT_EQ(60.0,f.d_);
}
TEST(AgradFwdMatrixProd,ffd_vector) {
  using stan::math::prod;
  using stan::math::vector_d;
  using stan::math::vector_ffd;
  using stan::math::fvar;

  vector_d vd;
  vector_ffd vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_.val());

  vd = vector_d(1);
  vv = vector_ffd(1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_.val());
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_.val());

  vd = vector_d(2);
  vd << 2.0, 3.0;
  vv = vector_ffd(2);
  vv << 2.0, 3.0;
   vv(0).d_ = 1.0;
   vv(1).d_ = 1.0;

  fvar<fvar<double> > f = prod(vv);
  EXPECT_FLOAT_EQ(6.0,prod(vd));
  EXPECT_FLOAT_EQ(6.0,f.val_.val());
  EXPECT_FLOAT_EQ(5.0,f.d_.val());
}

TEST(AgradFwdMatrixProd,ffd_rowvector) {
  using stan::math::prod;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;
  using stan::math::fvar;

  row_vector_d vd;
  row_vector_ffd vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_.val());

  vd = row_vector_d(1);
  vv = row_vector_ffd(1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_.val());
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_.val());

  vd = row_vector_d(2);
  vd << 2.0, 3.0;
  vv = row_vector_ffd(2);
  vv << 2.0, 3.0;
   vv(0).d_ = 1.0;
   vv(1).d_ = 1.0;
  row_vector_ffd x(2);
  x[0] = vv[0];
  x[1] = vv[1];

  fvar<fvar<double> > f = prod(vv);
  EXPECT_FLOAT_EQ(6.0,prod(vd));
  EXPECT_FLOAT_EQ(6.0,f.val_.val());
  EXPECT_FLOAT_EQ(5.0,f.d_.val());
}
TEST(AgradFwdMatrixProd,ffd_matrix) {
  using stan::math::prod;
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;
  using stan::math::fvar;

  matrix_d vd;
  matrix_ffd vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_.val());

  vd = matrix_d(1,1);
  vv = matrix_ffd(1,1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_.val());
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_.val());

  vd = matrix_d(2,2);
  vd << 2.0, 3.0,2.0, 3.0;
  vv = matrix_ffd(2,2);
  vv << 2.0, 3.0,2.0, 3.0;
   vv(0,0).d_ = 1.0;
   vv(0,1).d_ = 1.0;
   vv(1,0).d_ = 1.0;
   vv(1,1).d_ = 1.0;

  fvar<fvar<double> > f = prod(vv);
  EXPECT_FLOAT_EQ(36.0,prod(vd));
  EXPECT_FLOAT_EQ(36.0,f.val_.val());
  EXPECT_FLOAT_EQ(60.0,f.d_.val());
}

#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/math/matrix/prod.hpp>

TEST(AgradFwdMatrix,prod_vector) {
  using stan::math::prod;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;

  vector_d vd;
  vector_fv vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_);

  vd = vector_d(1);
  vv = vector_fv(1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_);
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_);

  vd = vector_d(2);
  vd << 2.0, 3.0;
  vv = vector_fv(2);
  vv << 2.0, 3.0;
   vv(0).d_ = 1.0;
   vv(1).d_ = 1.0;
  std::vector<fvar<double> > x(2);
  x[0] = vv[0];
  x[1] = vv[1];

  fvar<double> f = prod(vv);
  EXPECT_FLOAT_EQ(6.0,prod(vd));
  EXPECT_FLOAT_EQ(6.0,f.val_);
  EXPECT_FLOAT_EQ(5.0,f.d_);
}

TEST(AgradFwdMatrix,prod_rowvector) {
  using stan::math::prod;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;

  row_vector_d vd;
  row_vector_fv vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_);

  vd = row_vector_d(1);
  vv = row_vector_fv(1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_);
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_);

  vd = row_vector_d(2);
  vd << 2.0, 3.0;
  vv = row_vector_fv(2);
  vv << 2.0, 3.0;
   vv(0).d_ = 1.0;
   vv(1).d_ = 1.0;
  row_vector_fv x(2);
  x[0] = vv[0];
  x[1] = vv[1];

  fvar<double> f = prod(vv);
  EXPECT_FLOAT_EQ(6.0,prod(vd));
  EXPECT_FLOAT_EQ(6.0,f.val_);
  EXPECT_FLOAT_EQ(5.0,f.d_);
}
TEST(AgradFwdMatrix,prod_matrix) {
  using stan::math::prod;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::fvar;

  matrix_d vd;
  matrix_fv vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_);

  vd = matrix_d(1,1);
  vv = matrix_fv(1,1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_);
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_);

  vd = matrix_d(2,2);
  vd << 2.0, 3.0,2.0, 3.0;
  vv = matrix_fv(2,2);
  vv << 2.0, 3.0,2.0, 3.0;
   vv(0,0).d_ = 1.0;
   vv(0,1).d_ = 1.0;
   vv(1,0).d_ = 1.0;
   vv(1,1).d_ = 1.0;
  matrix_fv x(2,2);
  x(0,0) = vv(0,0);
  x(0,1) = vv(0,1);
  x(1,0) = vv(1,0);
  x(1,1) = vv(1,1);

  fvar<double> f = prod(vv);
  EXPECT_FLOAT_EQ(36.0,prod(vd));
  EXPECT_FLOAT_EQ(36.0,f.val_);
  EXPECT_FLOAT_EQ(60.0,f.d_);
}


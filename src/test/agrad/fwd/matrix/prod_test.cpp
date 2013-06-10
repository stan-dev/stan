#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/math/matrix/prod.hpp>
#include <stan/agrad/var.hpp>

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

  fvar<double> f = prod(vv);
  EXPECT_FLOAT_EQ(36.0,prod(vd));
  EXPECT_FLOAT_EQ(36.0,f.val_);
  EXPECT_FLOAT_EQ(60.0,f.d_);
}
TEST(AgradFwdFvarVarMatrix,prod_vector) {
  using stan::math::prod;
  using stan::math::vector_d;
  using stan::agrad::vector_fvv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  vector_d vd;
  vector_fvv vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_.val());

  vd = vector_d(1);
  vv = vector_fvv(1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_.val());
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_.val());

  vd = vector_d(2);
  vd << 2.0, 3.0;
  vv = vector_fvv(2);
  vv << 2.0, 3.0;
   vv(0).d_ = 1.0;
   vv(1).d_ = 1.0;
  std::vector<fvar<var> > x(2);
  x[0] = vv[0];
  x[1] = vv[1];

  fvar<var> f = prod(vv);
  EXPECT_FLOAT_EQ(6.0,prod(vd));
  EXPECT_FLOAT_EQ(6.0,f.val_.val());
  EXPECT_FLOAT_EQ(5.0,f.d_.val());
}

TEST(AgradFwdFvarVarMatrix,prod_rowvector) {
  using stan::math::prod;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fvv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  row_vector_d vd;
  row_vector_fvv vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_.val());

  vd = row_vector_d(1);
  vv = row_vector_fvv(1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_.val());
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_.val());

  vd = row_vector_d(2);
  vd << 2.0, 3.0;
  vv = row_vector_fvv(2);
  vv << 2.0, 3.0;
   vv(0).d_ = 1.0;
   vv(1).d_ = 1.0;

  fvar<var> f = prod(vv);
  EXPECT_FLOAT_EQ(6.0,prod(vd));
  EXPECT_FLOAT_EQ(6.0,f.val_.val());
  EXPECT_FLOAT_EQ(5.0,f.d_.val());
}
TEST(AgradFwdFvarVarMatrix,prod_matrix) {
  using stan::math::prod;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fvv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_d vd;
  matrix_fvv vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_.val());

  vd = matrix_d(1,1);
  vv = matrix_fvv(1,1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_.val());
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_.val());

  vd = matrix_d(2,2);
  vd << 2.0, 3.0,2.0, 3.0;
  vv = matrix_fvv(2,2);
  vv << 2.0, 3.0,2.0, 3.0;
   vv(0,0).d_ = 1.0;
   vv(0,1).d_ = 1.0;
   vv(1,0).d_ = 1.0;
   vv(1,1).d_ = 1.0;

  fvar<var> f = prod(vv);
  EXPECT_FLOAT_EQ(36.0,prod(vd));
  EXPECT_FLOAT_EQ(36.0,f.val_.val());
  EXPECT_FLOAT_EQ(60.0,f.d_.val());
}
TEST(AgradFwdFvarFvarMatrix,prod_vector) {
  using stan::math::prod;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  vector_d vd;
  vector_ffv vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_.val());

  vd = vector_d(1);
  vv = vector_ffv(1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_.val());
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_.val());

  vd = vector_d(2);
  vd << 2.0, 3.0;
  vv = vector_ffv(2);
  vv << 2.0, 3.0;
   vv(0).d_ = 1.0;
   vv(1).d_ = 1.0;

  fvar<fvar<double> > f = prod(vv);
  EXPECT_FLOAT_EQ(6.0,prod(vd));
  EXPECT_FLOAT_EQ(6.0,f.val_.val());
  EXPECT_FLOAT_EQ(5.0,f.d_.val());
}

TEST(AgradFwdFvarFvarMatrix,prod_rowvector) {
  using stan::math::prod;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  row_vector_d vd;
  row_vector_ffv vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_.val());

  vd = row_vector_d(1);
  vv = row_vector_ffv(1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_.val());
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_.val());

  vd = row_vector_d(2);
  vd << 2.0, 3.0;
  vv = row_vector_ffv(2);
  vv << 2.0, 3.0;
   vv(0).d_ = 1.0;
   vv(1).d_ = 1.0;
  row_vector_ffv x(2);
  x[0] = vv[0];
  x[1] = vv[1];

  fvar<fvar<double> > f = prod(vv);
  EXPECT_FLOAT_EQ(6.0,prod(vd));
  EXPECT_FLOAT_EQ(6.0,f.val_.val());
  EXPECT_FLOAT_EQ(5.0,f.d_.val());
}
TEST(AgradFwdFvarFvarMatrix,prod_matrix) {
  using stan::math::prod;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_d vd;
  matrix_ffv vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val_.val());

  vd = matrix_d(1,1);
  vv = matrix_ffv(1,1);
  vd << 2.0;
  vv << 2.0;
   vv(0).d_ = 1.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val_.val());
  EXPECT_FLOAT_EQ(1.0,prod(vv).d_.val());

  vd = matrix_d(2,2);
  vd << 2.0, 3.0,2.0, 3.0;
  vv = matrix_ffv(2,2);
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

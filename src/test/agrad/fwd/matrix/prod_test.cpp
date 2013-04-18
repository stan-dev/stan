#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/math/matrix/prod.hpp>

TEST(AgradFwdMatrix,prod) {
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

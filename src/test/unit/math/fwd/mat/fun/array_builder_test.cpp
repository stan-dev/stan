#include <stan/math/prim/mat/fun/array_builder.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/core.hpp>

using stan::math::fvar;

TEST(AgradFwdMatrixArrayBuilder,fvar_double) {
  using std::vector;
  using stan::math::array_builder;

  EXPECT_EQ(0U, array_builder<fvar<double> >().array().size());

  vector<fvar<double> > x
    = array_builder<fvar<double> >()
    .add(fvar<double>(1,4))
    .add(fvar<double>(3,4))
    .add(fvar<double>(2,4))
    .array();
  EXPECT_EQ(3U,x.size());
  EXPECT_FLOAT_EQ(1.0, x[0].val_);
  EXPECT_FLOAT_EQ(3.0, x[1].val_);
  EXPECT_FLOAT_EQ(2.0, x[2].val_);
  EXPECT_FLOAT_EQ(4.0, x[0].d_);
  EXPECT_FLOAT_EQ(4.0, x[1].d_);
  EXPECT_FLOAT_EQ(4.0, x[2].d_);

  vector<vector<fvar<double> > > xx
    = array_builder<vector<fvar<double> > >()
    .add(array_builder<fvar<double> >()
         .add(fvar<double>(1,4))
         .add(fvar<double>(2,4)).array())
    .add(array_builder<fvar<double> >()
         .add(fvar<double>(3,4))
         .add(fvar<double>(4,4)).array())
    .add(array_builder<fvar<double> >()
         .add(fvar<double>(5,4))
         .add(fvar<double>(6,4)).array())
    .array();

  EXPECT_EQ(3U,xx.size());
  for (size_t i = 0; i < 3; ++i)
    EXPECT_EQ(2U,xx[i].size());
  EXPECT_EQ(1,xx[0][0].val_);
  EXPECT_EQ(2,xx[0][1].val_);
  EXPECT_EQ(3,xx[1][0].val_);
  EXPECT_EQ(4,xx[1][1].val_);
  EXPECT_EQ(5,xx[2][0].val_);
  EXPECT_EQ(6,xx[2][1].val_);
  EXPECT_EQ(4,xx[0][0].d_);
  EXPECT_EQ(4,xx[0][1].d_);
  EXPECT_EQ(4,xx[1][0].d_);
  EXPECT_EQ(4,xx[1][1].d_);
  EXPECT_EQ(4,xx[2][0].d_);
  EXPECT_EQ(4,xx[2][1].d_);
}


TEST(AgradFwdMatrixArrayBuilder,fvar_fvar_double) {
  using std::vector;
  using stan::math::array_builder;

  EXPECT_EQ(0U, array_builder<fvar<fvar<double> > >().array().size());

  vector<fvar<fvar<double> > > x
    = array_builder<fvar<fvar<double> > >()
    .add(fvar<fvar<double> >(1,4))
    .add(fvar<fvar<double> >(3,4))
    .add(fvar<fvar<double> >(2,4))
    .array();
  EXPECT_EQ(3U,x.size());
  EXPECT_FLOAT_EQ(1.0, x[0].val_.val_);
  EXPECT_FLOAT_EQ(3.0, x[1].val_.val_);
  EXPECT_FLOAT_EQ(2.0, x[2].val_.val_);
  EXPECT_FLOAT_EQ(4.0, x[0].d_.val_);
  EXPECT_FLOAT_EQ(4.0, x[1].d_.val_);
  EXPECT_FLOAT_EQ(4.0, x[2].d_.val_);

  vector<vector<fvar<fvar<double> > > > xx
    = array_builder<vector<fvar<fvar<double> > > >()
    .add(array_builder<fvar<fvar<double> > >()
         .add(fvar<fvar<double> >(1,4))
         .add(fvar<fvar<double> >(2,4)).array())
    .add(array_builder<fvar<fvar<double> > >()
         .add(fvar<fvar<double> >(3,4))
         .add(fvar<fvar<double> >(4,4)).array())
    .add(array_builder<fvar<fvar<double> > >()
         .add(fvar<fvar<double> >(5,4))
         .add(fvar<fvar<double> >(6,4)).array())
    .array();

  EXPECT_EQ(3U,xx.size());
  for (size_t i = 0; i < 3; ++i)
    EXPECT_EQ(2U,xx[i].size());
  EXPECT_EQ(1,xx[0][0].val_.val_);
  EXPECT_EQ(2,xx[0][1].val_.val_);
  EXPECT_EQ(3,xx[1][0].val_.val_);
  EXPECT_EQ(4,xx[1][1].val_.val_);
  EXPECT_EQ(5,xx[2][0].val_.val_);
  EXPECT_EQ(6,xx[2][1].val_.val_);
  EXPECT_EQ(4,xx[0][0].d_.val_);
  EXPECT_EQ(4,xx[0][1].d_.val_);
  EXPECT_EQ(4,xx[1][0].d_.val_);
  EXPECT_EQ(4,xx[1][1].d_.val_);
  EXPECT_EQ(4,xx[2][0].d_.val_);
  EXPECT_EQ(4,xx[2][1].d_.val_);
}

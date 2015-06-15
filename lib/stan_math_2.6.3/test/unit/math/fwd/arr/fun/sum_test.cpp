#include <gtest/gtest.h>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/arr/fun/sum.hpp>
#include <stan/math/prim/mat/fun/sum.hpp>

TEST(AgradFwdMatrixSum, vectorFvar) {
  using stan::math::fvar;
  using stan::math::sum;
  using std::vector;

  vector<fvar<double> > v(6);
  
  for (int i = 0; i < 6; ++i) {
    v[i] = i + 1;
    v[i].d_ = 1.0;
  }
  
  fvar<double> output;
  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val_);  
  EXPECT_FLOAT_EQ( 6.0, output.d_);

  vector<fvar<double> > ve;
  EXPECT_FLOAT_EQ(0.0, sum(ve).val_);
  EXPECT_FLOAT_EQ(0.0, sum(ve).d_);
}

TEST(AgradFwdMatrixSum, ffd_vector) {
  using stan::math::fvar;
  using stan::math::sum;
  using std::vector;

  vector<fvar<fvar<double> > > v(6);
  
  for (int i = 0; i < 6; ++i) {
    v[i] = i + 1;
    v[i].d_ = 1.0;
  }
  
  fvar<fvar<double> > output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val_.val());  
  EXPECT_FLOAT_EQ( 6.0, output.d_.val());

  vector<fvar<fvar<double> > > ve;
  EXPECT_FLOAT_EQ(0.0, sum(ve).val_.val());
  EXPECT_FLOAT_EQ(0.0, sum(ve).d_.val());
}

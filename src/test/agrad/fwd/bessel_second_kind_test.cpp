#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>

TEST(AgradFvar, bessel_second_kind) {
  using stan::agrad::fvar;
  using stan::agrad::bessel_second_kind;

  fvar<double> a(4.0,1.0);
  int b = 0;
  fvar<double> x = bessel_second_kind(b,a);
  EXPECT_FLOAT_EQ(-0.01694073932506499190363513444715321824049258989801, 
                  x.val_);
  EXPECT_FLOAT_EQ(-0.39792571055710000525397997245079185227118918162290,
                  x.d_);

  fvar<double> c(3.0,2.0);

  x = bessel_second_kind(1, c);
  EXPECT_FLOAT_EQ(0.3246744247917999784370128392879532396692751433723549,
                  x.val_);
  EXPECT_FLOAT_EQ(0.5372504034977141116428784919345973293208741759303264,
                  x.d_);

  EXPECT_THROW(bessel_second_kind(0, -a), std::domain_error);
}

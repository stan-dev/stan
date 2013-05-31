#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>

TEST(AgradFvar, modified_bessel_first_kind) {
  using stan::agrad::fvar;
  using stan::agrad::modified_bessel_first_kind;

  fvar<double> a(4.0,1.0);
  int b = 1;
  fvar<double> x = modified_bessel_first_kind(b,a);
  EXPECT_FLOAT_EQ(9.75946515370444990947519256731268090, 
                  x.val_);
  EXPECT_FLOAT_EQ(8.862055663710218018987472041388932,
                  x.d_);

  fvar<double> c(-3.0,2.0);

  x = modified_bessel_first_kind(1, c);
  EXPECT_FLOAT_EQ(-3.95337021740260939647863574058058,
                  x.val_);
  EXPECT_FLOAT_EQ(2.0 * 3.5630025133974876201183569658,
                  x.d_);

  x = modified_bessel_first_kind(-1, c);
  EXPECT_FLOAT_EQ(-3.95337021740260939647863574058058,
                  x.val_);
  EXPECT_FLOAT_EQ(2.0 * 3.5630025133974876201183569658,
                  x.d_);
}

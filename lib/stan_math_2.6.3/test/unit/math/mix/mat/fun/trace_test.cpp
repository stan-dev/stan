#include <stan/math/prim/mat/fun/trace.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>

TEST(AgradMixMatrixTrace,fv) {
  using stan::math::trace;
  using stan::math::matrix_fv;
  using stan::math::fvar;
  using stan::math::var;

  matrix_fv a(2,2);
  a << -1.0, 2.0, 
    5.0, 10.0;
   a(0,0).d_ = 1.0;
   a(0,1).d_ = 1.0;
   a(1,0).d_ = 1.0;
   a(1,1).d_ = 1.0;
  
  fvar<var> s = trace(a);
  EXPECT_FLOAT_EQ(9.0,s.val_.val());
  EXPECT_FLOAT_EQ(2.0,s.d_.val());
}
TEST(AgradMixMatrixTrace,ffv) {
  using stan::math::trace;
  using stan::math::matrix_ffv;
  using stan::math::fvar;
  using stan::math::var;

  matrix_ffv a(2,2);
  a << -1.0, 2.0, 
    5.0, 10.0;
   a(0,0).d_ = 1.0;
   a(0,1).d_ = 1.0;
   a(1,0).d_ = 1.0;
   a(1,1).d_ = 1.0;
   a(0,0).val_.d_ = 1.0;
   a(0,1).val_.d_ = 1.0;
   a(1,0).val_.d_ = 1.0;
   a(1,1).val_.d_ = 1.0;
  
  fvar<fvar<var> > s = trace(a);
  EXPECT_FLOAT_EQ(9.0,s.val_.val().val());
  EXPECT_FLOAT_EQ(2.0,s.d_.val().val());

  AVEC q = createAVEC(a(0,0).val().val(),a(0,1).val().val(),
                      a(1,0).val().val(),a(1,1).val().val());
  VEC h;
  s.d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}

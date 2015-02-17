#include <stan/math/prim/mat/fun/trace.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/fwd/scal/fun/operator_addition.hpp>
#include <stan/math/fwd/scal/fun/operator_division.hpp>
#include <stan/math/fwd/scal/fun/operator_equal.hpp>
#include <stan/math/fwd/scal/fun/operator_greater_than.hpp>
#include <stan/math/fwd/scal/fun/operator_greater_than_or_equal.hpp>
#include <stan/math/fwd/scal/fun/operator_less_than.hpp>
#include <stan/math/fwd/scal/fun/operator_less_than_or_equal.hpp>
#include <stan/math/fwd/scal/fun/operator_multiplication.hpp>
#include <stan/math/fwd/scal/fun/operator_not_equal.hpp>
#include <stan/math/fwd/scal/fun/operator_subtraction.hpp>
#include <stan/math/fwd/scal/fun/operator_unary_minus.hpp>
#include <stan/math/rev/scal/fun/operator_addition.hpp>
#include <stan/math/rev/scal/fun/operator_divide_equal.hpp>
#include <stan/math/rev/scal/fun/operator_division.hpp>
#include <stan/math/rev/scal/fun/operator_equal.hpp>
#include <stan/math/rev/scal/fun/operator_greater_than.hpp>
#include <stan/math/rev/scal/fun/operator_greater_than_or_equal.hpp>
#include <stan/math/rev/scal/fun/operator_less_than.hpp>
#include <stan/math/rev/scal/fun/operator_less_than_or_equal.hpp>
#include <stan/math/rev/scal/fun/operator_minus_equal.hpp>
#include <stan/math/rev/scal/fun/operator_multiplication.hpp>
#include <stan/math/rev/scal/fun/operator_multiply_equal.hpp>
#include <stan/math/rev/scal/fun/operator_not_equal.hpp>
#include <stan/math/rev/scal/fun/operator_plus_equal.hpp>
#include <stan/math/rev/scal/fun/operator_subtraction.hpp>
#include <stan/math/rev/scal/fun/operator_unary_decrement.hpp>
#include <stan/math/rev/scal/fun/operator_unary_increment.hpp>
#include <stan/math/rev/scal/fun/operator_unary_negative.hpp>
#include <stan/math/rev/scal/fun/operator_unary_not.hpp>
#include <stan/math/rev/scal/fun/operator_unary_plus.hpp>

TEST(AgradFwdMatrixTrace,fd) {
  using stan::math::trace;
  using stan::agrad::matrix_fd;
  using stan::agrad::fvar;

  matrix_fd a(2,2);
  a << -1.0, 2.0, 
    5.0, 10.0;
   a(0,0).d_ = 1.0;
   a(0,1).d_ = 1.0;
   a(1,0).d_ = 1.0;
   a(1,1).d_ = 1.0;
  
  fvar<double> s = trace(a);
  EXPECT_FLOAT_EQ(9.0,s.val_);
  EXPECT_FLOAT_EQ(2.0,s.d_);
}  
TEST(AgradFwdMatrixTrace,fv) {
  using stan::math::trace;
  using stan::agrad::matrix_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

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
TEST(AgradFwdMatrixTrace,ffd) {
  using stan::math::trace;
  using stan::agrad::matrix_ffd;
  using stan::agrad::fvar;

  matrix_ffd a(2,2);
  a << -1.0, 2.0, 
    5.0, 10.0;
  a(0,0).d_ = 1.0;
  a(0,1).d_ = 1.0;
  a(1,0).d_ = 1.0;
  a(1,1).d_ = 1.0;
  
  fvar<fvar<double> > s = trace(a);
  EXPECT_FLOAT_EQ(9.0,s.val_.val());
  EXPECT_FLOAT_EQ(2.0,s.d_.val());
}  

TEST(AgradFwdMatrixTrace,ffv) {
  using stan::math::trace;
  using stan::agrad::matrix_ffv;
  using stan::agrad::fvar;
  using stan::agrad::var;

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

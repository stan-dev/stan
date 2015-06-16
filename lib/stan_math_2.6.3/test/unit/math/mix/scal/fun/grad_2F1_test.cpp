#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/grad_2F1.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>

TEST(ProbInternalMath, grad2F1_fd1) {
  using stan::math::fvar;

  fvar<double> a = 2.0;
  a.d_ = 1.0;
  fvar<double> b = 1.0;
  fvar<double> c = 2.0;
  fvar<double> z = 0.4;
  fvar<double> gradA;
  fvar<double> gradC;
  stan::math::grad_2F1(gradA,gradC,a, b, c, z);

  EXPECT_NEAR(0.4617734315397201318453321291834046302225919173588625242,gradA.val_,1e-6);
  EXPECT_NEAR(0.163714876516383746459968120418298168600425943651588679302872,gradA.d_,1e-5);
  EXPECT_NEAR(-0.461773435230326182245722531773361592054302268779753796048,gradC.val_,1e-6);
}
TEST(ProbInternalMath, grad2F1_fd2) {
  using stan::math::fvar;

  fvar<double> a = 2.0;
  fvar<double> b = 1.0;
  fvar<double> c = 2.0;
  c.d_ = 1.0;
  fvar<double> z = 0.4;
  fvar<double> gradA;
  fvar<double> gradC;
  stan::math::grad_2F1(gradA,gradC,a, b, c, z);

  EXPECT_NEAR(0.4617734315397201318453321291834046302225919173588625242,gradA.val_,1e-6);
  EXPECT_NEAR(-0.461773435230326182245722531773361592054302268779753796048,gradC.val_,1e-6);
  EXPECT_NEAR(0.5744063304437309685867184312646717864627845936245830896889,gradC.d_,1e-5);
}
TEST(ProbInternalMath, grad2F1_ffd1) {
  using stan::math::fvar;

  fvar<fvar<double> > a = 2.0;
  a.d_ = 1.0;
  fvar<fvar<double> > b = 1.0;
  fvar<fvar<double> > c = 2.0;
  fvar<fvar<double> > z = 0.4;
  fvar<fvar<double> > gradA;
  fvar<fvar<double> > gradC;

  stan::math::grad_2F1(gradA,gradC,a, b, c, z);
  EXPECT_NEAR(0.4617734315397201318453321291834046302225919173588625242928,gradA.val_.val_, 1e-6);
  EXPECT_NEAR(0.163714876516383746459968120418298168600425943651588679302872,gradA.d_.val_, 1e-5);
  EXPECT_NEAR(-0.46177343523032618224572253177336159205430226877975379604859,gradC.val_.val_, 1e-6);
}
TEST(ProbInternalMath, grad2F1_ffd2) {
  using stan::math::fvar;

  fvar<fvar<double> > a = 2.0;
  fvar<fvar<double> > b = 1.0;
  fvar<fvar<double> > c = 2.0;
  c.d_ = 1.0;
  fvar<fvar<double> > z = 0.4;
  fvar<fvar<double> > gradA;
  fvar<fvar<double> > gradC;

  stan::math::grad_2F1(gradA,gradC,a, b, c, z);
  EXPECT_NEAR(0.4617734315397201318453321291834046302225919173588625242928,gradA.val_.val_, 1e-6);
  EXPECT_NEAR(-0.46177343523032618224572253177336159205430226877975379604859,gradC.val_.val_, 1e-6);
  EXPECT_NEAR(0.5744063304437309685867184312646717864627845936245830896889,gradC.d_.val_, 1e-5);
}

TEST(ProbInternalMath, grad2F1_fv1) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a = 2.0;
  a.d_ = 1.0;
  fvar<var> b = 1.0;
  fvar<var> c = 2.0;
  fvar<var> z = 0.4;
  fvar<var> gradA; fvar<var> gradC;

  stan::math::grad_2F1(gradA,gradC,a, b, c, z);
  EXPECT_NEAR(0.4617734315397201318453321291834046302225919173588625242928, gradA.val_.val(),1e-6);
  EXPECT_NEAR(0.163714876516383746459968120418298168600425943651588679302872,gradA.d_.val(), 1e-5);
  EXPECT_NEAR(-0.4617734352303261822457225317733615920543022687797537960, gradC.val_.val(),1e-6);
}
TEST(ProbInternalMath, grad2F1_fv2) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a = 2.0;
  fvar<var> b = 1.0;
  fvar<var> c = 2.0;
  c.d_ = 1.0;
  fvar<var> z = 0.4;
  fvar<var> gradA; fvar<var> gradC;

  stan::math::grad_2F1(gradA,gradC,a, b, c, z);
  EXPECT_NEAR(0.4617734315397201318453321291834046302225919173588625242928, gradA.val_.val(),1e-6);
  EXPECT_NEAR(-0.4617734352303261822457225317733615920543022687797537960, gradC.val_.val(),1e-6);
  EXPECT_NEAR(0.5744063304437309685867184312646717864627845936245830896889,gradC.d_.val(), 1e-5);
}

TEST(ProbInternalMath, grad2F1_fv_1stderiv1) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a = 2.0;
  a.d_ = 1.0;
  fvar<var> b = 1.0;
  fvar<var> c = 2.0;
  fvar<var> z = 0.4;
  fvar<var> gradA; fvar<var> gradC;

  stan::math::grad_2F1(gradA,gradC,a, b, c, z);

  AVEC y1 = createAVEC(a.val_);
  VEC grad1;
  gradA.val_.grad(y1,grad1);
  EXPECT_NEAR(0.163714876516383746459968120418298168600425943651588679302872,grad1[0],1e-5);
}
TEST(ProbInternalMath, grad2F1_fv_1stderiv2) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a = 2.0;
  fvar<var> b = 1.0;
  fvar<var> c = 2.0;
  c.d_ = 1.0;
  fvar<var> z = 0.4;
  fvar<var> gradA; fvar<var> gradC;

  stan::math::grad_2F1(gradA,gradC,a, b, c, z);

  AVEC y1 = createAVEC(c.val_);
  VEC grad1;
  gradC.val_.grad(y1,grad1);
  EXPECT_NEAR(0.5744063304437309685867184312646717864627845936245830896889,grad1[0],1e-5);
}

TEST(ProbInternalMath, grad2F1_fv_2ndderiv1) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a = 2.0;
  a.d_ = 1.0;
  fvar<var> b = 1.0;
  fvar<var> c = 2.0;
  fvar<var> z = 0.4;
  fvar<var> gradA; fvar<var> gradC;
  
  stan::math::grad_2F1(gradA,gradC,a, b, c, z);

  AVEC y1 = createAVEC(a.val_);
  VEC grad1;
  gradA.d_.grad(y1,grad1);
  EXPECT_NEAR(0.06425652761307923044917291721823961650191124494852382302571,grad1[0],1e-5);
}

TEST(ProbInternalMath, grad2F1_fv_2ndderiv2) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a = 2.0;
  fvar<var> b = 1.0;
  fvar<var> c = 2.0;
  c.d_ = 1.0;
  fvar<var> z = 0.4;
  fvar<var> gradA; fvar<var> gradC;
  
  stan::math::grad_2F1(gradA,gradC,a, b, c, z);

  AVEC y1 = createAVEC(c.val_);
  VEC grad1;
  gradC.d_.grad(y1,grad1);
  EXPECT_NEAR(-1.000245537254470801442530432195413371212643855413220347277769,grad1[0],1e-5);
}

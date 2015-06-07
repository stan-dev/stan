#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/F32.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>

TEST(ProbInternalMath, F32_fd) {
  using stan::math::fvar;

  fvar<double> a = 1.0;
  fvar<double> b = 31.0;
  fvar<double> c = -27.0;
  fvar<double> d = 19.0;
  fvar<double> e = -41.0;
  fvar<double> z = 1.0;

  EXPECT_FLOAT_EQ(11.28915378492300834453857665243661995978358572684678329916652,stan::math::F32(a,b,c,d,e,z).val_);
}
TEST(ProbInternalMath, F32_ffd) {
  using stan::math::fvar;

  fvar<fvar<double> > a = 1.0;
  fvar<fvar<double> > b = 31.0;
  fvar<fvar<double> > c = -27.0;
  fvar<fvar<double> > d = 19.0;
  fvar<fvar<double> > e = -41.0;
  fvar<fvar<double> > z = 1.0;

  EXPECT_FLOAT_EQ(11.28915378492300834453857665243661995978358572684678329916652,stan::math::F32(a,b,c,d,e,z).val_.val_);
}
TEST(ProbInternalMath, F32_fv_1stderiv1) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a = 1.0;
  a.d_ = 1.0;
  fvar<var> b = 31.0;
  fvar<var> c = -27.0;
  fvar<var> d = 19.0;
  fvar<var> e = -41.0;
  fvar<var> z = 1.0;

  fvar<var> z1 = stan::math::F32(a,b,c,d,e,z);

  EXPECT_FLOAT_EQ(11.28915378492300834453857665243661995978358572684678329916652,z1.val_.val());

  AVEC y1 = createAVEC(a.val_);
  VEC grad1;
  z1.val_.grad(y1,grad1);
  EXPECT_NEAR(22.95829816018250585416491584581112223816561212219172212450836,grad1[0],1e-5);
}
TEST(ProbInternalMath, F32_fv_1stderiv2) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a = 1.0;
  fvar<var> b = 31.0;
  b.d_ = 1.0;
  fvar<var> c = -27.0;
  fvar<var> d = 19.0;
  fvar<var> e = -41.0;
  fvar<var> z = 1.0;

  fvar<var> z1 = stan::math::F32(a,b,c,d,e,z);

  EXPECT_FLOAT_EQ(11.28915378492300834453857665243661995978358572684678329916652,z1.val_.val());

  AVEC y1 = createAVEC(b.val_);
  VEC grad1;
  z1.val_.grad(y1,grad1);
  EXPECT_NEAR(1.740056451478897241488082512854205170874142224663970334770766,grad1[0],1e-5);
}
TEST(ProbInternalMath, F32_fv_1stderiv3) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a = 1.0;
  fvar<var> b = 31.0;
  fvar<var> c = -27.0;
  c.d_ = 1.0;
  fvar<var> d = 19.0;
  fvar<var> e = -41.0;
  fvar<var> z = 1.0;

  fvar<var> z1 = stan::math::F32(a,b,c,d,e,z);

  EXPECT_FLOAT_EQ(11.28915378492300834453857665243661995978358572684678329916652,z1.val_.val());

  AVEC y1 = createAVEC(c.val_);
  VEC grad1;
  z1.val_.grad(y1,grad1);
  EXPECT_NEAR(-2.6052400424887519,grad1[0],1e-5);
}
TEST(ProbInternalMath, F32_fv_1stderiv4) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a = 1.0;
  fvar<var> b = 31.0;
  fvar<var> c = -27.0;
  fvar<var> d = 19.0;
  d.d_ = 1.0;
  fvar<var> e = -41.0;
  fvar<var> z = 1.0;

  fvar<var> z1 = stan::math::F32(a,b,c,d,e,z);

  EXPECT_FLOAT_EQ(11.28915378492300834453857665243661995978358572684678329916652,z1.val_.val());

  AVEC y1 = createAVEC(d.val_);
  VEC grad1;
  z1.val_.grad(y1,grad1);
  EXPECT_NEAR(-2.69297893625022464634137707353872105148995696636392529052847,grad1[0],1e-5);
}
TEST(ProbInternalMath, F32_fv_1stderiv5) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a = 1.0;
  fvar<var> b = 31.0;
  fvar<var> c = -27.0;
  fvar<var> d = 19.0;
  fvar<var> e = -41.0;
  e.d_ = 1.0;
  fvar<var> z = 1.0;

  fvar<var> z1 = stan::math::F32(a,b,c,d,e,z);

  EXPECT_FLOAT_EQ(11.28915378492300834453857665243661995978358572684678329916652,z1.val_.val());

  AVEC y1 = createAVEC(e.val_);
  VEC grad1;
  z1.val_.grad(y1,grad1);
  EXPECT_NEAR(1.606519030743225019685406202522547937181609777973133379643512,grad1[0],1e-5);
}

TEST(ProbInternalMath, F32_fv_1stderiv6) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a = 1.0;
  fvar<var> b = 31.0;
  fvar<var> c = -27.0;
  fvar<var> d = 19.0;
  fvar<var> e = -41.0;
  fvar<var> z = 1.0;
  z.d_ = 1.0;

  fvar<var> z1 = stan::math::F32(a,b,c,d,e,z);

  EXPECT_FLOAT_EQ(11.28915378492300834453857665243661995978358572684678329916652,z1.val_.val());

  AVEC y1 = createAVEC(z.val_);
  VEC grad1;
  z1.val_.grad(y1,grad1);
  EXPECT_NEAR(59.65791128638963495870649759618269069328482053152963195380560,grad1[0],1e-5);
}

TEST(ProbInternalMath, F32_fv_2ndderiv1) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a = 1.0;
  a.d_ = 1.0;
  fvar<var> b = 31.0;
  fvar<var> c = -27.0;
  fvar<var> d = 19.0;
  fvar<var> e = -41.0;
  fvar<var> z = 1.0;

  fvar<var> z1 = stan::math::F32(a,b,c,d,e,z);

  EXPECT_FLOAT_EQ(11.28915378492300834453857665243661995978358572684678329916652,z1.val_.val());

  AVEC y1 = createAVEC(a.val_);
  VEC grad1;
  z1.d_.grad(y1,grad1);
  EXPECT_NEAR(41.01553475870347475023037358640582917147051389292162474016745,grad1[0],1e-5);
}

TEST(ProbInternalMath, F32_fv_2ndderiv2) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a = 1.0;
  fvar<var> b = 31.0;
  b.d_ = 1.0;
  fvar<var> c = -27.0;
  fvar<var> d = 19.0;
  fvar<var> e = -41.0;
  fvar<var> z = 1.0;

  fvar<var> z1 = stan::math::F32(a,b,c,d,e,z);

  EXPECT_FLOAT_EQ(11.28915378492300834453857665243661995978358572684678329916652,z1.val_.val());

  AVEC y1 = createAVEC(b.val_);
  VEC grad1;
  z1.d_.grad(y1,grad1);
  EXPECT_NEAR(0.342454543339724329115552426438001592723143365030924900588111,grad1[0],1e-5);
}
TEST(ProbInternalMath, F32_fv_2ndderiv3) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a = 1.0;
  fvar<var> b = 31.0;
  fvar<var> c = -27.0;
  c.d_ = 1.0;
  fvar<var> d = 19.0;
  fvar<var> e = -41.0;
  fvar<var> z = 1.0;

  fvar<var> z1 = stan::math::F32(a,b,c,d,e,z);

  EXPECT_FLOAT_EQ(11.28915378492300834453857665243661995978358572684678329916652,z1.val_.val());

  AVEC y1 = createAVEC(c.val_);
  VEC grad1;
  z1.d_.grad(y1,grad1);
  EXPECT_NEAR(0.90986472078762437,grad1[0],1e-5);
}
TEST(ProbInternalMath, F32_fv_2ndderiv4) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a = 1.0;
  fvar<var> b = 31.0;
  fvar<var> c = -27.0;
  fvar<var> d = 19.0;
  d.d_ = 1.0;
  fvar<var> e = -41.0;
  fvar<var> z = 1.0;

  fvar<var> z1 = stan::math::F32(a,b,c,d,e,z);

  AVEC y1 = createAVEC(d.val_);
  VEC grad1;
  z1.d_.grad(y1,grad1);
  EXPECT_NEAR(1.047024959065504556655904003595645684444382378830047020988218,grad1[0],1e-5);
}
TEST(ProbInternalMath, F32_fv_2ndderiv5) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a = 1.0;
  fvar<var> b = 31.0;
  fvar<var> c = -27.0;
  fvar<var> d = 19.0;
  fvar<var> e = -41.0;
  e.d_ = 1.0;
  fvar<var> z = 1.0;

  fvar<var> z1 = stan::math::F32(a,b,c,d,e,z);

  AVEC y1 = createAVEC(e.val_);
  VEC grad1;
  z1.d_.grad(y1,grad1);
  EXPECT_NEAR(0.415359887777218792995404669803015764396172842233556866773418,grad1[0],1e-5);
}

TEST(ProbInternalMath, F32_fv_2ndderiv6) {
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a = 1.0;
  fvar<var> b = 31.0;
  fvar<var> c = -27.0;
  fvar<var> d = 19.0;
  fvar<var> e = -41.0;
  fvar<var> z = 1.0;
  z.d_ = 1.0;

  fvar<var> z1 = stan::math::F32(a,b,c,d,e,z);

  EXPECT_FLOAT_EQ(11.28915378492300834453857665243661995978358572684678329916652,z1.val_.val());

  AVEC y1 = createAVEC(z.val_);
  VEC grad1;
  z1.d_.grad(y1,grad1);
  EXPECT_NEAR(424.5724606148232594702100102534498155985480235827583548085963,grad1[0],1e-5);
}

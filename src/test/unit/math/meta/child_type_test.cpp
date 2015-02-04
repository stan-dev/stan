
#include <stan/math/meta/child_type.hpp>
#include <test/unit/math/functions/promote_type_test_util.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/fwd/fvar.hpp>

TEST(MathMeta, value_type) {
  using stan::math::child_type;
  using stan::agrad::fvar;
  using stan::agrad::var;

  expect_same_type<double,child_type<double>::type>();
  expect_same_type<double,child_type<fvar<double> >::type>();
  expect_same_type<double,child_type<var>::type>();
  expect_same_type<fvar<double>,child_type<fvar<fvar<double> > >::type>();
  expect_same_type<fvar<var>,child_type<fvar<fvar<var> > >::type>();
  expect_same_type<var,child_type<fvar<var> >::type>();
}

#include <gtest/gtest.h>
#include <stan/math/prim/scal/meta/container_view.hpp>
#include <boost/type_traits/conditional.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>

TEST(MathMeta, container_view) {
  using stan::math::container_view;

  double y[1];
  container_view<double, double> view_test(4.0, y);
  view_test[0] = 1.0;
  EXPECT_FLOAT_EQ(1.0, view_test[0]);
  EXPECT_FLOAT_EQ(1.0, y[0]);
}

TEST(MathMeta, container_view_throw) {
  using stan::math::container_view;
  using boost::conditional;
  using stan::math::dummy;
  using stan::is_constant_struct;

  double arr[1];
  container_view<conditional<is_constant_struct<double>::value,dummy,double>::type, double> view_test(4.0, arr);
  EXPECT_THROW(view_test[0],std::out_of_range);
}

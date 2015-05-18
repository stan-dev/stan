#include <gtest/gtest.h>
#include <stan/math/prim/mat/meta/container_view.hpp>
#include <stan/math/prim/arr/meta/container_view.hpp>
#include <boost/type_traits/conditional.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/rev/core.hpp>

TEST(MathMeta, container_view_vector) {
  using stan::math::container_view;
  using stan::math::var;

  double y[10];
  std::vector<var> x;
  container_view<std::vector<var>, double> view_test(x, y);
  for (int i = 0; i < 10; ++i) {
    view_test[i] = i;
    EXPECT_FLOAT_EQ(i, view_test[i]);
    EXPECT_FLOAT_EQ(i, y[i]);
    view_test[i] = 0;
    EXPECT_FLOAT_EQ(0, y[i]);
  }
}


TEST(MathMeta, container_view_no_throw) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::container_view;
  using boost::conditional;
  using stan::math::dummy;
  using stan::math::var;
  using stan::is_constant_struct;

  double arr[1];
  std::vector<var> x;
  container_view<conditional<is_constant_struct<std::vector<var> >::value,dummy,std::vector<var> >::type, double> view_test(x, arr);
  EXPECT_NO_THROW(view_test[0]);
}

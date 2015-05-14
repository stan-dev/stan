#include <gtest/gtest.h>
#include <stan/math/prim/scal/meta/vector_view_map.hpp>
#include <boost/type_traits/conditional.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>

TEST(MathMeta, vector_view_map) {
  using stan::math::vector_view_map;
  using stan::math::var;

  stan::math::start_nested();
  double arr[1];
  var* test = static_cast<var*>(stan::math::chainable::operator new(sizeof(var) * 1));
  vector_view_map<var, double> view_test(var(4.0), arr);
  view_test[0] = 1.0;
  EXPECT_FLOAT_EQ(1.0, view_test[0]);
  vector_view_map<var, var> view_test_v(var(4.0), test);
  view_test_v[0] = 2.0;
  EXPECT_FLOAT_EQ(2.0,view_test_v[0].val());
  stan::math::recover_memory_nested();
}

TEST(MathMeta, vector_view_map_throw) {
  using stan::math::vector_view_map;
  using boost::conditional;
  using stan::math::dummy;
  using stan::is_constant_struct;

  double arr[1];
  vector_view_map<conditional<is_constant_struct<double>::value,dummy,double>::type, double> view_test(4.0, arr);
  EXPECT_THROW(view_test[0],std::out_of_range);
}

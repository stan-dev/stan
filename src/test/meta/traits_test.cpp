#include <gtest/gtest.h>
#include <boost/type_traits.hpp>
#include <stan/agrad/agrad.hpp>
#include <stan/meta/traits.hpp>

TEST(Traits, isConstant) {
  using stan::is_constant;
  EXPECT_TRUE(is_constant<double>::value);
  EXPECT_FALSE(is_constant<stan::agrad::var>::value);
}

TEST(Traits, scalar_type) {
  using boost::is_same;
  using stan::scalar_type;
  using std::vector;

  stan::scalar_type<double>::type x = 5.0;
  EXPECT_EQ(5.0,x);

  stan::scalar_type<std::vector<int> >::type n = 1;
  EXPECT_EQ(1,n);

  // hack to get value of template into Google test macro 
  bool b1 = is_same<double,double>::value;
  EXPECT_TRUE(b1);

  bool b2 = is_same<double,int>::value;
  EXPECT_FALSE(b2);

  bool b3 = is_same<double, scalar_type<vector<double> >::type>::value;
  EXPECT_TRUE(b3);

  bool b4 = is_same<double, scalar_type<double>::type>::value;
  EXPECT_TRUE(b4);

  bool b5 = is_same<int, scalar_type<double>::type>::value;
  EXPECT_FALSE(b5);
}

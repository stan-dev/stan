#ifndef TEST__MATH__UNIT__FUNCTIONS__PROMOTE_TYPE_TEST_UTIL_HPP
#define TEST__MATH__UNIT__FUNCTIONS__PROMOTE_TYPE_TEST_UTIL_HPP

#include <gtest/gtest.h>

#include <boost/typeof/typeof.hpp>
#include <boost/type_traits/is_same.hpp>

#include <stan/math/functions/promote_scalar_type.hpp>
#include <stan/math/functions/promote_scalar.hpp>

template <typename T, typename S>
void expect_type(S s) {
  typedef BOOST_TYPEOF_TPL(stan::math::promote_scalar<T>(s)) result_t;
  bool same = boost::is_same<S, result_t>::value;
  EXPECT_TRUE(same);
}

// pass:  expect_same_type<double,double>()
// fail:  expect_same_type<int,double>()
template <typename T, typename S>
void expect_same_type() {
  EXPECT_TRUE(( boost::is_same<S, T>::value ));
}

// E is expected value of promote_scalar_type<T,S>::type
template <typename E, typename T, typename S>
void expect_promote_type() {
  using stan::math::promote_scalar_type;
  return expect_same_type<E,typename promote_scalar_type<T,S>::type>();
}



#endif


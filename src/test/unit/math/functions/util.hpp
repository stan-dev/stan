#ifndef TEST__MATH__FUNCTIONS__UTIL_HPP
#define TEST__MATH__FUNCTIONS__UTIL_HPP

#include <gtest/gtest.h>

#include <boost/typeof/typeof.hpp>
#include <boost/type_traits/is_same.hpp>

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

#endif

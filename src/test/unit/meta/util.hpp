#ifndef TEST__UNIT__META__UTIL_HPP
#define TEST__UNIT__META__UTIL_HPP

#include <gtest/gtest.h>
#include <boost/type_traits/is_same.hpp> 


template <typename T, typename I>
void expect_eq_indexed() {
  EXPECT_TRUE(( boost::is_same<T,typename I::type>::value ));
}


#endif

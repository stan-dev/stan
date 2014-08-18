#ifndef TEST__UNIT_AGRAD_REV__NAN_UTIL_HPP
#define TEST__UNIT_AGRAD_REV__NAN_UTIL_HPP

#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

template <typename F>
void test_nan_vd(const F& f,
                 const double& arg1,
                 const double& arg2,
                 const bool& throws) {
  stan::agrad::var res;
  stan::agrad::var arg1_v = arg1;

  std::ostringstream fail_msg;
  fail_msg << "Failed for var,double version with first argument " << arg1_v
      << " and second argument " << arg2;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2), std::domain_error) << fail_msg;
  else {
    res = f(arg1_v, arg2);

    AVEC x = createAVEC(arg1_v);
    VEC g;
    res.grad(x,g);
  
    EXPECT_TRUE(boost::math::isnan(res.val())) << fail_msg;
    ASSERT_EQ(1U,g.size()) << fail_msg;
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg;
  }
}
template <typename F>
void test_nan_dv(const F& f,
                 const double& arg1,
                 const double& arg2,
                 const bool& throws) {
  stan::agrad::var res;
  stan::agrad::var arg2_v = arg2;

  std::ostringstream fail_msg;
  fail_msg << "Failed for double,var version with first argument " << arg1
      << " and second argument " << arg2_v;

  if (throws)
    EXPECT_THROW(f(arg1, arg2_v), std::domain_error) << fail_msg;
  else {
    res = f(arg1, arg2_v);

    AVEC x = createAVEC(arg2_v);
    VEC g;
    res.grad(x,g);
  
    EXPECT_TRUE(boost::math::isnan(res.val())) << fail_msg;
    ASSERT_EQ(1U,g.size()) << fail_msg;
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg;
  }
}
template <typename F>
void test_nan_vv(const F& f,
                 const double& arg1,
                 const double& arg2,
                 const bool& throws) {

  stan::agrad::var res;
  stan::agrad::var arg1_v = arg1;
  stan::agrad::var arg2_v = arg2;

  std::ostringstream fail_msg;
  fail_msg << "Failed for var,var version with first argument " << arg1_v 
      << " and second argument " << arg2_v;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2_v), std::domain_error) << fail_msg;
  else {
    res = f(arg1_v, arg2_v);

    AVEC x = createAVEC(arg1_v,arg2_v);
    VEC g;
    res.grad(x,g);
  
    EXPECT_TRUE(boost::math::isnan(res.val())) << fail_msg;
    ASSERT_EQ(2U,g.size()) << fail_msg;
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg;
    EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg;
  }
}

template <typename F>
void test_nan(const F& f,
              const double& arg1,
              const double& arg2,
              const bool& throws) {

  double nan = std::numeric_limits<double>::quiet_NaN();
  test_nan_vd(f, nan, arg2, throws);
  test_nan_vd(f, arg1, nan, throws);
  test_nan_vd(f, nan, nan, throws);
  test_nan_dv(f, nan, arg2, throws);
  test_nan_dv(f, arg1, nan, throws);
  test_nan_dv(f, nan, nan, throws);
  test_nan_vv(f, nan, arg2, throws);
  test_nan_vv(f, arg1, nan, throws);
  test_nan_vv(f, nan, nan, throws);
}

template <typename F>
void test_nan_v(const F& f,
                const double& arg1,
                const bool& throws) {
  stan::agrad::var res;
  stan::agrad::var arg1_v = arg1;
  if (throws)
    EXPECT_THROW(f(arg1_v), std::domain_error);
  else {
    res = f(arg1_v);

    AVEC x = createAVEC(arg1_v);
    VEC g;
    res.grad(x,g);
  
    EXPECT_TRUE(boost::math::isnan(res.val()));
    ASSERT_EQ(1U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
  }
}

template <typename F>
void test_nan(const F& f,
              const bool& throws) {

  double nan = std::numeric_limits<double>::quiet_NaN();
  test_nan_v(f, nan, throws);
}
#endif

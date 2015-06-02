#ifndef TEST__UNIT_AGRAD_REV__NAN_UTIL_HPP
#define TEST__UNIT_AGRAD_REV__NAN_UTIL_HPP

#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

template <typename F>
void test_nan_vd(const F& f,
                 const double& arg1,
                 const double& arg2,
                 const bool& throws,
                 const bool& is_grad_nan) {
  stan::math::var res;
  stan::math::var arg1_v = arg1;

  std::ostringstream fail_msg;
  fail_msg << "Failed for var,double version with first argument " << arg1_v
           << " and second argument " << arg2;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2), std::domain_error) << fail_msg.str();
  else {
    res = f(arg1_v, arg2);
    EXPECT_TRUE(boost::math::isnan(res.val())) << fail_msg.str();

    if (is_grad_nan) {
      AVEC x = createAVEC(arg1_v);
      VEC g;
      res.grad(x,g);
  
      ASSERT_EQ(1U,g.size()) << fail_msg.str();
      EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    }
  }
}
template <typename F>
void test_nan_dv(const F& f,
                 const double& arg1,
                 const double& arg2,
                 const bool& throws,
                 const bool& is_grad_nan) {
  stan::math::var res;
  stan::math::var arg2_v = arg2;

  std::ostringstream fail_msg;
  fail_msg << "Failed for double,var version with first argument " << arg1
           << " and second argument " << arg2_v;

  if (throws)
    EXPECT_THROW(f(arg1, arg2_v), std::domain_error) << fail_msg.str();
  else {
    res = f(arg1, arg2_v);
    EXPECT_TRUE(boost::math::isnan(res.val())) << fail_msg.str();

    if (is_grad_nan) {
      AVEC x = createAVEC(arg2_v);
      VEC g;
      res.grad(x,g);
  
      ASSERT_EQ(1U,g.size()) << fail_msg.str();
      EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    }
  }
}
template <typename F>
void test_nan_vv(const F& f,
                 const double& arg1,
                 const double& arg2,
                 const bool& throws,
                 const bool& is_grad_nan) {

  stan::math::var res;
  stan::math::var arg1_v = arg1;
  stan::math::var arg2_v = arg2;

  std::ostringstream fail_msg;
  fail_msg << "Failed for var,var version with first argument " << arg1_v 
           << " and second argument " << arg2_v;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2_v), std::domain_error) << fail_msg.str();
  else {
    res = f(arg1_v, arg2_v);
    EXPECT_TRUE(boost::math::isnan(res.val())) << fail_msg.str();

    if (is_grad_nan) {
      AVEC x = createAVEC(arg1_v,arg2_v);
      VEC g;
      res.grad(x,g);
  
      ASSERT_EQ(2U,g.size()) << fail_msg.str();
      EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
      EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
    }
  }
}

template <typename F>
void test_nan(const F& f,
              const double& arg1,
              const double& arg2,
              const bool& throws,
              const bool& is_grad_nan) {

  double nan = std::numeric_limits<double>::quiet_NaN();
  test_nan_vd(f, nan, arg2, throws, is_grad_nan);
  test_nan_vd(f, arg1, nan, throws, is_grad_nan);
  test_nan_vd(f, nan, nan, throws, is_grad_nan);
  test_nan_dv(f, nan, arg2, throws, is_grad_nan);
  test_nan_dv(f, arg1, nan, throws, is_grad_nan);
  test_nan_dv(f, nan, nan, throws, is_grad_nan);
  test_nan_vv(f, nan, arg2, throws, is_grad_nan);
  test_nan_vv(f, arg1, nan, throws, is_grad_nan);
  test_nan_vv(f, nan, nan, throws, is_grad_nan);
}

template <typename F>
void test_nan_v(const F& f,
                const double& arg1,
                const bool& throws,
                const bool& is_grad_nan) {
  stan::math::var res;
  stan::math::var arg1_v = arg1;
  if (throws)
    EXPECT_THROW(f(arg1_v), std::domain_error);
  else {
    res = f(arg1_v);
    EXPECT_TRUE(boost::math::isnan(res.val()));

    if (is_grad_nan) {
      AVEC x = createAVEC(arg1_v);
      VEC g;
      res.grad(x,g);
  
      ASSERT_EQ(1U,g.size());
      EXPECT_TRUE(boost::math::isnan(g[0]));
    }
  }
}

template <typename F>
void test_nan(const F& f,
              const bool& throws,
              const bool& is_grad_nan) {

  double nan = std::numeric_limits<double>::quiet_NaN();
  test_nan_v(f, nan, throws, is_grad_nan);
}


template <typename F>
void test_nan_vvv(const F& f,
                  const double& arg1,
                  const double& arg2,
                  const double& arg3,
                  const bool& throws,
                  const bool& is_grad_nan) {
  stan::math::var res;
  stan::math::var arg1_v = arg1;
  stan::math::var arg2_v = arg2;
  stan::math::var arg3_v = arg3;

  std::ostringstream fail_msg;
  fail_msg << "Failed for var,var,var version with first argument " 
           << arg1_v << " second argument " << arg2_v
           << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2_v, arg3_v), std::domain_error) << fail_msg.str();
  else {
    res = f(arg1_v, arg2_v, arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.val())) << fail_msg.str();

    if (is_grad_nan) {
      AVEC x = createAVEC(arg1_v,arg2_v, arg3_v);
      VEC g;
      res.grad(x,g);
  
      ASSERT_EQ(3U,g.size()) << fail_msg.str();
      EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
      EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
      EXPECT_TRUE(boost::math::isnan(g[2])) << fail_msg.str();
    }
  }
}

template <typename F>
void test_nan_dvv(const F& f,
                  const double& arg1,
                  const double& arg2,
                  const double& arg3,
                  const bool& throws,
                  const bool& is_grad_nan) {
  stan::math::var res;
  stan::math::var arg2_v = arg2;
  stan::math::var arg3_v = arg3;

  std::ostringstream fail_msg;
  fail_msg << "Failed for double,var,var version with first argument " 
           << arg1 << " second argument " << arg2_v
           << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1, arg2_v, arg3_v), std::domain_error) << fail_msg.str();
  else {
    res = f(arg1, arg2_v, arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.val())) << fail_msg.str();

    if (is_grad_nan) {
      AVEC x = createAVEC(arg2_v, arg3_v);
      VEC g;
      res.grad(x,g);
  
      ASSERT_EQ(2U,g.size()) << fail_msg.str();
      EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
      EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
    }
  }
}

template <typename F>
void test_nan_vdv(const F& f,
                  const double& arg1,
                  const double& arg2,
                  const double& arg3,
                  const bool& throws,
                  const bool& is_grad_nan) {
  stan::math::var res;
  stan::math::var arg1_v = arg1;
  stan::math::var arg3_v = arg3;

  std::ostringstream fail_msg;
  fail_msg << "Failed for var,double,var version with first argument " 
           << arg1_v << " second argument " << arg2
           << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2, arg3_v), std::domain_error) << fail_msg.str();
  else {
    res = f(arg1_v, arg2, arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.val())) << fail_msg.str();

    if (is_grad_nan) {
      AVEC x = createAVEC(arg1_v, arg3_v);
      VEC g;
      res.grad(x,g);
  
      ASSERT_EQ(2U,g.size()) << fail_msg.str();
      EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
      EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
    }
  }
}

template <typename F>
void test_nan_vvd(const F& f,
                  const double& arg1,
                  const double& arg2,
                  const double& arg3,
                  const bool& throws,
                  const bool& is_grad_nan) {
  stan::math::var res;
  stan::math::var arg1_v = arg1;
  stan::math::var arg2_v = arg2;

  std::ostringstream fail_msg;
  fail_msg << "Failed for var,var,double version with first argument " 
           << arg1_v << " second argument " << arg2_v
           << " and third argument " << arg3;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2_v, arg3), std::domain_error) << fail_msg.str();
  else {
    res = f(arg1_v, arg2_v, arg3);
    EXPECT_TRUE(boost::math::isnan(res.val())) << fail_msg.str();

    if (is_grad_nan) {
      AVEC x = createAVEC(arg1_v,arg2_v);
      VEC g;
      res.grad(x,g);
  
      ASSERT_EQ(2U,g.size()) << fail_msg.str();
      EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
      EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
    }
  }
}

template <typename F>
void test_nan_ddv(const F& f,
                  const double& arg1,
                  const double& arg2,
                  const double& arg3,
                  const bool& throws,
                  const bool& is_grad_nan) {
  stan::math::var res;
  stan::math::var arg3_v = arg3;

  std::ostringstream fail_msg;
  fail_msg << "Failed for double,double,var version with first argument " 
           << arg1 << " second argument " << arg2
           << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1, arg2, arg3_v), std::domain_error) << fail_msg.str();
  else {
    res = f(arg1, arg2, arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.val())) << fail_msg.str();

    if (is_grad_nan) {
      AVEC x = createAVEC(arg3_v);
      VEC g;
      res.grad(x,g);
  
      ASSERT_EQ(1U,g.size()) << fail_msg.str();
      EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    }
  }
}

template <typename F>
void test_nan_dvd(const F& f,
                  const double& arg1,
                  const double& arg2,
                  const double& arg3,
                  const bool& throws,
                  const bool& is_grad_nan) {
  stan::math::var res;
  stan::math::var arg2_v = arg2;

  std::ostringstream fail_msg;
  fail_msg << "Failed for double,var,double version with first argument " 
           << arg1 << " second argument " << arg2_v
           << " and third argument " << arg3;

  if (throws)
    EXPECT_THROW(f(arg1, arg2_v, arg3), std::domain_error) << fail_msg.str();
  else {
    res = f(arg1, arg2_v, arg3);
    EXPECT_TRUE(boost::math::isnan(res.val())) << fail_msg.str();

    if (is_grad_nan) {
      AVEC x = createAVEC(arg2_v);
      VEC g;
      res.grad(x,g);
  
      ASSERT_EQ(1U,g.size()) << fail_msg.str();
      EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    }
  }
}

template <typename F>
void test_nan_vdd(const F& f,
                  const double& arg1,
                  const double& arg2,
                  const double& arg3,
                  const bool& throws,
                  const bool& is_grad_nan) {
  stan::math::var res;
  stan::math::var arg1_v = arg1;

  std::ostringstream fail_msg;
  fail_msg << "Failed for var,double,double version with first argument " 
           << arg1_v << " second argument " << arg2
           << " and third argument " << arg3;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2, arg3), std::domain_error) << fail_msg.str();
  else {
    res = f(arg1_v, arg2, arg3);
    EXPECT_TRUE(boost::math::isnan(res.val())) << fail_msg.str();

    if (is_grad_nan) {
      AVEC x = createAVEC(arg1_v);
      VEC g;
      res.grad(x,g);
  
      ASSERT_EQ(1U,g.size()) << fail_msg.str();
      EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    }
  }
}
template <typename F>
void test_nan(const F& f,
              const double& arg1,
              const double& arg2,
              const double& arg3,
              const bool& throws,
              const bool& is_grad_nan) {

  double nan = std::numeric_limits<double>::quiet_NaN();
  test_nan_vvv(f, nan, arg2, arg3, throws, is_grad_nan);
  test_nan_vvv(f, arg1, nan, arg3, throws, is_grad_nan);
  test_nan_vvv(f, arg1, arg2, nan, throws, is_grad_nan);
  test_nan_vvv(f, nan, nan, arg3, throws, is_grad_nan);
  test_nan_vvv(f, nan, arg2, nan, throws, is_grad_nan);
  test_nan_vvv(f, arg1, nan, nan, throws, is_grad_nan);
  test_nan_vvv(f, nan, nan, nan, throws, is_grad_nan);

  test_nan_vvd(f, nan, arg2, arg3, throws, is_grad_nan);
  test_nan_vvd(f, arg1, nan, arg3, throws, is_grad_nan);
  test_nan_vvd(f, arg1, arg2, nan, throws, is_grad_nan);
  test_nan_vvd(f, nan, nan, arg3, throws, is_grad_nan);
  test_nan_vvd(f, nan, arg2, nan, throws, is_grad_nan);
  test_nan_vvd(f, arg1, nan, nan, throws, is_grad_nan);
  test_nan_vvd(f, nan, nan, nan, throws, is_grad_nan);

  test_nan_vdv(f, nan, arg2, arg3, throws, is_grad_nan);
  test_nan_vdv(f, arg1, nan, arg3, throws, is_grad_nan);
  test_nan_vdv(f, arg1, arg2, nan, throws, is_grad_nan);
  test_nan_vdv(f, nan, nan, arg3, throws, is_grad_nan);
  test_nan_vdv(f, nan, arg2, nan, throws, is_grad_nan);
  test_nan_vdv(f, arg1, nan, nan, throws, is_grad_nan);
  test_nan_vdv(f, nan, nan, nan, throws, is_grad_nan);

  test_nan_dvv(f, nan, arg2, arg3, throws, is_grad_nan);
  test_nan_dvv(f, arg1, nan, arg3, throws, is_grad_nan);
  test_nan_dvv(f, arg1, arg2, nan, throws, is_grad_nan);
  test_nan_dvv(f, nan, nan, arg3, throws, is_grad_nan);
  test_nan_dvv(f, nan, arg2, nan, throws, is_grad_nan);
  test_nan_dvv(f, arg1, nan, nan, throws, is_grad_nan);
  test_nan_dvv(f, nan, nan, nan, throws, is_grad_nan);

  test_nan_ddv(f, nan, arg2, arg3, throws, is_grad_nan);
  test_nan_ddv(f, arg1, nan, arg3, throws, is_grad_nan);
  test_nan_ddv(f, arg1, arg2, nan, throws, is_grad_nan);
  test_nan_ddv(f, nan, nan, arg3, throws, is_grad_nan);
  test_nan_ddv(f, nan, arg2, nan, throws, is_grad_nan);
  test_nan_ddv(f, arg1, nan, nan, throws, is_grad_nan);
  test_nan_ddv(f, nan, nan, nan, throws, is_grad_nan);

  test_nan_dvd(f, nan, arg2, arg3, throws, is_grad_nan);
  test_nan_dvd(f, arg1, nan, arg3, throws, is_grad_nan);
  test_nan_dvd(f, arg1, arg2, nan, throws, is_grad_nan);
  test_nan_dvd(f, nan, nan, arg3, throws, is_grad_nan);
  test_nan_dvd(f, nan, arg2, nan, throws, is_grad_nan);
  test_nan_dvd(f, arg1, nan, nan, throws, is_grad_nan);
  test_nan_dvd(f, nan, nan, nan, throws, is_grad_nan);

  test_nan_vdd(f, nan, arg2, arg3, throws, is_grad_nan);
  test_nan_vdd(f, arg1, nan, arg3, throws, is_grad_nan);
  test_nan_vdd(f, arg1, arg2, nan, throws, is_grad_nan);
  test_nan_vdd(f, nan, nan, arg3, throws, is_grad_nan);
  test_nan_vdd(f, nan, arg2, nan, throws, is_grad_nan);
  test_nan_vdd(f, arg1, nan, nan, throws, is_grad_nan);
  test_nan_vdd(f, nan, nan, nan, throws, is_grad_nan);
}
#endif

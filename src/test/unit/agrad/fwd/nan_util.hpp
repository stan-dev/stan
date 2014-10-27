#ifndef TEST__UNIT_AGRAD_FWD__NAN_UTIL_HPP
#define TEST__UNIT_AGRAD_FWD__NAN_UTIL_HPP

#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/agrad/fwd.hpp>
#include <gtest/gtest.h>

template <typename F>
void test_nan_fd(const F& f,
                 const double& arg1,
                 const bool& throws) {
  stan::agrad::fvar<double> arg1_v = arg1;
  arg1_v.d_ = 1.0;
  if (throws)
    EXPECT_THROW(f(arg1_v), std::domain_error);
  else {
    EXPECT_TRUE(boost::math::isnan(f(arg1_v).val_));
    EXPECT_TRUE(boost::math::isnan(f(arg1_v).d_));
  }
}

template <typename F>
void test_nan_fv1(const F& f,
                  const double& arg1,
                  const bool& throws) {
  using stan::agrad::var;
  stan::agrad::fvar<var> arg1_v = arg1;

  if (throws)
    EXPECT_THROW(f(arg1_v), std::domain_error);
  else {
    stan::agrad::fvar<var> res = f(arg1_v);
    EXPECT_TRUE(boost::math::isnan(res.val_.val()));

    AVEC x = createAVEC(arg1_v.val_);
    VEC g;
    res.val_.grad(x,g);
    
    ASSERT_EQ(1U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
  }
}

template <typename F>
void test_nan_fv2(const F& f,
                  const double& arg1,
                  const bool& throws) {
  using stan::agrad::var;
  stan::agrad::fvar<var> arg1_v = arg1;

  if (throws)
    EXPECT_THROW(f(arg1_v), std::domain_error);
  else {
    stan::agrad::fvar<var> res = f(arg1_v);
    EXPECT_TRUE(boost::math::isnan(res.d_.val()));

    AVEC x = createAVEC(arg1_v.val_);
    VEC g;
    res.d_.grad(x,g);
    
    ASSERT_EQ(1U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
  }
}
template <typename F>
void test_nan_ffd(const F& f,
                  const double& arg1,
                  const bool& throws) {
  using stan::agrad::fvar;
  fvar<fvar<double> > arg1_v(fvar<double>(arg1,1.0),fvar<double>(1.0,1.0));

  if (throws)
    EXPECT_THROW(f(arg1_v), std::domain_error);
  else {
    EXPECT_TRUE(boost::math::isnan(f(arg1_v).val_.val_));
    EXPECT_TRUE(boost::math::isnan(f(arg1_v).val_.d_));
    EXPECT_TRUE(boost::math::isnan(f(arg1_v).d_.val_));
    EXPECT_TRUE(boost::math::isnan(f(arg1_v).d_.d_));
  }
}
template <typename F>
void test_nan_ffv1(const F& f,
                   const double& arg1,
                   const bool& throws) {
  using stan::agrad::var;
  using stan::agrad::fvar;
  stan::agrad::fvar<fvar<var> > arg1_v = arg1;
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.val_ = 1.0;
  arg1_v.d_.d_ = 1.0;

  if (throws)
    EXPECT_THROW(f(arg1_v), std::domain_error);
  else {
    fvar<fvar<var> > res = f(arg1_v);
    EXPECT_TRUE(boost::math::isnan(res.val_.val_.val()));

    AVEC x = createAVEC(arg1_v.val_.val_);
    VEC g;
    res.val_.val_.grad(x,g);
    
    ASSERT_EQ(1U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
  }
}
template <typename F>
void test_nan_ffv2(const F& f,
                   const double& arg1,
                   const bool& throws) {
  using stan::agrad::var;
  using stan::agrad::fvar;
  stan::agrad::fvar<fvar<var> > arg1_v(fvar<var>(arg1,1.0),fvar<var>(1.0,1.0));

  if (throws)
    EXPECT_THROW(f(arg1_v), std::domain_error);
  else {
    fvar<fvar<var> > res = f(arg1_v);
    EXPECT_TRUE(boost::math::isnan(res.val_.d_.val()));

    AVEC x = createAVEC(arg1_v.val_.val_);
    VEC g;
    res.val_.d_.grad(x,g);
    
    ASSERT_EQ(1U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
  }
}
template <typename F>
void test_nan_ffv3(const F& f,
                   const double& arg1,
                   const bool& throws) {
  using stan::agrad::var;
  using stan::agrad::fvar;
  stan::agrad::fvar<fvar<var> > arg1_v = arg1;
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.val_ = 1.0;
  arg1_v.d_.d_ = 1.0;

  if (throws)
    EXPECT_THROW(f(arg1_v), std::domain_error);
  else {
    fvar<fvar<var> > res = f(arg1_v);
    EXPECT_TRUE(boost::math::isnan(res.d_.val_.val()));

    AVEC x = createAVEC(arg1_v.val_.val_);
    VEC g;
    res.d_.val_.grad(x,g);
    
    ASSERT_EQ(1U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
  }
}
template <typename F>
void test_nan_ffv4(const F& f,
                   const double& arg1,
                   const bool& throws) {
  using stan::agrad::var;
  using stan::agrad::fvar;
  stan::agrad::fvar<fvar<var> > arg1_v (fvar<var>(arg1,1.0), fvar<var>(1.0,1.0));

  if (throws)
    EXPECT_THROW(f(arg1_v), std::domain_error);
  else {
    fvar<fvar<var> > res = f(arg1_v);
    EXPECT_TRUE(boost::math::isnan(res.d_.d_.val()));

    AVEC x = createAVEC(arg1_v.val_.val_);
    VEC g;
    res.d_.d_.grad(x,g);
    
    ASSERT_EQ(1U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
  }
}

template <typename F>
void test_nan(const F& f,
              const bool& throws) {

  double nan = std::numeric_limits<double>::quiet_NaN();
  test_nan_fd(f, nan, throws);
  test_nan_fv1(f, nan, throws);
  test_nan_fv2(f, nan, throws);
  test_nan_ffd(f, nan, throws);
  test_nan_ffv1(f, nan, throws);
  test_nan_ffv2(f, nan, throws);
  test_nan_ffv3(f, nan, throws);
  test_nan_ffv4(f, nan, throws);
}

template <typename F>
void test_nan_fd(const F& f,
                 const double& arg1,
                 const double& arg2,
                 const bool& throws) {
  using stan::agrad::fvar;
  fvar<double> arg1_v(arg1,1.0);
  fvar<double> arg2_v(arg2,1.0);

  std::ostringstream fail_msg1;
  std::ostringstream fail_msg2;
  std::ostringstream fail_msg3;
  fail_msg1 << "Failed for fvar<double>,fvar<double> version with first argument " << arg1_v
           << " and second argument " << arg2_v;
  fail_msg2 << "Failed for fvar<double>,double version with first argument " << arg1_v
           << " and second argument " << arg2;
  fail_msg3 << "Failed for double,fvar<double> version with first argument " << arg1
           << " and second argument " << arg2_v;

  if (throws) {
    EXPECT_THROW(f(arg1_v, arg2_v), std::domain_error) << fail_msg1.str();
    EXPECT_THROW(f(arg1_v, arg2), std::domain_error) << fail_msg2.str();
    EXPECT_THROW(f(arg1, arg2_v), std::domain_error) << fail_msg3.str();
  }
  else {
    EXPECT_TRUE(boost::math::isnan(f(arg1_v, arg2_v).val_)) << fail_msg1.str();
    EXPECT_TRUE(boost::math::isnan(f(arg1_v, arg2).val_)) << fail_msg2.str();
    EXPECT_TRUE(boost::math::isnan(f(arg1, arg2_v).val_)) << fail_msg3.str();
    EXPECT_TRUE(boost::math::isnan(f(arg1_v, arg2_v).d_)) << fail_msg1.str();
    EXPECT_TRUE(boost::math::isnan(f(arg1_v, arg2).d_)) << fail_msg2.str();
    EXPECT_TRUE(boost::math::isnan(f(arg1, arg2_v).d_)) << fail_msg3.str();
  }
}

template <typename F>
void test_nan_fv_fv1(const F& f,
                    const double& arg1,
                    const double& arg2,
                    const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<var> arg1_v(arg1, 1.0);
  fvar<var> arg2_v(arg2, 1.0);

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2_v), std::domain_error);
  else {
    stan::agrad::fvar<var> res = f(arg1_v, arg2_v);
    EXPECT_TRUE(boost::math::isnan(res.val_.val()));

    AVEC x = createAVEC(arg1_v.val_, arg2_v.val_);
    VEC g;
    res.val_.grad(x,g);
    
    ASSERT_EQ(2U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
    EXPECT_TRUE(boost::math::isnan(g[1]));
  }
}

template <typename F>
void test_nan_fv_fv2(const F& f,
                    const double& arg1,
                    const double& arg2,
                    const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<var> arg1_v(arg1, 1.0);
  fvar<var> arg2_v(arg2, 1.0);

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2_v), std::domain_error);
  else {
    stan::agrad::fvar<var> res = f(arg1_v, arg2_v);
    EXPECT_TRUE(boost::math::isnan(res.d_.val()));

    AVEC x = createAVEC(arg1_v.val_, arg2_v.val_);
    VEC g;
    res.d_.grad(x,g);
    
    ASSERT_EQ(2U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
    EXPECT_TRUE(boost::math::isnan(g[1]));
  }
}

template <typename F>
void test_nan_fv_d1(const F& f,
                    const double& arg1,
                    const double& arg2,
                    const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<var> arg1_v(arg1, 1.0);

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2), std::domain_error);
  else {
    stan::agrad::fvar<var> res = f(arg1_v, arg2);
    EXPECT_TRUE(boost::math::isnan(res.val_.val()));

    AVEC x = createAVEC(arg1_v.val_);
    VEC g;
    res.val_.grad(x,g);
    
    ASSERT_EQ(1U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
  }
}

template <typename F>
void test_nan_fv_d2(const F& f,
                    const double& arg1,
                    const double& arg2,
                    const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<var> arg1_v(arg1, 1.0);

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2), std::domain_error);
  else {
    stan::agrad::fvar<var> res = f(arg1_v, arg2);
    EXPECT_TRUE(boost::math::isnan(res.d_.val()));

    AVEC x = createAVEC(arg1_v.val_);
    VEC g;
    res.d_.grad(x,g);
    
    ASSERT_EQ(1U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
  }
}

template <typename F>
void test_nan_d_fv1(const F& f,
                    const double& arg1,
                    const double& arg2,
                    const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<var> arg2_v(arg2, 1.0);

  if (throws)
    EXPECT_THROW(f(arg1, arg2_v), std::domain_error);
  else {
    stan::agrad::fvar<var> res = f(arg1, arg2_v);
    EXPECT_TRUE(boost::math::isnan(res.val_.val()));

    AVEC x = createAVEC(arg2_v.val_);
    VEC g;
    res.val_.grad(x,g);
    
    ASSERT_EQ(1U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
  }
}

template <typename F>
void test_nan_d_fv2(const F& f,
                    const double& arg1,
                    const double& arg2,
                    const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<var> arg2_v(arg2, 1.0);

  if (throws)
    EXPECT_THROW(f(arg1, arg2_v), std::domain_error);
  else {
    stan::agrad::fvar<var> res = f(arg1, arg2_v);
    EXPECT_TRUE(boost::math::isnan(res.d_.val()));

    AVEC x = createAVEC(arg2_v.val_);
    VEC g;
    res.d_.grad(x,g);
    
    ASSERT_EQ(1U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
  }
}

template <typename F>
void test_nan_ffd(const F& f,
                  const double& arg1,
                  const double& arg2,
                  const bool& throws) {
  using stan::agrad::fvar;
  fvar<fvar<double> > arg1_v = arg1;
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.val_ = 1.0;
  arg1_v.d_.d_ = 1.0;
  fvar<fvar<double> > arg2_v = arg2;
  arg2_v.val_.d_ = 1.0;
  arg2_v.d_.val_ = 1.0;
  arg2_v.d_.d_ = 1.0;
 
  std::ostringstream fail_msg1;
  std::ostringstream fail_msg2;
  std::ostringstream fail_msg3;
  fail_msg1 << "Failed for fvar<fvar<double>>,fvar<fvar<double>> version with first argument " << arg1_v
           << " and second argument " << arg2_v;
  fail_msg2 << "Failed for fvar<fvar<double>>,double version with first argument " << arg1_v
           << " and second argument " << arg2;
  fail_msg3 << "Failed for double,fvar<fvar<double>> version with first argument " << arg1
           << " and second argument " << arg2_v;

  if (throws) {
    EXPECT_THROW(f(arg1_v, arg2_v), std::domain_error) << fail_msg1.str();
    EXPECT_THROW(f(arg1_v, arg2), std::domain_error) << fail_msg2.str();
    EXPECT_THROW(f(arg1, arg2_v), std::domain_error) << fail_msg3.str();
  }
  else {
    EXPECT_TRUE(boost::math::isnan(f(arg1_v, arg2_v).val_.val_)) << fail_msg1.str();
    EXPECT_TRUE(boost::math::isnan(f(arg1_v, arg2).val_.val_)) << fail_msg2.str();
    EXPECT_TRUE(boost::math::isnan(f(arg1, arg2_v).val_.val_)) << fail_msg3.str();
    EXPECT_TRUE(boost::math::isnan(f(arg1_v, arg2_v).val_.d_)) << fail_msg1.str();
    EXPECT_TRUE(boost::math::isnan(f(arg1_v, arg2).val_.d_)) << fail_msg2.str();
    EXPECT_TRUE(boost::math::isnan(f(arg1, arg2_v).val_.d_)) << fail_msg3.str();
    EXPECT_TRUE(boost::math::isnan(f(arg1_v, arg2_v).d_.val_)) << fail_msg1.str();
    EXPECT_TRUE(boost::math::isnan(f(arg1_v, arg2).d_.val_)) << fail_msg2.str();
    EXPECT_TRUE(boost::math::isnan(f(arg1, arg2_v).d_.val_)) << fail_msg3.str();
    EXPECT_TRUE(boost::math::isnan(f(arg1_v, arg2_v).d_.d_)) << fail_msg1.str();
    EXPECT_TRUE(boost::math::isnan(f(arg1_v, arg2).d_.d_)) << fail_msg2.str();
    EXPECT_TRUE(boost::math::isnan(f(arg1, arg2_v).d_.d_)) << fail_msg3.str();
  }
}

template <typename F>
void test_nan_ffv_ffv1(const F& f,
                       const double& arg1,
                       const double& arg2,
                       const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg1_v = arg1;
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.val_ = 1.0;
  arg1_v.d_.d_ = 1.0;
  fvar<fvar<var> > arg2_v = arg2;
  arg2_v.val_.d_ = 1.0;
  arg2_v.d_.val_ = 1.0;
  arg2_v.d_.d_ = 1.0;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2_v), std::domain_error);
  else {
    fvar<fvar<var> > res = f(arg1_v, arg2_v);
    EXPECT_TRUE(boost::math::isnan(res.val_.val_.val()));

    AVEC x = createAVEC(arg1_v.val_.val_, arg2_v.val_.val_);
    VEC g;
    res.val_.val_.grad(x,g);
    
    ASSERT_EQ(2U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
    EXPECT_TRUE(boost::math::isnan(g[1]));
  }
}
template <typename F>
void test_nan_ffv_ffv2(const F& f,
                       const double& arg1,
                       const double& arg2,
                       const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg1_v = arg1;
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.val_ = 1.0;
  arg1_v.d_.d_ = 1.0;
  fvar<fvar<var> > arg2_v = arg2;
  arg2_v.val_.d_ = 1.0;
  arg2_v.d_.val_ = 1.0;
  arg2_v.d_.d_ = 1.0;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2_v), std::domain_error);
  else {
    fvar<fvar<var> > res = f(arg1_v, arg2_v);
    EXPECT_TRUE(boost::math::isnan(res.val_.d_.val()));

    AVEC x = createAVEC(arg1_v.val_.val_, arg2_v.val_.val_);
    VEC g;
    res.val_.d_.grad(x,g);
    
    ASSERT_EQ(2U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
    EXPECT_TRUE(boost::math::isnan(g[1]));
  }
}
template <typename F>
void test_nan_ffv_ffv3(const F& f,
                       const double& arg1,
                       const double& arg2,
                       const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg1_v = arg1;
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.val_ = 1.0;
  arg1_v.d_.d_ = 1.0;
  fvar<fvar<var> > arg2_v = arg2;
  arg2_v.val_.d_ = 1.0;
  arg2_v.d_.val_ = 1.0;
  arg2_v.d_.d_ = 1.0;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2_v), std::domain_error);
  else {
    fvar<fvar<var> > res = f(arg1_v, arg2_v);
    EXPECT_TRUE(boost::math::isnan(res.d_.val_.val()));

    AVEC x = createAVEC(arg1_v.val_.val_, arg2_v.val_.val_);
    VEC g;
    res.d_.val_.grad(x,g);
    
    ASSERT_EQ(2U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
    EXPECT_TRUE(boost::math::isnan(g[1]));
  }
}
template <typename F>
void test_nan_ffv_ffv4(const F& f,
                       const double& arg1,
                       const double& arg2,
                       const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg1_v = arg1;
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.val_ = 1.0;
  arg1_v.d_.d_ = 1.0;
  fvar<fvar<var> > arg2_v = arg2;
  arg2_v.val_.d_ = 1.0;
  arg2_v.d_.val_ = 1.0;
  arg2_v.d_.d_ = 1.0;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2_v), std::domain_error);
  else {
    fvar<fvar<var> > res = f(arg1_v, arg2_v);
    EXPECT_TRUE(boost::math::isnan(res.d_.d_.val()));

    AVEC x = createAVEC(arg1_v.val_.val_, arg2_v.val_.val_);
    VEC g;
    res.d_.d_.grad(x,g);
    
    ASSERT_EQ(2U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
    EXPECT_TRUE(boost::math::isnan(g[1]));
  }
}

template <typename F>
void test_nan_ffv_d1(const F& f,
                       const double& arg1,
                       const double& arg2,
                       const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg1_v = arg1;
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.val_ = 1.0;
  arg1_v.d_.d_ = 1.0;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2), std::domain_error);
  else {
    fvar<fvar<var> > res = f(arg1_v, arg2);
    EXPECT_TRUE(boost::math::isnan(res.val_.val_.val()));

    AVEC x = createAVEC(arg1_v.val_.val_);
    VEC g;
    res.val_.val_.grad(x,g);
    
    ASSERT_EQ(1U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
  }
}
template <typename F>
void test_nan_ffv_d2(const F& f,
                       const double& arg1,
                       const double& arg2,
                       const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg1_v = arg1;
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.val_ = 1.0;
  arg1_v.d_.d_ = 1.0;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2), std::domain_error);
  else {
    fvar<fvar<var> > res = f(arg1_v, arg2);
    EXPECT_TRUE(boost::math::isnan(res.val_.d_.val()));

    AVEC x = createAVEC(arg1_v.val_.val_);
    VEC g;
    res.val_.d_.grad(x,g);
    
    ASSERT_EQ(1U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
  }
}
template <typename F>
void test_nan_ffv_d3(const F& f,
                       const double& arg1,
                       const double& arg2,
                       const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg1_v = arg1;
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.val_ = 1.0;
  arg1_v.d_.d_ = 1.0;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2), std::domain_error);
  else {
    fvar<fvar<var> > res = f(arg1_v, arg2);
    EXPECT_TRUE(boost::math::isnan(res.d_.val_.val()));

    AVEC x = createAVEC(arg1_v.val_.val_);
    VEC g;
    res.d_.val_.grad(x,g);
    
    ASSERT_EQ(1U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
  }
}
template <typename F>
void test_nan_ffv_d4(const F& f,
                       const double& arg1,
                       const double& arg2,
                       const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg1_v = arg1;
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.val_ = 1.0;
  arg1_v.d_.d_ = 1.0;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2), std::domain_error);
  else {
    fvar<fvar<var> > res = f(arg1_v, arg2);
    EXPECT_TRUE(boost::math::isnan(res.d_.d_.val()));

    AVEC x = createAVEC(arg1_v.val_.val_);
    VEC g;
    res.d_.d_.grad(x,g);
    
    ASSERT_EQ(1U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
  }
}

template <typename F>
void test_nan_d_ffv1(const F& f,
                       const double& arg1,
                       const double& arg2,
                       const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg2_v = arg2;
  arg2_v.val_.d_ = 1.0;
  arg2_v.d_.val_ = 1.0;
  arg2_v.d_.d_ = 1.0;

  if (throws)
    EXPECT_THROW(f(arg1, arg2_v), std::domain_error);
  else {
    fvar<fvar<var> > res = f(arg1, arg2_v);
    EXPECT_TRUE(boost::math::isnan(res.val_.val_.val()));

    AVEC x = createAVEC(arg2_v.val_.val_);
    VEC g;
    res.val_.val_.grad(x,g);
    
    ASSERT_EQ(1U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
  }
}
template <typename F>
void test_nan_d_ffv2(const F& f,
                       const double& arg1,
                       const double& arg2,
                       const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg2_v = arg2;
  arg2_v.val_.d_ = 1.0;
  arg2_v.d_.val_ = 1.0;
  arg2_v.d_.d_ = 1.0;

  if (throws)
    EXPECT_THROW(f(arg1, arg2_v), std::domain_error);
  else {
    fvar<fvar<var> > res = f(arg1, arg2_v);
    EXPECT_TRUE(boost::math::isnan(res.val_.d_.val()));

    AVEC x = createAVEC(arg2_v.val_.val_);
    VEC g;
    res.val_.d_.grad(x,g);
    
    ASSERT_EQ(1U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
  }
}
template <typename F>
void test_nan_d_ffv3(const F& f,
                       const double& arg1,
                       const double& arg2,
                       const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg2_v = arg2;
  arg2_v.val_.d_ = 1.0;
  arg2_v.d_.val_ = 1.0;
  arg2_v.d_.d_ = 1.0;

  if (throws)
    EXPECT_THROW(f(arg1, arg2_v), std::domain_error);
  else {
    fvar<fvar<var> > res = f(arg1, arg2_v);
    EXPECT_TRUE(boost::math::isnan(res.d_.val_.val()));

    AVEC x = createAVEC(arg2_v.val_.val_);
    VEC g;
    res.d_.val_.grad(x,g);
    
    ASSERT_EQ(1U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
  }
}
template <typename F>
void test_nan_d_ffv4(const F& f,
                       const double& arg1,
                       const double& arg2,
                       const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg2_v = arg2;
  arg2_v.val_.d_ = 1.0;
  arg2_v.d_.val_ = 1.0;
  arg2_v.d_.d_ = 1.0;

  if (throws)
    EXPECT_THROW(f(arg1, arg2_v), std::domain_error);
  else {
    fvar<fvar<var> > res = f(arg1, arg2_v);
    EXPECT_TRUE(boost::math::isnan(res.d_.d_.val()));

    AVEC x = createAVEC(arg2_v.val_.val_);
    VEC g;
    res.d_.d_.grad(x,g);
    
    ASSERT_EQ(1U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
  }
}
template <typename F>
void test_nan(const F& f,
              const double& arg1,
              const double& arg2,
              const bool& throws) {

  double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::VectorXd arg1_vec(3);
  Eigen::VectorXd arg2_vec(3);
  arg1_vec << nan, arg1, nan;
  arg2_vec << arg2, nan, nan;
  for (int i = 0 ; i < arg1_vec.size(); i++) {
    test_nan_fd(f, arg1_vec(i), arg2_vec(i), throws);
    test_nan_ffd(f, arg1_vec(i), arg2_vec(i), throws);
    test_nan_fv_fv1(f, arg1_vec(i), arg2_vec(i), throws);
    test_nan_fv_fv2(f, arg1_vec(i), arg2_vec(i), throws);
    test_nan_fv_d1(f, arg1_vec(i), arg2_vec(i), throws);
    test_nan_fv_d2(f, arg1_vec(i), arg2_vec(i), throws);
    test_nan_d_fv1(f, arg1_vec(i), arg2_vec(i), throws);
    test_nan_d_fv2(f, arg1_vec(i), arg2_vec(i), throws);
    test_nan_ffv_ffv1(f, arg1_vec(i), arg2_vec(i), throws);
    test_nan_ffv_ffv2(f, arg1_vec(i), arg2_vec(i), throws);
    test_nan_ffv_ffv3(f, arg1_vec(i), arg2_vec(i), throws);
    test_nan_ffv_ffv4(f, arg1_vec(i), arg2_vec(i), throws);
    test_nan_ffv_d1(f, arg1_vec(i), arg2_vec(i), throws);
    test_nan_ffv_d2(f, arg1_vec(i), arg2_vec(i), throws);
    test_nan_ffv_d3(f, arg1_vec(i), arg2_vec(i), throws);
    test_nan_ffv_d4(f, arg1_vec(i), arg2_vec(i), throws);
    test_nan_d_ffv1(f, arg1_vec(i), arg2_vec(i), throws);
    test_nan_d_ffv2(f, arg1_vec(i), arg2_vec(i), throws);
    test_nan_d_ffv3(f, arg1_vec(i), arg2_vec(i), throws);
    test_nan_d_ffv4(f, arg1_vec(i), arg2_vec(i), throws);
  }
}


template <typename F>
void test_nan_fd(const F& f,
                 const double& arg1,
                 const double& arg2,
                 const double& arg3,
                 const bool& throws) {
  using stan::agrad::fvar;
  using boost::math::isnan;
  fvar<double> arg1_v(arg1,1.0);
  fvar<double> arg2_v(arg2,1.0);
  fvar<double> arg3_v(arg3,1.0);

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2_v 
            << " and third argument " << arg3_v;

  if (throws) {
    EXPECT_THROW(f(arg1, arg2_v, arg3_v), std::domain_error) << fail_msg.str();
    EXPECT_THROW(f(arg1_v, arg2, arg3_v), std::domain_error) << fail_msg.str();
    EXPECT_THROW(f(arg1_v, arg2_v, arg3), std::domain_error) << fail_msg.str();
    EXPECT_THROW(f(arg1_v, arg2, arg3), std::domain_error) << fail_msg.str();
    EXPECT_THROW(f(arg1, arg2_v, arg3), std::domain_error) << fail_msg.str();
    EXPECT_THROW(f(arg1, arg2, arg3_v), std::domain_error) << fail_msg.str();
    EXPECT_THROW(f(arg1_v, arg2_v, arg3_v), std::domain_error) << fail_msg.str();
  }
  else {
    EXPECT_TRUE(isnan(f(arg1_v, arg2_v, arg3_v).val_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1, arg2_v, arg3_v).val_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1_v, arg2, arg3_v).val_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1_v, arg2_v, arg3).val_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1, arg2, arg3_v).val_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1, arg2_v, arg3).val_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1_v, arg2, arg3).val_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1_v, arg2_v, arg3_v).d_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1, arg2_v, arg3_v).d_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1_v, arg2, arg3_v).d_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1_v, arg2_v, arg3).d_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1, arg2, arg3_v).d_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1, arg2_v, arg3).d_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1_v, arg2, arg3).d_)) << fail_msg.str();
  }
}

template <typename F>
void test_nan_fv_fv_fv1(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<var> arg1_v(arg1,1.0);
  fvar<var> arg2_v(arg2,1.0);
  fvar<var> arg3_v(arg3,1.0);

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2_v 
            << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2_v, arg3_v), std::domain_error) << fail_msg.str();
  else {
    stan::agrad::fvar<var> res = f(arg1_v,arg2_v,arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.val_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg1_v.val_,arg2_v.val_,arg3_v.val_);
    VEC g;
    res.val_.grad(x,g);
    
    ASSERT_EQ(3U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[2])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_fv_fv_fv2(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<var> arg1_v(arg1,1.0);
  fvar<var> arg2_v(arg2,1.0);
  fvar<var> arg3_v(arg3,1.0);

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2_v 
            << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2_v, arg3_v), std::domain_error) << fail_msg.str();
  else {
    stan::agrad::fvar<var> res = f(arg1_v,arg2_v,arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.d_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg1_v.val_,arg2_v.val_,arg3_v.val_);
    VEC g;
    res.d_.grad(x,g);
    
    ASSERT_EQ(3U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[2])) << fail_msg.str();
  }
}

template <typename F>
void test_nan_d_fv_fv1(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<var> arg2_v(arg2,1.0);
  fvar<var> arg3_v(arg3,1.0);

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1
            << " second argument " << arg2_v 
            << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1, arg2_v, arg3_v), std::domain_error) << fail_msg.str();
  else {
    stan::agrad::fvar<var> res = f(arg1,arg2_v,arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.val_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg2_v.val_,arg3_v.val_);
    VEC g;
    res.val_.grad(x,g);
    
    ASSERT_EQ(2U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_d_fv_fv2(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<var> arg2_v(arg2,1.0);
  fvar<var> arg3_v(arg3,1.0);

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1
            << " second argument " << arg2_v 
            << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1, arg2_v, arg3_v), std::domain_error) << fail_msg.str();
  else {
    stan::agrad::fvar<var> res = f(arg1,arg2_v,arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.d_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg2_v.val_,arg3_v.val_);
    VEC g;
    res.d_.grad(x,g);
    
    ASSERT_EQ(2U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_fv_d_fv1(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<var> arg1_v(arg1,1.0);
  fvar<var> arg3_v(arg3,1.0);

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2
            << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2, arg3_v), std::domain_error) << fail_msg.str();
  else {
    stan::agrad::fvar<var> res = f(arg1_v,arg2,arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.val_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg1_v.val_,arg3_v.val_);
    VEC g;
    res.val_.grad(x,g);
    
    ASSERT_EQ(2U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_fv_d_fv2(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<var> arg1_v(arg1,1.0);
  fvar<var> arg3_v(arg3,1.0);

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2
            << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2, arg3_v), std::domain_error) << fail_msg.str();
  else {
    stan::agrad::fvar<var> res = f(arg1_v,arg2,arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.d_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg1_v.val_,arg3_v.val_);
    VEC g;
    res.d_.grad(x,g);
    
    ASSERT_EQ(2U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_fv_fv_d1(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<var> arg1_v(arg1,1.0);
  fvar<var> arg2_v(arg2,1.0);

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2_v
            << " and third argument " << arg3;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2_v, arg3), std::domain_error) << fail_msg.str();
  else {
    stan::agrad::fvar<var> res = f(arg1_v,arg2_v,arg3);
    EXPECT_TRUE(boost::math::isnan(res.val_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg1_v.val_,arg2_v.val_);
    VEC g;
    res.val_.grad(x,g);
    
    ASSERT_EQ(2U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_fv_fv_d2(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<var> arg1_v(arg1,1.0);
  fvar<var> arg2_v(arg2,1.0);

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2_v
            << " and third argument " << arg3;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2_v, arg3), std::domain_error) << fail_msg.str();
  else {
    stan::agrad::fvar<var> res = f(arg1_v,arg2_v,arg3);
    EXPECT_TRUE(boost::math::isnan(res.d_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg1_v.val_,arg2_v.val_);
    VEC g;
    res.d_.grad(x,g);
    
    ASSERT_EQ(2U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_fv_d_d1(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<var> arg1_v(arg1,1.0);

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2
            << " and third argument " << arg3;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2, arg3), std::domain_error) << fail_msg.str();
  else {
    stan::agrad::fvar<var> res = f(arg1_v,arg2,arg3);
    EXPECT_TRUE(boost::math::isnan(res.val_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg1_v.val_);
    VEC g;
    res.val_.grad(x,g);
    
    ASSERT_EQ(1U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_fv_d_d2(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<var> arg1_v(arg1,1.0);

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2
            << " and third argument " << arg3;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2, arg3), std::domain_error) << fail_msg.str();
  else {
    stan::agrad::fvar<var> res = f(arg1_v,arg2,arg3);
    EXPECT_TRUE(boost::math::isnan(res.d_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg1_v.val_);
    VEC g;
    res.d_.grad(x,g);
    
    ASSERT_EQ(1U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
  }
}

template <typename F>
void test_nan_d_fv_d1(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<var> arg2_v(arg2,1.0);

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1
            << " second argument " << arg2_v
            << " and third argument " << arg3;

  if (throws)
    EXPECT_THROW(f(arg1, arg2_v, arg3), std::domain_error) << fail_msg.str();
  else {
    stan::agrad::fvar<var> res = f(arg1,arg2_v,arg3);
    EXPECT_TRUE(boost::math::isnan(res.val_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg2_v.val_);
    VEC g;
    res.val_.grad(x,g);
    
    ASSERT_EQ(1U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_d_fv_d2(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<var> arg2_v(arg2,1.0);

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1
            << " second argument " << arg2_v
            << " and third argument " << arg3;

  if (throws)
    EXPECT_THROW(f(arg1, arg2_v, arg3), std::domain_error) << fail_msg.str();
  else {
    stan::agrad::fvar<var> res = f(arg1,arg2_v,arg3);
    EXPECT_TRUE(boost::math::isnan(res.d_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg2_v.val_);
    VEC g;
    res.d_.grad(x,g);
    
    ASSERT_EQ(1U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
  }
}

template <typename F>
void test_nan_d_d_fv1(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<var> arg3_v(arg3,1.0);

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1
            << " second argument " << arg2
            << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1, arg2, arg3_v), std::domain_error) << fail_msg.str();
  else {
    stan::agrad::fvar<var> res = f(arg1,arg2,arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.val_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg3_v.val_);
    VEC g;
    res.val_.grad(x,g);
    
    ASSERT_EQ(1U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_d_d_fv2(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<var> arg3_v(arg3,1.0);

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1
            << " second argument " << arg2
            << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1, arg2, arg3_v), std::domain_error) << fail_msg.str();
  else {
    stan::agrad::fvar<var> res = f(arg1,arg2,arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.d_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg3_v.val_);
    VEC g;
    res.d_.grad(x,g);
    
    ASSERT_EQ(1U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_ffd(const F& f,
                 const double& arg1,
                 const double& arg2,
                 const double& arg3,
                 const bool& throws) {
  using stan::agrad::fvar;
  using boost::math::isnan;
  fvar<fvar<double> > arg1_v(arg1,1.0);
  fvar<fvar<double> > arg2_v(arg2,1.0);
  fvar<fvar<double> > arg3_v(arg3,1.0);
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.d_ = 1.0;
  arg2_v.val_.d_ = 1.0;
  arg2_v.d_.d_ = 1.0;
  arg3_v.val_.d_ = 1.0;
  arg3_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2_v 
            << " and third argument " << arg3_v;

  if (throws) {
    EXPECT_THROW(f(arg1, arg2_v, arg3_v), std::domain_error) << fail_msg.str();
    EXPECT_THROW(f(arg1_v, arg2, arg3_v), std::domain_error) << fail_msg.str();
    EXPECT_THROW(f(arg1_v, arg2_v, arg3), std::domain_error) << fail_msg.str();
    EXPECT_THROW(f(arg1_v, arg2, arg3), std::domain_error) << fail_msg.str();
    EXPECT_THROW(f(arg1, arg2_v, arg3), std::domain_error) << fail_msg.str();
    EXPECT_THROW(f(arg1, arg2, arg3_v), std::domain_error) << fail_msg.str();
    EXPECT_THROW(f(arg1_v, arg2_v, arg3_v), std::domain_error) << fail_msg.str();
  }
  else {
    EXPECT_TRUE(isnan(f(arg1_v, arg2_v, arg3_v).val_.val_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1, arg2_v, arg3_v).val_.val_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1_v, arg2, arg3_v).val_.val_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1_v, arg2_v, arg3).val_.val_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1, arg2, arg3_v).val_.val_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1, arg2_v, arg3).val_.val_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1_v, arg2, arg3).val_.val_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1_v, arg2_v, arg3_v).val_.d_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1, arg2_v, arg3_v).val_.d_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1_v, arg2, arg3_v).val_.d_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1_v, arg2_v, arg3).val_.d_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1, arg2, arg3_v).val_.d_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1, arg2_v, arg3).val_.d_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1_v, arg2, arg3).val_.d_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1_v, arg2_v, arg3_v).d_.val_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1, arg2_v, arg3_v).d_.val_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1_v, arg2, arg3_v).d_.val_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1_v, arg2_v, arg3).d_.val_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1, arg2, arg3_v).d_.val_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1, arg2_v, arg3).d_.val_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1_v, arg2, arg3).d_.val_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1_v, arg2_v, arg3_v).d_.d_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1, arg2_v, arg3_v).d_.d_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1_v, arg2, arg3_v).d_.d_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1_v, arg2_v, arg3).d_.d_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1, arg2, arg3_v).d_.d_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1, arg2_v, arg3).d_.d_)) << fail_msg.str();
    EXPECT_TRUE(isnan(f(arg1_v, arg2, arg3).d_.d_)) << fail_msg.str();
  }
}

template <typename F>
void test_nan_ffv_ffv_ffv1(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg1_v(arg1,1.0);
  fvar<fvar<var> > arg2_v(arg2,1.0);
  fvar<fvar<var> > arg3_v(arg3,1.0);
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.d_ = 1.0;
  arg2_v.val_.d_ = 1.0;
  arg2_v.d_.d_ = 1.0;
  arg3_v.val_.d_ = 1.0;
  arg3_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2_v 
            << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2_v, arg3_v), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1_v,arg2_v,arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.val_.val_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg1_v.val_.val_,arg2_v.val_.val_,arg3_v.val_.val_);
    VEC g;
    res.val_.val_.grad(x,g);
    
    ASSERT_EQ(3U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[2])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_ffv_ffv_ffv2(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg1_v(arg1,1.0);
  fvar<fvar<var> > arg2_v(arg2,1.0);
  fvar<fvar<var> > arg3_v(arg3,1.0);
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.d_ = 1.0;
  arg2_v.val_.d_ = 1.0;
  arg2_v.d_.d_ = 1.0;
  arg3_v.val_.d_ = 1.0;
  arg3_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2_v 
            << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2_v, arg3_v), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1_v,arg2_v,arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.d_.val_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg1_v.val_.val_,arg2_v.val_.val_,arg3_v.val_.val_);
    VEC g;
    res.d_.val_.grad(x,g);
    
    ASSERT_EQ(3U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[2])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_ffv_ffv_ffv3(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg1_v(arg1,1.0);
  fvar<fvar<var> > arg2_v(arg2,1.0);
  fvar<fvar<var> > arg3_v(arg3,1.0);
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.d_ = 1.0;
  arg2_v.val_.d_ = 1.0;
  arg2_v.d_.d_ = 1.0;
  arg3_v.val_.d_ = 1.0;
  arg3_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2_v 
            << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2_v, arg3_v), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1_v,arg2_v,arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.val_.d_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg1_v.val_.val_,arg2_v.val_.val_,arg3_v.val_.val_);
    VEC g;
    res.val_.d_.grad(x,g);
    
    ASSERT_EQ(3U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[2])) << fail_msg.str();
  }
}

template <typename F>
void test_nan_ffv_ffv_ffv4(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg1_v(arg1,1.0);
  fvar<fvar<var> > arg2_v(arg2,1.0);
  fvar<fvar<var> > arg3_v(arg3,1.0);
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.d_ = 1.0;
  arg2_v.val_.d_ = 1.0;
  arg2_v.d_.d_ = 1.0;
  arg3_v.val_.d_ = 1.0;
  arg3_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2_v 
            << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2_v, arg3_v), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1_v,arg2_v,arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.d_.d_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg1_v.val_.val_,arg2_v.val_.val_,arg3_v.val_.val_);
    VEC g;
    res.d_.d_.grad(x,g);
    
    ASSERT_EQ(3U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[2])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_d_ffv_ffv1(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg2_v(arg2,1.0);
  fvar<fvar<var> > arg3_v(arg3,1.0);
  arg2_v.val_.d_ = 1.0;
  arg2_v.d_.d_ = 1.0;
  arg3_v.val_.d_ = 1.0;
  arg3_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1
            << " second argument " << arg2_v 
            << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1, arg2_v, arg3_v), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1,arg2_v,arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.val_.val_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg2_v.val_.val_,arg3_v.val_.val_);
    VEC g;
    res.val_.val_.grad(x,g);
    
    ASSERT_EQ(2U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_d_ffv_ffv2(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg2_v(arg2,1.0);
  fvar<fvar<var> > arg3_v(arg3,1.0);
  arg2_v.val_.d_ = 1.0;
  arg2_v.d_.d_ = 1.0;
  arg3_v.val_.d_ = 1.0;
  arg3_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1
            << " second argument " << arg2_v 
            << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1, arg2_v, arg3_v), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1,arg2_v,arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.d_.val_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg2_v.val_.val_,arg3_v.val_.val_);
    VEC g;
    res.d_.val_.grad(x,g);
    
    ASSERT_EQ(2U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_d_ffv_ffv3(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg2_v(arg2,1.0);
  fvar<fvar<var> > arg3_v(arg3,1.0);
  arg2_v.val_.d_ = 1.0;
  arg2_v.d_.d_ = 1.0;
  arg3_v.val_.d_ = 1.0;
  arg3_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1
            << " second argument " << arg2_v 
            << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1, arg2_v, arg3_v), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1,arg2_v,arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.val_.d_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg2_v.val_.val_,arg3_v.val_.val_);
    VEC g;
    res.val_.d_.grad(x,g);
    
    ASSERT_EQ(2U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_d_ffv_ffv4(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg2_v(arg2,1.0);
  fvar<fvar<var> > arg3_v(arg3,1.0);
  arg2_v.val_.d_ = 1.0;
  arg2_v.d_.d_ = 1.0;
  arg3_v.val_.d_ = 1.0;
  arg3_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1
            << " second argument " << arg2_v 
            << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1, arg2_v, arg3_v), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1,arg2_v,arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.d_.d_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg2_v.val_.val_,arg3_v.val_.val_);
    VEC g;
    res.d_.d_.grad(x,g);
    
    ASSERT_EQ(2U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_ffv_d_ffv1(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg1_v(arg1,1.0);
  fvar<fvar<var> > arg3_v(arg3,1.0);
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.d_ = 1.0;
  arg3_v.val_.d_ = 1.0;
  arg3_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2
            << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2, arg3_v), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1_v,arg2,arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.val_.val_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg1_v.val_.val_,arg3_v.val_.val_);
    VEC g;
    res.val_.val_.grad(x,g);
    
    ASSERT_EQ(2U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_ffv_d_ffv2(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg1_v(arg1,1.0);
  fvar<fvar<var> > arg3_v(arg3,1.0);
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.d_ = 1.0;
  arg3_v.val_.d_ = 1.0;
  arg3_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2
            << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2, arg3_v), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1_v,arg2,arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.d_.val_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg1_v.val_.val_,arg3_v.val_.val_);
    VEC g;
    res.d_.val_.grad(x,g);
    
    ASSERT_EQ(2U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_ffv_d_ffv3(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg1_v(arg1,1.0);
  fvar<fvar<var> > arg3_v(arg3,1.0);
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.d_ = 1.0;
  arg3_v.val_.d_ = 1.0;
  arg3_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2
            << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2, arg3_v), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1_v,arg2,arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.val_.d_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg1_v.val_.val_,arg3_v.val_.val_);
    VEC g;
    res.val_.d_.grad(x,g);
    
    ASSERT_EQ(2U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_ffv_d_ffv4(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg1_v(arg1,1.0);
  fvar<fvar<var> > arg3_v(arg3,1.0);
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.d_ = 1.0;
  arg3_v.val_.d_ = 1.0;
  arg3_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2
            << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2, arg3_v), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1_v,arg2,arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.d_.d_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg1_v.val_.val_,arg3_v.val_.val_);
    VEC g;
    res.d_.d_.grad(x,g);
    
    ASSERT_EQ(2U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_ffv_ffv_d1(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg1_v(arg1,1.0);
  fvar<fvar<var> > arg2_v(arg2,1.0);
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.d_ = 1.0;
  arg2_v.val_.d_ = 1.0;
  arg2_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2_v
            << " and third argument " << arg3;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2_v, arg3), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1_v,arg2_v,arg3);
    EXPECT_TRUE(boost::math::isnan(res.val_.val_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg1_v.val_.val_,arg2_v.val_.val_);
    VEC g;
    res.val_.val_.grad(x,g);
    
    ASSERT_EQ(2U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_ffv_ffv_d2(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg1_v(arg1,1.0);
  fvar<fvar<var> > arg2_v(arg2,1.0);
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.d_ = 1.0;
  arg2_v.val_.d_ = 1.0;
  arg2_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2_v
            << " and third argument " << arg3;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2_v, arg3), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1_v,arg2_v,arg3);
    EXPECT_TRUE(boost::math::isnan(res.d_.val_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg1_v.val_.val_,arg2_v.val_.val_);
    VEC g;
    res.d_.val_.grad(x,g);
    
    ASSERT_EQ(2U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_ffv_ffv_d3(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg1_v(arg1,1.0);
  fvar<fvar<var> > arg2_v(arg2,1.0);
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.d_ = 1.0;
  arg2_v.val_.d_ = 1.0;
  arg2_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2_v
            << " and third argument " << arg3;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2_v, arg3), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1_v,arg2_v,arg3);
    EXPECT_TRUE(boost::math::isnan(res.val_.d_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg1_v.val_.val_,arg2_v.val_.val_);
    VEC g;
    res.val_.d_.grad(x,g);
    
    ASSERT_EQ(2U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_ffv_ffv_d4(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg1_v(arg1,1.0);
  fvar<fvar<var> > arg2_v(arg2,1.0);
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.d_ = 1.0;
  arg2_v.val_.d_ = 1.0;
  arg2_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2_v
            << " and third argument " << arg3;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2_v, arg3), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1_v,arg2_v,arg3);
    EXPECT_TRUE(boost::math::isnan(res.d_.d_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg1_v.val_.val_,arg2_v.val_.val_);
    VEC g;
    res.d_.d_.grad(x,g);
    
    ASSERT_EQ(2U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[1])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_ffv_d_d1(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg1_v(arg1,1.0);
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2
            << " and third argument " << arg3;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2, arg3), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1_v,arg2,arg3);
    EXPECT_TRUE(boost::math::isnan(res.val_.val_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg1_v.val_.val_);
    VEC g;
    res.val_.val_.grad(x,g);
    
    ASSERT_EQ(1U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_ffv_d_d2(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg1_v(arg1,1.0);
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2
            << " and third argument " << arg3;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2, arg3), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1_v,arg2,arg3);
    EXPECT_TRUE(boost::math::isnan(res.d_.val_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg1_v.val_.val_);
    VEC g;
    res.d_.val_.grad(x,g);
    
    ASSERT_EQ(1U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_ffv_d_d3(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg1_v(arg1,1.0);
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2
            << " and third argument " << arg3;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2, arg3), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1_v,arg2,arg3);
    EXPECT_TRUE(boost::math::isnan(res.val_.d_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg1_v.val_.val_);
    VEC g;
    res.val_.d_.grad(x,g);
    
    ASSERT_EQ(1U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_ffv_d_d4(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg1_v(arg1,1.0);
  arg1_v.val_.d_ = 1.0;
  arg1_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2
            << " and third argument " << arg3;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2, arg3), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1_v,arg2,arg3);
    EXPECT_TRUE(boost::math::isnan(res.d_.d_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg1_v.val_.val_);
    VEC g;
    res.d_.d_.grad(x,g);
    
    ASSERT_EQ(1U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
  }
}

template <typename F>
void test_nan_d_ffv_d1(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg2_v(arg2,1.0);
  arg2_v.val_.d_ = 1.0;
  arg2_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1
            << " second argument " << arg2_v
            << " and third argument " << arg3;

  if (throws)
    EXPECT_THROW(f(arg1, arg2_v, arg3), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1,arg2_v,arg3);
    EXPECT_TRUE(boost::math::isnan(res.val_.val_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg2_v.val_.val_);
    VEC g;
    res.val_.val_.grad(x,g);
    
    ASSERT_EQ(1U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_d_ffv_d2(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg2_v(arg2,1.0);
  arg2_v.val_.d_ = 1.0;
  arg2_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1
            << " second argument " << arg2_v
            << " and third argument " << arg3;

  if (throws)
    EXPECT_THROW(f(arg1, arg2_v, arg3), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1,arg2_v,arg3);
    EXPECT_TRUE(boost::math::isnan(res.d_.val_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg2_v.val_.val_);
    VEC g;
    res.d_.val_.grad(x,g);
    
    ASSERT_EQ(1U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_d_ffv_d3(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg2_v(arg2,1.0);
  arg2_v.val_.d_ = 1.0;
  arg2_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1
            << " second argument " << arg2_v
            << " and third argument " << arg3;

  if (throws)
    EXPECT_THROW(f(arg1, arg2_v, arg3), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1,arg2_v,arg3);
    EXPECT_TRUE(boost::math::isnan(res.val_.d_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg2_v.val_.val_);
    VEC g;
    res.val_.d_.grad(x,g);
    
    ASSERT_EQ(1U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_d_ffv_d4(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg2_v(arg2,1.0);
  arg2_v.val_.d_ = 1.0;
  arg2_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1
            << " second argument " << arg2_v
            << " and third argument " << arg3;

  if (throws)
    EXPECT_THROW(f(arg1, arg2_v, arg3), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1,arg2_v,arg3);
    EXPECT_TRUE(boost::math::isnan(res.d_.d_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg2_v.val_.val_);
    VEC g;
    res.d_.d_.grad(x,g);
    
    ASSERT_EQ(1U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
  }
}

template <typename F>
void test_nan_d_d_ffv1(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg3_v(arg3,1.0);
  arg3_v.val_.d_ = 1.0;
  arg3_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1
            << " second argument " << arg2
            << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1, arg2, arg3_v), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1,arg2,arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.val_.val_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg3_v.val_.val_);
    VEC g;
    res.val_.val_.grad(x,g);
    
    ASSERT_EQ(1U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_d_d_ffv2(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg3_v(arg3,1.0);
  arg3_v.val_.d_ = 1.0;
  arg3_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1
            << " second argument " << arg2
            << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1, arg2, arg3_v), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1,arg2,arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.d_.val_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg3_v.val_.val_);
    VEC g;
    res.d_.val_.grad(x,g);
    
    ASSERT_EQ(1U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_d_d_ffv3(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg3_v(arg3,1.0);
  arg3_v.val_.d_ = 1.0;
  arg3_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1
            << " second argument " << arg2
            << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1, arg2, arg3_v), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1,arg2,arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.val_.d_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg3_v.val_.val_);
    VEC g;
    res.val_.d_.grad(x,g);
    
    ASSERT_EQ(1U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_d_d_ffv4(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  fvar<fvar<var> > arg3_v(arg3,1.0);
  arg3_v.val_.d_ = 1.0;
  arg3_v.d_.d_ = 1.0;

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1
            << " second argument " << arg2
            << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1, arg2, arg3_v), std::domain_error) << fail_msg.str();
  else {
    fvar<fvar<var> > res = f(arg1,arg2,arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.d_.d_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg3_v.val_.val_);
    VEC g;
    res.d_.d_.grad(x,g);
    
    ASSERT_EQ(1U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
  }
}

template <typename F>
void test_nan(const F& f,
              const double& arg1,
              const double& arg2,
              const double& arg3,
              const bool& throws) {

  double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::VectorXd arg1_vec(7);
  Eigen::VectorXd arg2_vec(7);
  Eigen::VectorXd arg3_vec(7);
  arg1_vec << nan, arg1, arg1, nan, nan, arg1, nan;
  arg2_vec << arg2, nan, arg2, nan, arg2, nan, nan;
  arg3_vec << arg3, arg3, nan, arg3, nan, nan, nan;

  for (int i = 0; i < arg1_vec.size() ;i++) {
    test_nan_fd(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_ffd(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_fv_fv_fv1(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_fv_fv_fv2(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_d_fv_fv1(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_d_fv_fv2(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_fv_d_fv1(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_fv_d_fv2(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_fv_fv_d1(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_fv_fv_d2(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_d_d_fv1(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_d_d_fv2(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_d_fv_d1(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_d_fv_d2(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_fv_d_d1(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_fv_d_d2(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_ffv_ffv_ffv1(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_ffv_ffv_ffv2(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_d_ffv_ffv1(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_d_ffv_ffv2(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_ffv_d_ffv1(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_ffv_d_ffv2(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_ffv_ffv_d1(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_ffv_ffv_d2(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_d_d_ffv1(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_d_d_ffv2(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_d_ffv_d1(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_d_ffv_d2(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_ffv_d_d1(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_ffv_d_d2(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_ffv_ffv_ffv3(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_ffv_ffv_ffv4(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_d_ffv_ffv3(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_d_ffv_ffv4(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_ffv_d_ffv3(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_ffv_d_ffv4(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_ffv_ffv_d3(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_ffv_ffv_d4(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_d_d_ffv3(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_d_d_ffv4(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_d_ffv_d3(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_d_ffv_d4(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_ffv_d_d3(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
    test_nan_ffv_d_d4(f, arg1_vec(i), arg2_vec(i), arg3_vec(i), throws);
  }
}

#endif

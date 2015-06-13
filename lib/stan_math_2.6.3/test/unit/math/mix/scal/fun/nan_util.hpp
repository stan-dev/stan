#ifndef TEST__UNIT_AGRAD_MIX__NAN_UTIL_HPP
#define TEST__UNIT_AGRAD_MIX__NAN_UTIL_HPP

#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/rev/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/rev/scal/fun/abs.hpp>
#include <gtest/gtest.h>

template <typename F>
void test_nan_fv1(const F& f,
                  const double& arg1,
                  const bool& throws) {
  using stan::math::var;
  stan::math::fvar<var> arg1_v = arg1;

  if (throws)
    EXPECT_THROW(f(arg1_v), std::domain_error);
  else {
    stan::math::fvar<var> res = f(arg1_v);
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
  using stan::math::var;
  stan::math::fvar<var> arg1_v = arg1;

  if (throws)
    EXPECT_THROW(f(arg1_v), std::domain_error);
  else {
    stan::math::fvar<var> res = f(arg1_v);
    EXPECT_TRUE(boost::math::isnan(res.d_.val()));

    AVEC x = createAVEC(arg1_v.val_);
    VEC g;
    res.d_.grad(x,g);
    
    ASSERT_EQ(1U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
  }
}
template <typename F>
void test_nan_ffv1(const F& f,
                   const double& arg1,
                   const bool& throws) {
  using stan::math::var;
  using stan::math::fvar;
  stan::math::fvar<fvar<var> > arg1_v = arg1;
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
  using stan::math::var;
  using stan::math::fvar;
  stan::math::fvar<fvar<var> > arg1_v(fvar<var>(arg1,1.0),fvar<var>(1.0,1.0));

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
  using stan::math::var;
  using stan::math::fvar;
  stan::math::fvar<fvar<var> > arg1_v = arg1;
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
  using stan::math::var;
  using stan::math::fvar;
  stan::math::fvar<fvar<var> > arg1_v (fvar<var>(arg1,1.0), fvar<var>(1.0,1.0));

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
void test_nan_mix(const F& f,
                  const bool& throws) {

  double nan = std::numeric_limits<double>::quiet_NaN();
  test_nan_fv1(f, nan, throws);
  test_nan_fv2(f, nan, throws);
  test_nan_ffv1(f, nan, throws);
  test_nan_ffv2(f, nan, throws);
  test_nan_ffv3(f, nan, throws);
  test_nan_ffv4(f, nan, throws);
}

template <typename F>
void test_nan_fv_fv1(const F& f,
                    const double& arg1,
                    const double& arg2,
                    const bool& throws) {
  using stan::math::fvar;
  using stan::math::var;
  fvar<var> arg1_v(arg1, 1.0);
  fvar<var> arg2_v(arg2, 1.0);

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2_v), std::domain_error);
  else {
    stan::math::fvar<var> res = f(arg1_v, arg2_v);
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
  using stan::math::fvar;
  using stan::math::var;
  fvar<var> arg1_v(arg1, 1.0);
  fvar<var> arg2_v(arg2, 1.0);

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2_v), std::domain_error);
  else {
    stan::math::fvar<var> res = f(arg1_v, arg2_v);
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
  using stan::math::fvar;
  using stan::math::var;
  fvar<var> arg1_v(arg1, 1.0);

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2), std::domain_error);
  else {
    stan::math::fvar<var> res = f(arg1_v, arg2);
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
  using stan::math::fvar;
  using stan::math::var;
  fvar<var> arg1_v(arg1, 1.0);

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2), std::domain_error);
  else {
    stan::math::fvar<var> res = f(arg1_v, arg2);
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
  using stan::math::fvar;
  using stan::math::var;
  fvar<var> arg2_v(arg2, 1.0);

  if (throws)
    EXPECT_THROW(f(arg1, arg2_v), std::domain_error);
  else {
    stan::math::fvar<var> res = f(arg1, arg2_v);
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
  using stan::math::fvar;
  using stan::math::var;
  fvar<var> arg2_v(arg2, 1.0);

  if (throws)
    EXPECT_THROW(f(arg1, arg2_v), std::domain_error);
  else {
    stan::math::fvar<var> res = f(arg1, arg2_v);
    EXPECT_TRUE(boost::math::isnan(res.d_.val()));

    AVEC x = createAVEC(arg2_v.val_);
    VEC g;
    res.d_.grad(x,g);
    
    ASSERT_EQ(1U,g.size());
    EXPECT_TRUE(boost::math::isnan(g[0]));
  }
}
template <typename F>
void test_nan_ffv_ffv1(const F& f,
                       const double& arg1,
                       const double& arg2,
                       const bool& throws) {
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
void test_nan_mix(const F& f,
                  const double& arg1,
                  const double& arg2,
                  const bool& throws) {

  double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::VectorXd arg1_vec(3);
  Eigen::VectorXd arg2_vec(3);
  arg1_vec << nan, arg1, nan;
  arg2_vec << arg2, nan, nan;
  for (int i = 0 ; i < arg1_vec.size(); i++) {
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
void test_nan_fv_fv_fv1(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::math::fvar;
  using stan::math::var;
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
    stan::math::fvar<var> res = f(arg1_v,arg2_v,arg3_v);
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
  using stan::math::fvar;
  using stan::math::var;
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
    stan::math::fvar<var> res = f(arg1_v,arg2_v,arg3_v);
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
  using stan::math::fvar;
  using stan::math::var;
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
    stan::math::fvar<var> res = f(arg1,arg2_v,arg3_v);
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
  using stan::math::fvar;
  using stan::math::var;
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
    stan::math::fvar<var> res = f(arg1,arg2_v,arg3_v);
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
  using stan::math::fvar;
  using stan::math::var;
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
    stan::math::fvar<var> res = f(arg1_v,arg2,arg3_v);
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
  using stan::math::fvar;
  using stan::math::var;
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
    stan::math::fvar<var> res = f(arg1_v,arg2,arg3_v);
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
  using stan::math::fvar;
  using stan::math::var;
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
    stan::math::fvar<var> res = f(arg1_v,arg2_v,arg3);
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
  using stan::math::fvar;
  using stan::math::var;
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
    stan::math::fvar<var> res = f(arg1_v,arg2_v,arg3);
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
  using stan::math::fvar;
  using stan::math::var;
  fvar<var> arg1_v(arg1,1.0);

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2
            << " and third argument " << arg3;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2, arg3), std::domain_error) << fail_msg.str();
  else {
    stan::math::fvar<var> res = f(arg1_v,arg2,arg3);
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
  using stan::math::fvar;
  using stan::math::var;
  fvar<var> arg1_v(arg1,1.0);

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1_v
            << " second argument " << arg2
            << " and third argument " << arg3;

  if (throws)
    EXPECT_THROW(f(arg1_v, arg2, arg3), std::domain_error) << fail_msg.str();
  else {
    stan::math::fvar<var> res = f(arg1_v,arg2,arg3);
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
  using stan::math::fvar;
  using stan::math::var;
  fvar<var> arg2_v(arg2,1.0);

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1
            << " second argument " << arg2_v
            << " and third argument " << arg3;

  if (throws)
    EXPECT_THROW(f(arg1, arg2_v, arg3), std::domain_error) << fail_msg.str();
  else {
    stan::math::fvar<var> res = f(arg1,arg2_v,arg3);
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
  using stan::math::fvar;
  using stan::math::var;
  fvar<var> arg2_v(arg2,1.0);

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1
            << " second argument " << arg2_v
            << " and third argument " << arg3;

  if (throws)
    EXPECT_THROW(f(arg1, arg2_v, arg3), std::domain_error) << fail_msg.str();
  else {
    stan::math::fvar<var> res = f(arg1,arg2_v,arg3);
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
  using stan::math::fvar;
  using stan::math::var;
  fvar<var> arg3_v(arg3,1.0);

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1
            << " second argument " << arg2
            << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1, arg2, arg3_v), std::domain_error) << fail_msg.str();
  else {
    stan::math::fvar<var> res = f(arg1,arg2,arg3_v);
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
  using stan::math::fvar;
  using stan::math::var;
  fvar<var> arg3_v(arg3,1.0);

  std::ostringstream fail_msg;
  fail_msg << "Failed for "
            << "first argument " << arg1
            << " second argument " << arg2
            << " and third argument " << arg3_v;

  if (throws)
    EXPECT_THROW(f(arg1, arg2, arg3_v), std::domain_error) << fail_msg.str();
  else {
    stan::math::fvar<var> res = f(arg1,arg2,arg3_v);
    EXPECT_TRUE(boost::math::isnan(res.d_.val())) << fail_msg.str();

    AVEC x = createAVEC(arg3_v.val_);
    VEC g;
    res.d_.grad(x,g);
    
    ASSERT_EQ(1U,g.size()) << fail_msg.str();
    EXPECT_TRUE(boost::math::isnan(g[0])) << fail_msg.str();
  }
}
template <typename F>
void test_nan_ffv_ffv_ffv1(const F& f,
                        const double& arg1,
                        const double& arg2,
                        const double& arg3,
                        const bool& throws) {
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
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
void test_nan_mix(const F& f,
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

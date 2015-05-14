#ifndef TEST__UNIT_AGRAD_FWD__NAN_UTIL_HPP
#define TEST__UNIT_AGRAD_FWD__NAN_UTIL_HPP

#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <gtest/gtest.h>

template <typename F>
void test_nan_fd(const F& f,
                 const double& arg1,
                 const bool& throws) {
  stan::math::fvar<double> arg1_v = arg1;
  arg1_v.d_ = 1.0;
  if (throws)
    EXPECT_THROW(f(arg1_v), std::domain_error);
  else {
    EXPECT_TRUE(boost::math::isnan(f(arg1_v).val_));
    EXPECT_TRUE(boost::math::isnan(f(arg1_v).d_));
  }
}

template <typename F>
void test_nan_ffd(const F& f,
                  const double& arg1,
                  const bool& throws) {
  using stan::math::fvar;
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
void test_nan_fwd(const F& f,
                  const bool& throws) {

  double nan = std::numeric_limits<double>::quiet_NaN();
  test_nan_fd(f, nan, throws);
  test_nan_ffd(f, nan, throws);
}

template <typename F>
void test_nan_fd(const F& f,
                 const double& arg1,
                 const double& arg2,
                 const bool& throws) {
  using stan::math::fvar;
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
void test_nan_ffd(const F& f,
                  const double& arg1,
                  const double& arg2,
                  const bool& throws) {
  using stan::math::fvar;
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
void test_nan_fwd(const F& f,
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
  }
}


template <typename F>
void test_nan_fd(const F& f,
                 const double& arg1,
                 const double& arg2,
                 const double& arg3,
                 const bool& throws) {
  using stan::math::fvar;
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
void test_nan_ffd(const F& f,
                 const double& arg1,
                 const double& arg2,
                 const double& arg3,
                 const bool& throws) {
  using stan::math::fvar;
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
void test_nan_fwd(const F& f,
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
  }
}

#endif

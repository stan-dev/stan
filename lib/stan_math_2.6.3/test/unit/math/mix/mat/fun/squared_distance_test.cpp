#include <stan/math/prim/mat/fun/squared_distance.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>

using stan::math::var;
using stan::math::fvar;

TEST(AgradMixMatrixSquaredDistance, vector_fv_vector_fv1) {
  stan::math::vector_fv v1, v2;
  
  v1.resize(3);
  v2.resize(3);
  v1 << 1, 3, -5;
  v2 << 4, -2, -1;
  v1(0).d_ = 1.0;
  v1(1).d_ = 2.0;
  v1(2).d_ = 3.0;
  v2(0).d_ = 4.0;
  v2(1).d_ = 5.0;
  v2(2).d_ = 6.0;
  
  stan::math::fvar<var> a = stan::math::squared_distance(v1, v2);

  EXPECT_FLOAT_EQ(50, a.val_.val());
  EXPECT_FLOAT_EQ(12, a.d_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(v1(0).val_);
  vars.push_back(v1(1).val_);
  vars.push_back(v1(2).val_);
  vars.push_back(v2(0).val_);
  vars.push_back(v2(1).val_);
  vars.push_back(v2(2).val_);

  a.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-6, grads[0]);
  EXPECT_FLOAT_EQ(10, grads[1]);
  EXPECT_FLOAT_EQ(-8, grads[2]);
  EXPECT_FLOAT_EQ(6, grads[3]);
  EXPECT_FLOAT_EQ(-10, grads[4]);
  EXPECT_FLOAT_EQ(8, grads[5]);

  v1.resize(0);
  v2.resize(0);
  EXPECT_FLOAT_EQ(0, stan::math::squared_distance(v1, v2).val_.val());

  v1.resize(1);
  v2.resize(2);
  v1 << 1;
  v2 << 2, 3;
  EXPECT_THROW(stan::math::squared_distance(v1, v2), std::invalid_argument);
}

TEST(AgradMixMatrixSquaredDistance, vector_fv_vector_fv2) {
  stan::math::vector_fv v1, v2;
  
  v1.resize(3);
  v2.resize(3);
  v1 << 1, 3, -5;
  v2 << 4, -2, -1;
  v1(0).d_ = 1.0;
  v1(1).d_ = 2.0;
  v1(2).d_ = 3.0;
  v2(0).d_ = 4.0;
  v2(1).d_ = 5.0;
  v2(2).d_ = 6.0;
  
  stan::math::fvar<var> a = stan::math::squared_distance(v1, v2);

  EXPECT_FLOAT_EQ(50, a.val_.val());
  EXPECT_FLOAT_EQ(12, a.d_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(v1(0).val_);
  vars.push_back(v1(1).val_);
  vars.push_back(v1(2).val_);
  vars.push_back(v2(0).val_);
  vars.push_back(v2(1).val_);
  vars.push_back(v2(2).val_);

  a.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-6, grads[0]);
  EXPECT_FLOAT_EQ(-6, grads[1]);
  EXPECT_FLOAT_EQ(-6, grads[2]);
  EXPECT_FLOAT_EQ(6, grads[3]);
  EXPECT_FLOAT_EQ(6, grads[4]);
  EXPECT_FLOAT_EQ(6, grads[5]);
}

TEST(AgradMixMatrixSquaredDistance, rowvector_fv_vector_fv1) {
  stan::math::row_vector_fv rv;
  stan::math::vector_fv v;
  
  rv.resize(3);
  v.resize(3);
  rv << 1, 3, -5;
  v << 4, -2, -1;
  rv(0).d_ = 1.0;
  rv(1).d_ = 2.0;
  rv(2).d_ = 3.0;
  v(0).d_ = 4.0;
  v(1).d_ = 5.0;
  v(2).d_ = 6.0;

  stan::math::fvar<var> a = stan::math::squared_distance(rv, v);

  EXPECT_FLOAT_EQ(50, a.val_.val());
  EXPECT_FLOAT_EQ(12, a.d_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(rv(0).val_);
  vars.push_back(rv(1).val_);
  vars.push_back(rv(2).val_);
  vars.push_back(v(0).val_);
  vars.push_back(v(1).val_);
  vars.push_back(v(2).val_);

  a.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-6, grads[0]);
  EXPECT_FLOAT_EQ(10, grads[1]);
  EXPECT_FLOAT_EQ(-8, grads[2]);
  EXPECT_FLOAT_EQ(6, grads[3]);
  EXPECT_FLOAT_EQ(-10, grads[4]);
  EXPECT_FLOAT_EQ(8, grads[5]);

  rv.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ(0, stan::math::squared_distance(rv, v).val_.val());

  rv.resize(1);
  v.resize(2);
  rv << 1;
  v << 2, 3;
  EXPECT_THROW(stan::math::squared_distance(rv, v), std::invalid_argument);
}

TEST(AgradMixMatrixSquaredDistance, rowvector_fv_vector_fv2) {
  stan::math::row_vector_fv rv;
  stan::math::vector_fv v;
  
  rv.resize(3);
  v.resize(3);
  rv << 1, 3, -5;
  v << 4, -2, -1;
  rv(0).d_ = 1.0;
  rv(1).d_ = 2.0;
  rv(2).d_ = 3.0;
  v(0).d_ = 4.0;
  v(1).d_ = 5.0;
  v(2).d_ = 6.0;

  stan::math::fvar<var> a = stan::math::squared_distance(rv, v);

  EXPECT_FLOAT_EQ(50, a.val_.val());
  EXPECT_FLOAT_EQ(12, a.d_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(rv(0).val_);
  vars.push_back(rv(1).val_);
  vars.push_back(rv(2).val_);
  vars.push_back(v(0).val_);
  vars.push_back(v(1).val_);
  vars.push_back(v(2).val_);

  a.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-6, grads[0]);
  EXPECT_FLOAT_EQ(-6, grads[1]);
  EXPECT_FLOAT_EQ(-6, grads[2]);
  EXPECT_FLOAT_EQ(6, grads[3]);
  EXPECT_FLOAT_EQ(6, grads[4]);
  EXPECT_FLOAT_EQ(6, grads[5]);
}

TEST(AgradMixMatrixSquaredDistance, vector_fv_rowvector_fv1) {
  stan::math::row_vector_fv rv;
  stan::math::vector_fv v;
  
  rv.resize(3);
  v.resize(3);
  rv << 1, 3, -5;
  v << 4, -2, -1;
  rv(0).d_ = 1.0;
  rv(1).d_ = 2.0;
  rv(2).d_ = 3.0;
  v(0).d_ = 4.0;
  v(1).d_ = 5.0;
  v(2).d_ = 6.0;

  stan::math::fvar<var> a = stan::math::squared_distance(v, rv);

  EXPECT_FLOAT_EQ(50, a.val_.val());
  EXPECT_FLOAT_EQ(12, a.d_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(rv(0).val_);
  vars.push_back(rv(1).val_);
  vars.push_back(rv(2).val_);
  vars.push_back(v(0).val_);
  vars.push_back(v(1).val_);
  vars.push_back(v(2).val_);

  a.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-6, grads[0]);
  EXPECT_FLOAT_EQ(10, grads[1]);
  EXPECT_FLOAT_EQ(-8, grads[2]);
  EXPECT_FLOAT_EQ(6, grads[3]);
  EXPECT_FLOAT_EQ(-10, grads[4]);
  EXPECT_FLOAT_EQ(8, grads[5]);

  v.resize(0);
  rv.resize(0);
  EXPECT_FLOAT_EQ(0, stan::math::squared_distance(v, rv).val_.val());

  v.resize(1);
  rv.resize(2);
  v << 1;
  rv << 2, 3;
  EXPECT_THROW(stan::math::squared_distance(v, rv), std::invalid_argument);
}

TEST(AgradMixMatrixSquaredDistance, vector_fv_rowvector_fv2) {
  stan::math::row_vector_fv rv;
  stan::math::vector_fv v;
  
  rv.resize(3);
  v.resize(3);
  rv << 1, 3, -5;
  v << 4, -2, -1;
  rv(0).d_ = 1.0;
  rv(1).d_ = 2.0;
  rv(2).d_ = 3.0;
  v(0).d_ = 4.0;
  v(1).d_ = 5.0;
  v(2).d_ = 6.0;

  stan::math::fvar<var> a = stan::math::squared_distance(v, rv);

  EXPECT_FLOAT_EQ(50, a.val_.val());
  EXPECT_FLOAT_EQ(12, a.d_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(rv(0).val_);
  vars.push_back(rv(1).val_);
  vars.push_back(rv(2).val_);
  vars.push_back(v(0).val_);
  vars.push_back(v(1).val_);
  vars.push_back(v(2).val_);

  a.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-6, grads[0]);
  EXPECT_FLOAT_EQ(-6, grads[1]);
  EXPECT_FLOAT_EQ(-6, grads[2]);
  EXPECT_FLOAT_EQ(6, grads[3]);
  EXPECT_FLOAT_EQ(6, grads[4]);
  EXPECT_FLOAT_EQ(6, grads[5]);
}
TEST(AgradMixMatrixSquaredDistance, special_values_fv) {
  stan::math::vector_fv v1, v2;
  v1.resize(1);
  v2.resize(1);
  
  v1 << 0;
  v2 << std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(boost::math::isnan(stan::math::squared_distance(v1, v2)));
  EXPECT_TRUE(boost::math::isnan(stan::math::squared_distance(v2, v1)));

  v1 << 0;
  v2 << std::numeric_limits<double>::infinity();
  EXPECT_TRUE(boost::math::isinf(stan::math::squared_distance(v1, v2)));
  EXPECT_TRUE(boost::math::isinf(stan::math::squared_distance(v2, v1)));

  v1 << std::numeric_limits<double>::infinity();
  v2 << std::numeric_limits<double>::infinity();
  EXPECT_TRUE(boost::math::isnan(stan::math::squared_distance(v1, v2)));
  EXPECT_TRUE(boost::math::isnan(stan::math::squared_distance(v2, v1)));

  v1 << -std::numeric_limits<double>::infinity();
  v2 << std::numeric_limits<double>::infinity();
  EXPECT_TRUE(boost::math::isinf(stan::math::squared_distance(v1, v2)));
  EXPECT_TRUE(boost::math::isinf(stan::math::squared_distance(v2, v1)));
}

TEST(AgradMixMatrixSquaredDistance, vector_fv_vector_ffv1) {
  stan::math::vector_ffv v1, v2;
  
  v1.resize(3);
  v2.resize(3);
  v1 << 1, 3, -5;
  v2 << 4, -2, -1;
  v1(0).d_ = 1.0;
  v1(1).d_ = 2.0;
  v1(2).d_ = 3.0;
  v2(0).d_ = 4.0;
  v2(1).d_ = 5.0;
  v2(2).d_ = 6.0;
  
  stan::math::fvar<fvar<var> > a = stan::math::squared_distance(v1, v2);

  EXPECT_FLOAT_EQ(50, a.val_.val_.val());
  EXPECT_FLOAT_EQ(12, a.d_.val_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(v1(0).val_.val_);
  vars.push_back(v1(1).val_.val_);
  vars.push_back(v1(2).val_.val_);
  vars.push_back(v2(0).val_.val_);
  vars.push_back(v2(1).val_.val_);
  vars.push_back(v2(2).val_.val_);

  a.val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-6, grads[0]);
  EXPECT_FLOAT_EQ(10, grads[1]);
  EXPECT_FLOAT_EQ(-8, grads[2]);
  EXPECT_FLOAT_EQ(6, grads[3]);
  EXPECT_FLOAT_EQ(-10, grads[4]);
  EXPECT_FLOAT_EQ(8, grads[5]);

  v1.resize(0);
  v2.resize(0);
  EXPECT_FLOAT_EQ(0, stan::math::squared_distance(v1, v2).val_.val_.val());

  v1.resize(1);
  v2.resize(2);
  v1 << 1;
  v2 << 2, 3;
  EXPECT_THROW(stan::math::squared_distance(v1, v2), std::invalid_argument);
}

TEST(AgradMixMatrixSquaredDistance, vector_fv_vector_ffv2) {
  stan::math::vector_ffv v1, v2;
  
  v1.resize(3);
  v2.resize(3);
  v1 << 1, 3, -5;
  v2 << 4, -2, -1;
  v1(0).d_ = 1.0;
  v1(1).d_ = 2.0;
  v1(2).d_ = 3.0;
  v2(0).d_ = 4.0;
  v2(1).d_ = 5.0;
  v2(2).d_ = 6.0;
  v1(0).val_.d_ = 1.0;
  v1(1).val_.d_ = 2.0;
  v1(2).val_.d_ = 3.0;
  v2(0).val_.d_ = 4.0;
  v2(1).val_.d_ = 5.0;
  v2(2).val_.d_ = 6.0;
  
  stan::math::fvar<fvar<var> > a = stan::math::squared_distance(v1, v2);

  EXPECT_FLOAT_EQ(50, a.val_.val_.val());
  EXPECT_FLOAT_EQ(12, a.d_.val_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(v1(0).val_.val_);
  vars.push_back(v1(1).val_.val_);
  vars.push_back(v1(2).val_.val_);
  vars.push_back(v2(0).val_.val_);
  vars.push_back(v2(1).val_.val_);
  vars.push_back(v2(2).val_.val_);

  a.val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-6, grads[0]);
  EXPECT_FLOAT_EQ(-6, grads[1]);
  EXPECT_FLOAT_EQ(-6, grads[2]);
  EXPECT_FLOAT_EQ(6, grads[3]);
  EXPECT_FLOAT_EQ(6, grads[4]);
  EXPECT_FLOAT_EQ(6, grads[5]);
}

TEST(AgradMixMatrixSquaredDistance, vector_fv_vector_ffv3) {
  stan::math::vector_ffv v1, v2;
  
  v1.resize(3);
  v2.resize(3);
  v1 << 1, 3, -5;
  v2 << 4, -2, -1;
  v1(0).d_ = 1.0;
  v1(1).d_ = 2.0;
  v1(2).d_ = 3.0;
  v2(0).d_ = 4.0;
  v2(1).d_ = 5.0;
  v2(2).d_ = 6.0;
  v1(0).val_.d_ = 1.0;
  v1(1).val_.d_ = 2.0;
  v1(2).val_.d_ = 3.0;
  v2(0).val_.d_ = 4.0;
  v2(1).val_.d_ = 5.0;
  v2(2).val_.d_ = 6.0;
  
  stan::math::fvar<fvar<var> > a = stan::math::squared_distance(v1, v2);

  EXPECT_FLOAT_EQ(50, a.val_.val_.val());
  EXPECT_FLOAT_EQ(12, a.d_.val_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(v1(0).val_.val_);
  vars.push_back(v1(1).val_.val_);
  vars.push_back(v1(2).val_.val_);
  vars.push_back(v2(0).val_.val_);
  vars.push_back(v2(1).val_.val_);
  vars.push_back(v2(2).val_.val_);

  a.d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-6, grads[0]);
  EXPECT_FLOAT_EQ(-6, grads[1]);
  EXPECT_FLOAT_EQ(-6, grads[2]);
  EXPECT_FLOAT_EQ(6, grads[3]);
  EXPECT_FLOAT_EQ(6, grads[4]);
  EXPECT_FLOAT_EQ(6, grads[5]);
}

TEST(AgradMixMatrixSquaredDistance, vector_fv_vector_ffv4) {
  stan::math::vector_ffv v1, v2;
  
  v1.resize(3);
  v2.resize(3);
  v1 << 1, 3, -5;
  v2 << 4, -2, -1;
  v1(0).d_ = 1.0;
  v1(1).d_ = 2.0;
  v1(2).d_ = 3.0;
  v2(0).d_ = 4.0;
  v2(1).d_ = 5.0;
  v2(2).d_ = 6.0;
  v1(0).val_.d_ = 1.0;
  v1(1).val_.d_ = 2.0;
  v1(2).val_.d_ = 3.0;
  v2(0).val_.d_ = 4.0;
  v2(1).val_.d_ = 5.0;
  v2(2).val_.d_ = 6.0;
  
  stan::math::fvar<fvar<var> > a = stan::math::squared_distance(v1, v2);

  EXPECT_FLOAT_EQ(50, a.val_.val_.val());
  EXPECT_FLOAT_EQ(12, a.d_.val_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(v1(0).val_.val_);
  vars.push_back(v1(1).val_.val_);
  vars.push_back(v1(2).val_.val_);
  vars.push_back(v2(0).val_.val_);
  vars.push_back(v2(1).val_.val_);
  vars.push_back(v2(2).val_.val_);

  a.d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(-0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(-0, grads[3]);
  EXPECT_FLOAT_EQ(0, grads[4]);
  EXPECT_FLOAT_EQ(-0, grads[5]);
}

TEST(AgradMixMatrixSquaredDistance, rowvector_fv_vector_ffv1) {
  stan::math::row_vector_ffv rv;
  stan::math::vector_ffv v;
  
  rv.resize(3);
  v.resize(3);
  rv << 1, 3, -5;
  v << 4, -2, -1;
  rv(0).d_ = 1.0;
  rv(1).d_ = 2.0;
  rv(2).d_ = 3.0;
  v(0).d_ = 4.0;
  v(1).d_ = 5.0;
  v(2).d_ = 6.0;

  stan::math::fvar<fvar<var> > a = stan::math::squared_distance(rv, v);

  EXPECT_FLOAT_EQ(50, a.val_.val_.val());
  EXPECT_FLOAT_EQ(12, a.d_.val_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(rv(0).val_.val_);
  vars.push_back(rv(1).val_.val_);
  vars.push_back(rv(2).val_.val_);
  vars.push_back(v(0).val_.val_);
  vars.push_back(v(1).val_.val_);
  vars.push_back(v(2).val_.val_);

  a.val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-6, grads[0]);
  EXPECT_FLOAT_EQ(10, grads[1]);
  EXPECT_FLOAT_EQ(-8, grads[2]);
  EXPECT_FLOAT_EQ(6, grads[3]);
  EXPECT_FLOAT_EQ(-10, grads[4]);
  EXPECT_FLOAT_EQ(8, grads[5]);

  rv.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ(0, stan::math::squared_distance(rv, v).val_.val_.val());

  rv.resize(1);
  v.resize(2);
  rv << 1;
  v << 2, 3;
  EXPECT_THROW(stan::math::squared_distance(rv, v), std::invalid_argument);
}

TEST(AgradMixMatrixSquaredDistance, rowvector_fv_vector_ffv2) {
  stan::math::row_vector_ffv rv;
  stan::math::vector_ffv v;
  
  rv.resize(3);
  v.resize(3);
  rv << 1, 3, -5;
  v << 4, -2, -1;
  rv(0).d_ = 1.0;
  rv(1).d_ = 2.0;
  rv(2).d_ = 3.0;
  v(0).d_ = 4.0;
  v(1).d_ = 5.0;
  v(2).d_ = 6.0;
  rv(0).val_.d_ = 1.0;
  rv(1).val_.d_ = 2.0;
  rv(2).val_.d_ = 3.0;
  v(0).val_.d_ = 4.0;
  v(1).val_.d_ = 5.0;
  v(2).val_.d_ = 6.0;

  stan::math::fvar<fvar<var> > a = stan::math::squared_distance(rv, v);

  EXPECT_FLOAT_EQ(50, a.val_.val_.val());
  EXPECT_FLOAT_EQ(12, a.d_.val_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(rv(0).val_.val_);
  vars.push_back(rv(1).val_.val_);
  vars.push_back(rv(2).val_.val_);
  vars.push_back(v(0).val_.val_);
  vars.push_back(v(1).val_.val_);
  vars.push_back(v(2).val_.val_);

  a.val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-6, grads[0]);
  EXPECT_FLOAT_EQ(-6, grads[1]);
  EXPECT_FLOAT_EQ(-6, grads[2]);
  EXPECT_FLOAT_EQ(6, grads[3]);
  EXPECT_FLOAT_EQ(6, grads[4]);
  EXPECT_FLOAT_EQ(6, grads[5]);
}
TEST(AgradMixMatrixSquaredDistance, rowvector_fv_vector_ffv3) {
  stan::math::row_vector_ffv rv;
  stan::math::vector_ffv v;
  
  rv.resize(3);
  v.resize(3);
  rv << 1, 3, -5;
  v << 4, -2, -1;
  rv(0).d_ = 1.0;
  rv(1).d_ = 2.0;
  rv(2).d_ = 3.0;
  v(0).d_ = 4.0;
  v(1).d_ = 5.0;
  v(2).d_ = 6.0;
  rv(0).val_.d_ = 1.0;
  rv(1).val_.d_ = 2.0;
  rv(2).val_.d_ = 3.0;
  v(0).val_.d_ = 4.0;
  v(1).val_.d_ = 5.0;
  v(2).val_.d_ = 6.0;

  stan::math::fvar<fvar<var> > a = stan::math::squared_distance(rv, v);

  EXPECT_FLOAT_EQ(50, a.val_.val_.val());
  EXPECT_FLOAT_EQ(12, a.d_.val_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(rv(0).val_.val_);
  vars.push_back(rv(1).val_.val_);
  vars.push_back(rv(2).val_.val_);
  vars.push_back(v(0).val_.val_);
  vars.push_back(v(1).val_.val_);
  vars.push_back(v(2).val_.val_);

  a.d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-6, grads[0]);
  EXPECT_FLOAT_EQ(-6, grads[1]);
  EXPECT_FLOAT_EQ(-6, grads[2]);
  EXPECT_FLOAT_EQ(6, grads[3]);
  EXPECT_FLOAT_EQ(6, grads[4]);
  EXPECT_FLOAT_EQ(6, grads[5]);
}
TEST(AgradMixMatrixSquaredDistance, rowvector_fv_vector_ffv4) {
  stan::math::row_vector_ffv rv;
  stan::math::vector_ffv v;
  
  rv.resize(3);
  v.resize(3);
  rv << 1, 3, -5;
  v << 4, -2, -1;
  rv(0).d_ = 1.0;
  rv(1).d_ = 2.0;
  rv(2).d_ = 3.0;
  v(0).d_ = 4.0;
  v(1).d_ = 5.0;
  v(2).d_ = 6.0;
  rv(0).val_.d_ = 1.0;
  rv(1).val_.d_ = 2.0;
  rv(2).val_.d_ = 3.0;
  v(0).val_.d_ = 4.0;
  v(1).val_.d_ = 5.0;
  v(2).val_.d_ = 6.0;

  stan::math::fvar<fvar<var> > a = stan::math::squared_distance(rv, v);

  EXPECT_FLOAT_EQ(50, a.val_.val_.val());
  EXPECT_FLOAT_EQ(12, a.d_.val_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(rv(0).val_.val_);
  vars.push_back(rv(1).val_.val_);
  vars.push_back(rv(2).val_.val_);
  vars.push_back(v(0).val_.val_);
  vars.push_back(v(1).val_.val_);
  vars.push_back(v(2).val_.val_);

  a.d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(-0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(-0, grads[3]);
  EXPECT_FLOAT_EQ(0, grads[4]);
  EXPECT_FLOAT_EQ(-0, grads[5]);
}

TEST(AgradMixMatrixSquaredDistance, vector_fv_rowvector_ffv1) {
  stan::math::row_vector_ffv rv;
  stan::math::vector_ffv v;
  
  rv.resize(3);
  v.resize(3);
  rv << 1, 3, -5;
  v << 4, -2, -1;
  rv(0).d_ = 1.0;
  rv(1).d_ = 2.0;
  rv(2).d_ = 3.0;
  v(0).d_ = 4.0;
  v(1).d_ = 5.0;
  v(2).d_ = 6.0;

  stan::math::fvar<fvar<var> > a = stan::math::squared_distance(v, rv);

  EXPECT_FLOAT_EQ(50, a.val_.val_.val());
  EXPECT_FLOAT_EQ(12, a.d_.val_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(rv(0).val_.val_);
  vars.push_back(rv(1).val_.val_);
  vars.push_back(rv(2).val_.val_);
  vars.push_back(v(0).val_.val_);
  vars.push_back(v(1).val_.val_);
  vars.push_back(v(2).val_.val_);

  a.val_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-6, grads[0]);
  EXPECT_FLOAT_EQ(10, grads[1]);
  EXPECT_FLOAT_EQ(-8, grads[2]);
  EXPECT_FLOAT_EQ(6, grads[3]);
  EXPECT_FLOAT_EQ(-10, grads[4]);
  EXPECT_FLOAT_EQ(8, grads[5]);

  v.resize(0);
  rv.resize(0);
  EXPECT_FLOAT_EQ(0, stan::math::squared_distance(v, rv).val_.val_.val());

  v.resize(1);
  rv.resize(2);
  v << 1;
  rv << 2, 3;
  EXPECT_THROW(stan::math::squared_distance(v, rv), std::invalid_argument);
}

TEST(AgradMixMatrixSquaredDistance, vector_fv_rowvector_ffv2) {
  stan::math::row_vector_ffv rv;
  stan::math::vector_ffv v;
  
  rv.resize(3);
  v.resize(3);
  rv << 1, 3, -5;
  v << 4, -2, -1;
  rv(0).d_ = 1.0;
  rv(1).d_ = 2.0;
  rv(2).d_ = 3.0;
  v(0).d_ = 4.0;
  v(1).d_ = 5.0;
  v(2).d_ = 6.0;
  rv(0).val_.d_ = 1.0;
  rv(1).val_.d_ = 2.0;
  rv(2).val_.d_ = 3.0;
  v(0).val_.d_ = 4.0;
  v(1).val_.d_ = 5.0;
  v(2).val_.d_ = 6.0;


  stan::math::fvar<fvar<var> > a = stan::math::squared_distance(v, rv);

  EXPECT_FLOAT_EQ(50, a.val_.val_.val());
  EXPECT_FLOAT_EQ(12, a.d_.val_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(rv(0).val_.val_);
  vars.push_back(rv(1).val_.val_);
  vars.push_back(rv(2).val_.val_);
  vars.push_back(v(0).val_.val_);
  vars.push_back(v(1).val_.val_);
  vars.push_back(v(2).val_.val_);

  a.d_.val_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-6, grads[0]);
  EXPECT_FLOAT_EQ(-6, grads[1]);
  EXPECT_FLOAT_EQ(-6, grads[2]);
  EXPECT_FLOAT_EQ(6, grads[3]);
  EXPECT_FLOAT_EQ(6, grads[4]);
  EXPECT_FLOAT_EQ(6, grads[5]);
}
TEST(AgradMixMatrixSquaredDistance, vector_fv_rowvector_ffv3) {
  stan::math::row_vector_ffv rv;
  stan::math::vector_ffv v;
  
  rv.resize(3);
  v.resize(3);
  rv << 1, 3, -5;
  v << 4, -2, -1;
  rv(0).d_ = 1.0;
  rv(1).d_ = 2.0;
  rv(2).d_ = 3.0;
  v(0).d_ = 4.0;
  v(1).d_ = 5.0;
  v(2).d_ = 6.0;
  rv(0).val_.d_ = 1.0;
  rv(1).val_.d_ = 2.0;
  rv(2).val_.d_ = 3.0;
  v(0).val_.d_ = 4.0;
  v(1).val_.d_ = 5.0;
  v(2).val_.d_ = 6.0;


  stan::math::fvar<fvar<var> > a = stan::math::squared_distance(v, rv);

  EXPECT_FLOAT_EQ(50, a.val_.val_.val());
  EXPECT_FLOAT_EQ(12, a.d_.val_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(rv(0).val_.val_);
  vars.push_back(rv(1).val_.val_);
  vars.push_back(rv(2).val_.val_);
  vars.push_back(v(0).val_.val_);
  vars.push_back(v(1).val_.val_);
  vars.push_back(v(2).val_.val_);

  a.val_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(-6, grads[0]);
  EXPECT_FLOAT_EQ(-6, grads[1]);
  EXPECT_FLOAT_EQ(-6, grads[2]);
  EXPECT_FLOAT_EQ(6, grads[3]);
  EXPECT_FLOAT_EQ(6, grads[4]);
  EXPECT_FLOAT_EQ(6, grads[5]);
}
TEST(AgradMixMatrixSquaredDistance, vector_fv_rowvector_ffv4) {
  stan::math::row_vector_ffv rv;
  stan::math::vector_ffv v;
  
  rv.resize(3);
  v.resize(3);
  rv << 1, 3, -5;
  v << 4, -2, -1;
  rv(0).d_ = 1.0;
  rv(1).d_ = 2.0;
  rv(2).d_ = 3.0;
  v(0).d_ = 4.0;
  v(1).d_ = 5.0;
  v(2).d_ = 6.0;
  rv(0).val_.d_ = 1.0;
  rv(1).val_.d_ = 2.0;
  rv(2).val_.d_ = 3.0;
  v(0).val_.d_ = 4.0;
  v(1).val_.d_ = 5.0;
  v(2).val_.d_ = 6.0;


  stan::math::fvar<fvar<var> > a = stan::math::squared_distance(v, rv);

  EXPECT_FLOAT_EQ(50, a.val_.val_.val());
  EXPECT_FLOAT_EQ(12, a.d_.val_.val());

  std::vector<double> grads;
  std::vector<var> vars;
  vars.push_back(rv(0).val_.val_);
  vars.push_back(rv(1).val_.val_);
  vars.push_back(rv(2).val_.val_);
  vars.push_back(v(0).val_.val_);
  vars.push_back(v(1).val_.val_);
  vars.push_back(v(2).val_.val_);

  a.d_.d_.grad(vars, grads);
  EXPECT_FLOAT_EQ(0, grads[0]);
  EXPECT_FLOAT_EQ(-0, grads[1]);
  EXPECT_FLOAT_EQ(0, grads[2]);
  EXPECT_FLOAT_EQ(-0, grads[3]);
  EXPECT_FLOAT_EQ(0, grads[4]);
  EXPECT_FLOAT_EQ(-0, grads[5]);
}
TEST(AgradMixMatrixSquaredDistance, special_values_ffv) {
  stan::math::vector_ffv v1, v2;
  v1.resize(1);
  v2.resize(1);
  
  v1 << 0;
  v2 << std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(boost::math::isnan(stan::math::squared_distance(v1, v2)));
  EXPECT_TRUE(boost::math::isnan(stan::math::squared_distance(v2, v1)));

  v1 << 0;
  v2 << std::numeric_limits<double>::infinity();
  EXPECT_TRUE(boost::math::isinf(stan::math::squared_distance(v1, v2)));
  EXPECT_TRUE(boost::math::isinf(stan::math::squared_distance(v2, v1)));

  v1 << std::numeric_limits<double>::infinity();
  v2 << std::numeric_limits<double>::infinity();
  EXPECT_TRUE(boost::math::isnan(stan::math::squared_distance(v1, v2)));
  EXPECT_TRUE(boost::math::isnan(stan::math::squared_distance(v2, v1)));

  v1 << -std::numeric_limits<double>::infinity();
  v2 << std::numeric_limits<double>::infinity();
  EXPECT_TRUE(boost::math::isinf(stan::math::squared_distance(v1, v2)));
  EXPECT_TRUE(boost::math::isinf(stan::math::squared_distance(v2, v1)));
}

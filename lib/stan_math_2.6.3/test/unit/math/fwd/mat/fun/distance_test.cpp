#include <stan/math/prim/mat/fun/distance.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/math/fwd/scal/fun/sqrt.hpp>
#include <stan/math/fwd/core.hpp>

using stan::math::fvar;

TEST(AgradFwdMatrixDistance, vector_fd_vector_fd) {
  stan::math::vector_fd v1, v2;
  
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
  
  stan::math::fvar<double> a = stan::math::distance(v1, v2);

  EXPECT_FLOAT_EQ(7.071068, a.val_);
  EXPECT_FLOAT_EQ(0.84852815, a.d_);

  v1.resize(0);
  v2.resize(0);
  EXPECT_FLOAT_EQ(0, stan::math::distance(v1, v2).val_);

  v1.resize(1);
  v2.resize(2);
  v1 << 1;
  v2 << 2, 3;
  EXPECT_THROW(stan::math::distance(v1, v2), std::invalid_argument);
}

TEST(AgradFwdMatrixDistance, rowvector_fd_vector_fd) {
  stan::math::row_vector_fd rv;
  stan::math::vector_fd v;
  
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

  stan::math::fvar<double> a = stan::math::distance(rv, v);

  EXPECT_FLOAT_EQ(7.071068, a.val_);
  EXPECT_FLOAT_EQ(0.84852815, a.d_);

  rv.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ(0, stan::math::distance(rv, v).val_);

  rv.resize(1);
  v.resize(2);
  rv << 1;
  v << 2, 3;
  EXPECT_THROW(stan::math::distance(rv, v), std::invalid_argument);
}

TEST(AgradFwdMatrixDistance, vector_fd_rowvector_fd) {
  stan::math::row_vector_fd rv;
  stan::math::vector_fd v;
  
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

  stan::math::fvar<double> a = stan::math::distance(v, rv);

  EXPECT_FLOAT_EQ(7.071068, a.val_);
  EXPECT_FLOAT_EQ(0.84852815, a.d_);

  v.resize(0);
  rv.resize(0);
  EXPECT_FLOAT_EQ(0, stan::math::distance(v, rv).val_);

  v.resize(1);
  rv.resize(2);
  v << 1;
  rv << 2, 3;
  EXPECT_THROW(stan::math::distance(v, rv), std::invalid_argument);
}

TEST(AgradFwdMatrixDistance, special_values_fd) {
  stan::math::vector_fd v1, v2;
  v1.resize(1);
  v2.resize(1);
  
  v1 << 0;
  v2 << std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(boost::math::isnan(stan::math::distance(v1, v2)));
  EXPECT_TRUE(boost::math::isnan(stan::math::distance(v2, v1)));

  v1 << 0;
  v2 << std::numeric_limits<double>::infinity();
  EXPECT_TRUE(boost::math::isinf(stan::math::distance(v1, v2)));
  EXPECT_TRUE(boost::math::isinf(stan::math::distance(v2, v1)));

  v1 << std::numeric_limits<double>::infinity();
  v2 << std::numeric_limits<double>::infinity();
  EXPECT_TRUE(boost::math::isnan(stan::math::distance(v1, v2)));
  EXPECT_TRUE(boost::math::isnan(stan::math::distance(v2, v1)));

  v1 << -std::numeric_limits<double>::infinity();
  v2 << std::numeric_limits<double>::infinity();
  EXPECT_TRUE(boost::math::isinf(stan::math::distance(v1, v2)));
  EXPECT_TRUE(boost::math::isinf(stan::math::distance(v2, v1)));
}

TEST(AgradFwdMatrixDistance, vector_ffd_vector_ffd) {
  stan::math::vector_ffd v1, v2;
  
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
  
  stan::math::fvar<fvar<double> > a = stan::math::distance(v1, v2);

  EXPECT_FLOAT_EQ(7.071068, a.val_.val_);
  EXPECT_FLOAT_EQ(0.84852815, a.d_.val_);

  v1.resize(0);
  v2.resize(0);
  EXPECT_FLOAT_EQ(0, stan::math::distance(v1, v2).val_.val_);

  v1.resize(1);
  v2.resize(2);
  v1 << 1;
  v2 << 2, 3;
  EXPECT_THROW(stan::math::distance(v1, v2), std::invalid_argument);
}

TEST(AgradFwdMatrixDistance, rowvector_ffd_vector_ffd) {
  stan::math::row_vector_ffd rv;
  stan::math::vector_ffd v;
  
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

  stan::math::fvar<fvar<double> > a = stan::math::distance(rv, v);

  EXPECT_FLOAT_EQ(7.071068, a.val_.val_);
  EXPECT_FLOAT_EQ(0.84852815, a.d_.val_);

  rv.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ(0, stan::math::distance(rv, v).val_.val_);

  rv.resize(1);
  v.resize(2);
  rv << 1;
  v << 2, 3;
  EXPECT_THROW(stan::math::distance(rv, v), std::invalid_argument);
}

TEST(AgradFwdMatrixDistance, vector_ffd_rowvector_ffd) {
  stan::math::row_vector_ffd rv;
  stan::math::vector_ffd v;
  
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

  stan::math::fvar<fvar<double> > a = stan::math::distance(v, rv);

  EXPECT_FLOAT_EQ(7.071068, a.val_.val_);
  EXPECT_FLOAT_EQ(0.84852815, a.d_.val_);

  v.resize(0);
  rv.resize(0);
  EXPECT_FLOAT_EQ(0, stan::math::distance(v, rv).val_.val_);

  v.resize(1);
  rv.resize(2);
  v << 1;
  rv << 2, 3;
  EXPECT_THROW(stan::math::distance(v, rv), std::invalid_argument);
}

TEST(AgradFwdMatrixDistance, special_values_ffd) {
  stan::math::vector_ffd v1, v2;
  v1.resize(1);
  v2.resize(1);
  
  v1 << 0;
  v2 << std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(boost::math::isnan(stan::math::distance(v1, v2)));
  EXPECT_TRUE(boost::math::isnan(stan::math::distance(v2, v1)));

  v1 << 0;
  v2 << std::numeric_limits<double>::infinity();
  EXPECT_TRUE(boost::math::isinf(stan::math::distance(v1, v2)));
  EXPECT_TRUE(boost::math::isinf(stan::math::distance(v2, v1)));

  v1 << std::numeric_limits<double>::infinity();
  v2 << std::numeric_limits<double>::infinity();
  EXPECT_TRUE(boost::math::isnan(stan::math::distance(v1, v2)));
  EXPECT_TRUE(boost::math::isnan(stan::math::distance(v2, v1)));

  v1 << -std::numeric_limits<double>::infinity();
  v2 << std::numeric_limits<double>::infinity();
  EXPECT_TRUE(boost::math::isinf(stan::math::distance(v1, v2)));
  EXPECT_TRUE(boost::math::isinf(stan::math::distance(v2, v1)));
}

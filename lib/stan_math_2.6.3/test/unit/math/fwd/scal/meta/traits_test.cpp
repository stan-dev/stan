#include <gtest/gtest.h>
#include <boost/type_traits.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/prim/scal/meta/error_index.hpp>
#include <stan/math/prim/scal/meta/is_constant.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/meta/contains_nonconstant_struct.hpp>
#include <stan/math/prim/scal/meta/is_vector.hpp>
#include <stan/math/prim/scal/meta/contains_vector.hpp>
#include <stan/math/prim/scal/meta/scalar_type.hpp>
#include <stan/math/prim/scal/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/length.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/scal/meta/length.hpp>
#include <stan/math/prim/scal/meta/size_of.hpp>
#include <stan/math/prim/scal/meta/max_size.hpp>
#include <stan/math/prim/scal/meta/VectorView.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <stan/math/prim/scal/meta/is_fvar.hpp>
#include <stan/math/prim/scal/meta/is_var.hpp>
#include <stan/math/prim/scal/meta/partials_type.hpp>
#include <stan/math/prim/scal/meta/contains_fvar.hpp>
#include <stan/math/prim/scal/meta/partials_return_type.hpp>
#include <stan/math/prim/scal/meta/is_var_or_arithmetic.hpp>
#include <stan/math/prim/scal/meta/scalar_type_pre.hpp>
#include <stan/math/prim/scal/meta/VectorViewMvt.hpp>
#include <stan/math/prim/scal/meta/length.hpp>
#include <stan/math/prim/scal/meta/max_size_mvt.hpp>
#include <stan/math/fwd/scal/meta/is_fvar.hpp>
#include <stan/math/fwd/scal/meta/partials_type.hpp>

using stan::length;

TEST(MetaTraits, error_index) {
  EXPECT_EQ(1, int(stan::error_index::value));
}

TEST(MetaTraits, isConstant) {
  using stan::is_constant;
  using stan::math::fvar;

  EXPECT_FALSE(is_constant<fvar<double> >::value);
}

TEST(MetaTraits,containsFvar) {
  using stan::math::fvar;
  using stan::contains_fvar;
  EXPECT_TRUE((contains_fvar<fvar<double> >::value));
  EXPECT_TRUE((contains_fvar<double, fvar<double> >::value));
  EXPECT_TRUE((contains_fvar<fvar<fvar<double> > >::value));
}
TEST(MetaTraits, partials_type) {
  using stan::math::fvar;
  using stan::partials_type;

  stan::partials_type<fvar<double> >::type a(2.0);
  EXPECT_EQ(2.0,a);
  stan::partials_type<fvar<double> >::type b(4.0);
  EXPECT_EQ(4.0,b);
  stan::partials_type<fvar<fvar<double> > >::type c(7.0,1.0);
  EXPECT_EQ(7.0,c.val_);
  EXPECT_EQ(1.0,c.d_);
}
TEST(MetaTraits, partials_return_type) {
  using stan::math::fvar;
  using stan::partials_return_type;

  partials_return_type<double,fvar<double>, std::vector<fvar<double> > >::type a(5.0);
  EXPECT_EQ(5.0,a);

  partials_return_type<double,fvar<fvar<double> > >::type b(3.0,2.0);
  EXPECT_EQ(3.0,b.val_);
  EXPECT_EQ(2.0,b.d_);

  partials_return_type<double,double, fvar<fvar<double> >, fvar<fvar<double> > >::type e(3.0,2.0);
  EXPECT_EQ(3.0,e.val_);
  EXPECT_EQ(2.0,e.d_);

}

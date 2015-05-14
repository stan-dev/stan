#include <gtest/gtest.h>
#include <boost/type_traits.hpp>
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
#include <stan/math/rev/scal/meta/is_var.hpp>
#include <stan/math/rev/scal/meta/partials_type.hpp>

using stan::length;

TEST(MetaTraits, error_index) {
  EXPECT_EQ(1, int(stan::error_index::value));
}

TEST(MetaTraits, isConstant) {
  using stan::is_constant;
  using stan::math::var;

  EXPECT_FALSE(is_constant<var>::value);
}

typedef Eigen::Matrix<stan::math::var,Eigen::Dynamic,Eigen::Dynamic> var_t1;
typedef std::vector<var_t1> var_t2;
typedef std::vector<var_t2> var_t3;

typedef Eigen::Matrix<stan::math::var,Eigen::Dynamic,1> var_u1;
typedef std::vector<var_u1> var_u2;
typedef std::vector<var_u2> var_u3;

typedef Eigen::Matrix<stan::math::var,1,Eigen::Dynamic> var_v1;
typedef std::vector<var_v1> var_v2;
typedef std::vector<var_v2> var_v3;


TEST(MetaTraits, isConstantStruct) {
  using stan::is_constant_struct;
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  EXPECT_FALSE(is_constant_struct<stan::math::var>::value);
  EXPECT_FALSE(is_constant_struct<vector<stan::math::var> >::value);
  EXPECT_FALSE(is_constant_struct<vector<vector<stan::math::var> > >::value);
  EXPECT_FALSE(is_constant_struct<vector<vector<vector<stan::math::var> > > >::value);
  EXPECT_FALSE(is_constant_struct<var_t1>::value);
  EXPECT_FALSE(is_constant_struct<var_t2>::value);
  EXPECT_FALSE(is_constant_struct<var_t3>::value);
  EXPECT_FALSE(is_constant_struct<var_u1>::value);
  EXPECT_FALSE(is_constant_struct<var_u2>::value);
  EXPECT_FALSE(is_constant_struct<var_u3>::value);
  EXPECT_FALSE(is_constant_struct<var_v1>::value);
  EXPECT_FALSE(is_constant_struct<var_v2>::value);
  EXPECT_FALSE(is_constant_struct<var_v3>::value);

}

TEST(MetaTraits, containsNonconstantStruct) {
  using stan::contains_nonconstant_struct;
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  EXPECT_TRUE(contains_nonconstant_struct<stan::math::var>::value);
  EXPECT_TRUE(contains_nonconstant_struct<vector<stan::math::var> >::value);
  EXPECT_TRUE(contains_nonconstant_struct<vector<vector<stan::math::var> > >::value);
  EXPECT_TRUE(contains_nonconstant_struct<vector<vector<vector<stan::math::var> > > >::value);
  EXPECT_TRUE(contains_nonconstant_struct<var_t1>::value);
  EXPECT_TRUE(contains_nonconstant_struct<var_t2>::value);
  EXPECT_TRUE(contains_nonconstant_struct<var_t3>::value);
  EXPECT_TRUE(contains_nonconstant_struct<var_u1>::value);
  EXPECT_TRUE(contains_nonconstant_struct<var_u2>::value);
  EXPECT_TRUE(contains_nonconstant_struct<var_u3>::value);
  EXPECT_TRUE(contains_nonconstant_struct<var_v1>::value);
  EXPECT_TRUE(contains_nonconstant_struct<var_v2>::value);
  EXPECT_TRUE(contains_nonconstant_struct<var_v3>::value);

  bool temp = contains_nonconstant_struct<var_v3,var_v2,var_v1,double,int>::value;
  EXPECT_TRUE(temp);
}

TEST(MetaTraits, VectorView_var) {
  using stan::VectorView;
  using stan::math::var;
  
  var d(10);
  VectorView<var> dv(d);
  EXPECT_FLOAT_EQ(d.val(), dv[0].val());
  dv[1] = 2.0;
  EXPECT_FLOAT_EQ(2.0, dv[0].val());
  EXPECT_FLOAT_EQ(2.0, d.val());

  const var c(10);
  VectorView<const var> cv(c);
  EXPECT_FLOAT_EQ(c.val(), cv[0].val());
}

TEST(MetaTraits, VectorView_vector_var) {
  using stan::VectorView;
  using std::vector;
  using stan::math::var;
  
  vector<var> x(10);
  for (size_t n = 0; n < 10; ++n) 
    x[n] = n;
  VectorView<vector<var> > xv(x);
  for (size_t n = 0; n < 10; ++n)
    EXPECT_FLOAT_EQ(x[n].val(), xv[n].val());
  for (size_t n = 0; n < 10; ++n)
    xv[n] = 10+n;
  for (size_t n = 0; n < 10; ++n) {
    EXPECT_FLOAT_EQ(x[n].val(), xv[n].val());
    EXPECT_FLOAT_EQ(10+n, xv[n].val());
  }

  const vector<var> y(x);
  VectorView<const vector<var> > yv(y);
  for (size_t n = 0; n < 10; ++n)
    EXPECT_FLOAT_EQ(y[n].val(), yv[n].val());
}

TEST(MetaTraits, VectorView_matrix_var) {
  using stan::VectorView;
  using stan::math::var;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  
  Matrix<var,Dynamic,1> a(10);
  for (size_t n = 0; n < 10; ++n) 
    a[n] = n;
  VectorView<Matrix<var,Dynamic,1> > av(a);
  for (size_t n = 0; n < 10; ++n)
    EXPECT_FLOAT_EQ(a[n].val(), av[n].val());
  for (size_t n = 0; n < 10; ++n)
    av[n] = n+10;
  for (size_t n = 0; n < 10; ++n) {
    EXPECT_FLOAT_EQ(10+n, av[n].val());
    EXPECT_FLOAT_EQ(10+n, a[n].val());
  }

  const Matrix<var,Dynamic,1> b(a);
  VectorView<const Matrix<var,Dynamic,1> > bv(b);
  for (size_t n = 0; n < 10; ++n)
    EXPECT_FLOAT_EQ(b[n].val(), bv[n].val());
  
  Matrix<var,1,Dynamic> c(10);
  for (size_t n = 0; n < 10; ++n) 
    c[n] = n;
  VectorView<Matrix<var,1,Dynamic> > cv(c);
  for (size_t n = 0; n < 10; ++n)
    EXPECT_FLOAT_EQ(c[n].val(), cv[n].val());
  for (size_t n = 0; n < 10; ++n) 
    cv[n] = n+10;
  for (size_t n = 0; n < 10; ++n) {
    EXPECT_FLOAT_EQ(10+n, cv[n].val());
    EXPECT_FLOAT_EQ(10+n, c[n].val());
  }

  const Matrix<var,1,Dynamic> d(c);
  VectorView<const Matrix<var,1,Dynamic> > dv(d);
  for (size_t n = 0; n < 10; ++n)
    EXPECT_FLOAT_EQ(d[n].val(), dv[n].val());
}

TEST(MetaTraits, VectorBuilderHelper_false_true) {
  using std::vector;
  using stan::VectorBuilderHelper;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;

  var a_var(1);
  std::vector<var> a_std_vector(3);
  Matrix<var,Dynamic,1> a_vector(4);
  Matrix<var,1,Dynamic> a_row_vector(5);

  VectorBuilderHelper<double,false,true> dvv1(length(a_var));
  EXPECT_THROW(dvv1[0], std::logic_error);

  VectorBuilderHelper<double,false,true> dvv2(length(a_std_vector));
  EXPECT_THROW(dvv2[0], std::logic_error);
  
  VectorBuilderHelper<double,false,true> dvv3(length(a_vector));
  EXPECT_THROW(dvv3[0], std::logic_error);
  
  VectorBuilderHelper<double,false,true> dvv4(length(a_row_vector));
  EXPECT_THROW(dvv4[0], std::logic_error);
}

TEST(MetaTraits, VectorBuilderHelper_true_true) {
  using std::vector;
  using stan::VectorBuilderHelper;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;

  var a_var(1);
  std::vector<var> a_std_vector(3);
  Matrix<var,Dynamic,1> a_vector(4);
  Matrix<var,1,Dynamic> a_row_vector(5);

  VectorBuilderHelper<double,true,true> dvv1(length(a_var));
  dvv1[0] = 0.0;
  EXPECT_FLOAT_EQ(0.0, dvv1[0]);

  VectorBuilderHelper<double,true,true> dvv2(length(a_std_vector));
  dvv2[0] = 0.0;
  dvv2[1] = 1.0;
  dvv2[2] = 2.0;
  EXPECT_FLOAT_EQ(0.0, dvv2[0]);
  EXPECT_FLOAT_EQ(1.0, dvv2[1]);
  EXPECT_FLOAT_EQ(2.0, dvv2[2]);  
  
  VectorBuilderHelper<double,true,true> dvv3(length(a_vector));
  dvv3[0] = 0.0;
  dvv3[1] = 1.0;
  dvv3[2] = 2.0;
  EXPECT_FLOAT_EQ(0.0, dvv3[0]);
  EXPECT_FLOAT_EQ(1.0, dvv3[1]);
  EXPECT_FLOAT_EQ(2.0, dvv3[2]);  
  
  VectorBuilderHelper<double,true,true> dvv4(length(a_row_vector));
  dvv4[0] = 0.0;
  dvv4[1] = 1.0;
  dvv4[2] = 2.0;
  EXPECT_FLOAT_EQ(0.0, dvv4[0]);
  EXPECT_FLOAT_EQ(1.0, dvv4[1]);
  EXPECT_FLOAT_EQ(2.0, dvv4[2]);
}

TEST(MetaTraits, VectorBuilder_false_true) {
  using std::vector;
  using stan::VectorBuilder;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;

  var a_var(1);
  std::vector<var> a_std_vector(3);
  Matrix<var,Dynamic,1> a_vector(4);
  Matrix<var,1,Dynamic> a_row_vector(5);

  VectorBuilder<false,double,std::vector<var> > dvv1(length(a_var));
  EXPECT_THROW(dvv1[0], std::logic_error);

  VectorBuilder<false,double,std::vector<var> > dvv2(length(a_std_vector));
  EXPECT_THROW(dvv2[0], std::logic_error);
  
  VectorBuilder<false,double,Matrix<var,Dynamic,1> > dvv3(length(a_vector));
  EXPECT_THROW(dvv3[0], std::logic_error);
  
  VectorBuilder<false,double,Matrix<var,1,Dynamic> > dvv4(length(a_row_vector));
  EXPECT_THROW(dvv4[0], std::logic_error);
}

TEST(MetaTraits, VectorBuilder_true_true) {
  using std::vector;
  using stan::VectorBuilder;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;

  var a_var(1);
  std::vector<var> a_std_vector(3);
  Matrix<var,Dynamic,1> a_vector(4);
  Matrix<var,1,Dynamic> a_row_vector(5);

  VectorBuilder<true,double,std::vector<var> > dvv1(length(a_var));
  dvv1[0] = 0.0;
  EXPECT_FLOAT_EQ(0.0, dvv1[0]);

  VectorBuilder<true,double,std::vector<var> > dvv2(length(a_std_vector));
  dvv2[0] = 0.0;
  dvv2[1] = 1.0;
  dvv2[2] = 2.0;
  EXPECT_FLOAT_EQ(0.0, dvv2[0]);
  EXPECT_FLOAT_EQ(1.0, dvv2[1]);
  EXPECT_FLOAT_EQ(2.0, dvv2[2]);  
  
  VectorBuilder<true,double,Matrix<var,Dynamic,1> > dvv3(length(a_vector));
  dvv3[0] = 0.0;
  dvv3[1] = 1.0;
  dvv3[2] = 2.0;
  EXPECT_FLOAT_EQ(0.0, dvv3[0]);
  EXPECT_FLOAT_EQ(1.0, dvv3[1]);
  EXPECT_FLOAT_EQ(2.0, dvv3[2]);  
  
  VectorBuilder<true,double,Matrix<var,1,Dynamic> > dvv4(length(a_row_vector));
  dvv4[0] = 0.0;
  dvv4[1] = 1.0;
  dvv4[2] = 2.0;
  EXPECT_FLOAT_EQ(0.0, dvv4[0]);
  EXPECT_FLOAT_EQ(1.0, dvv4[1]);
  EXPECT_FLOAT_EQ(2.0, dvv4[2]);
}

TEST(MetaTraits, partials_type) {
  using stan::math::var;
  using stan::partials_type;

  stan::partials_type<var>::type f(2.0);
  EXPECT_EQ(2.0,f);
}

TEST(MetaTraits, partials_return_type) {
  using stan::math::var;
  using stan::partials_return_type;

  partials_return_type<double,stan::math::var>::type f(5.0);
  EXPECT_EQ(5.0,f);

  partials_return_type<double,stan::math::var,std::vector<stan::math::var> >::type g(5.0);
  EXPECT_EQ(5.0,g);
}

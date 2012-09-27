#include <gtest/gtest.h>
#include <boost/type_traits.hpp>
#include <stan/agrad/agrad.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/matrix.hpp>

TEST(MetaTraits, isConstant) {
  using stan::is_constant;
  EXPECT_TRUE(is_constant<double>::value);
  EXPECT_TRUE(is_constant<float>::value);
  EXPECT_TRUE(is_constant<unsigned int>::value);
  EXPECT_TRUE(is_constant<int>::value);
  EXPECT_FALSE(is_constant<stan::agrad::var>::value);
}

TEST(MetaTraits, is_vector) {
  using stan::is_vector;
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  EXPECT_FALSE(is_vector<double>::value);
  EXPECT_FALSE(is_vector<int>::value);
  EXPECT_FALSE(is_vector<size_t>::value);

  EXPECT_FALSE(is_vector<const double>::value);
  EXPECT_FALSE(is_vector<const int>::value);
  EXPECT_FALSE(is_vector<const size_t>::value);

  EXPECT_TRUE(is_vector<std::vector<double> >::value);
  EXPECT_TRUE(is_vector<std::vector<int> >::value);
  EXPECT_TRUE(is_vector<std::vector<const double> >::value);
  EXPECT_TRUE(is_vector<std::vector<const int> >::value);

  typedef Matrix<double,Dynamic,1> temp_vec_d;
  EXPECT_TRUE(is_vector<temp_vec_d>::value);
  EXPECT_TRUE(is_vector<const temp_vec_d>::value);
  
  typedef Matrix<double,1,Dynamic> temp_rowvec_d;
  EXPECT_TRUE(is_vector<temp_rowvec_d>::value);
  EXPECT_TRUE(is_vector<const temp_rowvec_d>::value);

  typedef Matrix<double,Dynamic,Dynamic> temp_matrix_d;
  EXPECT_FALSE(is_vector<temp_matrix_d>::value);
  EXPECT_FALSE(is_vector<const temp_matrix_d>::value);
}

typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> const_t1;
typedef std::vector<const_t1> const_t2;
typedef std::vector<const_t2> const_t3;

typedef Eigen::Matrix<double,Eigen::Dynamic,1> const_u1;
typedef std::vector<const_u1> const_u2;
typedef std::vector<const_u2> const_u3;

typedef Eigen::Matrix<double,1,Eigen::Dynamic> const_v1;
typedef std::vector<const_v1> const_v2;
typedef std::vector<const_v2> const_v3;

typedef Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,Eigen::Dynamic> var_t1;
typedef std::vector<var_t1> var_t2;
typedef std::vector<var_t2> var_t3;

typedef Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,1> var_u1;
typedef std::vector<var_u1> var_u2;
typedef std::vector<var_u2> var_u3;

typedef Eigen::Matrix<stan::agrad::var,1,Eigen::Dynamic> var_v1;
typedef std::vector<var_v1> var_v2;
typedef std::vector<var_v2> var_v3;


TEST(MetaTraits, isConstantStruct) {
  using stan::is_constant_struct;
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  EXPECT_TRUE(is_constant_struct<int>::value);
  EXPECT_TRUE(is_constant_struct<double>::value);
  EXPECT_TRUE(is_constant_struct<float>::value);
  EXPECT_TRUE(is_constant_struct<long>::value);
  EXPECT_TRUE(is_constant_struct<vector<double> >::value);
  EXPECT_TRUE(is_constant_struct<vector<vector<double> > >::value);
  EXPECT_TRUE(is_constant_struct<vector<vector<vector<double> > > >::value);
  EXPECT_TRUE(is_constant_struct<const_t1>::value);
  EXPECT_TRUE(is_constant_struct<const_t2>::value);
  EXPECT_TRUE(is_constant_struct<const_t3>::value);
  EXPECT_TRUE(is_constant_struct<const_u1>::value);
  EXPECT_TRUE(is_constant_struct<const_u2>::value);
  EXPECT_TRUE(is_constant_struct<const_u3>::value);
  EXPECT_TRUE(is_constant_struct<const_v1>::value);
  EXPECT_TRUE(is_constant_struct<const_v2>::value);
  EXPECT_TRUE(is_constant_struct<const_v3>::value);

  EXPECT_FALSE(is_constant_struct<stan::agrad::var>::value);
  EXPECT_FALSE(is_constant_struct<vector<stan::agrad::var> >::value);
  EXPECT_FALSE(is_constant_struct<vector<vector<stan::agrad::var> > >::value);
  EXPECT_FALSE(is_constant_struct<vector<vector<vector<stan::agrad::var> > > >::value);
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

TEST(MetaTraits, length) {
  using stan::length;
  EXPECT_EQ(1U, length(27.0));
  EXPECT_EQ(1U, length(3));
  std::vector<double> x(10);
  EXPECT_EQ(10U, length(x));
}
TEST(MetaTraits, VectorView_new_double)  {
  using stan::VectorView_new;

  double d(10);
  VectorView_new<double> dv(d);
  EXPECT_FLOAT_EQ(d, dv[0]);
  dv[1] = 2.0;
  EXPECT_FLOAT_EQ(2.0, dv[0]);
  EXPECT_FLOAT_EQ(2.0, d);

  const double c(10);
  VectorView_new<const double> cv(c);
  EXPECT_FLOAT_EQ(c, cv[0]);
}
TEST(MetaTraits, VectorView_new_var) {
  using stan::VectorView_new;
  using stan::agrad::var;
  
  var d(10);
  VectorView_new<var> dv(d);
  EXPECT_FLOAT_EQ(d.val(), dv[0].val());
  dv[1] = 2.0;
  EXPECT_FLOAT_EQ(2.0, dv[0].val());
  EXPECT_FLOAT_EQ(2.0, d.val());

  const var c(10);
  VectorView_new<const var> cv(c);
  EXPECT_FLOAT_EQ(c.val(), cv[0].val());
}
TEST(MetaTraits, VectorView_new_vector_double) {
  using stan::VectorView_new;
  using std::vector;
  
  vector<double> x(10);
  for (size_t n = 0; n < 10; ++n) 
    x[n] = n;
  VectorView_new<vector<double> > xv(x);
  for (size_t n = 0; n < 10; ++n)
    EXPECT_FLOAT_EQ(x[n], xv[n]);
  for (size_t n = 0; n < 10; ++n)
    xv[n] = 10+n;
  for (size_t n = 0; n < 10; ++n) {
    EXPECT_FLOAT_EQ(x[n], xv[n]);
    EXPECT_FLOAT_EQ(10+n, xv[n]);
  }

  const vector<double> y(x);
  VectorView_new<const vector<double> > yv(y);
  for (size_t n = 0; n < 10; ++n)
    EXPECT_FLOAT_EQ(y[n], yv[n]);
}
TEST(MetaTraits, VectorView_new_vector_var) {
  using stan::VectorView_new;
  using std::vector;
  using stan::agrad::var;
  
  vector<var> x(10);
  for (size_t n = 0; n < 10; ++n) 
    x[n] = n;
  VectorView_new<vector<var> > xv(x);
  for (size_t n = 0; n < 10; ++n)
    EXPECT_FLOAT_EQ(x[n].val(), xv[n].val());
  for (size_t n = 0; n < 10; ++n)
    xv[n] = 10+n;
  for (size_t n = 0; n < 10; ++n) {
    EXPECT_FLOAT_EQ(x[n].val(), xv[n].val());
    EXPECT_FLOAT_EQ(10+n, xv[n].val());
  }

  const vector<var> y(x);
  VectorView_new<const vector<var> > yv(y);
  for (size_t n = 0; n < 10; ++n)
    EXPECT_FLOAT_EQ(y[n].val(), yv[n].val());
}
TEST(MetaTraits, VectorView_new_matrix_double) {
  using stan::VectorView_new;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  
  Matrix<double,Dynamic,1> a(10);
  for (size_t n = 0; n < 10; ++n) 
    a[n] = n;
  VectorView_new<Matrix<double,Dynamic,1> > av(a);
  for (size_t n = 0; n < 10; ++n)
    EXPECT_FLOAT_EQ(a[n], av[n]);
  for (size_t n = 0; n < 10; ++n)
    av[n] = n+10;
  for (size_t n = 0; n < 10; ++n) {
    EXPECT_FLOAT_EQ(10+n, av[n]);
    EXPECT_FLOAT_EQ(10+n, a[n]);
  }

  const Matrix<double,Dynamic,1> b(a);
  VectorView_new<const Matrix<double,Dynamic,1> > bv(b);
  for (size_t n = 0; n < 10; ++n)
    EXPECT_FLOAT_EQ(b[n], bv[n]);
  
  Matrix<double,1,Dynamic> c(10);
  for (size_t n = 0; n < 10; ++n) 
    c[n] = n;
  VectorView_new<Matrix<double,1,Dynamic> > cv(c);
  for (size_t n = 0; n < 10; ++n)
    EXPECT_FLOAT_EQ(c[n], cv[n]);
  for (size_t n = 0; n < 10; ++n) 
    cv[n] = n+10;
  for (size_t n = 0; n < 10; ++n) {
    EXPECT_FLOAT_EQ(10+n, cv[n]);
    EXPECT_FLOAT_EQ(10+n, c[n]);
  }

  const Matrix<double,1,Dynamic> d(c);
  VectorView_new<const Matrix<double,Dynamic,1> > dv(d);
  for (size_t n = 0; n < 10; ++n)
    EXPECT_FLOAT_EQ(d[n], dv[n]);
}
TEST(MetaTraits, VectorView_new_matrix_var) {
  using stan::VectorView_new;
  using stan::agrad::var;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  
  Matrix<var,Dynamic,1> a(10);
  for (size_t n = 0; n < 10; ++n) 
    a[n] = n;
  VectorView_new<Matrix<var,Dynamic,1> > av(a);
  for (size_t n = 0; n < 10; ++n)
    EXPECT_FLOAT_EQ(a[n].val(), av[n].val());
  for (size_t n = 0; n < 10; ++n)
    av[n] = n+10;
  for (size_t n = 0; n < 10; ++n) {
    EXPECT_FLOAT_EQ(10+n, av[n].val());
    EXPECT_FLOAT_EQ(10+n, a[n].val());
  }

  const Matrix<var,Dynamic,1> b(a);
  VectorView_new<const Matrix<var,Dynamic,1> > bv(b);
  for (size_t n = 0; n < 10; ++n)
    EXPECT_FLOAT_EQ(b[n].val(), bv[n].val());
  
  Matrix<var,1,Dynamic> c(10);
  for (size_t n = 0; n < 10; ++n) 
    c[n] = n;
  VectorView_new<Matrix<var,1,Dynamic> > cv(c);
  for (size_t n = 0; n < 10; ++n)
    EXPECT_FLOAT_EQ(c[n].val(), cv[n].val());
  for (size_t n = 0; n < 10; ++n) 
    cv[n] = n+10;
  for (size_t n = 0; n < 10; ++n) {
    EXPECT_FLOAT_EQ(10+n, cv[n].val());
    EXPECT_FLOAT_EQ(10+n, c[n].val());
  }

  const Matrix<var,1,Dynamic> d(c);
  VectorView_new<const Matrix<var,Dynamic,1> > dv(d);
  for (size_t n = 0; n < 10; ++n)
    EXPECT_FLOAT_EQ(d[n].val(), dv[n].val());
}
TEST(MetaTraits, VectorView_new_double_star) {
  using stan::VectorView_new;
  double a[10];
  double *a_star = &a[0];
  for (size_t n = 0; n < 10; ++n)
    a[n] = n;
  VectorView_new<double*,true> av(a_star);
  for (size_t n = 0; n < 10; ++n) 
    EXPECT_FLOAT_EQ(a[n], av[n]);
  for (size_t n = 0; n < 10; ++n)
    av[n] = n+10;
  for (size_t n = 0; n < 10; ++n) {
    EXPECT_FLOAT_EQ(n+10, a[n]);
    EXPECT_FLOAT_EQ(n+10, av[n]);
  }

  double b(20);
  double *b_star = &b;
  VectorView_new<double*,false> bv(b_star);
  for (size_t n = 0; n < 10; ++n) 
    EXPECT_FLOAT_EQ(20, bv[n]);
  bv[1] = 10;
  EXPECT_FLOAT_EQ(10, bv[0]);
  EXPECT_FLOAT_EQ(10, b);
}
TEST(MetaTraits, DoubleVectorView_false_double) {
  using std::vector;
  using stan::DoubleVectorView;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::agrad::var;

  double a_double(1);
  std::vector<double> a_std_vector(3);
  Matrix<double,Dynamic,1> a_vector(4);
  Matrix<double,1,Dynamic> a_row_vector(5);

  DoubleVectorView<false,double> dvv1(a_double);
  EXPECT_THROW(dvv1[0], std::runtime_error);

  DoubleVectorView<false,std::vector<double> > dvv2(a_std_vector);
  EXPECT_THROW(dvv2[0], std::runtime_error);
  
  DoubleVectorView<false,Matrix<double,Dynamic,1> > dvv3(a_vector);
  EXPECT_THROW(dvv3[0], std::runtime_error);
  
  DoubleVectorView<false,Matrix<double,1,Dynamic> > dvv4(a_row_vector);
  EXPECT_THROW(dvv4[0], std::runtime_error);
}


TEST(MetaTraits, DoubleVectorView_true_double) {
  using std::vector;
  using stan::DoubleVectorView;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  double a_double(1);
  std::vector<double> a_std_vector(3);
  Matrix<double,Dynamic,1> a_vector(4);
  Matrix<double,1,Dynamic> a_row_vector(5);

  DoubleVectorView<true,double> dvv1(a_double);
  EXPECT_FLOAT_EQ(0.0, dvv1[0]);
  EXPECT_FLOAT_EQ(0.0, dvv1[1]);
  EXPECT_FLOAT_EQ(0.0, dvv1[100]);

  DoubleVectorView<true,std::vector<double> > dvv2(a_std_vector);
  EXPECT_FLOAT_EQ(0.0, dvv2[0]);
  EXPECT_FLOAT_EQ(0.0, dvv2[1]);
  EXPECT_FLOAT_EQ(0.0, dvv2[2]);  
  
  DoubleVectorView<true,Matrix<double,Dynamic,1> > dvv3(a_vector);
  EXPECT_FLOAT_EQ(0.0, dvv3[0]);
  EXPECT_FLOAT_EQ(0.0, dvv3[1]);
  EXPECT_FLOAT_EQ(0.0, dvv3[2]);  
  
  DoubleVectorView<true,Matrix<double,1,Dynamic> > dvv4(a_row_vector);
  EXPECT_FLOAT_EQ(0.0, dvv4[0]);
  EXPECT_FLOAT_EQ(0.0, dvv4[1]);
  EXPECT_FLOAT_EQ(0.0, dvv4[2]);
}


TEST(MetaTraits, DoubleVectorView_false_var) {
  using std::vector;
  using stan::DoubleVectorView;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::agrad::var;

  var a_var(1);
  std::vector<var> a_std_vector(3);
  Matrix<var,Dynamic,1> a_vector(4);
  Matrix<var,1,Dynamic> a_row_vector(5);

  DoubleVectorView<false,var> dvv1(a_var);
  EXPECT_THROW(dvv1[0], std::runtime_error);

  DoubleVectorView<false,std::vector<var> > dvv2(a_std_vector);
  EXPECT_THROW(dvv2[0], std::runtime_error);
  
  DoubleVectorView<false,Matrix<var,Dynamic,1> > dvv3(a_vector);
  EXPECT_THROW(dvv3[0], std::runtime_error);
  
  DoubleVectorView<false,Matrix<var,1,Dynamic> > dvv4(a_row_vector);
  EXPECT_THROW(dvv4[0], std::runtime_error);
}


TEST(MetaTraits, DoubleVectorView_true_var) {
  using std::vector;
  using stan::DoubleVectorView;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::agrad::var;

  var a_var(1);
  std::vector<var> a_std_vector(3);
  Matrix<var,Dynamic,1> a_vector(4);
  Matrix<var,1,Dynamic> a_row_vector(5);

  DoubleVectorView<true,var> dvv1(a_var);
  EXPECT_FLOAT_EQ(0.0, dvv1[0]);
  EXPECT_FLOAT_EQ(0.0, dvv1[1]);
  EXPECT_FLOAT_EQ(0.0, dvv1[100]);

  DoubleVectorView<true,std::vector<var> > dvv2(a_std_vector);
  EXPECT_FLOAT_EQ(0.0, dvv2[0]);
  EXPECT_FLOAT_EQ(0.0, dvv2[1]);
  EXPECT_FLOAT_EQ(0.0, dvv2[2]);  
  
  DoubleVectorView<true,Matrix<var,Dynamic,1> > dvv3(a_vector);
  EXPECT_FLOAT_EQ(0.0, dvv3[0]);
  EXPECT_FLOAT_EQ(0.0, dvv3[1]);
  EXPECT_FLOAT_EQ(0.0, dvv3[2]);  
  
  DoubleVectorView<true,Matrix<var,1,Dynamic> > dvv4(a_row_vector);
  EXPECT_FLOAT_EQ(0.0, dvv4[0]);
  EXPECT_FLOAT_EQ(0.0, dvv4[1]);
  EXPECT_FLOAT_EQ(0.0, dvv4[2]);
}


TEST(MetaTraits, DoubleVectorView_false_double_const) {
  using std::vector;
  using stan::DoubleVectorView;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::agrad::var;

  const double a_double(1);
  const std::vector<double> a_std_vector(3);
  const Matrix<double,Dynamic,1> a_vector(4);
  const Matrix<double,1,Dynamic> a_row_vector(5);

  DoubleVectorView<false,const double> dvv1(a_double);
  EXPECT_THROW(dvv1[0], std::runtime_error);

  DoubleVectorView<false,const std::vector<double> > dvv2(a_std_vector);
  EXPECT_THROW(dvv2[0], std::runtime_error);
  
  DoubleVectorView<false,const Matrix<double,Dynamic,1> > dvv3(a_vector);
  EXPECT_THROW(dvv3[0], std::runtime_error);
  
  DoubleVectorView<false,const Matrix<double,1,Dynamic> > dvv4(a_row_vector);
  EXPECT_THROW(dvv4[0], std::runtime_error);
}


TEST(MetaTraits, DoubleVectorView_true_double_const) {
  using std::vector;
  using stan::DoubleVectorView;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  const double a_double(1);
  const std::vector<double> a_std_vector(3);
  const Matrix<double,Dynamic,1> a_vector(4);
  const Matrix<double,1,Dynamic> a_row_vector(5);

  DoubleVectorView<true,const double> dvv1(a_double);
  EXPECT_FLOAT_EQ(0.0, dvv1[0]);
  EXPECT_FLOAT_EQ(0.0, dvv1[1]);
  EXPECT_FLOAT_EQ(0.0, dvv1[100]);

  DoubleVectorView<true,const std::vector<double> > dvv2(a_std_vector);
  EXPECT_FLOAT_EQ(0.0, dvv2[0]);
  EXPECT_FLOAT_EQ(0.0, dvv2[1]);
  EXPECT_FLOAT_EQ(0.0, dvv2[2]);  
  
  DoubleVectorView<true,const Matrix<double,Dynamic,1> > dvv3(a_vector);
  EXPECT_FLOAT_EQ(0.0, dvv3[0]);
  EXPECT_FLOAT_EQ(0.0, dvv3[1]);
  EXPECT_FLOAT_EQ(0.0, dvv3[2]);  
  
  DoubleVectorView<true,const Matrix<double,1,Dynamic> > dvv4(a_row_vector);
  EXPECT_FLOAT_EQ(0.0, dvv4[0]);
  EXPECT_FLOAT_EQ(0.0, dvv4[1]);
  EXPECT_FLOAT_EQ(0.0, dvv4[2]);
}


TEST(MetaTraits, DoubleVectorView_false_var_const) {
  using std::vector;
  using stan::DoubleVectorView;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::agrad::var;

  const var a_var(1);
  const std::vector<var> a_std_vector(3);
  const Matrix<var,Dynamic,1> a_vector(4);
  const Matrix<var,1,Dynamic> a_row_vector(5);

  DoubleVectorView<false,const var> dvv1(a_var);
  EXPECT_THROW(dvv1[0], std::runtime_error);

  DoubleVectorView<false,const std::vector<var> > dvv2(a_std_vector);
  EXPECT_THROW(dvv2[0], std::runtime_error);
  
  DoubleVectorView<false,const Matrix<var,Dynamic,1> > dvv3(a_vector);
  EXPECT_THROW(dvv3[0], std::runtime_error);
  
  DoubleVectorView<false,const Matrix<var,1,Dynamic> > dvv4(a_row_vector);
  EXPECT_THROW(dvv4[0], std::runtime_error);
}


TEST(MetaTraits, DoubleVectorView_true_var_const) {
  using std::vector;
  using stan::DoubleVectorView;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::agrad::var;

  const var a_var(1);
  const std::vector<var> a_std_vector(3);
  const Matrix<var,Dynamic,1> a_vector(4);
  const Matrix<var,1,Dynamic> a_row_vector(5);

  DoubleVectorView<true,const var> dvv1(a_var);
  EXPECT_FLOAT_EQ(0.0, dvv1[0]);
  EXPECT_FLOAT_EQ(0.0, dvv1[1]);
  EXPECT_FLOAT_EQ(0.0, dvv1[100]);

  DoubleVectorView<true,const std::vector<var> > dvv2(a_std_vector);
  EXPECT_FLOAT_EQ(0.0, dvv2[0]);
  EXPECT_FLOAT_EQ(0.0, dvv2[1]);
  EXPECT_FLOAT_EQ(0.0, dvv2[2]);  
  
  DoubleVectorView<true,const Matrix<var,Dynamic,1> > dvv3(a_vector);
  EXPECT_FLOAT_EQ(0.0, dvv3[0]);
  EXPECT_FLOAT_EQ(0.0, dvv3[1]);
  EXPECT_FLOAT_EQ(0.0, dvv3[2]);  
  
  DoubleVectorView<true,const Matrix<var,1,Dynamic> > dvv4(a_row_vector);
  EXPECT_FLOAT_EQ(0.0, dvv4[0]);
  EXPECT_FLOAT_EQ(0.0, dvv4[1]);
  EXPECT_FLOAT_EQ(0.0, dvv4[2]);
}


TEST(MetaTraits, scalar_type) {
  using boost::is_same;
  using stan::scalar_type;
  using std::vector;

  stan::scalar_type<double>::type x = 5.0;
  EXPECT_EQ(5.0,x);

  stan::scalar_type<std::vector<int> >::type n = 1;
  EXPECT_EQ(1,n);

  // hack to get value of template into Google test macro 
  bool b1 = is_same<double,double>::value;
  EXPECT_TRUE(b1);

  bool b2 = is_same<double,int>::value;
  EXPECT_FALSE(b2);

  bool b3 = is_same<double, scalar_type<vector<double> >::type>::value;
  EXPECT_TRUE(b3);

  bool b4 = is_same<double, scalar_type<double>::type>::value;
  EXPECT_TRUE(b4);

  bool b5 = is_same<int, scalar_type<double>::type>::value;
  EXPECT_FALSE(b5);
}

#include <stan/math/prim/mat/fun/subtract.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/fwd/core.hpp>

using stan::math::fvar;
TEST(AgradFwdMatrixOperatorSubtraction,fd_scalar_matrix) {
  using stan::math::subtract;
  using stan::math::matrix_fd;

  matrix_fd v(2,2);
  v << 1, 2, 3, 4;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
  matrix_fd result;

  result = subtract(2.0,v);
  EXPECT_FLOAT_EQ(1.0,result(0,0).val_);
  EXPECT_FLOAT_EQ(0.0,result(0,1).val_);
  EXPECT_FLOAT_EQ(-1.0,result(1,0).val_);
  EXPECT_FLOAT_EQ(-2.0,result(1,1).val_);
  EXPECT_FLOAT_EQ(-1.0,result(0,0).d_);
  EXPECT_FLOAT_EQ(-1.0,result(0,1).d_);
  EXPECT_FLOAT_EQ(-1.0,result(1,0).d_);
  EXPECT_FLOAT_EQ(-1.0,result(1,1).d_);

  result = subtract(v,2.0);
  EXPECT_FLOAT_EQ(-1.0,result(0,0).val_);
  EXPECT_FLOAT_EQ(0.0,result(0,1).val_);
  EXPECT_FLOAT_EQ(1.0,result(1,0).val_);
  EXPECT_FLOAT_EQ(2.0,result(1,1).val_);
  EXPECT_FLOAT_EQ(1.0,result(0,0).d_);
  EXPECT_FLOAT_EQ(1.0,result(0,1).d_);
  EXPECT_FLOAT_EQ(1.0,result(1,0).d_);
  EXPECT_FLOAT_EQ(1.0,result(1,1).d_);
}
TEST(AgradFwdMatrixOperatorSubtraction,fd_scalar_vector) {
  using stan::math::subtract;
  using stan::math::vector_fd;

  vector_fd v(4);
  v << 1, 2, 3, 4;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
  vector_fd result;

  result = subtract(2.0,v);
  EXPECT_FLOAT_EQ(1.0,result(0).val_);
  EXPECT_FLOAT_EQ(0.0,result(1).val_);
  EXPECT_FLOAT_EQ(-1.0,result(2).val_);
  EXPECT_FLOAT_EQ(-2.0,result(3).val_);
  EXPECT_FLOAT_EQ(-1.0,result(0).d_);
  EXPECT_FLOAT_EQ(-1.0,result(1).d_);
  EXPECT_FLOAT_EQ(-1.0,result(3).d_);
  EXPECT_FLOAT_EQ(-1.0,result(3).d_);

  result = subtract(v,2.0);
  EXPECT_FLOAT_EQ(-1.0,result(0).val_);
  EXPECT_FLOAT_EQ(0.0,result(1).val_);
  EXPECT_FLOAT_EQ(1.0,result(2).val_);
  EXPECT_FLOAT_EQ(2.0,result(3).val_);
  EXPECT_FLOAT_EQ(1.0,result(0).d_);
  EXPECT_FLOAT_EQ(1.0,result(1).d_);
  EXPECT_FLOAT_EQ(1.0,result(3).d_);
  EXPECT_FLOAT_EQ(1.0,result(3).d_);
}
TEST(AgradFwdMatrixOperatorSubtraction,fd_scalar_rowvector) {
  using stan::math::subtract;
  using stan::math::row_vector_fd;

  row_vector_fd v(4);
  v << 1, 2, 3, 4;
   v(0).d_ = 1.0;
   v(1).d_ = 1.0;
   v(2).d_ = 1.0;
   v(3).d_ = 1.0;
  row_vector_fd result;

  result = subtract(2.0,v);
  EXPECT_FLOAT_EQ(1.0,result(0).val_);
  EXPECT_FLOAT_EQ(0.0,result(1).val_);
  EXPECT_FLOAT_EQ(-1.0,result(2).val_);
  EXPECT_FLOAT_EQ(-2.0,result(3).val_);
  EXPECT_FLOAT_EQ(-1.0,result(0).d_);
  EXPECT_FLOAT_EQ(-1.0,result(1).d_);
  EXPECT_FLOAT_EQ(-1.0,result(3).d_);
  EXPECT_FLOAT_EQ(-1.0,result(3).d_);

  result = subtract(v,2.0);
  EXPECT_FLOAT_EQ(-1.0,result(0).val_);
  EXPECT_FLOAT_EQ(0.0,result(1).val_);
  EXPECT_FLOAT_EQ(1.0,result(2).val_);
  EXPECT_FLOAT_EQ(2.0,result(3).val_);
  EXPECT_FLOAT_EQ(1.0,result(0).d_);
  EXPECT_FLOAT_EQ(1.0,result(1).d_);
  EXPECT_FLOAT_EQ(1.0,result(3).d_);
  EXPECT_FLOAT_EQ(1.0,result(3).d_);
}
TEST(AgradFwdMatrixOperatorSubtraction,fd_vector_vector) {
  using stan::math::subtract;
  using stan::math::vector_d;
  using stan::math::vector_fd;

  vector_d expected_output(5);
  vector_fd output;
  vector_d output_d;
  vector_d vd_1(5), vd_2(5);
  vector_fd vv_1(5), vv_2(5);

  vd_1 << 0, 2, -6, 10, 6;
  vv_1 << 0, 2, -6, 10, 6;
   vv_1(0).d_ = 1.0;
   vv_1(1).d_ = 1.0;
   vv_1(2).d_ = 1.0;
   vv_1(3).d_ = 1.0;
   vv_1(4).d_ = 1.0;
  vd_2 << 2, 3, 4, 5, 6;
  vv_2 << 2, 3, 4, 5, 6;
   vv_2(0).d_ = 1.0;
   vv_2(1).d_ = 1.0;
   vv_2(2).d_ = 1.0;
   vv_2(3).d_ = 1.0;
   vv_2(4).d_ = 1.0;

  expected_output << -2, -1, -10, 5, 0;
  
  output_d = subtract(vd_1, vd_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_d(0));
  EXPECT_FLOAT_EQ(expected_output(1), output_d(1));
  EXPECT_FLOAT_EQ(expected_output(2), output_d(2));
  EXPECT_FLOAT_EQ(expected_output(3), output_d(3));
  EXPECT_FLOAT_EQ(expected_output(4), output_d(4));  

  output = subtract(vv_1, vd_2);
  EXPECT_FLOAT_EQ(expected_output(0), output(0).val_);
  EXPECT_FLOAT_EQ(expected_output(1), output(1).val_);
  EXPECT_FLOAT_EQ(expected_output(2), output(2).val_);
  EXPECT_FLOAT_EQ(expected_output(3), output(3).val_);
  EXPECT_FLOAT_EQ(expected_output(4), output(4).val_);  
  EXPECT_FLOAT_EQ(1, output(0).d_);
  EXPECT_FLOAT_EQ(1, output(1).d_);
  EXPECT_FLOAT_EQ(1, output(2).d_);
  EXPECT_FLOAT_EQ(1, output(3).d_);
  EXPECT_FLOAT_EQ(1, output(4).d_);  

  output = subtract(vd_1, vv_2);
  EXPECT_FLOAT_EQ(expected_output(0), output(0).val_);
  EXPECT_FLOAT_EQ(expected_output(1), output(1).val_);
  EXPECT_FLOAT_EQ(expected_output(2), output(2).val_);
  EXPECT_FLOAT_EQ(expected_output(3), output(3).val_);
  EXPECT_FLOAT_EQ(expected_output(4), output(4).val_);  
  EXPECT_FLOAT_EQ(-1.0, output(0).d_);
  EXPECT_FLOAT_EQ(-1.0, output(1).d_);
  EXPECT_FLOAT_EQ(-1.0, output(2).d_);
  EXPECT_FLOAT_EQ(-1.0, output(3).d_);
  EXPECT_FLOAT_EQ(-1.0, output(4).d_);  

  output = subtract(vv_1, vv_2);
  EXPECT_FLOAT_EQ(expected_output(0), output(0).val_);
  EXPECT_FLOAT_EQ(expected_output(1), output(1).val_);
  EXPECT_FLOAT_EQ(expected_output(2), output(2).val_);
  EXPECT_FLOAT_EQ(expected_output(3), output(3).val_);
  EXPECT_FLOAT_EQ(expected_output(4), output(4).val_);  
  EXPECT_FLOAT_EQ(0, output(0).d_);
  EXPECT_FLOAT_EQ(0, output(1).d_);
  EXPECT_FLOAT_EQ(0, output(2).d_);
  EXPECT_FLOAT_EQ(0, output(3).d_);
  EXPECT_FLOAT_EQ(0, output(4).d_);  
}
TEST(AgradFwdMatrixOperatorSubtraction,fd_vector_vector_exception) {
  using stan::math::subtract;
  using stan::math::vector_d;
  using stan::math::vector_fd;

  vector_d d1(5), d2(1);
  vector_fd v1(5), v2(1);
  
  vector_fd output;
  EXPECT_THROW(subtract(d1, d2), std::invalid_argument);
  EXPECT_THROW(subtract(v1, d2), std::invalid_argument);
  EXPECT_THROW(subtract(d1, v2), std::invalid_argument);
  EXPECT_THROW(subtract(v1, v2), std::invalid_argument);
}
TEST(AgradFwdMatrixOperatorSubtraction,fd_rowvector_rowvector) {
  using stan::math::subtract;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

  row_vector_d expected_output(5);
  row_vector_d  output_d;
  row_vector_fd  output;
  row_vector_d rvd_1(5), rvd_2(5);
  row_vector_fd rvv_1(5), rvv_2(5);

  rvd_1 << 0, 2, -6, 10, 6;
  rvv_1 << 0, 2, -6, 10, 6;
   rvv_1(0).d_ = 1.0;
   rvv_1(1).d_ = 1.0;
   rvv_1(2).d_ = 1.0;
   rvv_1(3).d_ = 1.0;
   rvv_1(4).d_ = 1.0;
  rvd_2 << 2, 3, 4, 5, 6;
  rvv_2 << 2, 3, 4, 5, 6;
   rvv_2(0).d_ = 1.0;
   rvv_2(1).d_ = 1.0;
   rvv_2(2).d_ = 1.0;
   rvv_2(3).d_ = 1.0;
   rvv_2(4).d_ = 1.0;
  
  expected_output << -2, -1, -10, 5, 0;
  
  output_d = subtract(rvd_1, rvd_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_d(0));
  EXPECT_FLOAT_EQ(expected_output(1), output_d(1));
  EXPECT_FLOAT_EQ(expected_output(2), output_d(2));
  EXPECT_FLOAT_EQ(expected_output(3), output_d(3));
  EXPECT_FLOAT_EQ(expected_output(4), output_d(4));

  output = subtract(rvv_1, rvd_2);
  EXPECT_FLOAT_EQ(expected_output(0), output(0).val_);
  EXPECT_FLOAT_EQ(expected_output(1), output(1).val_);
  EXPECT_FLOAT_EQ(expected_output(2), output(2).val_);
  EXPECT_FLOAT_EQ(expected_output(3), output(3).val_);
  EXPECT_FLOAT_EQ(expected_output(4), output(4).val_);  
  EXPECT_FLOAT_EQ(1, output(0).d_);
  EXPECT_FLOAT_EQ(1, output(1).d_);
  EXPECT_FLOAT_EQ(1, output(2).d_);
  EXPECT_FLOAT_EQ(1, output(3).d_);
  EXPECT_FLOAT_EQ(1, output(4).d_);  

  output = subtract(rvd_1, rvv_2);
  EXPECT_FLOAT_EQ(expected_output(0), output(0).val_);
  EXPECT_FLOAT_EQ(expected_output(1), output(1).val_);
  EXPECT_FLOAT_EQ(expected_output(2), output(2).val_);
  EXPECT_FLOAT_EQ(expected_output(3), output(3).val_);
  EXPECT_FLOAT_EQ(expected_output(4), output(4).val_);  
  EXPECT_FLOAT_EQ(-1, output(0).d_);
  EXPECT_FLOAT_EQ(-1, output(1).d_);
  EXPECT_FLOAT_EQ(-1, output(2).d_);
  EXPECT_FLOAT_EQ(-1, output(3).d_);
  EXPECT_FLOAT_EQ(-1, output(4).d_);   

  output = subtract(rvv_1, rvv_2);
  EXPECT_FLOAT_EQ(expected_output(0), output(0).val_);
  EXPECT_FLOAT_EQ(expected_output(1), output(1).val_);
  EXPECT_FLOAT_EQ(expected_output(2), output(2).val_);
  EXPECT_FLOAT_EQ(expected_output(3), output(3).val_);
  EXPECT_FLOAT_EQ(expected_output(4), output(4).val_);  
  EXPECT_FLOAT_EQ(0, output(0).d_);
  EXPECT_FLOAT_EQ(0, output(1).d_);
  EXPECT_FLOAT_EQ(0, output(2).d_);
  EXPECT_FLOAT_EQ(0, output(3).d_);
  EXPECT_FLOAT_EQ(0, output(4).d_);  
}
TEST(AgradFwdMatrixOperatorSubtraction,fd_rowvector_rowvector_exception) {
  using stan::math::subtract;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

  row_vector_d d1(5), d2(2);
  row_vector_fd v1(5), v2(2);

  row_vector_fd output;
  EXPECT_THROW(subtract(d1, d2), std::invalid_argument);
  EXPECT_THROW(subtract(d1, v2), std::invalid_argument);
  EXPECT_THROW(subtract(v1, d2), std::invalid_argument);
  EXPECT_THROW(subtract(v1, v2), std::invalid_argument);
}
TEST(AgradFwdMatrixOperatorSubtraction,fd_matrix_matrix) {
  using stan::math::subtract;
  using stan::math::matrix_d;
  using stan::math::matrix_fd;
  
  matrix_d expected_output(2,2);
  matrix_fd output;
  matrix_d md_1(2,2), md_2(2,2);
  matrix_fd mv_1(2,2), mv_2(2,2);
  matrix_d md_mis (2, 3);
  matrix_fd mv_mis (1, 1);

  md_1 << -10, 1, 10, 0;
  mv_1 << -10, 1, 10, 0;
   mv_1(0,0).d_ = 1.0;
   mv_1(0,1).d_ = 1.0;
   mv_1(1,0).d_ = 1.0;
   mv_1(1,1).d_ = 1.0;
  md_2 << 10, -10, 1, 2;
  mv_2 << 10, -10, 1, 2;
   mv_2(0,0).d_ = 1.0;
   mv_2(0,1).d_ = 1.0;
   mv_2(1,0).d_ = 1.0;
   mv_2(1,1).d_ = 1.0;
  
  expected_output << -20, 11, 9, -2;
  
  matrix_d output_d = subtract(md_1, md_2);
  EXPECT_FLOAT_EQ(expected_output(0,0), output_d(0,0));
  EXPECT_FLOAT_EQ(expected_output(0,1), output_d(0,1));
  EXPECT_FLOAT_EQ(expected_output(1,0), output_d(1,0));
  EXPECT_FLOAT_EQ(expected_output(1,1), output_d(1,1));

  output = subtract(mv_1, md_2);
  EXPECT_FLOAT_EQ(expected_output(0,0), output(0,0).val_);
  EXPECT_FLOAT_EQ(expected_output(0,1), output(0,1).val_);
  EXPECT_FLOAT_EQ(expected_output(1,0), output(1,0).val_);
  EXPECT_FLOAT_EQ(expected_output(1,1), output(1,1).val_);
  EXPECT_FLOAT_EQ(1, output(0,0).d_);
  EXPECT_FLOAT_EQ(1, output(0,1).d_);
  EXPECT_FLOAT_EQ(1, output(1,0).d_);
  EXPECT_FLOAT_EQ(1, output(1,1).d_);

  output = subtract(md_1, mv_2);
  EXPECT_FLOAT_EQ(expected_output(0,0), output(0,0).val_);
  EXPECT_FLOAT_EQ(expected_output(0,1), output(0,1).val_);
  EXPECT_FLOAT_EQ(expected_output(1,0), output(1,0).val_);
  EXPECT_FLOAT_EQ(expected_output(1,1), output(1,1).val_);
  EXPECT_FLOAT_EQ(-1, output(0,0).d_);
  EXPECT_FLOAT_EQ(-1, output(0,1).d_);
  EXPECT_FLOAT_EQ(-1, output(1,0).d_);
  EXPECT_FLOAT_EQ(-1, output(1,1).d_);

  output = subtract(mv_1, mv_2);
  EXPECT_FLOAT_EQ(expected_output(0,0), output(0,0).val_);
  EXPECT_FLOAT_EQ(expected_output(0,1), output(0,1).val_);
  EXPECT_FLOAT_EQ(expected_output(1,0), output(1,0).val_);
  EXPECT_FLOAT_EQ(expected_output(1,1), output(1,1).val_);
  EXPECT_FLOAT_EQ(0, output(0,0).d_);
  EXPECT_FLOAT_EQ(0, output(0,1).d_);
  EXPECT_FLOAT_EQ(0, output(1,0).d_);
  EXPECT_FLOAT_EQ(0, output(1,1).d_);
}
TEST(AgradFwdMatrixOperatorSubtraction,fd_matrix_matrix_exception) {
  using stan::math::subtract;
  using stan::math::matrix_d;
  using stan::math::matrix_fd;

  matrix_d d1(2,2), d2(1,2);
  matrix_fd v1(2,2), v2(1,2);

  EXPECT_THROW(subtract(d1, d2), std::invalid_argument);
  EXPECT_THROW(subtract(d1, v2), std::invalid_argument);
  EXPECT_THROW(subtract(v1, d2), std::invalid_argument);
  EXPECT_THROW(subtract(v1, v2), std::invalid_argument);
}
TEST(AgradFwdMatrixOperatorSubtraction,ffd_scalar_matrix) {
  using stan::math::subtract;
  using stan::math::matrix_ffd;

 fvar<fvar<double> > a,b,c,d,e,f,g,h;
  a.val_.val_ = 1.0;
  b.val_.val_ = 2.0;
  c.val_.val_ = 3.0;
  d.val_.val_ = 4.0;
  a.d_.val_ = 1.0;
  b.d_.val_ = 1.0;
  c.d_.val_ = 1.0;
  d.d_.val_ = 1.0;

  matrix_ffd v(2,2);
  v << a,b,c,d;
  matrix_ffd result;

  result = subtract(2.0,v);
  EXPECT_FLOAT_EQ(1.0,result(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.0,result(0,1).val_.val());
  EXPECT_FLOAT_EQ(-1.0,result(1,0).val_.val());
  EXPECT_FLOAT_EQ(-2.0,result(1,1).val_.val());
  EXPECT_FLOAT_EQ(-1.0,result(0,0).d_.val());
  EXPECT_FLOAT_EQ(-1.0,result(0,1).d_.val());
  EXPECT_FLOAT_EQ(-1.0,result(1,0).d_.val());
  EXPECT_FLOAT_EQ(-1.0,result(1,1).d_.val());

  result = subtract(v,2.0);
  EXPECT_FLOAT_EQ(-1.0,result(0,0).val_.val());
  EXPECT_FLOAT_EQ(0.0,result(0,1).val_.val());
  EXPECT_FLOAT_EQ(1.0,result(1,0).val_.val());
  EXPECT_FLOAT_EQ(2.0,result(1,1).val_.val());
  EXPECT_FLOAT_EQ(1.0,result(0,0).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(0,1).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(1,0).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(1,1).d_.val());
}
TEST(AgradFwdMatrixOperatorSubtraction,ffd_scalar_vector) {
  using stan::math::subtract;
  using stan::math::vector_ffd;

  fvar<fvar<double> > a,b,c,d;
  a.val_.val_ = 1.0;
  b.val_.val_ = 2.0;
  c.val_.val_ = 3.0;
  d.val_.val_ = 4.0;
  a.d_.val_ = 1.0;
  b.d_.val_ = 1.0;
  c.d_.val_ = 1.0;
  d.d_.val_ = 1.0;

  vector_ffd v(4);
  v << a,b,c,d;
  vector_ffd result;

  result = subtract(2.0,v);
  EXPECT_FLOAT_EQ(1.0,result(0).val_.val());
  EXPECT_FLOAT_EQ(0.0,result(1).val_.val());
  EXPECT_FLOAT_EQ(-1.0,result(2).val_.val());
  EXPECT_FLOAT_EQ(-2.0,result(3).val_.val());
  EXPECT_FLOAT_EQ(-1.0,result(0).d_.val());
  EXPECT_FLOAT_EQ(-1.0,result(1).d_.val());
  EXPECT_FLOAT_EQ(-1.0,result(3).d_.val());
  EXPECT_FLOAT_EQ(-1.0,result(3).d_.val());

  result = subtract(v,2.0);
  EXPECT_FLOAT_EQ(-1.0,result(0).val_.val());
  EXPECT_FLOAT_EQ(0.0,result(1).val_.val());
  EXPECT_FLOAT_EQ(1.0,result(2).val_.val());
  EXPECT_FLOAT_EQ(2.0,result(3).val_.val());
  EXPECT_FLOAT_EQ(1.0,result(0).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(1).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(3).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(3).d_.val());
}
TEST(AgradFwdMatrixOperatorSubtraction,ffd_scalar_rowvector) {
  using stan::math::subtract;
  using stan::math::row_vector_ffd;

 fvar<fvar<double> > a,b,c,d;
  a.val_.val_ = 1.0;
  b.val_.val_ = 2.0;
  c.val_.val_ = 3.0;
  d.val_.val_ = 4.0;
  a.d_.val_ = 1.0;
  b.d_.val_ = 1.0;
  c.d_.val_ = 1.0;
  d.d_.val_ = 1.0;

  row_vector_ffd v(4);
  v << a,b,c,d;
  row_vector_ffd result;

  result = subtract(2.0,v);
  EXPECT_FLOAT_EQ(1.0,result(0).val_.val());
  EXPECT_FLOAT_EQ(0.0,result(1).val_.val());
  EXPECT_FLOAT_EQ(-1.0,result(2).val_.val());
  EXPECT_FLOAT_EQ(-2.0,result(3).val_.val());
  EXPECT_FLOAT_EQ(-1.0,result(0).d_.val());
  EXPECT_FLOAT_EQ(-1.0,result(1).d_.val());
  EXPECT_FLOAT_EQ(-1.0,result(3).d_.val());
  EXPECT_FLOAT_EQ(-1.0,result(3).d_.val());

  result = subtract(v,2.0);
  EXPECT_FLOAT_EQ(-1.0,result(0).val_.val());
  EXPECT_FLOAT_EQ(0.0,result(1).val_.val());
  EXPECT_FLOAT_EQ(1.0,result(2).val_.val());
  EXPECT_FLOAT_EQ(2.0,result(3).val_.val());
  EXPECT_FLOAT_EQ(1.0,result(0).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(1).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(3).d_.val());
  EXPECT_FLOAT_EQ(1.0,result(3).d_.val());
}
TEST(AgradFwdMatrixOperatorSubtraction,ffd_vector_vector) {
  using stan::math::subtract;
  using stan::math::vector_d;
  using stan::math::vector_ffd;

  vector_d expected_output(5);
  vector_ffd output;
  vector_d output_d;
  vector_d vd_1(5), vd_2(5);
  vector_ffd vv_1(5), vv_2(5);

  fvar<fvar<double> > a,b,c,d,e,f,g,h;
  a.val_.val_ = 0.0;
  b.val_.val_ = 2.0;
  c.val_.val_ = -6.0;
  d.val_.val_ = 10.0;
  e.val_.val_ = 6.0;
  f.val_.val_ = 3.0;
  g.val_.val_ = 4.0;
  h.val_.val_ = 5.0;
  a.d_.val_ = 1.0;
  b.d_.val_ = 1.0;
  c.d_.val_ = 1.0;
  d.d_.val_ = 1.0;
  e.d_.val_ = 1.0;
  f.d_.val_ = 1.0;
  g.d_.val_ = 1.0;
  h.d_.val_ = 1.0;

  vd_1 << 0, 2, -6, 10, 6;
  vv_1 << a,b,c,d,e;
  vd_2 << 2, 3, 4, 5, 6;
  vv_2 << b,f,g,h,e;

  expected_output << -2, -1, -10, 5, 0;
  
  output_d = subtract(vd_1, vd_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_d(0));
  EXPECT_FLOAT_EQ(expected_output(1), output_d(1));
  EXPECT_FLOAT_EQ(expected_output(2), output_d(2));
  EXPECT_FLOAT_EQ(expected_output(3), output_d(3));
  EXPECT_FLOAT_EQ(expected_output(4), output_d(4));  

  output = subtract(vv_1, vd_2);
  EXPECT_FLOAT_EQ(expected_output(0), output(0).val_.val());
  EXPECT_FLOAT_EQ(expected_output(1), output(1).val_.val());
  EXPECT_FLOAT_EQ(expected_output(2), output(2).val_.val());
  EXPECT_FLOAT_EQ(expected_output(3), output(3).val_.val());
  EXPECT_FLOAT_EQ(expected_output(4), output(4).val_.val());  
  EXPECT_FLOAT_EQ(1, output(0).d_.val());
  EXPECT_FLOAT_EQ(1, output(1).d_.val());
  EXPECT_FLOAT_EQ(1, output(2).d_.val());
  EXPECT_FLOAT_EQ(1, output(3).d_.val());
  EXPECT_FLOAT_EQ(1, output(4).d_.val());  

  output = subtract(vd_1, vv_2);
  EXPECT_FLOAT_EQ(expected_output(0), output(0).val_.val());
  EXPECT_FLOAT_EQ(expected_output(1), output(1).val_.val());
  EXPECT_FLOAT_EQ(expected_output(2), output(2).val_.val());
  EXPECT_FLOAT_EQ(expected_output(3), output(3).val_.val());
  EXPECT_FLOAT_EQ(expected_output(4), output(4).val_.val());  
  EXPECT_FLOAT_EQ(-1.0, output(0).d_.val());
  EXPECT_FLOAT_EQ(-1.0, output(1).d_.val());
  EXPECT_FLOAT_EQ(-1.0, output(2).d_.val());
  EXPECT_FLOAT_EQ(-1.0, output(3).d_.val());
  EXPECT_FLOAT_EQ(-1.0, output(4).d_.val());  

  output = subtract(vv_1, vv_2);
  EXPECT_FLOAT_EQ(expected_output(0), output(0).val_.val());
  EXPECT_FLOAT_EQ(expected_output(1), output(1).val_.val());
  EXPECT_FLOAT_EQ(expected_output(2), output(2).val_.val());
  EXPECT_FLOAT_EQ(expected_output(3), output(3).val_.val());
  EXPECT_FLOAT_EQ(expected_output(4), output(4).val_.val());  
  EXPECT_FLOAT_EQ(0, output(0).d_.val());
  EXPECT_FLOAT_EQ(0, output(1).d_.val());
  EXPECT_FLOAT_EQ(0, output(2).d_.val());
  EXPECT_FLOAT_EQ(0, output(3).d_.val());
  EXPECT_FLOAT_EQ(0, output(4).d_.val());  
}
TEST(AgradFwdMatrixOperatorSubtraction,ffd_vector_vector_exception) {
  using stan::math::subtract;
  using stan::math::vector_d;
  using stan::math::vector_ffd;

  vector_d d1(5), d2(1);
  vector_ffd v1(5), v2(1);
  
  vector_ffd output;
  EXPECT_THROW(subtract(d1, d2), std::invalid_argument);
  EXPECT_THROW(subtract(v1, d2), std::invalid_argument);
  EXPECT_THROW(subtract(d1, v2), std::invalid_argument);
  EXPECT_THROW(subtract(v1, v2), std::invalid_argument);
}
TEST(AgradFwdMatrixOperatorSubtraction,ffd_rowvector_rowvector) {
  using stan::math::subtract;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;

  row_vector_d expected_output(5);
  row_vector_d  output_d;
  row_vector_ffd  output;
  row_vector_d rvd_1(5), rvd_2(5);
  row_vector_ffd rvv_1(5), rvv_2(5);

  fvar<fvar<double> > a,b,c,d,e,f,g,h;
  a.val_.val_ = 0.0;
  b.val_.val_ = 2.0;
  c.val_.val_ = -6.0;
  d.val_.val_ = 10.0;
  e.val_.val_ = 6.0;
  f.val_.val_ = 3.0;
  g.val_.val_ = 4.0;
  h.val_.val_ = 5.0;
  a.d_.val_ = 1.0;
  b.d_.val_ = 1.0;
  c.d_.val_ = 1.0;
  d.d_.val_ = 1.0;
  e.d_.val_ = 1.0;
  f.d_.val_ = 1.0;
  g.d_.val_ = 1.0;
  h.d_.val_ = 1.0;

  rvd_1 << 0, 2, -6, 10, 6;
  rvv_1 << a,b,c,d,e;
  rvd_2 << 2, 3, 4, 5, 6;
  rvv_2 << b,f,g,h,e;
  
  expected_output << -2, -1, -10, 5, 0;
  
  output_d = subtract(rvd_1, rvd_2);
  EXPECT_FLOAT_EQ(expected_output(0), output_d(0));
  EXPECT_FLOAT_EQ(expected_output(1), output_d(1));
  EXPECT_FLOAT_EQ(expected_output(2), output_d(2));
  EXPECT_FLOAT_EQ(expected_output(3), output_d(3));
  EXPECT_FLOAT_EQ(expected_output(4), output_d(4));

  output = subtract(rvv_1, rvd_2);
  EXPECT_FLOAT_EQ(expected_output(0), output(0).val_.val());
  EXPECT_FLOAT_EQ(expected_output(1), output(1).val_.val());
  EXPECT_FLOAT_EQ(expected_output(2), output(2).val_.val());
  EXPECT_FLOAT_EQ(expected_output(3), output(3).val_.val());
  EXPECT_FLOAT_EQ(expected_output(4), output(4).val_.val());  
  EXPECT_FLOAT_EQ(1, output(0).d_.val());
  EXPECT_FLOAT_EQ(1, output(1).d_.val());
  EXPECT_FLOAT_EQ(1, output(2).d_.val());
  EXPECT_FLOAT_EQ(1, output(3).d_.val());
  EXPECT_FLOAT_EQ(1, output(4).d_.val());  

  output = subtract(rvd_1, rvv_2);
  EXPECT_FLOAT_EQ(expected_output(0), output(0).val_.val());
  EXPECT_FLOAT_EQ(expected_output(1), output(1).val_.val());
  EXPECT_FLOAT_EQ(expected_output(2), output(2).val_.val());
  EXPECT_FLOAT_EQ(expected_output(3), output(3).val_.val());
  EXPECT_FLOAT_EQ(expected_output(4), output(4).val_.val());  
  EXPECT_FLOAT_EQ(-1, output(0).d_.val());
  EXPECT_FLOAT_EQ(-1, output(1).d_.val());
  EXPECT_FLOAT_EQ(-1, output(2).d_.val());
  EXPECT_FLOAT_EQ(-1, output(3).d_.val());
  EXPECT_FLOAT_EQ(-1, output(4).d_.val());   

  output = subtract(rvv_1, rvv_2);
  EXPECT_FLOAT_EQ(expected_output(0), output(0).val_.val());
  EXPECT_FLOAT_EQ(expected_output(1), output(1).val_.val());
  EXPECT_FLOAT_EQ(expected_output(2), output(2).val_.val());
  EXPECT_FLOAT_EQ(expected_output(3), output(3).val_.val());
  EXPECT_FLOAT_EQ(expected_output(4), output(4).val_.val());  
  EXPECT_FLOAT_EQ(0, output(0).d_.val());
  EXPECT_FLOAT_EQ(0, output(1).d_.val());
  EXPECT_FLOAT_EQ(0, output(2).d_.val());
  EXPECT_FLOAT_EQ(0, output(3).d_.val());
  EXPECT_FLOAT_EQ(0, output(4).d_.val());  
}
TEST(AgradFwdMatrixOperatorSubtraction,ffd_rowvector_rowvector_exception) {
  using stan::math::subtract;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;

  row_vector_d d1(5), d2(2);
  row_vector_ffd v1(5), v2(2);

  row_vector_ffd output;
  EXPECT_THROW(subtract(d1, d2), std::invalid_argument);
  EXPECT_THROW(subtract(d1, v2), std::invalid_argument);
  EXPECT_THROW(subtract(v1, d2), std::invalid_argument);
  EXPECT_THROW(subtract(v1, v2), std::invalid_argument);
}
TEST(AgradFwdMatrixOperatorSubtraction,ffd_matrix_matrix) {
  using stan::math::subtract;
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;
  
  matrix_d expected_output(2,2);
  matrix_ffd output;
  matrix_d md_1(2,2), md_2(2,2);
  matrix_ffd mv_1(2,2), mv_2(2,2);
  matrix_d md_mis (2, 3);
  matrix_ffd mv_mis (1, 1);

  fvar<fvar<double> > a,b,c,d,e;
  a.val_.val_ = -10.0;
  b.val_.val_ = 1.0;
  c.val_.val_ = 10.0;
  d.val_.val_ = 0.0;
  e.val_.val_ = 2.0;
  a.d_.val_ = 1.0;
  b.d_.val_ = 1.0;
  c.d_.val_ = 1.0;
  d.d_.val_ = 1.0;
  e.d_.val_ = 1.0;

  md_1 << -10, 1, 10, 0;
  mv_1 << a,b,c,d;
  md_2 << 10, -10, 1, 2;
  mv_2 << c,a,b,e;
  
  expected_output << -20, 11, 9, -2;
  
  matrix_d output_d = subtract(md_1, md_2);
  EXPECT_FLOAT_EQ(expected_output(0,0), output_d(0,0));
  EXPECT_FLOAT_EQ(expected_output(0,1), output_d(0,1));
  EXPECT_FLOAT_EQ(expected_output(1,0), output_d(1,0));
  EXPECT_FLOAT_EQ(expected_output(1,1), output_d(1,1));

  output = subtract(mv_1, md_2);
  EXPECT_FLOAT_EQ(expected_output(0,0), output(0,0).val_.val());
  EXPECT_FLOAT_EQ(expected_output(0,1), output(0,1).val_.val());
  EXPECT_FLOAT_EQ(expected_output(1,0), output(1,0).val_.val());
  EXPECT_FLOAT_EQ(expected_output(1,1), output(1,1).val_.val());
  EXPECT_FLOAT_EQ(1, output(0,0).d_.val());
  EXPECT_FLOAT_EQ(1, output(0,1).d_.val());
  EXPECT_FLOAT_EQ(1, output(1,0).d_.val());
  EXPECT_FLOAT_EQ(1, output(1,1).d_.val());

  output = subtract(md_1, mv_2);
  EXPECT_FLOAT_EQ(expected_output(0,0), output(0,0).val_.val());
  EXPECT_FLOAT_EQ(expected_output(0,1), output(0,1).val_.val());
  EXPECT_FLOAT_EQ(expected_output(1,0), output(1,0).val_.val());
  EXPECT_FLOAT_EQ(expected_output(1,1), output(1,1).val_.val());
  EXPECT_FLOAT_EQ(-1, output(0,0).d_.val());
  EXPECT_FLOAT_EQ(-1, output(0,1).d_.val());
  EXPECT_FLOAT_EQ(-1, output(1,0).d_.val());
  EXPECT_FLOAT_EQ(-1, output(1,1).d_.val());

  output = subtract(mv_1, mv_2);
  EXPECT_FLOAT_EQ(expected_output(0,0), output(0,0).val_.val());
  EXPECT_FLOAT_EQ(expected_output(0,1), output(0,1).val_.val());
  EXPECT_FLOAT_EQ(expected_output(1,0), output(1,0).val_.val());
  EXPECT_FLOAT_EQ(expected_output(1,1), output(1,1).val_.val());
  EXPECT_FLOAT_EQ(0, output(0,0).d_.val());
  EXPECT_FLOAT_EQ(0, output(0,1).d_.val());
  EXPECT_FLOAT_EQ(0, output(1,0).d_.val());
  EXPECT_FLOAT_EQ(0, output(1,1).d_.val());
}
TEST(AgradFwdMatrixOperatorSubtraction,ffd_matrix_matrix_exception) {
  using stan::math::subtract;
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;

  matrix_d d1(2,2), d2(1,2);
  matrix_ffd v1(2,2), v2(1,2);

  EXPECT_THROW(subtract(d1, d2), std::invalid_argument);
  EXPECT_THROW(subtract(d1, v2), std::invalid_argument);
  EXPECT_THROW(subtract(v1, d2), std::invalid_argument);
  EXPECT_THROW(subtract(v1, v2), std::invalid_argument);
}

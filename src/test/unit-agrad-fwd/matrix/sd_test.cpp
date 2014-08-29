#include <stan/math/matrix/sd.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

TEST(AgradFwdMatrixSD, fd_vector) {
  using stan::math::sd;
  using stan::math::vector_d;
  using stan::agrad::vector_fd;

  vector_d v(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));

  vector_d d1(6);
  vector_fd v1(6);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
                   
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_);
  EXPECT_FLOAT_EQ(std::sqrt(1.0/14.0), sd(v1).d_);
  
  d1.resize(1);
  v1.resize(1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_);
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_);
}
TEST(AgradFwdMatrixSD, fd_vector_exception) {
  using stan::math::sd;
  using stan::math::vector_d;
  using stan::agrad::vector_fd;

  vector_d d1;
  vector_fd v1;
  EXPECT_THROW(sd(d1), std::domain_error);
  EXPECT_THROW(sd(v1), std::domain_error);
}
TEST(AgradFwdMatrixSD, fd_rowvector) {
  using stan::math::sd;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fd;

  row_vector_d v(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));


  row_vector_d d1(6);
  row_vector_fd v1(6);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
                   
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_);
  EXPECT_FLOAT_EQ(std::sqrt(1.0/14.0), sd(v1).d_);

  d1.resize(1);
  v1.resize(1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_);
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_);
}
TEST(AgradFwdMatrixSD, fd_rowvector_exception) {
  using stan::math::sd;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fd;

  row_vector_d d;
  row_vector_fd v;
  
  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);
}
TEST(AgradFwdMatrixSD, fd_matrix) {
  using stan::math::sd;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fd;

  matrix_d v(1,1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));

  matrix_d d1(2, 3);
  matrix_fd v1(2, 3);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_);
  EXPECT_FLOAT_EQ(std::sqrt(1.0/14.0), sd(v1).d_);

  d1.resize(1, 1);
  v1.resize(1, 1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_);
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_);
}
TEST(AgradFwdMatrixSD, fd_matrix_exception) {
  using stan::math::sd;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fd;

  matrix_d d;
  matrix_fd v;

  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);

  d.resize(1, 0);
  v.resize(1, 0);
  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);

  d.resize(0, 1);
  v.resize(0, 1);
  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);
}
TEST(AgradFwdMatrixSD, fv_vector_1stDeriv) {
  using stan::math::sd;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_d v(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));

  vector_d d1(6);
  vector_fv v1(6);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
                   
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_.val());
  EXPECT_FLOAT_EQ(std::sqrt(1.0/14.0), sd(v1).d_.val());

  AVEC q = createAVEC(v1(0).val(),v1(1).val(),v1(2).val(),v1(3).val(),v1(4).val(),v1(5).val());
  VEC h;
  sd(v1).val_.grad(q,h);
  EXPECT_FLOAT_EQ(-std::sqrt(1.0/14.0),h[0]);
  EXPECT_FLOAT_EQ(-std::sqrt(9.0/350.0),h[1]);
  EXPECT_FLOAT_EQ(-std::sqrt(1.0/350.0),h[2]);
  EXPECT_FLOAT_EQ(std::sqrt(1.0/350.0),h[3]);
  EXPECT_FLOAT_EQ(std::sqrt(9.0/350.0),h[4]);
  EXPECT_FLOAT_EQ(std::sqrt(1.0/14.0),h[5]);
  
  d1.resize(1);
  v1.resize(1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_.val());
}
TEST(AgradFwdMatrixSD, fv_vector_2ndDeriv) {
  using stan::math::sd;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_fv v1(6);
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;

  AVEC q = createAVEC(v1(0).val(),v1(1).val(),v1(2).val(),v1(3).val(),v1(4).val(),v1(5).val());
  VEC h;
  sd(v1).d_.grad(q,h);
  EXPECT_FLOAT_EQ(-std::sqrt(8.0/3087.0),h[0]);
  EXPECT_FLOAT_EQ(8.0 / 105.0 * std::sqrt(2.0/7.0),h[1]);
  EXPECT_FLOAT_EQ(std::sqrt(2.0/7.0/441.0),h[2]);
  EXPECT_FLOAT_EQ(2.0 /105.0 * std::sqrt(2.0/7.0),h[3]);
  EXPECT_FLOAT_EQ(-std::sqrt(2.0/7.0)/105.0,h[4]);
  EXPECT_FLOAT_EQ(-4.0/105.0 * std::sqrt(2.0/7.0),h[5]);
}
TEST(AgradFwdMatrixSD, fv_vector_exception) {
  using stan::math::sd;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_d d1;
  vector_fv v1;
  EXPECT_THROW(sd(d1), std::domain_error);
  EXPECT_THROW(sd(v1), std::domain_error);
}
TEST(AgradFwdMatrixSD, fv_rowvector_1stDeriv) {
  using stan::math::sd;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d v(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));


  row_vector_d d1(6);
  row_vector_fv v1(6);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
                   
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.26726124, sd(v1).d_.val());

  AVEC q = createAVEC(v1(0).val(),v1(1).val(),v1(2).val(),v1(3).val(),v1(4).val(),v1(5).val());
  VEC h;
  sd(v1).val_.grad(q,h);
  EXPECT_FLOAT_EQ(-std::sqrt(1.0/14.0),h[0]);
  EXPECT_FLOAT_EQ(-std::sqrt(9.0/350.0),h[1]);
  EXPECT_FLOAT_EQ(-std::sqrt(1.0/350.0),h[2]);
  EXPECT_FLOAT_EQ(std::sqrt(1.0/350.0),h[3]);
  EXPECT_FLOAT_EQ(std::sqrt(9.0/350.0),h[4]);
  EXPECT_FLOAT_EQ(std::sqrt(1.0/14.0),h[5]);

  d1.resize(1);
  v1.resize(1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_.val());
}
TEST(AgradFwdMatrixSD, fv_rowvector_2ndDeriv) {
  using stan::math::sd;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_fv v1(6);
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  AVEC q = createAVEC(v1(0).val(),v1(1).val(),v1(2).val(),v1(3).val(),v1(4).val(),v1(5).val());
  VEC h;
  sd(v1).d_.grad(q,h);
  EXPECT_FLOAT_EQ(-std::sqrt(8.0/3087.0),h[0]);
  EXPECT_FLOAT_EQ(8.0 / 105.0 * std::sqrt(2.0/7.0),h[1]);
  EXPECT_FLOAT_EQ(std::sqrt(2.0/7.0/441.0),h[2]);
  EXPECT_FLOAT_EQ(2.0 /105.0 * std::sqrt(2.0/7.0),h[3]);
  EXPECT_FLOAT_EQ(-std::sqrt(2.0/7.0)/105.0,h[4]);
  EXPECT_FLOAT_EQ(-4.0/105.0 * std::sqrt(2.0/7.0),h[5]);
}
TEST(AgradFwdMatrixSD, fv_rowvector_exception) {
  using stan::math::sd;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d d;
  row_vector_fv v;
  
  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);
}
TEST(AgradFwdMatrixSD, fv_matrix_1stDeriv) {
  using stan::math::sd;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;

  matrix_d v(1,1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));

  matrix_d d1(2, 3);
  matrix_fv v1(2, 3);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.26726124, sd(v1).d_.val());
  
  AVEC q = createAVEC(v1(0,0).val(),v1(0,1).val(),v1(0,2).val(),v1(1,0).val(),v1(1,1).val(),v1(1,2).val());
  VEC h;
  sd(v1).val_.grad(q,h);
  EXPECT_FLOAT_EQ(-std::sqrt(1.0/14.0),h[0]);
  EXPECT_FLOAT_EQ(-std::sqrt(9.0/350.0),h[1]);
  EXPECT_FLOAT_EQ(-std::sqrt(1.0/350.0),h[2]);
  EXPECT_FLOAT_EQ(std::sqrt(1.0/350.0),h[3]);
  EXPECT_FLOAT_EQ(std::sqrt(9.0/350.0),h[4]);
  EXPECT_FLOAT_EQ(std::sqrt(1.0/14.0),h[5]);

  d1.resize(1, 1);
  v1.resize(1, 1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_.val());
}
TEST(AgradFwdMatrixSD, fv_matrix_2ndDeriv) {
  using stan::math::sd;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;

  matrix_fv v1(2, 3);
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  AVEC q = createAVEC(v1(0,0).val(),v1(0,1).val(),v1(0,2).val(),v1(1,0).val(),v1(1,1).val(),v1(1,2).val());
  VEC h;
  sd(v1).d_.grad(q,h);
  EXPECT_FLOAT_EQ(-std::sqrt(8.0/3087.0),h[0]);
  EXPECT_FLOAT_EQ(8.0 / 105.0 * std::sqrt(2.0/7.0),h[1]);
  EXPECT_FLOAT_EQ(std::sqrt(2.0/7.0/441.0),h[2]);
  EXPECT_FLOAT_EQ(2.0 /105.0 * std::sqrt(2.0/7.0),h[3]);
  EXPECT_FLOAT_EQ(-std::sqrt(2.0/7.0)/105.0,h[4]);
  EXPECT_FLOAT_EQ(-4.0/105.0 * std::sqrt(2.0/7.0),h[5]);
}
TEST(AgradFwdMatrixSD, fv_matrix_exception) {
  using stan::math::sd;
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;

  matrix_d d;
  matrix_fv v;

  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);

  d.resize(1, 0);
  v.resize(1, 0);
  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);

  d.resize(0, 1);
  v.resize(0, 1);
  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);
}
TEST(AgradFwdMatrixSD, ffd_vector) {
  using stan::math::sd;
  using stan::math::vector_d;
  using stan::agrad::vector_ffd;

  vector_d v(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));

  vector_d d1(6);
  vector_ffd v1(6);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
                   
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.26726124, sd(v1).d_.val());
  
  d1.resize(1);
  v1.resize(1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_.val());
}
TEST(AgradFwdMatrixSD, ffd_vector_exception) {
  using stan::math::sd;
  using stan::math::vector_d;
  using stan::agrad::vector_ffd;

  vector_d d1;
  vector_ffd v1;
  EXPECT_THROW(sd(d1), std::domain_error);
  EXPECT_THROW(sd(v1), std::domain_error);
}
TEST(AgradFwdMatrixSD, ffd_rowvector) {
  using stan::math::sd;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffd;

  row_vector_d v(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));


  row_vector_d d1(6);
  row_vector_ffd v1(6);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
                   
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.26726124, sd(v1).d_.val());

  d1.resize(1);
  v1.resize(1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_.val());
}
TEST(AgradFwdMatrixSD, ffd_rowvector_exception) {
  using stan::math::sd;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffd;

  row_vector_d d;
  row_vector_ffd v;
  
  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);
}
TEST(AgradFwdMatrixSD, ffd_matrix) {
  using stan::math::sd;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffd;

  matrix_d v(1,1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));

  matrix_d d1(2, 3);
  matrix_ffd v1(2, 3);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.26726124, sd(v1).d_.val());

  d1.resize(1, 1);
  v1.resize(1, 1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_.val());
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_.val());
}
TEST(AgradFwdMatrixSD, ffd_matrix_exception) {
  using stan::math::sd;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffd;

  matrix_d d;
  matrix_ffd v;

  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);

  d.resize(1, 0);
  v.resize(1, 0);
  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);

  d.resize(0, 1);
  v.resize(0, 1);
  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);
}
TEST(AgradFwdMatrixSD, ffv_vector_1stDeriv) {
  using stan::math::sd;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;

  vector_d v(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));

  vector_d d1(6);
  vector_ffv v1(6);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
                   
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_.val().val());
  EXPECT_FLOAT_EQ(std::sqrt(1.0/14.0), sd(v1).d_.val().val());

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val(),v1(3).val().val(),v1(4).val().val(),v1(5).val().val());
  VEC h;
  sd(v1).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-std::sqrt(1.0/14.0),h[0]);
  EXPECT_FLOAT_EQ(-std::sqrt(9.0/350.0),h[1]);
  EXPECT_FLOAT_EQ(-std::sqrt(1.0/350.0),h[2]);
  EXPECT_FLOAT_EQ(std::sqrt(1.0/350.0),h[3]);
  EXPECT_FLOAT_EQ(std::sqrt(9.0/350.0),h[4]);
  EXPECT_FLOAT_EQ(std::sqrt(1.0/14.0),h[5]);
  
  d1.resize(1);
  v1.resize(1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_.val().val());
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_.val().val());
}
TEST(AgradFwdMatrixSD, ffv_vector_2ndDeriv_1) {
  using stan::math::sd;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;

  vector_ffv v1(6);
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val(),v1(3).val().val(),v1(4).val().val(),v1(5).val().val());
  VEC h;
  sd(v1).val().d_.grad(q,h);  
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradFwdMatrixSD, ffv_vector_2ndDeriv_2) {
  using stan::math::sd;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;

  vector_ffv v1(6);
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val(),v1(3).val().val(),v1(4).val().val(),v1(5).val().val());
  VEC h;
  sd(v1).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-std::sqrt(8.0/3087.0),h[0]);
  EXPECT_FLOAT_EQ(8.0 / 105.0 * std::sqrt(2.0/7.0),h[1]);
  EXPECT_FLOAT_EQ(std::sqrt(2.0/7.0/441.0),h[2]);
  EXPECT_FLOAT_EQ(2.0 /105.0 * std::sqrt(2.0/7.0),h[3]);
  EXPECT_FLOAT_EQ(-std::sqrt(2.0/7.0)/105.0,h[4]);
  EXPECT_FLOAT_EQ(-4.0/105.0 * std::sqrt(2.0/7.0),h[5]);
}
TEST(AgradFwdMatrixSD, ffv_vector_3rdDeriv) {
  using stan::math::sd;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;

  vector_ffv v1(6);
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_.val_ = 1.0;
   v1(1).d_.val_ = 1.0;
   v1(2).d_.val_ = 1.0;
   v1(3).d_.val_ = 1.0;
   v1(4).d_.val_ = 1.0;
   v1(5).d_.val_ = 1.0;
   v1(0).val_.d_ = 1.0;
   v1(1).val_.d_ = 1.0;
   v1(2).val_.d_ = 1.0;
   v1(3).val_.d_ = 1.0;
   v1(4).val_.d_ = 1.0;
   v1(5).val_.d_ = 1.0;

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val(),v1(3).val().val(),v1(4).val().val(),v1(5).val().val());
  VEC h;
  sd(v1).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradFwdMatrixSD, ffv_vector_exception) {
  using stan::math::sd;
  using stan::math::vector_d;
  using stan::agrad::vector_ffv;

  vector_d d1;
  vector_ffv v1;
  EXPECT_THROW(sd(d1), std::domain_error);
  EXPECT_THROW(sd(v1), std::domain_error);
}
TEST(AgradFwdMatrixSD, ffv_rowvector_1stDeriv) {
  using stan::math::sd;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;

  row_vector_d v(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));


  row_vector_d d1(6);
  row_vector_ffv v1(6);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
                   
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_.val().val());
  EXPECT_FLOAT_EQ(0.26726124, sd(v1).d_.val().val());

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val(),v1(3).val().val(),v1(4).val().val(),v1(5).val().val());
  VEC h;
  sd(v1).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-std::sqrt(1.0/14.0),h[0]);
  EXPECT_FLOAT_EQ(-std::sqrt(9.0/350.0),h[1]);
  EXPECT_FLOAT_EQ(-std::sqrt(1.0/350.0),h[2]);
  EXPECT_FLOAT_EQ(std::sqrt(1.0/350.0),h[3]);
  EXPECT_FLOAT_EQ(std::sqrt(9.0/350.0),h[4]);
  EXPECT_FLOAT_EQ(std::sqrt(1.0/14.0),h[5]);

  d1.resize(1);
  v1.resize(1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_.val().val());
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_.val().val());
}
TEST(AgradFwdMatrixSD, ffv_rowvector_2ndDeriv_1) {
  using stan::math::sd;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;

  row_vector_ffv v1(6);
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val(),v1(3).val().val(),v1(4).val().val(),v1(5).val().val());
  VEC h;
  sd(v1).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradFwdMatrixSD, ffv_rowvector_2ndDeriv_2) {
  using stan::math::sd;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;

  row_vector_ffv v1(6);
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val(),v1(3).val().val(),v1(4).val().val(),v1(5).val().val());
  VEC h;
  sd(v1).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-std::sqrt(8.0/3087.0),h[0]);
  EXPECT_FLOAT_EQ(8.0 / 105.0 * std::sqrt(2.0/7.0),h[1]);
  EXPECT_FLOAT_EQ(std::sqrt(2.0/7.0/441.0),h[2]);
  EXPECT_FLOAT_EQ(2.0 /105.0 * std::sqrt(2.0/7.0),h[3]);
  EXPECT_FLOAT_EQ(-std::sqrt(2.0/7.0)/105.0,h[4]);
  EXPECT_FLOAT_EQ(-4.0/105.0 * std::sqrt(2.0/7.0),h[5]);
}
TEST(AgradFwdMatrixSD, ffv_rowvector_3rdDeriv) {
  using stan::math::sd;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;

  row_vector_ffv v1(6);
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_.val_ = 1.0;
   v1(1).d_.val_ = 1.0;
   v1(2).d_.val_ = 1.0;
   v1(3).d_.val_ = 1.0;
   v1(4).d_.val_ = 1.0;
   v1(5).d_.val_ = 1.0;
   v1(0).val_.d_ = 1.0;
   v1(1).val_.d_ = 1.0;
   v1(2).val_.d_ = 1.0;
   v1(3).val_.d_ = 1.0;
   v1(4).val_.d_ = 1.0;
   v1(5).val_.d_ = 1.0;
  
  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val(),v1(3).val().val(),v1(4).val().val(),v1(5).val().val());
  VEC h;
  sd(v1).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradFwdMatrixSD, ffv_rowvector_exception) {
  using stan::math::sd;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_ffv;

  row_vector_d d;
  row_vector_ffv v;
  
  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);
}
TEST(AgradFwdMatrixSD, ffv_matrix_1stDeriv) {
  using stan::math::sd;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;

  matrix_d v(1,1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0, sd(v));

  matrix_d d1(2, 3);
  matrix_ffv v1(2, 3);
  
  d1 << 1, 2, 3, 4, 5, 6;
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(d1));
  EXPECT_FLOAT_EQ(std::sqrt(17.5/5.0), sd(v1).val_.val().val());
  EXPECT_FLOAT_EQ(0.26726124, sd(v1).d_.val().val());
  
  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(0,2).val().val(),v1(1,0).val().val(),v1(1,1).val().val(),v1(1,2).val().val());
  VEC h;
  sd(v1).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-std::sqrt(1.0/14.0),h[0]);
  EXPECT_FLOAT_EQ(-std::sqrt(9.0/350.0),h[1]);
  EXPECT_FLOAT_EQ(-std::sqrt(1.0/350.0),h[2]);
  EXPECT_FLOAT_EQ(std::sqrt(1.0/350.0),h[3]);
  EXPECT_FLOAT_EQ(std::sqrt(9.0/350.0),h[4]);
  EXPECT_FLOAT_EQ(std::sqrt(1.0/14.0),h[5]);

  d1.resize(1, 1);
  v1.resize(1, 1);
  EXPECT_FLOAT_EQ(0.0, sd(d1));
  EXPECT_FLOAT_EQ(0.0, sd(v1).val_.val().val());
  EXPECT_FLOAT_EQ(0.0, sd(v1).d_.val().val());
}
TEST(AgradFwdMatrixSD, ffv_matrix_2ndDeriv_1) {
  using stan::math::sd;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;

  matrix_ffv v1(2, 3);
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(0,2).val().val(),v1(1,0).val().val(),v1(1,1).val().val(),v1(1,2).val().val());
  VEC h;
  sd(v1).val_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradFwdMatrixSD, ffv_matrix_2ndDeriv_2) {
  using stan::math::sd;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;

  matrix_ffv v1(2, 3);
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_ = 1.0;
   v1(1).d_ = 2.0;
   v1(2).d_ = 2.0;
   v1(3).d_ = 2.0;
   v1(4).d_ = 2.0;
   v1(5).d_ = 2.0;
  
  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(0,2).val().val(),v1(1,0).val().val(),v1(1,1).val().val(),v1(1,2).val().val());
  VEC h;
  sd(v1).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-std::sqrt(8.0/3087.0),h[0]);
  EXPECT_FLOAT_EQ(8.0 / 105.0 * std::sqrt(2.0/7.0),h[1]);
  EXPECT_FLOAT_EQ(std::sqrt(2.0/7.0/441.0),h[2]);
  EXPECT_FLOAT_EQ(2.0 /105.0 * std::sqrt(2.0/7.0),h[3]);
  EXPECT_FLOAT_EQ(-std::sqrt(2.0/7.0)/105.0,h[4]);
  EXPECT_FLOAT_EQ(-4.0/105.0 * std::sqrt(2.0/7.0),h[5]);
}
TEST(AgradFwdMatrixSD, ffv_matrix_3rdDeriv) {
  using stan::math::sd;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;

  matrix_ffv v1(2, 3);
  v1 << 1, 2, 3, 4, 5, 6;
   v1(0).d_.val_ = 1.0;
   v1(1).d_.val_ = 1.0;
   v1(2).d_.val_ = 1.0;
   v1(3).d_.val_ = 1.0;
   v1(4).d_.val_ = 1.0;
   v1(5).d_.val_ = 1.0;
   v1(0).val_.d_ = 1.0;
   v1(1).val_.d_ = 1.0;
   v1(2).val_.d_ = 1.0;
   v1(3).val_.d_ = 1.0;
   v1(4).val_.d_ = 1.0;
   v1(5).val_.d_ = 1.0;
  
  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(0,2).val().val(),v1(1,0).val().val(),v1(1,1).val().val(),v1(1,2).val().val());
  VEC h;
  sd(v1).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradFwdMatrixSD, ffv_matrix_exception) {
  using stan::math::sd;
  using stan::math::matrix_d;
  using stan::agrad::matrix_ffv;

  matrix_d d;
  matrix_ffv v;

  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);

  d.resize(1, 0);
  v.resize(1, 0);
  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);

  d.resize(0, 1);
  v.resize(0, 1);
  EXPECT_THROW(sd(d), std::domain_error);
  EXPECT_THROW(sd(v), std::domain_error);
}

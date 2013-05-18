#include <stan/agrad/rev/matrix/squared_dist.hpp>
#include <gtest/gtest.h>
#include <test/agrad/util.hpp>
#include <stan/agrad/agrad.hpp>

TEST(AgradRevMatrix, squared_dist_vector_vector) {
  using stan::math::vector_d;
  using stan::agrad::vector_v;

  vector_d vd_1(3), vd_2(3);
  vector_v vv_1(3), vv_2(3);
  
  vd_1 << 1, 3, -5;
  vv_1 << 1, 3, -5;
  vd_2 << 4, -2, -1;
  vv_2 << 4, -2, -1;

  EXPECT_FLOAT_EQ(50, stan::agrad::squared_dist(vv_1, vd_2).val());
  EXPECT_FLOAT_EQ(50, stan::agrad::squared_dist(vd_1, vv_2).val());
  EXPECT_FLOAT_EQ(50, stan::agrad::squared_dist(vv_1, vv_2).val());
}
TEST(AgradRevMatrix, squared_dist_vector_vector_exception) {
  using stan::math::vector_d;
  using stan::agrad::vector_v;

  vector_d d1(3);
  vector_v v1(3);
  vector_d d2(2);
  vector_v v2(4);

  EXPECT_THROW(stan::agrad::squared_dist(v1, d2), std::domain_error);
  EXPECT_THROW(stan::agrad::squared_dist(d1, v2), std::domain_error);
  EXPECT_THROW(stan::agrad::squared_dist(v1, v2), std::domain_error);
}
TEST(AgradRevMatrix, squared_dist_rowvector_vector) {
  using stan::math::vector_d;
  using stan::agrad::vector_v;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_v;

  row_vector_d d1(3);
  row_vector_v v1(3);
  vector_d d2(3);
  vector_v v2(3);
  
  d1 << 1, 3, -5;
  v1 << 1, 3, -5;
  d2 << 4, -2, -1;
  v2 << 4, -2, -1;

  EXPECT_FLOAT_EQ(50, stan::agrad::squared_dist(v1, d2).val());
  EXPECT_FLOAT_EQ(50, stan::agrad::squared_dist(d1, v2).val());
  EXPECT_FLOAT_EQ(50, stan::agrad::squared_dist(v1, v2).val());
}
TEST(AgradRevMatrix, squared_dist_rowvector_vector_exception) {
  using stan::math::vector_d;
  using stan::agrad::vector_v;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_v;

  row_vector_d d1(3);
  row_vector_v v1(3);
  vector_d d2(2);
  vector_v v2(4);

  EXPECT_THROW(stan::agrad::squared_dist(v1, d2), std::domain_error);
  EXPECT_THROW(stan::agrad::squared_dist(d1, v2), std::domain_error);
  EXPECT_THROW(stan::agrad::squared_dist(v1, v2), std::domain_error);
}
TEST(AgradRevMatrix, squared_dist_vector_rowvector) {
  using stan::math::vector_d;
  using stan::agrad::vector_v;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_v;

  vector_d d1(3);
  vector_v v1(3);
  row_vector_d d2(3);
  row_vector_v v2(3);
  
  d1 << 1, 3, -5;
  v1 << 1, 3, -5;
  d2 << 4, -2, -1;
  v2 << 4, -2, -1;
  
  EXPECT_FLOAT_EQ(50, stan::agrad::squared_dist(v1, d2).val());
  EXPECT_FLOAT_EQ(50, stan::agrad::squared_dist(d1, v2).val());
  EXPECT_FLOAT_EQ(50, stan::agrad::squared_dist(v1, v2).val());
}
TEST(AgradRevMatrix, squared_dist_vector_rowvector_exception) {
  using stan::math::vector_d;
  using stan::agrad::vector_v;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_v;

  vector_d d1(3);
  vector_v v1(3);
  row_vector_d d2(2);
  row_vector_v v2(4);

  EXPECT_THROW(stan::agrad::squared_dist(v1, d2), std::domain_error);
  EXPECT_THROW(stan::agrad::squared_dist(d1, v2), std::domain_error);
  EXPECT_THROW(stan::agrad::squared_dist(v1, v2), std::domain_error);
}
TEST(AgradRevMatrix, squared_dist_rowvector_rowvector) {
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_v;

  row_vector_d d1(3), d2(3);
  row_vector_v v1(3), v2(3);
  
  d1 << 1, 3, -5;
  v1 << 1, 3, -5;
  d2 << 4, -2, -1;
  v2 << 4, -2, -1;

  EXPECT_FLOAT_EQ(50, stan::agrad::squared_dist(v1, d2).val());
  EXPECT_FLOAT_EQ(50, stan::agrad::squared_dist(d1, v2).val());
  EXPECT_FLOAT_EQ(50, stan::agrad::squared_dist(v1, v2).val());
}
TEST(AgradRevMatrix, squared_dist_rowvector_rowvector_exception) {
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_v;

  row_vector_d d1(3), d2(2);
  row_vector_v v1(3), v2(4);

  EXPECT_THROW(stan::agrad::squared_dist(v1, d2), std::domain_error);
  EXPECT_THROW(stan::agrad::squared_dist(d1, v2), std::domain_error);
  EXPECT_THROW(stan::agrad::squared_dist(v1, v2), std::domain_error);
}

TEST(AgradRevMatrix, squared_dist_vv) {
  using stan::agrad::vector_v;

  vector_v a(3), b(3);
  AVAR c;
  for (int i = -1; i < 2; i++) { // a = (-1, 0, 1), b = (1, 2, 3)
    a(i+1) = i;
    b(i+1) = i + 2;
  }
  c = squared_dist(a, b);
  EXPECT_EQ(12, c);
  AVEC ab;
  VEC grad;
  for (size_t i = 0; i < 3; i++) {
    ab.push_back(a[i]);
    ab.push_back(b[i]);
  }
  c.grad(ab, grad);
  EXPECT_EQ(grad[0],  2*(a(0).val() - b(0).val()));
  EXPECT_EQ(grad[1], -2*(a(0).val() - b(0).val()));
  EXPECT_EQ(grad[2],  2*(a(1).val() - b(1).val()));
  EXPECT_EQ(grad[3], -2*(a(1).val() - b(1).val()));
  EXPECT_EQ(grad[4],  2*(a(2).val() - b(2).val()));
  EXPECT_EQ(grad[5], -2*(a(2).val() - b(2).val()));
}
TEST(AgradRevMatrix, squared_dist_dv) {
  using stan::math::vector_d;
  using stan::agrad::vector_v;

  vector_d a(3);
  vector_v b(3);
  AVAR c;
  for (int i = -1; i < 2; i++) { // a = (-1, 0, 1), b = (1, 2, 3)
    a(i+1) = i;
    b(i+1) = i + 2;
  }
  c = squared_dist(a, b);
  EXPECT_EQ(12, c);
  AVEC bv;
  VEC grad;
  for (size_t i = 0; i < 3; i++) {
    bv.push_back(b[i]);
  }
  c.grad(bv, grad);
  EXPECT_EQ(grad[0], -2*(a(0) - b(0).val()));
  EXPECT_EQ(grad[1], -2*(a(1) - b(1).val()));
  EXPECT_EQ(grad[2], -2*(a(2) - b(2).val()));
}
TEST(AgradRevMatrix, squared_dist_vd) {
  using stan::math::vector_d;
  using stan::agrad::vector_v;

  vector_v a(3);
  vector_d b(3);
  AVAR c;
  for (int i = -1; i < 2; i++) { // a = (-1, 0, 1), b = (1, 2, 3)
    a(i+1) = i;
    b(i+1) = i + 2;
  }
  c = squared_dist(a, b);
  EXPECT_EQ(12, c);
  AVEC av;
  VEC grad;
  for (size_t i = 0; i < 3; i++) {
    av.push_back(a[i]);
  }
  c.grad(av, grad);
  EXPECT_EQ(grad[0], 2*(a(0).val() - b(0)));
  EXPECT_EQ(grad[1], 2*(a(1).val() - b(1)));
  EXPECT_EQ(grad[2], 2*(a(2).val() - b(2)));
}

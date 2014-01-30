#include <stan/agrad/rev/matrix/dot_product.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/agrad/rev.hpp>

TEST(AgradRevMatrix, dot_product_vector_vector) {
  using stan::math::vector_d;
  using stan::agrad::vector_v;

  vector_d vd_1(3), vd_2(3);
  vector_v vv_1(3), vv_2(3);
  
  vd_1 << 1, 3, -5;
  vv_1 << 1, 3, -5;
  vd_2 << 4, -2, -1;
  vv_2 << 4, -2, -1;

  EXPECT_FLOAT_EQ(3, stan::agrad::dot_product(vv_1, vd_2).val());
  EXPECT_FLOAT_EQ(3, stan::agrad::dot_product(vd_1, vv_2).val());
  EXPECT_FLOAT_EQ(3, stan::agrad::dot_product(vv_1, vv_2).val());
}
TEST(AgradRevMatrix, dot_product_vector_vector_exception) {
  using stan::math::vector_d;
  using stan::agrad::vector_v;

  vector_d d1(3);
  vector_v v1(3);
  vector_d d2(2);
  vector_v v2(4);

  EXPECT_THROW(stan::agrad::dot_product(v1, d2), std::domain_error);
  EXPECT_THROW(stan::agrad::dot_product(d1, v2), std::domain_error);
  EXPECT_THROW(stan::agrad::dot_product(v1, v2), std::domain_error);
}
TEST(AgradRevMatrix, dot_product_rowvector_vector) {
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

  EXPECT_FLOAT_EQ(3, stan::agrad::dot_product(v1, d2).val());
  EXPECT_FLOAT_EQ(3, stan::agrad::dot_product(d1, v2).val());
  EXPECT_FLOAT_EQ(3, stan::agrad::dot_product(v1, v2).val());
}
TEST(AgradRevMatrix, dot_product_rowvector_vector_exception) {
  using stan::math::vector_d;
  using stan::agrad::vector_v;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_v;

  row_vector_d d1(3);
  row_vector_v v1(3);
  vector_d d2(2);
  vector_v v2(4);

  EXPECT_THROW(stan::agrad::dot_product(v1, d2), std::domain_error);
  EXPECT_THROW(stan::agrad::dot_product(d1, v2), std::domain_error);
  EXPECT_THROW(stan::agrad::dot_product(v1, v2), std::domain_error);
}
TEST(AgradRevMatrix, dot_product_vector_rowvector) {
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
  
  EXPECT_FLOAT_EQ(3, stan::agrad::dot_product(v1, d2).val());
  EXPECT_FLOAT_EQ(3, stan::agrad::dot_product(d1, v2).val());
  EXPECT_FLOAT_EQ(3, stan::agrad::dot_product(v1, v2).val());
}
TEST(AgradRevMatrix, dot_product_vector_rowvector_exception) {
  using stan::math::vector_d;
  using stan::agrad::vector_v;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_v;

  vector_d d1(3);
  vector_v v1(3);
  row_vector_d d2(2);
  row_vector_v v2(4);

  EXPECT_THROW(stan::agrad::dot_product(v1, d2), std::domain_error);
  EXPECT_THROW(stan::agrad::dot_product(d1, v2), std::domain_error);
  EXPECT_THROW(stan::agrad::dot_product(v1, v2), std::domain_error);
}
TEST(AgradRevMatrix, dot_product_rowvector_rowvector) {
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_v;

  row_vector_d d1(3), d2(3);
  row_vector_v v1(3), v2(3);
  
  d1 << 1, 3, -5;
  v1 << 1, 3, -5;
  d2 << 4, -2, -1;
  v2 << 4, -2, -1;

  EXPECT_FLOAT_EQ(3, stan::agrad::dot_product(v1, d2).val());
  EXPECT_FLOAT_EQ(3, stan::agrad::dot_product(d1, v2).val());
  EXPECT_FLOAT_EQ(3, stan::agrad::dot_product(v1, v2).val());
}
TEST(AgradRevMatrix, dot_product_rowvector_rowvector_exception) {
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_v;

  row_vector_d d1(3), d2(2);
  row_vector_v v1(3), v2(4);

  EXPECT_THROW(stan::agrad::dot_product(v1, d2), std::domain_error);
  EXPECT_THROW(stan::agrad::dot_product(d1, v2), std::domain_error);
  EXPECT_THROW(stan::agrad::dot_product(v1, v2), std::domain_error);
}

TEST(AgradRevMatrix, dot_product_vv) {
  AVEC a, b;
  AVAR c;
  for (int i = -1; i < 2; i++) { // a = (-1, 0, 1), b = (1, 2, 3)
    a.push_back(i);
    b.push_back(i + 2);
  }
  c = dot_product(&a[0], &b[0], 3);
  EXPECT_EQ(2, c);
  AVEC ab;
  VEC grad;
  for (size_t i = 0; i < 3; i++) {
    ab.push_back(a[i]);
    ab.push_back(b[i]);
  }
  c.grad(ab, grad);
  EXPECT_EQ(grad[0], 1);
  EXPECT_EQ(grad[1], -1);
  EXPECT_EQ(grad[2], 2);
  EXPECT_EQ(grad[3], 0);
  EXPECT_EQ(grad[4], 3);
  EXPECT_EQ(grad[5], 1);
}
TEST(AgradRevMatrix, dot_product_dv) {
  VEC a;
  AVEC b;
  AVAR c;
  for (int i = -1; i < 2; i++) { // a = (-1, 0, 1), b = (1, 2, 3)
    a.push_back(i);
    b.push_back(i + 2);
  }
  c = dot_product(&a[0], &b[0], 3);
  EXPECT_EQ(2, c);
  VEC grad;
  c.grad(b, grad);
  EXPECT_EQ(grad[0], -1);
  EXPECT_EQ(grad[1], 0);
  EXPECT_EQ(grad[2], 1);
}
TEST(AgradRevMatrix, dot_product_vd) {
  AVEC a;
  VEC b;
  AVAR c;
  for (int i = -1; i < 2; i++) { // a = (-1, 0, 1), b = (1, 2, 3)
    a.push_back(i);
    b.push_back(i + 2);
  }
  c = dot_product(&a[0], &b[0], 3);
  EXPECT_EQ(2, c);
  VEC grad;
  c.grad(a, grad);
  EXPECT_EQ(grad[0], 1);
  EXPECT_EQ(grad[1], 2);
  EXPECT_EQ(grad[2], 3);
}
TEST(AgradRevMatrix, dot_product_vv_vec) {
  AVEC a, b;
  AVAR c;
  for (int i = -1; i < 2; i++) { // a = (-1, 0, 1), b = (1, 2, 3)
    a.push_back(i);
    b.push_back(i + 2);
  }
  c = dot_product(a, b);
  EXPECT_EQ(2, c);
  AVEC ab;
  VEC grad;
  for (size_t i = 0; i < 3; i++) {
    ab.push_back(a[i]);
    ab.push_back(b[i]);
  }
  c.grad(ab, grad);
  EXPECT_EQ(grad[0], 1);
  EXPECT_EQ(grad[1], -1);
  EXPECT_EQ(grad[2], 2);
  EXPECT_EQ(grad[3], 0);
  EXPECT_EQ(grad[4], 3);
  EXPECT_EQ(grad[5], 1);
}
TEST(AgradRevMatrix, dot_product_dv_vec) {
  VEC a;
  AVEC b;
  AVAR c;
  for (int i = -1; i < 2; i++) { // a = (-1, 0, 1), b = (1, 2, 3)
    a.push_back(i);
    b.push_back(i + 2);
  }
  c = dot_product(a, b);
  EXPECT_EQ(2, c);
  VEC grad;
  c.grad(b, grad);
  EXPECT_EQ(grad[0], -1);
  EXPECT_EQ(grad[1], 0);
  EXPECT_EQ(grad[2], 1);
}
TEST(AgradRevMatrix, dot_product_vd_vec) {
  AVEC a;
  VEC b;
  AVAR c;
  for (int i = -1; i < 2; i++) { // a = (-1, 0, 1), b = (1, 2, 3)
    a.push_back(i);
    b.push_back(i + 2);
  }
  c = dot_product(a, b);
  EXPECT_EQ(2, c);
  VEC grad;
  c.grad(a, grad);
  EXPECT_EQ(grad[0], 1);
  EXPECT_EQ(grad[1], 2);
  EXPECT_EQ(grad[2], 3);
}


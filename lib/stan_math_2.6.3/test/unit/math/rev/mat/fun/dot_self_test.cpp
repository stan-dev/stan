#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/rev/mat/fun/dot_self.hpp>
#include <stan/math/rev/mat/fun/columns_dot_self.hpp>
#include <stan/math/prim/mat/fun/dot_self.hpp>
#include <stan/math/prim/mat/fun/columns_dot_self.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>

template <int R, int C>
void assert_val_grad(Eigen::Matrix<stan::math::var,R,C>& v) {
  v << -1.0, 0.0, 3.0;
  AVEC x = createAVEC(v(0),v(1),v(2));
  AVAR f = dot_self(v);
  VEC g;
  f.grad(x,g);
  
  EXPECT_FLOAT_EQ(-2.0,g[0]);
  EXPECT_FLOAT_EQ(0.0,g[1]);
  EXPECT_FLOAT_EQ(6.0,g[2]);
}  


TEST(AgradRevMatrix, dot_self_vec) {
  using stan::math::dot_self;

  Eigen::Matrix<AVAR,Eigen::Dynamic,1> v1(1);
  v1 << 2.0;
  EXPECT_NEAR(4.0,dot_self(v1).val(),1E-12);
  Eigen::Matrix<AVAR,Eigen::Dynamic,1> v2(2);
  v2 << 2.0, 3.0;
  EXPECT_NEAR(13.0,dot_self(v2).val(),1E-12);
  Eigen::Matrix<AVAR,Eigen::Dynamic,1> v3(3);
  v3 << 2.0, 3.0, 4.0;
  EXPECT_NEAR(29.0,dot_self(v3).val(),1E-12);  

  Eigen::Matrix<AVAR,Eigen::Dynamic,1> v(3);
  assert_val_grad(v);

  Eigen::Matrix<AVAR,1,Eigen::Dynamic> vv(3);
  assert_val_grad(vv);

  Eigen::Matrix<AVAR,Eigen::Dynamic,Eigen::Dynamic> vvv(3,1);
  assert_val_grad(vvv);

  Eigen::Matrix<AVAR,Eigen::Dynamic,Eigen::Dynamic> vvvv(1,3);
  assert_val_grad(vvvv);
}

TEST(AgradRevMatrix,columns_dot_self) {
  using stan::math::columns_dot_self;

  Eigen::Matrix<AVAR,Eigen::Dynamic,Eigen::Dynamic> m1(1,1);
  m1 << 2.0;
  EXPECT_NEAR(4.0,columns_dot_self(m1)(0,0).val(),1E-12);
  Eigen::Matrix<AVAR,Eigen::Dynamic,Eigen::Dynamic> m2(1,2);
  m2 << 2.0, 3.0;
  Eigen::Matrix<AVAR,Eigen::Dynamic,Eigen::Dynamic> x;
  x = columns_dot_self(m2);
  EXPECT_NEAR(4.0,x(0,0).val(),1E-12);
  EXPECT_NEAR(9.0,x(0,1).val(),1E-12);
  Eigen::Matrix<AVAR,Eigen::Dynamic,Eigen::Dynamic> m3(2,2);
  m3 << 2.0, 3.0, 4.0, 5.0;
  x = columns_dot_self(m3);
  EXPECT_NEAR(20.0,x(0,0).val(),1E-12);
  EXPECT_NEAR(34.0,x(0,1).val(),1E-12);

  Eigen::Matrix<AVAR,Eigen::Dynamic,Eigen::Dynamic> vvv(3,1);
  assert_val_grad(vvv);

  Eigen::Matrix<AVAR,Eigen::Dynamic,Eigen::Dynamic> vvvv(1,3);
  assert_val_grad(vvvv);
}

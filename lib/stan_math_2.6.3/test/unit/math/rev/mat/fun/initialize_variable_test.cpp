#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/rev/mat/fun/initialize_variable.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>

TEST(AgradRevMatrix, initializeVariable) {
  using stan::math::initialize_variable;
  using std::vector;

  using Eigen::Matrix;
  using Eigen::Dynamic;

  AVAR a;
  initialize_variable(a, AVAR(1.0));
  EXPECT_FLOAT_EQ(1.0, a.val());

  AVEC b(3);
  initialize_variable(b, AVAR(2.0));
  EXPECT_EQ(3U,b.size());
  EXPECT_FLOAT_EQ(2.0, b[0].val());
  EXPECT_FLOAT_EQ(2.0, b[1].val());
  EXPECT_FLOAT_EQ(2.0, b[2].val());

  vector<AVEC > c(4,AVEC(3));
  initialize_variable(c, AVAR(3.0));
  for (size_t m = 0; m < c.size(); ++m)
    for (size_t n = 0; n < c[0].size(); ++n)
      EXPECT_FLOAT_EQ(3.0,c[m][n].val());

  Matrix<AVAR, Dynamic, Dynamic> aa(5,7);
  initialize_variable(aa, AVAR(4.0));
  for (int m = 0; m < aa.rows(); ++m)
    for (int n = 0; n < aa.cols(); ++n)
      EXPECT_FLOAT_EQ(4.0, aa(m,n).val());

  Matrix<AVAR, Dynamic, 1> bb(5);
  initialize_variable(bb, AVAR(5.0));
  for (int m = 0; m < bb.size(); ++m) 
    EXPECT_FLOAT_EQ(5.0, bb(m).val());

  Matrix<AVAR,1,Dynamic> cc(12);
  initialize_variable(cc, AVAR(7.0));
  for (int m = 0; m < cc.size(); ++m) 
    EXPECT_FLOAT_EQ(7.0, cc(m).val());
  
  Matrix<AVAR,Dynamic,Dynamic> init_val(3,4);
  vector<Matrix<AVAR,Dynamic,Dynamic> > dd(5, init_val);
  initialize_variable(dd, AVAR(11.0));
  for (size_t i = 0; i < dd.size(); ++i)
    for (int m = 0; m < dd[0].rows(); ++m)
      for (int n = 0; n < dd[0].cols(); ++n)
        EXPECT_FLOAT_EQ(11.0, dd[i](m,n).val());
}

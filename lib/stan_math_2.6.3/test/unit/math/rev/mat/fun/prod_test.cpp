#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/prod.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>
#include <stan/math/rev/core.hpp>

TEST(AgradRevMatrix,prod) {
  using stan::math::prod;
  using stan::math::vector_d;
  using stan::math::vector_v;

  vector_d vd;
  vector_v vv;
  EXPECT_FLOAT_EQ(1.0,prod(vd));
  EXPECT_FLOAT_EQ(1.0,prod(vv).val());

  vd = vector_d(1);
  vv = vector_v(1);
  vd << 2.0;
  vv << 2.0;
  EXPECT_FLOAT_EQ(2.0,prod(vd));
  EXPECT_FLOAT_EQ(2.0,prod(vv).val());

  vd = vector_d(2);
  vd << 2.0, 3.0;
  vv = vector_v(2);
  vv << 2.0, 3.0;
  AVEC x(2);
  x[0] = vv[0];
  x[1] = vv[1];
  AVAR f = prod(vv);
  EXPECT_FLOAT_EQ(6.0,prod(vd));
  EXPECT_FLOAT_EQ(6.0,f.val());
  VEC g;
  f.grad(x,g);
  EXPECT_FLOAT_EQ(3.0,g[0]);
  EXPECT_FLOAT_EQ(2.0,g[1]);
}

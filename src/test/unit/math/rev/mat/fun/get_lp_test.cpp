#include <stan/math/prim/mat/fun/accumulator.hpp>
#include <stan/math/prim/mat/fun/get_lp.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core.hpp>


TEST(mathMatrix,getLp) {
  using stan::math::accumulator;
  using stan::math::get_lp;
  using stan::agrad::var;

  var lp = 12.5;
  accumulator<var> lp_accum;
  EXPECT_FLOAT_EQ(12.5, get_lp(lp,lp_accum).val());

  lp_accum.add(2);
  lp_accum.add(3);
  EXPECT_FLOAT_EQ(17.5, get_lp(lp,lp_accum).val());
}


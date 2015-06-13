#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/prob/multinomial_log.hpp>
#include <stan/math/prim/mat/prob/multinomial_rng.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/fwd/scal/fun/value_of.hpp>

using Eigen::Matrix;
using Eigen::Dynamic;

TEST(ProbDistributionsMultinomial,fvar_var) {
  using stan::math::var;
  using stan::math::fvar;
  std::vector<int> ns;
  ns.push_back(1);
  ns.push_back(2);
  ns.push_back(3);
  Matrix<fvar<var>,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;
  for (int i = 0; i < 3; i++)
    theta(i).d_ = 1.0;

  EXPECT_FLOAT_EQ(-2.002481, stan::math::multinomial_log(ns,theta).val_.val());
  EXPECT_FLOAT_EQ(17.666666, stan::math::multinomial_log(ns,theta).d_.val());
}

TEST(ProbDistributionsMultinomial,fvar_fvar_var) {
  using stan::math::var;
  using stan::math::fvar;
  std::vector<int> ns;
  ns.push_back(1);
  ns.push_back(2);
  ns.push_back(3);
  Matrix<fvar<fvar<var> >,Dynamic,1> theta(3,1);
  theta << 0.2, 0.3, 0.5;
  for (int i = 0; i < 3; i++)
    theta(i).d_.val_ = 1.0;

  EXPECT_FLOAT_EQ(-2.002481, stan::math::multinomial_log(ns,theta).val_.val_.val());
  EXPECT_FLOAT_EQ(17.666666, stan::math::multinomial_log(ns,theta).d_.val_.val());
}

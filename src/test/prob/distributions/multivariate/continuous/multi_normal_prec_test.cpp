#include <gtest/gtest.h>
#include "stan/prob/distributions/multivariate/continuous/multi_normal_prec.hpp"
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <stan/agrad/agrad.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;

TEST(ProbDistributionsMultiNormalPrec, MultiNormalPrec) {
  // FIXME:  dummy, of course
  EXPECT_EQ(1,1);
}

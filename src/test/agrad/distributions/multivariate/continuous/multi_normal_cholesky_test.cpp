#include <gtest/gtest.h>
#include <test/agrad/distributions/expect_eq_diffs.hpp>
#include <stan/prob/distributions/multivariate/continuous/multi_normal_cholesky.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;

using stan::agrad::var;
using stan::agrad::to_var;


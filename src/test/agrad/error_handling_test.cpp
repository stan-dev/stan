#include <stan/math/error_handling.hpp>
#include <stan/agrad/rev/numeric_limits.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad.hpp>
#include <gtest/gtest.h>


typedef boost::math::policies::policy<
  boost::math::policies::domain_error<boost::math::policies::errno_on_error>
  > errno_policy;


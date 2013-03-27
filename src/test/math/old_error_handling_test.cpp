#include <gtest/gtest.h>
#include <stan/math/error_handling.hpp>

typedef boost::math::policies::policy<
  boost::math::policies::domain_error<boost::math::policies::errno_on_error>
  > errno_policy;
using namespace stan::math;




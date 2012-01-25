#ifndef __STAN__AGRAD__ERROR_HANDLING_HPP__
#define __STAN__AGRAD__ERROR_HANDLING_HPP__

#include <limits>

#include <boost/math/policies/policy.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/distributions/detail/common_error_handling.hpp>

#include <stan/maths/matrix.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/transform.hpp>

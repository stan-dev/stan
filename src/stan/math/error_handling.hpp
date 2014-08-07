#ifndef STAN__MATH__ERROR_HANDLING_HPP
#define STAN__MATH__ERROR_HANDLING_HPP

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <iostream>

#include <boost/type_traits/is_unsigned.hpp>

#include <stan/meta/traits.hpp>
#include <stan/math/error_handling/dom_err.hpp>
#include <stan/math/error_handling/dom_err_vec.hpp>
#include <stan/math/error_handling/check_not_nan.hpp>
#include <stan/math/error_handling/check_equal.hpp>
#include <stan/math/error_handling/check_finite.hpp>
#include <stan/math/error_handling/check_greater.hpp>
#include <stan/math/error_handling/check_greater_or_equal.hpp>
#include <stan/math/error_handling/check_less.hpp>
#include <stan/math/error_handling/check_less_or_equal.hpp>
#include <stan/math/error_handling/check_bounded.hpp>
#include <stan/math/error_handling/check_nonnegative.hpp>
#include <stan/math/error_handling/check_positive.hpp>
#include <stan/math/error_handling/check_positive_finite.hpp>
#include <stan/math/error_handling/check_consistent_size.hpp>
#include <stan/math/error_handling/check_consistent_sizes.hpp>

#endif


#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(langParser, integrate_ode_deprecation) {
  test_warning("integrate_ode_deprecation",
               "the integrate_ode() function is deprecated in the"
               " Stan language; use the following functions instead.\n"
               " integrate_ode_rk45() [explicit, order 5, for non-stiff problems]\n"
               " integrate_ode_adams() [implicit, up to order 12, for non-stiff problems]\n"
               " integrate_ode_bdf() [implicit, up to order 5, for stiff problems].");
}

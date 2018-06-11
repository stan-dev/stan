#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(langParser, integrate_ode_deprecation) {
  test_warning("integrate_ode_deprecation",
               "the integrate_ode() function is deprecated in the"
               " Stan language; use the following functions instead."
               "  integrate_ode_rk45() [non-stiff]"
               "  integrate_ode_adams() [non-stiff]"
               "  integrate_ode_bdf() [stiff].");
}

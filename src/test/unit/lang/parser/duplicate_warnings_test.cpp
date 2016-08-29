#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(langParser, duplicateWarnings) {
  test_num_warnings("duplicate-warns",
                    "assignment operator <- deprecated", 1);
  test_num_warnings("duplicate-warns",
                    "increment_log_prob(...); is deprecated", 1);
  test_num_warnings("duplicate-warns",
                    "get_lp() function deprecated", 1);
  test_num_warnings("duplicate-warns",
                    "'multiply_log' is deprecated", 1);
  test_num_warnings("duplicate-warns",
                    "'binomial_coefficient_log' is deprecated", 1);
  test_num_warnings("duplicate-warns",
                    "Deprecated function 'normal_log'", 1);
  test_num_warnings("duplicate-warns",
                    "Deprecated function 'normal_cdf_log'", 1);
  test_num_warnings("duplicate-warns",
                    "Deprecated function 'normal_ccdf_log'", 1);
}

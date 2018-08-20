#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(parserReservedWords, for) {
  test_throws("reserved/for",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, in) {
  test_throws("reserved/in",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, while) {
  test_throws("reserved/while",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, repeat) {
  test_throws("reserved/repeat",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, until) {
  test_throws("reserved/until",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, if) {
  test_throws("reserved/if",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, then) {
  test_throws("reserved/then",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, else) {
  test_throws("reserved/else",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, true) {
  test_throws("reserved/true",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, false) {
  test_throws("reserved/false",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, int) {
  test_throws("reserved/int",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, real) {
  test_throws("reserved/real",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, vector) {
  test_throws("reserved/vector",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, simplex) {
  test_throws("reserved/simplex",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, unit_vector) {
  test_throws("reserved/unit_vector",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, ordered) {
  test_throws("reserved/ordered",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, positive_ordered) {
  test_throws("reserved/positive_ordered",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, row_vector) {
  test_throws("reserved/row_vector",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, matrix) {
  test_throws("reserved/matrix",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, cholesky_factor_corr) {
  test_throws("reserved/cholesky_factor_corr",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, cholesky_factor_cov) {
  test_throws("reserved/cholesky_factor_cov",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, corr_matrix) {
  test_throws("reserved/corr_matrix",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, cov_matrix) {
  test_throws("reserved/cov_matrix",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, var) {
  test_throws("reserved/var",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, fvar) {
  test_throws("reserved/fvar",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, macro_STAN_MAJOR) {
  test_throws("reserved/STAN_MAJOR",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, macro_STAN_MINOR) {
  test_throws("reserved/STAN_MINOR",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, macro_STAN_PATCH) {
  test_throws("reserved/STAN_PATCH",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, macro_STAN_MATH_MAJOR) {
  test_throws("reserved/STAN_MATH_MAJOR",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, macro_STAN_MATH_MINOR) {
  test_throws("reserved/STAN_MATH_MINOR",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, macro_STAN_MATH_PATCH) {
  test_throws("reserved/STAN_MATH_PATCH",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, alignas) {
  test_throws("reserved/alignas",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, alignof) {
  test_throws("reserved/alignof",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, and) {
  test_throws("reserved/and",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, and_eq) {
  test_throws("reserved/and_eq",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, asm) {
  test_throws("reserved/asm",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, auto) {
  test_throws("reserved/auto",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, bitand) {
  test_throws("reserved/bitand",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, bitor) {
  test_throws("reserved/bitor",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, bool) {
  test_throws("reserved/bool",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, break) {
  test_throws("reserved/break",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, case) {
  test_throws("reserved/case",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, catch) {
  test_throws("reserved/catch",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, char) {
  test_throws("reserved/char",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, char16_t) {
  test_throws("reserved/char16_t",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, char32_t) {
  test_throws("reserved/char32_t",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, class) {
  test_throws("reserved/class",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, compl) {
  test_throws("reserved/compl",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, const) {
  test_throws("reserved/const",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, constexpr) {
  test_throws("reserved/constexpr",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, const_cast) {
  test_throws("reserved/const_cast",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, continue) {
  test_throws("reserved/continue",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, decltype) {
  test_throws("reserved/decltype",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, default) {
  test_throws("reserved/default",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, delete) {
  test_throws("reserved/delete",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, do) {
  test_throws("reserved/do",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, double) {
  test_throws("reserved/double",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, dynamic_cast) {
  test_throws("reserved/dynamic_cast",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, enum) {
  test_throws("reserved/enum",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, explicit) {
  test_throws("reserved/explicit",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, export) {
  test_throws("reserved/export",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, extern) {
  test_throws("reserved/extern",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, float) {
  test_throws("reserved/float",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, friend) {
  test_throws("reserved/friend",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, goto) {
  test_throws("reserved/goto",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, inline) {
  test_throws("reserved/inline",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, long) {
  test_throws("reserved/long",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, mutable) {
  test_throws("reserved/mutable",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, namespace) {
  test_throws("reserved/namespace",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, new) {
  test_throws("reserved/new",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, noexcept) {
  test_throws("reserved/noexcept",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, not) {
  test_throws("reserved/not",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, not_eq) {
  test_throws("reserved/not_eq",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, keyword_nullptr) {
  test_throws("reserved/nullptr",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, operator) {
  test_throws("reserved/operator",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, or) {
  test_throws("reserved/or",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, or_eq) {
  test_throws("reserved/or_eq",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, private) {
  test_throws("reserved/private",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, protected) {
  test_throws("reserved/protected",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, public) {
  test_throws("reserved/public",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, register) {
  test_throws("reserved/register",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, reinterpret_cast) {
  test_throws("reserved/reinterpret_cast",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, return) {
  test_throws("reserved/return",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, short) {
  test_throws("reserved/short",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, signed) {
  test_throws("reserved/signed",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, sizeof) {
  test_throws("reserved/sizeof",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, static) {
  test_throws("reserved/static",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, static_assert) {
  test_throws("reserved/static_assert",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, static_cast) {
  test_throws("reserved/static_cast",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, struct) {
  test_throws("reserved/struct",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, switch) {
  test_throws("reserved/switch",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, template) {
  test_throws("reserved/template",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, this) {
  test_throws("reserved/this",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, thread_local) {
  test_throws("reserved/thread_local",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, throw) {
  test_throws("reserved/throw",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, try) {
  test_throws("reserved/try",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, typedef) {
  test_throws("reserved/typedef",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, typeid) {
  test_throws("reserved/typeid",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, typename) {
  test_throws("reserved/typename",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, union) {
  test_throws("reserved/union",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, unsigned) {
  test_throws("reserved/unsigned",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, using) {
  test_throws("reserved/using",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, virtual) {
  test_throws("reserved/virtual",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, void) {
  test_throws("reserved/void",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, volatile) {
  test_throws("reserved/volatile",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, wchar_t) {
  test_throws("reserved/wchar_t",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, xor) {
  test_throws("reserved/xor",
              "Variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, xor_eq) {
  test_throws("reserved/xor_eq",
              "Variable identifier (name) may not be reserved word");
}

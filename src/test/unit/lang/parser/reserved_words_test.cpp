#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(parserReservedWords, for) {
  test_throws("reserved/for", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, in) {
  test_throws("reserved/in", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, while) {
  test_throws("reserved/while", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, repeat) {
  test_throws("reserved/repeat", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, until) {
  test_throws("reserved/until", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, if) {
  test_throws("reserved/if", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, then) {
  test_throws("reserved/then", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, else) {
  test_throws("reserved/else", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, true) {
  test_throws("reserved/true", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, false) {
  test_throws("reserved/false", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, int) {
  test_throws("reserved/int", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, real) {
  test_throws("reserved/real", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, vector) {
  test_throws("reserved/vector", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, simplex) {
  test_throws("reserved/simplex", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, unit_vector) {
  test_throws("reserved/unit_vector", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, ordered) {
  test_throws("reserved/ordered", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, positive_ordered) {
  test_throws("reserved/positive_ordered", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, row_vector) {
  test_throws("reserved/row_vector", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, matrix) {
  test_throws("reserved/matrix", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, cholesky_factor_corr) {
  test_throws("reserved/cholesky_factor_corr", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, cholesky_factor_cov) {
  test_throws("reserved/cholesky_factor_cov", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, corr_matrix) {
  test_throws("reserved/corr_matrix", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, cov_matrix) {
  test_throws("reserved/cov_matrix", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, var) {
  test_throws("reserved/var", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, fvar) {
  test_throws("reserved/fvar", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, macro_STAN_MAJOR) {
  test_throws("reserved/STAN_MAJOR", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, macro_STAN_MINOR) {
  test_throws("reserved/STAN_MINOR", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, macro_STAN_PATCH) {
  test_throws("reserved/STAN_PATCH", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, macro_STAN_MATH_MAJOR) {
  test_throws("reserved/STAN_MATH_MAJOR", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, macro_STAN_MATH_MINOR) {
  test_throws("reserved/STAN_MATH_MINOR", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, macro_STAN_MATH_PATCH) {
  test_throws("reserved/STAN_MATH_PATCH", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, alignas) {
  test_throws("reserved/alignas", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, alignof) {
  test_throws("reserved/alignof", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, and) {
  test_throws("reserved/and", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, and_eq) {
  test_throws("reserved/and_eq", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, asm) {
  test_throws("reserved/asm", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, auto) {
  test_throws("reserved/auto", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, bitand) {
  test_throws("reserved/bitand", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, bitor) {
  test_throws("reserved/bitor", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, bool) {
  test_throws("reserved/bool", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, break) {
  test_throws("reserved/break", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, case) {
  test_throws("reserved/case", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, catch) {
  test_throws("reserved/catch", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, char) {
  test_throws("reserved/char", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, char16_t) {
  test_throws("reserved/char16_t", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, char32_t) {
  test_throws("reserved/char32_t", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, class) {
  test_throws("reserved/class", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, compl) {
  test_throws("reserved/compl", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, const) {
  test_throws("reserved/const", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, constexpr) {
  test_throws("reserved/constexpr", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, const_cast) {
  test_throws("reserved/const_cast", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, continue) {
  test_throws("reserved/continue", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, decltype) {
  test_throws("reserved/decltype", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, default) {
  test_throws("reserved/default", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, delete) {
  test_throws("reserved/delete", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, do) {
  test_throws("reserved/do", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, double) {
  test_throws("reserved/double", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, dynamic_cast) {
  test_throws("reserved/dynamic_cast", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, enum) {
  test_throws("reserved/enum", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, explicit) {
  test_throws("reserved/explicit", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, export) {
  test_throws("reserved/export", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, extern) {
  test_throws("reserved/extern", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, float) {
  test_throws("reserved/float", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, friend) {
  test_throws("reserved/friend", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, goto) {
  test_throws("reserved/goto", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, inline) {
  test_throws("reserved/inline", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, long) {
  test_throws("reserved/long", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, mutable) {
  test_throws("reserved/mutable", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, namespace) {
  test_throws("reserved/namespace", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, new) {
  test_throws("reserved/new", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, noexcept) {
  test_throws("reserved/noexcept", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, not) {
  test_throws("reserved/not", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, not_eq) {
  test_throws("reserved/not_eq", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, keyword_nullptr) {
  test_throws("reserved/nullptr", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, operator) {
  test_throws("reserved/operator", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, or) {
  test_throws("reserved/or", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, or_eq) {
  test_throws("reserved/or_eq", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, private) {
  test_throws("reserved/private", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, protected) {
  test_throws("reserved/protected", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, public) {
  test_throws("reserved/public", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, register) {
  test_throws("reserved/register", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, reinterpret_cast) {
  test_throws("reserved/reinterpret_cast", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, return) {
  test_throws("reserved/return", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, short) {
  test_throws("reserved/short", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, signed) {
  test_throws("reserved/signed", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, sizeof) {
  test_throws("reserved/sizeof", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, static) {
  test_throws("reserved/static", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, static_assert) {
  test_throws("reserved/static_assert", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, static_cast) {
  test_throws("reserved/static_cast", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, struct) {
  test_throws("reserved/struct", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, switch) {
  test_throws("reserved/switch", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, template) {
  test_throws("reserved/template", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, this) {
  test_throws("reserved/this", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, thread_local) {
  test_throws("reserved/thread_local", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, throw) {
  test_throws("reserved/throw", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, try) {
  test_throws("reserved/try", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, typedef) {
  test_throws("reserved/typedef", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, typeid) {
  test_throws("reserved/typeid", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, typename) {
  test_throws("reserved/typename", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, union) {
  test_throws("reserved/union", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, unsigned) {
  test_throws("reserved/unsigned", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, using) {
  test_throws("reserved/using", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, virtual) {
  test_throws("reserved/virtual", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, void) {
  test_throws("reserved/void", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, volatile) {
  test_throws("reserved/volatile", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, wchar_t) {
  test_throws("reserved/wchar_t", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, xor) {
  test_throws("reserved/xor", "variable identifier (name) may not be reserved word");
}

TEST(parserReservedWords, xor_eq) {
  test_throws("reserved/xor_eq", "variable identifier (name) may not be reserved word");
}

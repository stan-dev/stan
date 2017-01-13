#ifndef STAN_LANG_GENERATOR_GENERATE_INDEXED_EXPR_HPP
#define STAN_LANG_GENERATOR_GENERATE_INDEXED_EXPR_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/generate_indexed_expr_user.hpp>
#include <stan/lang/generator/generate_quoted_string.hpp>
#include <ostream>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    void generate_expression(const expression& e, bool user_facing,
                             std::ostream& o);

    /**
     * Generate the specified expression indexed with the specified
     * indices with the specified base type of expression being
     * indexed, number of dimensions, and a flag indicating whether
     * the generation is for user output or C++ compilation.
     * Depending on the base type, two layers of parens may be written
     * in the underlying code.
     *
     * @tparam isLHS true if indexed expression appears on left-hand
     * side of an assignment
     * @param[in] expr string for expression
     * @param[in] indexes indexes for expression
     * @param[in] base_type base type of expression
     * @param[in] e_num_dims number of array dimensions in expression
     * @param[in] user_facing true if expression generated for user
     * output
     * @param[in,out] o stream for generating
     */
    template <bool isLHS>
    void generate_indexed_expr(const std::string& expr,
                               const std::vector<expression>& indexes,
                               base_expr_type base_type, size_t e_num_dims,
                               bool user_facing, std::ostream& o) {
      if (user_facing) {
        generate_indexed_expr_user(expr, indexes, o);
        return;
      }
      size_t ai_size = indexes.size();
      if (ai_size == 0) {
        o << expr;
        return;
      }
      if (ai_size <= (e_num_dims + 1) || base_type != MATRIX_T) {
        for (size_t n = 0; n < ai_size; ++n)
          o << (isLHS ? "get_base1_lhs(" : "get_base1(");
        o << expr;
        for (size_t n = 0; n < ai_size; ++n) {
          o << ',';
          generate_expression(indexes[n], user_facing, o);
          o << ',';
          generate_quoted_string(expr, o);
          o << ',' << (n + 1) << ')';
        }
      } else {
        for (size_t n = 0; n < ai_size - 1; ++n)
          o << (isLHS ? "get_base1_lhs(" : "get_base1(");
        o << expr;
        for (size_t n = 0; n < ai_size - 2; ++n) {
          o << ',';
          generate_expression(indexes[n], user_facing, o);
          o << ',';
          generate_quoted_string(expr, o);
          o << ',' << (n+1) << ')';
        }
        o << ',';
        generate_expression(indexes[ai_size - 2U], user_facing, o);
        o << ',';
        generate_expression(indexes[ai_size - 1U], user_facing, o);
        o << ',';
        generate_quoted_string(expr, o);
        o << ',' << (ai_size - 1U) << ')';
      }
    }

  }
}
#endif

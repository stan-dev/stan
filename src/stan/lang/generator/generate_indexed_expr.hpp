#ifndef STAN_LANG_GENERATOR_GENERATE_INDEXED_EXPR_HPP
#define STAN_LANG_GENERATOR_GENERATE_INDEXED_EXPR_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/generate_indexed_expr_user.hpp>
#include <stan/lang/generator/generate_quoted_string.hpp>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Generate the specified expression indexed with the specified
     * indices with the specified bare type of expression being
     * indexed, number of dimensions, and a flag indicating whether
     * the generation is for user output or C++ compilation.
     *
     * Generated code is a call stan mathlib function which does the indexing.
     * Row-major indexing, twisty logic.
     *
     * @tparam isLHS true if indexed expression appears on left-hand
     * side of an assignment
     * @param[in] expr string for expression
     * @param[in] indexes indexes for expression
     * @param[in] bare_type bare type of expression
     * @param[in] e_num_dims number of array dimensions in expression
     * @param[in] user_facing true if expression might be reported to user
     * @param[in,out] o stream for generating
     */
    template <bool isLHS>
    void generate_indexed_expr(const std::string& expr,
                               const std::vector<expression>& indexes,
                               bare_expr_type bare_type, size_t e_num_dims,
                               bool user_facing, std::ostream& o) {
      if (user_facing) {
        generate_indexed_expr_user(expr, indexes, o);
        return;
      }
      if (indexes.size() == 0) {
        o << expr;
        return;
      }

      // indexing logic depends on variable shape - array dimensions vs. row/col indices 
      size_t max_ar_dims = bare_type.base().is_primitive() ? bare_type.array_dims() : bare_type.array_dims() + 1;
      size_t indexed_ar_dims = (indexes.size() < max_ar_dims ? indexes.size() : max_ar_dims);

      // open get_base stmts
      for (size_t i = 0; i < indexed_ar_dims; ++ i) {
        o << (isLHS ? "get_base1_lhs(" : "get_base1(");
      }        
      o << expr << ", ";
      // get first index (nested)
      for (size_t i = 0; i < indexed_ar_dims; ++ i) {
        generate_expression(indexes[i], user_facing, o);
        if (i < indexed_ar_dims - 1)
          o << ", \"" << expr << "\", " << i + 1 << "), ";
      }
      // remaining indexes
      for (size_t i = indexed_ar_dims; i < indexes.size(); ++i) {
        o << ", ";
        generate_expression(indexes[i], user_facing, o);
      }
      // close
      o << ", " << "\"" << expr << "\", " << indexed_ar_dims << ")";
    }

  }
}
#endif

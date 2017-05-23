#ifndef STAN_LANG_GENERATOR_GENERATE_EXPRESSION_HPP
#define STAN_LANG_GENERATOR_GENERATE_EXPRESSION_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/expression_visgen.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <ostream>

namespace stan {
  namespace lang {

    /**
     * Generate the specified expression to the specified stream with
     * user-facing/C++ format and parameter/data format controlled by
     * the flags.
     *
     * @param[in] e expression to generate
     * @param[in] user_facing true if generation is to read by user, false
     * for code generation in C++
     * @param[in] is_var_context true if generation in parameter var
     * context, false for data context
     * @param[in,out] o stream for generating
     */
    void generate_expression(const expression& e, bool user_facing,
                             bool is_var_context, std::ostream& o) {
      expression_visgen vis(o, user_facing, is_var_context);
      boost::apply_visitor(vis, e.expr_);
    }

    /**
     * Generate the specified expression to the specified stream, with
     * user-facing or C++ syntax as specified by the flag.  The
     * generation is done for a double context rather than a var
     * (parameter) context.
     *
     * @param[in] e expression to generate
     * @param[in] user_facing true if generation is to read by user, false
     * for code generation in C++
     * @param[in,out] o stream for generating
     */
    void generate_expression(const expression& e, bool user_facing,
                             std::ostream& o) {
      static const bool is_var_context = false;
      expression_visgen vis(o, user_facing, is_var_context);
      boost::apply_visitor(vis, e.expr_);
    }

    /**
     * Generate the specified expression to the specified stream, with
     * C++ syntax.
     *
     * @param[in] e expression to generate
     * @param[in,out] o stream for generating
     */
    void generate_expression(const expression& e, std::ostream& o) {
      static const bool user_facing = false;
      static const bool is_var_context = false;
      generate_expression(e, user_facing, is_var_context, o);
    }


  }
}
#endif

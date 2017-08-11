#ifndef STAN_LANG_GENERATOR_GENERATE_STATEMENTS_HPP
#define STAN_LANG_GENERATOR_GENERATE_STATEMENTS_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/generate_statement.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Generate the set of statements in a program block with
     * the specified indentation level on the specified stream
     * with flags indicating whether sampling statements are allowed
     * and whether generation is in a variable context or
     * a function return context.
     *
     * @param[in] statements vector of statements
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     * @param[in] include_sampling true if sampling statements are
     * included
     * @param[in] is_var_context true if in context to generate
     * variables
     * @param[in] is_fun_return true if in context of function return
     */
    void generate_statements(const std::vector<statement> statements,
                             int indent, std::ostream& o,
                             bool include_sampling, bool is_var_context,
                             bool is_fun_return) {
      for (size_t i = 0; i < statements.size(); ++i)
        generate_statement(statements[i], indent, o, include_sampling,
                           is_var_context, is_fun_return);
    }

  }
}
#endif


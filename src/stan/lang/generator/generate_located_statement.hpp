#ifndef STAN_LANG_GENERATOR_GENERATE_LOCATED_STATEMENT_HPP
#define STAN_LANG_GENERATOR_GENERATE_LOCATED_STATEMENT_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/generate_catch_throw_located.hpp>
#include <stan/lang/generator/generate_try.hpp>
#include <stan/lang/generator/generate_statement.hpp>
#include <ostream>

namespace stan {
  namespace lang {

    /**
     * Generate the specified statement at the specified indentation
     * level to the specified stream, with flags indicating if
     * sampling should be included and if the context requires
     * generating variables or function return types.
     *
     * @param[in] s statement to generate
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     * @param[in] include_sampling true if sampling should be included
     * @param[in] is_var_context true if variable types should be
     * generated
     * @param[in] is_fun_return true if function return types should
     * be generated
     */
    void generate_located_statement(const statement& s, int indent,
                                    std::ostream& o, bool include_sampling,
                                    bool is_var_context, bool is_fun_return) {
      generate_try(indent, o);
      generate_statement(s, indent+1, o, include_sampling, is_var_context,
                         is_fun_return);
      generate_catch_throw_located(indent, o);
    }


  }
}
#endif

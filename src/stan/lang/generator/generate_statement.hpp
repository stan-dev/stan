#ifndef STAN_LANG_GENERATOR_GENERATE_STATEMENT_HPP
#define STAN_LANG_GENERATOR_GENERATE_STATEMENT_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/is_numbered_statement_vis.hpp>
#include <stan/lang/generator/statement_visgen.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <ostream>

namespace stan {
  namespace lang {

    /**
     * Generate the specified statement with the specified indentation
     * level on the specified output stream with flags indicating
     * whether sampling statements are allowed and whether the
     * generation is in a variable context or function return context.
     *
     * @param[in] s statement to generate
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     * @param[in] include_sampling true if sampling statements are
     * included
     * @param[in] is_var_context true if in context to generate
     * variables
     * @param[in] is_fun_return true if in context of function return
     */
    void generate_statement(const statement& s, int indent, std::ostream& o,
                            bool include_sampling, bool is_var_context,
                            bool is_fun_return) {
      is_numbered_statement_vis vis_is_numbered;
      if (boost::apply_visitor(vis_is_numbered, s.statement_)) {
        generate_indent(indent, o);
        o << "current_statement_begin__ = " <<  s.begin_line_ << ";"
          << EOL;
      }
      statement_visgen vis(indent, include_sampling, is_var_context,
                           is_fun_return, o);
      boost::apply_visitor(vis, s.statement_);
    }


  }
}
#endif

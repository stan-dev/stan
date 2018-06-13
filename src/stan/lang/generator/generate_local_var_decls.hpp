#ifndef STAN_LANG_GENERATOR_GENERATE_LOCAL_VAR_DECLS_HPP
#define STAN_LANG_GENERATOR_GENERATE_LOCAL_VAR_DECLS_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/init_vars_visgen.hpp>
#include <stan/lang/generator/local_var_decl_visgen.hpp>
#include <stan/lang/generator/local_var_init_nan_visgen.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Generate local variable declarations, including
     * initializations, for the specified declarations, indentation
     * level, writing to the specified stream.
     * Generated code is preceeded by stmt updating global variable
     * `current_statement_begin__` to src file line number where
     * variable is declared.
     *
     * @param[in] vs variable declarations
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void generate_local_var_decls(const std::vector<var_decl>& vs, int indent,
                                  std::ostream& o) {
      local_var_decl_visgen vis_decl(indent, o);
      local_var_init_nan_visgen vis_init(indent, o);
      init_vars_visgen vis_filler(indent, o);
      for (size_t i = 0; i < vs.size(); ++i) {
        generate_indent(indent, o);
        o << "current_statement_begin__ = " <<  vs[i].begin_line_ << ";"
          << EOL;
        boost::apply_visitor(vis_decl, vs[i].decl_);
        boost::apply_visitor(vis_init, vs[i].decl_);
        boost::apply_visitor(vis_filler, vs[i].decl_);
        if (vs[i].has_def()) {
          generate_indent(indent, o);
          o << "stan::math::assign("
            << vs[i].name()
            << ",";
          generate_expression(vs[i].def(), NOT_USER_FACING, o);
          o << ");" << EOL;
        }
      }
      o << EOL;
    }

  }
}
#endif

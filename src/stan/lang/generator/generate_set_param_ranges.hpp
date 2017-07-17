#ifndef STAN_LANG_GENERATOR_GENERATE_SET_PARAM_RANGES_HPP
#define STAN_LANG_GENERATOR_GENERATE_SET_PARAM_RANGES_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/set_param_ranges_visgen.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {

    /*
     * Generate statements in constructor body which cumulatively
     * determine the size required for the vector of param ranges and
     * the range for each parameter in the model by iterating over the
     * list of parameter variable declarations.
     * Generated code is preceeded by stmt updating global variable
     * `current_statement_begin__` to src file line number where
     * parameter variable is declared.
     *
     * @param[in] var_decls sequence of variable declarations
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void generate_set_param_ranges(const std::vector<var_decl>& var_decls,
                                   int indent, std::ostream& o) {
      generate_indent(indent, o);
      o << "num_params_r__ = 0U;" << EOL;
      generate_indent(indent, o);
      o << "param_ranges_i__.clear();" << EOL;
      set_param_ranges_visgen vis(o);
      for (size_t i = 0; i < var_decls.size(); ++i) {
        generate_indent(indent, o);
        o << "current_statement_begin__ = " <<  var_decls[i].begin_line_ << ";"
          << EOL;
        boost::apply_visitor(vis, var_decls[i].decl_);
      }
    }



  }
}
#endif

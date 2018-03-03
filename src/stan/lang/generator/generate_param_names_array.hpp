#ifndef STAN_LANG_GENERATOR_GENERATE_PARAM_NAMES_ARRAY_HPP
#define STAN_LANG_GENERATOR_GENERATE_PARAM_NAMES_ARRAY_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_expression.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/get_block_var_dims.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Generate the parameter names for the specified parameter variable.
     *
     * @param[in] indent level of indentation
     * @param[in,out] o stream for generating
     * @param[in] var_decl block_var_decl
     */
    void
    generate_param_names_array(size_t indent, std::ostream& o,
                               const block_var_decl& var_decl) {
      std::vector<expression> dims = get_block_var_dims(var_decl);
      for (size_t i = dims.size(); i-- > 0; ) {
        generate_indent(indent + dims.size() - i, o);
        o << "for (int k_" << i << "__ = 1;"
           << " k_" << i << "__ <= ";
        generate_expression(dims[i].expr_, NOT_USER_FACING, o);
        o << "; ++k_" << i << "__) {" << EOL;  // begin (1)
      }
      generate_indent(indent + 1 + dims.size(), o);
      o << "param_name_stream__.str(std::string());" << EOL;

      generate_indent(indent + 1 + dims.size(), o);
      o << "param_name_stream__ << \"" << var_decl.name() << '"';

      for (size_t i = 0; i < dims.size(); ++i)
        o << " << '.' << k_" << i << "__";
      o << ';' << EOL;

      generate_indent(indent + 1 + dims.size(), o);
      o << "param_names__.push_back(param_name_stream__.str());" << EOL;

      // end for loop dims
      for (size_t i = 0; i < dims.size(); ++i) {
        generate_indent(indent + dims.size() - i, o);
        o << "}" << EOL;  // end (1)
      }
    }

  }
}
#endif

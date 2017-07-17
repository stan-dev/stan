#ifndef STAN_LANG_GENERATOR_GENERATE_VALIDATE_VAR_DECLS_HPP
#define STAN_LANG_GENERATOR_GENERATE_VALIDATE_VAR_DECLS_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/generate_validate_var_decl.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Generate code to validate the specified variable declarations at
     * the specified indentation level to the specified stream.
     * Generated code is preceeded by stmt updating global variable
     * `current_statement_begin__` to src file line number where
     * variable is declared.
     *
     * @param[in] decls variable declarations
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void generate_validate_var_decls(const std::vector<var_decl> decls,
                                     int indent, std::ostream& o) {
      for (size_t i = 0; i < decls.size(); ++i) {
        generate_indent(indent, o);
        o << "current_statement_begin__ = " <<  decls[i].begin_line_ << ";"
          << EOL;

        generate_validate_var_decl(decls[i], indent, o);
      }
    }
  }
}
#endif

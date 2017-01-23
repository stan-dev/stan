#ifndef STAN_LANG_GENERATOR_GENERATE_VALIDATE_VAR_DECLS_HPP
#define STAN_LANG_GENERATOR_GENERATE_VALIDATE_VAR_DECLS_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/generate_validate_var_decl.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Generate code to validate the specified variable declarations at
     * the specified indentation level to the specified stream.
     *
     * @param[in] decls variable declarations
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void generate_validate_var_decls(const std::vector<var_decl> decls,
                                     int indent, std::ostream& o) {
      for (size_t i = 0; i < decls.size(); ++i)
        generate_validate_var_decl(decls[i], indent, o);
    }


  }
}
#endif

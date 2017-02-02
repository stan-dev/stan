#ifndef STAN_LANG_GENERATOR_GENERATE_VALIDATE_VAR_DECL_HPP
#define STAN_LANG_GENERATOR_GENERATE_VALIDATE_VAR_DECL_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/validate_var_decl_visgen.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <ostream>

namespace stan {
  namespace lang {

    /**
     * Generate code to validate the specified variable declaration at
     * the specified indentation level to the specified stream.
     *
     * @param[in] decl variable declaration
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void generate_validate_var_decl(const var_decl& decl, int indent,
                                    std::ostream& o) {
      validate_var_decl_visgen vis(indent, o);
      boost::apply_visitor(vis, decl.decl_);
    }

  }
}
#endif

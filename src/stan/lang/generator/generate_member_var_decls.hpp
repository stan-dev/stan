#ifndef STAN_LANG_GENERATOR_GENERATE_MEMBER_VAR_DECLS_HPP
#define STAN_LANG_GENERATOR_GENERATE_MEMBER_VAR_DECLS_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/member_var_decl_visgen.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Generate member variable declarations for the specified
     * variable declarations at the specified indentation level to the
     * specified stream.
     *
     * @param[in] vs variable declarations
     * @param[in] indent indentation level
     * @param[in] o stream for writing
     */
    void generate_member_var_decls(const std::vector<var_decl>& vs,
                                   int indent, std::ostream& o) {
      member_var_decl_visgen vis(indent, o);
      for (size_t i = 0; i < vs.size(); ++i)
        boost::apply_visitor(vis, vs[i].decl_);
    }



  }
}
#endif

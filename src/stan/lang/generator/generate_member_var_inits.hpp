#ifndef STAN_LANG_GENERATOR_GENERATE_MEMBER_VAR_INITS_HPP
#define STAN_LANG_GENERATOR_GENERATE_MEMBER_VAR_INITS_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/dump_member_var_visgen.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Generate initializations for member variables by reading from
     * constructor variable context.
     *
     * @param[in] vs member variable declarations
     * @param[in,out] o stream for generating
     */
    void generate_member_var_inits(const std::vector<var_decl>& vs,
                                   std::ostream& o) {
      dump_member_var_visgen vis(o);
      for (size_t i = 0; i < vs.size(); ++i)
        boost::apply_visitor(vis, vs[i].decl_);
    }

  }
}
#endif

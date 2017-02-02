#ifndef STAN_LANG_GENERATOR_GENERATE_LOCAL_VAR_INITS_HPP
#define STAN_LANG_GENERATOR_GENERATE_LOCAL_VAR_INITS_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/init_local_var_visgen.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Generate initializations for the specified local variables,
     * with flags indicating whether the generation is in a variable
     * context and whether variables need to be declared, writing to
     * the specified stream.
     *
     * @param[in] vs variable declarations
     * @param[in] is_var_context true if generating in variable
     * context
     * @param[in] declare_vars true if variables should be declared
     * @param[in,out] o stream for generating
     */
    void generate_local_var_inits(std::vector<var_decl> vs, bool is_var_context,
                                  bool declare_vars, std::ostream& o) {
      o << INDENT2
        << "stan::io::reader<"
        << (is_var_context ? "T__" : "double")
        << "> in__(params_r__,params_i__);" << EOL2;
      init_local_var_visgen vis_init(declare_vars, is_var_context, o);
      for (size_t i = 0; i < vs.size(); ++i)
        boost::apply_visitor(vis_init, vs[i].decl_);
    }

  }
}
#endif

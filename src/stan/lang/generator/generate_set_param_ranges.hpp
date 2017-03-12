#ifndef STAN_LANG_GENERATOR_GENERATE_SET_PARAM_RANGES_HPP
#define STAN_LANG_GENERATOR_GENERATE_SET_PARAM_RANGES_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
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
     *
     * @param[in] var_decls sequence of variable declarations
     * @param[in,out] o stream for generating
     */
    void generate_set_param_ranges(const std::vector<var_decl>& var_decls,
                                   std::ostream& o) {
      o << INDENT2 << "num_params_r__ = 0U;" << EOL;
      o << INDENT2 << "param_ranges_i__.clear();" << EOL;
      set_param_ranges_visgen vis(o);
      for (size_t i = 0; i < var_decls.size(); ++i)
        boost::apply_visitor(vis, var_decls[i].decl_);
    }



  }
}
#endif

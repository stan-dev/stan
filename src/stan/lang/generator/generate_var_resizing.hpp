#ifndef STAN_LANG_GENERATOR_GENERATE_VAR_RESIZING_HPP
#define STAN_LANG_GENERATOR_GENERATE_VAR_RESIZING_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/var_resizing_visgen.hpp>
#include <stan/lang/generator/init_vars_visgen.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Generate code to to the specified stream to resize the
     * variables in the specified declarations and fill them with
     * dummy values.
     *
     * @param[in] vs variable declarations
     * @param[in,out] o stream for generating
     */
    void generate_var_resizing(const std::vector<var_decl>& vs,
                               std::ostream& o) {
      var_resizing_visgen vis_resizer(o);
      init_vars_visgen vis_filler(2, o);
      for (size_t i = 0; i < vs.size(); ++i) {
        boost::apply_visitor(vis_resizer, vs[i].decl_);
        boost::apply_visitor(vis_filler, vs[i].decl_);
        if (vs[i].has_def()) {
          o << INDENT2 << "stan::math::assign(" << vs[i].name() << ",";
          generate_expression(vs[i].def(), o);
          o << ");" << EOL;
        }
      }
    }


  }
}
#endif

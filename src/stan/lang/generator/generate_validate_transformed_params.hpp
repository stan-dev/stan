#ifndef STAN_LANG_GENERATOR_GENERATE_VALIDATE_TRANSFORMED_PARAMS_HPP
#define STAN_LANG_GENERATOR_GENERATE_VALIDATE_TRANSFORMED_PARAMS_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_comment.hpp>
#include <stan/lang/generator/validate_transformed_params_visgen.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Generate code to validate the specified transformed parameters,
     * generating at the specified indentation level to the specified
     * stream.
     *
     * @param[in] vs variable declarations
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void generate_validate_transformed_params(const std::vector<var_decl>& vs,
                                              int indent, std::ostream& o) {
      generate_comment("validate transformed parameters", indent, o);
      validate_transformed_params_visgen vis(indent, o);
      for (size_t i = 0; i < vs.size(); ++i)
        boost::apply_visitor(vis, vs[i].decl_);
      o << EOL;
    }

  }
}
#endif

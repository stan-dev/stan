#ifndef STAN_LANG_GENERATOR_GET_CONSTRAIN_FN_PREFIX_HPP
#define STAN_LANG_GENERATOR_GET_CONSTRAIN_FN_PREFIX_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <ostream>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Generate the name of the constrain function together
     * with expressions for the bounds parameters, if any.
     *
     * NOTE: expecting that parser disallows integer params.
     *
     * @param[in] btype block var type
     */
    std::string
    get_constrain_fn_prefix(const block_var_type& btype) {
      std::stringstream ss;
      if (btype.bare_type().is_double_type())
        ss << "scalar";
      else
        ss << btype.name();
      if (btype.has_def_bounds()) {
        if (btype.bounds().has_low() && btype.bounds().has_high()) {
          ss << "_lub_constrain(";
          generate_expression(btype.bounds().low_.expr_, NOT_USER_FACING, ss);
          ss << ", ";
          generate_expression(btype.bounds().high_.expr_, NOT_USER_FACING, ss);
          ss << ", ";
        } else if (btype.bounds().has_low()) {
          ss << "_lb_constrain(";
          generate_expression(btype.bounds().low_.expr_, NOT_USER_FACING, ss);
          ss << ", ";
        } else {
          ss << "_ub_constrain(";
          generate_expression(btype.bounds().high_.expr_, NOT_USER_FACING, ss);
          ss << ", ";
        }
      } else {
        ss << "_constrain(";
      }
      if (!is_nil(btype.arg1())) {
        generate_expression(btype.arg1(), NOT_USER_FACING, ss);
      }
      if (btype.name() == "matrix" || btype.name() == "cholesky_factor_cov") {
        ss << ", ";
        generate_expression(btype.arg2(), NOT_USER_FACING, ss);
      }
      return ss.str();
    }

  }
}
#endif

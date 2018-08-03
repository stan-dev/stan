#ifndef STAN_LANG_GENERATOR_WRITE_CONSTRAINTS_FN_HPP
#define STAN_LANG_GENERATOR_WRITE_CONSTRAINTS_FN_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_expression.hpp>
#include <ostream>
#include <sstream>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Generate the name of the constrain function together
     * with expressions for the bounds parameters, if any.
     * Constrain and unconstrain functions both take bounds
     * constrain function also needs row, column size args.
     *
     * NOTE: expecting that parser disallows integer params.
     *
     * @param[in] btype block var type
     * @param[in] fn_name either "constrain" or "unconstrain"
     */
    std::string write_constraints_fn(const block_var_type& btype,
                                   std::string fn_name) {
      std::stringstream ss;

      if (btype.bare_type().is_double_type())
        ss << "scalar";
      else
        ss << btype.name();

      if (btype.has_def_bounds()) {
        if (btype.bounds().has_low() && btype.bounds().has_high()) {
          ss << "_lub_" << fn_name << "(";
          generate_expression(btype.bounds().low_.expr_, NOT_USER_FACING, ss);
          ss << ", ";
          generate_expression(btype.bounds().high_.expr_, NOT_USER_FACING, ss);
        } else if (btype.bounds().has_low()) {
          ss << "_lb_" << fn_name << "(";
          generate_expression(btype.bounds().low_.expr_, NOT_USER_FACING, ss);
        } else {
          ss << "_ub_" << fn_name << "(";
          generate_expression(btype.bounds().high_.expr_, NOT_USER_FACING, ss);
        }
      } else {
        ss << "_" << fn_name << "(";
      }

      if ((fn_name.compare("unconstrain") == 0)) {
        if (btype.has_def_bounds())
          ss << ", ";
        return ss.str();
      }

      if (!is_nil(btype.arg1())) {
        if (btype.has_def_bounds())
          ss << ", ";
        generate_expression(btype.arg1(), NOT_USER_FACING, ss);
      }
      if (btype.name() == "matrix"
          || btype.name() == "cholesky_factor_cov") {
        ss << ", ";
        generate_expression(btype.arg2(), NOT_USER_FACING, ss);
      }
      return ss.str();
    }

  }
}
#endif

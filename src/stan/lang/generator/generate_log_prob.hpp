#ifndef STAN_LANG_GENERATOR_GENERATE_LOG_PROB_HPP
#define STAN_LANG_GENERATOR_GENERATE_LOG_PROB_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_comment.hpp>
#include <stan/lang/generator/generate_local_var_decls.hpp>
#include <stan/lang/generator/generate_local_var_inits.hpp>
#include <stan/lang/generator/generate_located_statement.hpp>
#include <stan/lang/generator/generate_located_statements.hpp>
#include <stan/lang/generator/generate_validate_transformed_params.hpp>
#include <stan/lang/generator/generate_validate_var_decls.hpp>
#include <ostream>

namespace stan {
  namespace lang {

    /**
     * Generate the log_prob method for the model class for the
     * specified program on the specified stream.
     *
     * @param p program
     * @param o stream for generating
     */
    void generate_log_prob(const program& p, std::ostream& o) {
      o << EOL;
      o << INDENT << "template <bool propto__, bool jacobian__, typename T__>"
        << EOL;
      o << INDENT << "T__ log_prob(vector<T__>& params_r__,"
        << EOL;
      o << INDENT << "             vector<int>& params_i__,"
        << EOL;
      o << INDENT << "             std::ostream* pstream__ = 0) const {"
        << EOL2;

      // use this dummy for inits
      o << INDENT2
        << "T__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());"
        << EOL;
      o << INDENT2 << "(void) DUMMY_VAR__;  // suppress unused var warning"
        << EOL2;

      o << INDENT2 << "T__ lp__(0.0);"
        << EOL;
      o << INDENT2 << "stan::math::accumulator<T__> lp_accum__;"
        << EOL2;

      bool is_var_context = true;
      bool is_fun_return = false;

      generate_comment("model parameters", 2, o);
      generate_local_var_inits(p.parameter_decl_, is_var_context, true, o);
      o << EOL;

      generate_comment("transformed parameters", 2, o);
      generate_local_var_decls(p.derived_decl_.first, 2, o, is_var_context,
                               is_fun_return);
      o << EOL;

      bool include_sampling = true;
      generate_located_statements(p.derived_decl_.second, 2, o,
                                  include_sampling, is_var_context,
                                  is_fun_return);
      o << EOL;

      generate_validate_transformed_params(p.derived_decl_.first, 2, o);
      o << INDENT2
        << "const char* function__ = \"validate transformed params\";"
        << EOL;
      o << INDENT2
        << "(void) function__;  // dummy to suppress unused var warning"
        << EOL;

      generate_validate_var_decls(p.derived_decl_.first, 2, o);

      o << EOL;
      generate_comment("model body", 2, o);


      generate_located_statement(p.statement_, 2, o, include_sampling,
                                 is_var_context, is_fun_return);


      o << EOL;
      o << INDENT2 << "lp_accum__.add(lp__);" << EOL;
      o << INDENT2 << "return lp_accum__.sum();" << EOL2;
      o << INDENT << "} // log_prob()" << EOL2;

      o << INDENT
        << "template <bool propto, bool jacobian, typename T_>" << EOL;
      o << INDENT
        << "T_ log_prob(Eigen::Matrix<T_,Eigen::Dynamic,1>& params_r," << EOL;
      o << INDENT << "           std::ostream* pstream = 0) const {" << EOL;
      o << INDENT << "  std::vector<T_> vec_params_r;" << EOL;
      o << INDENT << "  vec_params_r.reserve(params_r.size());" << EOL;
      o << INDENT << "  for (int i = 0; i < params_r.size(); ++i)" << EOL;
      o << INDENT << "    vec_params_r.push_back(params_r(i));" << EOL;
      o << INDENT << "  std::vector<int> vec_params_i;" << EOL;
      o << INDENT
        << "  return log_prob<propto,jacobian,T_>(vec_params_r, "
        << "vec_params_i, pstream);" << EOL;
      o << INDENT << "}" << EOL2;
    }

  }
}
#endif

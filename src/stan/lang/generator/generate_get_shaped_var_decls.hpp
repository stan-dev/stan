#ifndef STAN_LANG_GENERATOR_GENERATE_GET_SHAPED_VAR_DECLS_HPP
#define STAN_LANG_GENERATOR_GENERATE_GET_SHAPED_VAR_DECLS_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/get_shaped_var_decls_visgen.hpp>
#include <stan/lang/generator/constants.hpp>
#include <ostream>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Generates the variables and declarations with pushbacks into an
     * accumulator as a helper function.
     *
     * @param[in] flag name of boolean variable to generate to
     * condition execution of pushbacks
     * @param[in] vs variable declaraitons for tis block
     * @param[in, out] o stream to which code is generated
     */
    void generate_get_shaped_var_decls(const std::string& flag,
                      const std::vector<stan::lang::var_decl>& vs,
                      std::ostream& o) {
      o << EOL << INDENT2 << "if (" << flag << ") {" << EOL;
      get_shaped_var_decls_visgen vis(o);
      for (size_t i = 0; i < vs.size(); ++i)
        boost::apply_visitor(vis, vs[i].decl_);
      o << INDENT2 << "}" << EOL;
    }

    /**
     * Generates the static class function
     * <code>shaped_var_decls()</code> for the specified program on
     * the specified stream.
     *
     * @param[in] prog the Stan program AST node
     * @param[in, out] o stream to which code is generated
     */
    void generate_get_shaped_var_decls(const stan::lang::program& prog,
                                       std::ostream& o) {
      o << EOL2 << INDENT << "std::vector<stan::model::shaped_var_decl>"
        << EOL << INDENT << "shaped_var_decls(bool data__,"
        << EOL << INDENT3 << "bool transformed_data__,"
        << EOL << INDENT3 << "bool parameters__,"
        << EOL << INDENT3 << "bool transformed_parameters__,"
        << EOL << INDENT3 << "bool generated_quantities__,"
        << EOL << INDENT3 << "std::vector<double>& params_r__,"
        << EOL << INDENT3 << "unsigned int random_seed__) {"
        << EOL << INDENT2 << "using stan::model::shaped_var_decl;"
        << EOL << INDENT2 << "using std::vector;"
        << EOL << INDENT2 << "vector<stan::model::shaped_var_decl> decls__;"
        << EOL << INDENT2 << "vector<int> params_i__;"
        << EOL << INDENT2 << "std::stringstream ss__;"
        << EOL << INDENT2 << "std::ostream* pstream__ = &ss__;"
        << EOL << INDENT2 << "boost::ecuyer1988 base_rng__"
        << EOL << INDENT3
        << "= stan::services::util::create_rng(random_seed__, 1);"
        << EOL;

      // DATA

      generate_get_shaped_var_decls("data__", prog.data_decl_, o);

      // TRANSFORMED DATA

      generate_get_shaped_var_decls("transformed_data__",
                                    prog.derived_data_decl_.first, o);

      // PARAMETERS

      o << INDENT2 << "if (!parameters__" << EOL;
      o << INDENT2 << "    && !transformed_parameters__" << EOL;
      o << INDENT2 << "    && !generated_quantities__) return decls__;"
        << EOL;

      // similar to generate_write_array_method.hpp (without writing)
      o << INDENT2 << "stan::io::reader<double> in__(params_r__, params_i__);"
        << EOL;
      o << INDENT2 << "static const char* function__"
        << "= \"shaped_var_decls\";" << EOL;
      generate_void_statement("function__", 2, o);
      generate_comment("read-transform, write parameters", 2, o);
      write_array_visgen vis(o);
      for (size_t i = 0; i < prog.parameter_decl_.size(); ++i)
        boost::apply_visitor(vis, prog.parameter_decl_[i].decl_);

      generate_get_shaped_var_decls("parameters__", prog.parameter_decl_, o);

      // TRANSFORMED PARAMETERS

      o << INDENT2 << "if (!transformed_parameters__" << EOL;
      o << INDENT2 << "    && !generated_quantities__) return decls__;"
        << EOL;

      generate_comment("declare and define transformed parameters", 2, o);
      o << INDENT2 <<  "double lp__ = 0.0;" << EOL;
      generate_void_statement("lp__", 2, o);
      o << INDENT2 << "stan::math::accumulator<double> lp_accum__;" << EOL2;
      o << INDENT2
        << "double DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());"
        << EOL;
      generate_void_statement("DUMMY_VAR__", 2, o);

      generate_try(2, o);  // matching generate catch at end

      static const bool is_var_context = false;
      static const bool is_fun_return = false;
      generate_local_var_decls(prog.derived_decl_.first, 3, o, is_var_context,
                               is_fun_return);
      o << EOL;
      static const bool include_sampling = false;
      generate_statements(prog.derived_decl_.second, 3, o,
                         include_sampling, is_var_context,
                         is_fun_return);
      o << EOL;
      generate_comment("validate transformed parameters", 3, o);
      generate_validate_var_decls(prog.derived_decl_.first, 3, o);
      o << EOL;

      generate_get_shaped_var_decls("transformed_parameters__",
                                    prog.derived_decl_.first, o);

      // GENERATED QUANTITIES

      o << INDENT3 << "if (!generated_quantities__) return decls__;" << EOL;

      generate_comment("declare and define generated quantities", 3, o);
      generate_local_var_decls(prog.generated_decl_.first, 3, o,
                               is_var_context, is_fun_return);
      o << EOL;
      generate_statements(prog.generated_decl_.second,
                           3, o, include_sampling, is_var_context,
                           is_fun_return);
      o << EOL;
      generate_comment("validate generated quantities", 3, o);
      generate_validate_var_decls(prog.generated_decl_.first, 3, o);
      o << EOL;

      generate_get_shaped_var_decls("generated_quantities__",
                                    prog.generated_decl_.first, o);

      generate_catch_throw_located(2, o);
      o << EOL << INDENT2 << "return decls__;"
        << EOL << INDENT << "}"
        << EOL2;
    }

  }
}
#endif

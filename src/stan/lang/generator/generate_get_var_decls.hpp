#ifndef STAN_LANG_GENERATOR_GENERATE_GET_VAR_DECLS_HPP
#define STAN_LANG_GENERATOR_GENERATE_GET_VAR_DECLS_HPP

#include <stan/lang/ast.hpp>
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
    void generate_get_var_decls(const std::string& flag,
                                const std::vector<stan::lang::var_decl>& vs,
                                std::ostream& o) {
      o << EOL << INDENT2 << "if (" << flag << ") {" << EOL;

      get_var_decls_visgen vis(o);
      for (size_t i = 0; i < vs.size(); ++i)
        boost::apply_visitor(vis, vs[i].decl_);

      o << INDENT2 << "}" << EOL;
    }

    /**
     * Generates the static class function get_var_decls().
     *
     * @param[in] prog the Stan program AST node
     * @param[in, out] o stream to which code is generated
     */
    void generate_get_var_decls(const stan::lang::program& prog,
                                std::ostream& o) {
      o << EOL
        << EOL << INDENT << "static std::vector<stan::model::var_decl>"
        << EOL << INDENT << "get_var_decls(bool data__,"
        << EOL << INDENT3 << "bool transformed_data__,"
        << EOL << INDENT3 << "bool parameters__,"
        << EOL << INDENT3 << "bool transformed_parameters__,"
        << EOL << INDENT3 << "bool generated_quantities__) {"
        << EOL << INDENT2 << "using stan::model::var_decl;"
        << EOL << INDENT2 << "using std::vector;"
        << EOL << INDENT2 << "vector<var_decl> decls__;";

      generate_get_var_decls("data__", prog.data_decl_, o);
      generate_get_var_decls("transformed_data__",
                             prog.derived_data_decl_.first, o);
      generate_get_var_decls("parameters__", prog.parameter_decl_, o);
      generate_get_var_decls("transformed_parameters__",
                             prog.derived_decl_.first, o);
      generate_get_var_decls("generated_quantities__",
                             prog.generated_decl_.first, o);

      o << EOL << INDENT2 << "return decls__;"
        << EOL << INDENT << "}"
        << EOL << EOL;
    }

  }
}
#endif

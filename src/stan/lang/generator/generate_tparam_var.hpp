#ifndef STAN_LANG_GENERATOR_GENERATE_TPARAM_VAR_HPP
#define STAN_LANG_GENERATOR_GENERATE_TPARAM_VAR_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/generate_bare_type.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_expression.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/generate_initializer.hpp>
#include <stan/lang/generator/generate_validate_var_dims.hpp>
#include <stan/lang/generator/generate_var_constructor.hpp>
#include <stan/lang/generator/generate_void_statement.hpp>
#include <iostream>
#include <ostream>
#include <vector>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Generate transformed parameter declaration, fill statements.
     *
     * @param[in] var_decl transformed parameter block variable
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void generate_tparam_var(const block_var_decl& var_decl,
                             int indent, std::ostream& o) {

      std::string var_name(var_decl.name());
      if (var_decl.type().num_dims() > 0)
        generate_validate_var_dims(var_decl, indent, o);

      generate_indent(indent, o);
      generate_bare_type(var_decl.type().bare_type(), "local_scalar_t__", o);
      o << " " << var_name;
      if (var_decl.type().num_dims() == 0) {
        o << ";" << EOL;
        generate_void_statement(var_name, indent, o);
      } else {
        o << "(";
        generate_initializer(var_decl.type(), "local_scalar_t__", o);
        o << ");" << EOL;
      }        

      generate_indent(indent, o);
      o << "stan::math::initialize(" << var_decl.name() << ", DUMMY_VAR__);" << EOL;

      generate_indent(indent, o);
      o << "stan::math::fill(" << var_decl.name() << ", DUMMY_VAR__);" << EOL;

      if (var_decl.has_def()) {
        generate_indent(indent, o);
        o << "stan::math::assign("
          << var_decl.name()
          << ",";
        generate_expression(var_decl.def(), NOT_USER_FACING, o);
        o << ");" << EOL;
      }
    }        

  }
}
#endif

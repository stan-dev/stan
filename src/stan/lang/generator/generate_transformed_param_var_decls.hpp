#ifndef STAN_LANG_GENERATOR_GENERATE_TRANSFORMED_PARAM_VAR_DECLS_HPP
#define STAN_LANG_GENERATOR_GENERATE_TRANSFORMED_PARAM_VAR_DECLS_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/generate_validate_positive.hpp>
#include <stan/lang/generator/write_var_decl_arg.hpp>
#include <stan/lang/generator/write_var_decl_type.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Generate transformed_param variable declarations, including
     * initializations, for the specified declarations, indentation
     * level, writing to the specified stream.
     * Generated code is preceeded by stmt updating global variable
     * `current_statement_begin__` to src file line number where
     * variable is declared.
     *
     * @param[in] vs variable declarations
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void generate_transformed_param_var_decls(const std::vector<block_var_decl>& vs, int indent,
                                  std::ostream& o) {
      for (size_t i = 0; i < vs.size(); ++i) {
        generate_indent(indent, o);
        o << "current_statement_begin__ = " <<  vs[i].begin_line_ << ";"
          << EOL;

        int ar_dims = vs[i].type().array_dims();
        std::vector<expression> ar_lens(vs[i].type().array_lens());
        std::string var_name(vs[i].name());
        // unfold array type to get array element info
        block_var_type ltype = (vs[i].type());
        if (ltype.is_array_type())
          ltype = ltype.array_contains();

        // validate dimensions before declaration
        for (int i = 1; i < ar_dims; ++i)
          generate_validate_positive(var_name, ar_lens[i], indent, o);

        // declare 
        write_var_decl_type(ltype.bare_type(), 
                            get_verbose_var_type(btype.bare_type()),
                            ar_dims, indent, o);
        o << " " << var_name;
        write_var_decl_arg(ltype.bare_type(),
                           ar_lens, ltype.arg1(), ltype.arg2(), o);
        o << ";" << EOL;

        // fill
        generate_indent(indent, o);
        if (ltype.bare_type().is_int_type()) {
          o << "stan::math::fill(" << var_name
            << ", std::numeric_limits<int>::min());"
            << EOL;
        } else {
          o << "stan::math::fill(" << var_name << ", DUMMY_VAR__);" << EOL;
        }

        // define
        if (vs[i].has_def()) {
          generate_indent(indent, o);
          o << "stan::math::assign("
            << vs[i].name()
            << ",";
          generate_expression(vs[i].def(), NOT_USER_FACING, o);
          o << ");" << EOL;
        }

        // validate
        // TODO:mitzi
        // check initialized
        // apply constraints
        


      }
      o << EOL;
    }

  }
}
#endif

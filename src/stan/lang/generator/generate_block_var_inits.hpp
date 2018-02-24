#ifndef STAN_LANG_GENERATOR_GENERATE_BLOCK_VAR_INITS_HPP
#define STAN_LANG_GENERATOR_GENERATE_BLOCK_VAR_INITS_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/get_block_var_dims.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Generate code to to the specified stream to resize the
     * variables in the specified declarations and fill them with
     * dummy values.
     * Generated code is preceeded by stmt updating global variable
     * `current_statement_begin__` to src file line number where
     * variable is declared.
     *
     * @param[in] vs block variable declarations
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void generate_block_var_inits(const std::vector<block_var_decl>& vs,
                                  int indent, std::ostream& o) {

      for (size_t i = 0; i < vs.size(); ++i) {
        generate_indent(indent, o);
        o << "current_statement_begin__ = " <<  vs[i].begin_line_ << ";"
          << EOL;

        // unfold recursive array type to get array element info
        std::string var_name(vs[i].name());
        block_var_type btype = (vs[i].type());
        if (btype.is_array_type())
          btype = btype.array_contains();
        std::string cpp_typename(btype.cpp_typename());
        expression el_arg1 = btype.arg1();
        expression el_arg2 = btype.arg2();

        generate_initialization(o, indent, var_name, cpp_typename,
                                el_arg1, el_arg2, vs[i].type().array_lens());

        generate_indent(indent, o);
        if (btype.bare_type().is_int_type()) {
          o << "stan::math::fill(" << var_name
            << ", std::numeric_limits<int>::min());"
            << EOL;
        } else {
          o << "stan::math::fill(" << var_name << ", DUMMY_VAR__);" << EOL;
        }

        if (vs[i].has_def()) {
          generate_indent(indent, o);
          o << "stan::math::assign(" << vs[i].name() << ", ";
          generate_expression(vs[i].def(), NOT_USER_FACING, o);
          o << ");" << EOL;
        }
      }
    }
  }
}
#endif

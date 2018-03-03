#ifndef STAN_LANG_GENERATOR_GENERATE_BLOCK_VAR_DECLS_HPP
#define STAN_LANG_GENERATOR_GENERATE_BLOCK_VAR_DECLS_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Generate local variable declarations, including
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
    void generate_block_var_decls(const std::vector<block_var_decl>& vs, int indent,
                                  std::ostream& o) {
      for (size_t i = 0; i < vs.size(); ++i) {
        generate_indent(indent, o);
        o << "current_statement_begin__ = " <<  vs[i].begin_line_ << ";"
          << EOL;

        //   // unfold array type to get array element info
        //   std::string var_name(vs[i].name());
        //   block_var_type btype = (vs[i].type());
        //   if (btype.is_array_type())
        //     btype = btype.array_contains();
        //   expression el_arg1 = btype.arg1();
        //   expression el_arg2 = btype.arg2();
        //   // declare
        //   std::string typedef_var_type = get_typedef_var_type(btype.bare_type());
        //   int ar_dims = vs[i].type().array_dims();
        //   for (int i = 0; i < indent; ++i)
        //     o << INDENT;
        //   for (int i = 0; i < ar_dims; ++i)
        //     o << "vector<";
        //   o << typedef_var_type;
        //   if (ar_dims > 0)
        //     o << ">";
        //   for (int i = 1; i < ar_dims; ++i)
        //     o << " >";
        //   o << " " << var_name << ";" << EOL;

        //   generate_initialization(o, indent, var_name, btype.bare_type(),
        //                           el_arg1, el_arg2, vs[i].type().array_lens());

        //   // initialize, fill
        //   if (btype.bare_type().is_int_type()) {
        //     generate_indent(indent, o);
        //     o << "stan::math::fill(" << var_name
        //       << ", std::numeric_limits<int>::min());"
        //       << EOL;
        //   } else {
        //     generate_indent(indent, o);
        //     o << "stan::math::initialize(" << var_name << ", DUMMY_VAR__);" << EOL;
        //     generate_indent(indent, o);
        //     o << "stan::math::fill(" << var_name << ", DUMMY_VAR__);" << EOL;
        //   }
        //   if (vs[i].has_def()) {
        //     generate_indent(indent, o);
        //     o << "stan::math::assign("
        //       << vs[i].name()
        //       << ",";
        //     generate_expression(vs[i].def(), NOT_USER_FACING, o);
        //     o << ");" << EOL;
        //      }
        o << EOL;
      }
    }
  }
}
#endif

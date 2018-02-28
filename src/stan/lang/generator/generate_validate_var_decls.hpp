#ifndef STAN_LANG_GENERATOR_GENERATE_VALIDATE_VAR_DECLS_HPP
#define STAN_LANG_GENERATOR_GENERATE_VALIDATE_VAR_DECLS_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/write_dims_for_loop_begin.hpp>
#include <stan/lang/generator/write_dims_for_loop_end.hpp>
#include <stan/lang/generator/write_dims_for_loop_var.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Generate code to validate the specified variable declarations at
     * the specified indentation level to the specified stream.
     * Checks bounds on block var decls and constraints on
     * specialized vector and matrix types.
     * NOTE:  bounded / specialized types are mutually exclusive
     *
     * Generated code is preceeded by stmt updating global variable
     * `current_statement_begin__` to src file line number where
     * variable is declared.
     *
     * @param[in] decls variable declarations
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void generate_validate_var_decls(const std::vector<block_var_decl> decls,
                                     int indent, std::ostream& o) {
      for (size_t i = 0; i < decls.size(); ++i) {
        generate_indent(indent, o);
        o << "current_statement_begin__ = " <<  decls[i].begin_line_ << ";"
          << EOL;

        std::string var_name(decls[i].name());
        std::vector<expression> ar_lens(decls[i].type().array_lens());
        // unfold array type to get array element info
        block_var_type btype = (decls[i].type());
        if (btype.is_array_type())
          btype = btype.array_contains();

        if (btype.has_def_bounds()) {
          write_dims_for_loop_begin(ar_lens, indent, o);
          range bounds = btype.bounds();
          if (bounds.has_low()) {
            generate_indent(indent + ar_lens.size(), o);
            o << "check_greater_or_equal(function__, ";
            o << "\"";
            write_dims_for_loop_var(var_name, ar_lens.size(), indent, o);
            o << "\", ";
            write_dims_for_loop_var(var_name, ar_lens.size(), indent, o);
            o << ", ";
            generate_expression(bounds.low_.expr_, NOT_USER_FACING, o);
            o << ");" << EOL;
          }
          if (bounds.has_high()) {
            generate_indent(indent + ar_lens.size(), o);
            o << "check_less_or_equal(function__, ";
            o << "\"";
            write_dims_for_loop_var(var_name, ar_lens.size(), indent, o);
            o << "\", ";
            write_dims_for_loop_var(var_name, ar_lens.size(), indent, o);
            o << ", ";
            generate_expression(bounds.high_.expr_, NOT_USER_FACING, o);
            o << ");" << EOL;
          }
          write_dims_for_loop_end(ar_lens.size(), indent, o);
        } else if (btype.is_specialized()) {
          write_dims_for_loop_begin(ar_lens, indent, o);
          generate_indent(indent + ar_lens.size(), o);
          o << "stan::math::check_" << btype.name() << "(function__, ";
          o << "\"";
          write_dims_for_loop_var(var_name, ar_lens.size(), indent, o);
          o << "\", ";
          write_dims_for_loop_var(var_name, ar_lens.size(), indent, o);
          o << ");" << EOL;
          write_dims_for_loop_end(ar_lens.size(), indent, o);
        }
      }
    }
  }
}
#endif

#ifndef STAN_LANG_GENERATOR_GENERATE_READ_TRANSFORM_PARAMS_HPP
#define STAN_LANG_GENERATOR_GENERATE_READ_TRANSFORM_PARAMS_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/get_constrain_fn_prefix.hpp>
#include <ostream>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Generate declarations for parameters on the constrained scale.
     *
     * @param[in] vs variable declarations
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void generate_read_transform_params(const std::vector<block_var_decl>& vs,
                                        int indent, std::ostream& o) {
      for (size_t i = 0; i < vs.size(); ++i) {
        // generate_indent(indent, o);
        // o << "current_statement_begin__ = " <<  vs[i].begin_line_ << ";"
        //   << EOL;

        // setup - name, type, and var shape
        std::string var_name(vs[i].name());
        // unfold array type to get array element info
        block_var_type btype = (vs[i].type());
        if (btype.is_array_type())
          btype = btype.array_contains();
        std::string cpp_type_str = get_typedef_var_type(btype.bare_type());
        // dimension sizes and type - array or matrix/vec rows, columns
        std::vector<expression> ar_lens(vs[i].type().array_lens());
        expression arg1 = btype.arg1();
        expression arg2 = btype.arg2();
        // use parallel arrays of index names and dim_size names for read loop
        std::vector<std::string> idx_names;
        std::vector<std::string> idx_dimsize_names;
        for (size_t k = 0; k < ar_lens.size(); ++k) {
          std::stringstream ss_name;
          ss_name << "k__" << k << "__";
          idx_names.push_back(ss_name.str());
          std::stringstream ss_dim;
          ss_dim <<  "dim_" << var_name << k << "__";
          idx_dimsize_names.push_back(ss_dim.str());
        }

        // declaration
        write_var_decl_type(btype.bare_type(), cpp_type_str,
                            ar_lens.size(), indent, o);
        o << " " << var_name;
        if (ar_lens.size() == 0) {
          o << " = " << get_constrain_fn_prefix(btype) << ");" << EOL;
        } else {
          o << ";" << EOL;
          // open for loop: declare dim var, (resize)
          for (size_t k = 0; k < ar_lens.size(); ++k) {
            generate_indent(indent + k, o);
            o << "size_t " << idx_dimsize_names[k] << " = ";
            generate_expression(ar_lens[k], NOT_USER_FACING, o);
            o << ";" << EOL;

            // only resize outer dims
            if (k < ar_lens.size() - 1) {
              generate_indent(indent + k, o);
              o << var_name;
              if (k > 0)
                o << "[" << idx_names[k-1] << "]";
              o << ".resize(" << idx_dimsize_names[k] << ");" << EOL;
            }

            generate_indent(indent + k, o);
            o << "for (size_t " << idx_names[k] << " = 0; "
              << idx_names[k] << " < " <<  idx_dimsize_names[k]
              << "; ++" << idx_names[k] << ") {" << EOL;
          }

          // assign to element
          generate_indent(indent + ar_lens.size(), o);
          o << var_name;
          for (size_t k = 1; k < ar_lens.size(); ++k)
            o << "[" << idx_names[k-1] << "]";
          o << ".push_back(in__." 
            << get_constrain_fn_prefix(btype)
            << "));" << EOL;

          // close for loop
          for (size_t k = ar_lens.size(); k > 0; --k) {
            generate_indent(indent + k - 1, o);
            o << "}" << EOL;
          }
        }
        o << EOL;
      }
      o << EOL;
    }

  }
}
#endif

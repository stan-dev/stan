#ifndef STAN_LANG_GENERATOR_GENERATE_DATA_BLOCK_VAR_INITS_HPP
#define STAN_LANG_GENERATOR_GENERATE_DATA_BLOCK_VAR_INITS_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/generate_validate_context_size.hpp>
#include <stan/lang/generator/generate_validate_positive.hpp>
#include <iostream>
#include <ostream>
#include <vector>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Generate initializations for data block variables by reading
     * dump format data from constructor variable context.
     * In dump format data, arrays are indexed in last-index major fashion,
     * which corresponds to column-major order for matrices
     * represented as two-dimensional arrays.  As a result, the first
     * indices change fastest.  Therefore loops must be constructed:
     * (col) (row) (array-dim-N) ... (array-dim-1)
     * 
     * Generated initializations are preceeded by stmt updating global variable
     * `current_statement_begin__` to src file line number where
     * variable is declared.
     *
     * @param[in] vs data block variable declarations
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void generate_data_block_var_inits(const std::vector<block_var_decl>& vs,
                                   int indent, std::ostream& o) {
      for (size_t i = 0; i < vs.size(); ++i) {
        generate_indent(indent, o);
        o << "current_statement_begin__ = " <<  vs[i].begin_line_ << ";"
          << EOL;

        // setup - name, type, and var shape
        std::string var_name(vs[i].name());
        // unfold array type to get array element info
        block_var_type btype = (vs[i].type());
        if (btype.is_array_type())
          btype = btype.array_contains();
        // dimension sizes and type - array or matrix/vec rows, columns
        std::vector<expression> ar_lens(vs[i].type().array_lens());
        expression arg1 = btype.arg1();
        expression arg2 = btype.arg2();
        // use parallel arrays of index names and limits for read loop
        std::vector<std::string> idx_names;
        std::vector<std::string> idx_limit_names;
        std::vector<expression> limits;
        int start_ar_idx = (!is_nil(arg2)) ? 2 : ((!is_nil(arg1)) ? 1 : 0);
        if (start_ar_idx == 2) {
          idx_names.push_back("arg2__");
          idx_limit_names.push_back(std::string(var_name + "_arg2_lim__"));
          limits.push_back(arg2);
        }
        if (start_ar_idx > 0) {
          idx_names.push_back("arg1__");
          idx_limit_names.push_back(std::string(var_name + "_arg1_lim__"));
          limits.push_back(arg1);
        }
        for (size_t i = ar_lens.size(); i > 0; --i) {
          std::stringstream ss_name;
          ss_name << "i__" << i - 1 << "__";
          idx_names.push_back(ss_name.str());
          std::stringstream ss_limit;
          ss_limit <<  var_name << "_limit_" << i - 1 << "__";
          idx_limit_names.push_back(ss_limit.str());
          limits.push_back(ar_lens[i - 1]);
        }
        std::string vals("vals_r");
        if (btype.bare_type().is_int_type())
          vals = "vals_i";

        // validate all dims
        for (size_t i = 0; i < ar_lens.size() ; ++i) {
          generate_validate_positive(var_name, ar_lens[i], indent, o);
        }
        if (!is_nil(arg1))
          generate_validate_positive(var_name, arg1, indent, o);
        if (!is_nil(arg2))
          generate_validate_positive(var_name, arg2, indent, o);

        // validate context
        generate_validate_context_size(indent, o, "data initialization",
                                       var_name, get_typedef_var_type(btype.bare_type()),
                                       ar_lens, arg1, arg2);
        generate_indent(indent, o);
        o << vals << "__ = context__." << vals << "(\"" << var_name << "\");" << EOL;
        generate_indent(indent, o);
        o << "pos__ = 0;" << EOL;

        // declare, initialize limits
        for (size_t i = 0; i < limits.size(); ++i) {
          generate_indent(indent, o);
          o << "size_t " << idx_limit_names[i]  << " = ";
          generate_expression(limits[i], NOT_USER_FACING, o);
          o << EOL;
        }

        // nested for stmts open
        int indentation = indent;
        for (size_t i = 0; i < limits.size(); ++i) {
          generate_indent(indentation, o);
          o << "for (size_t i_" << idx_names[i] << " = 0; "
            << idx_names[i] << " < " << idx_limit_names[i]
            << "; ++" << idx_names[i] << ") {" << EOL;
          indentation++;                                    
        }               
        // innermost assign - update pos__
        generate_indent(indentation, o);
        o << var_name;
        // array idxs
        for (size_t i = start_ar_idx; i < limits.size(); ++i)
          o << "[" << idx_names[i] << "]";
        if (start_ar_idx == 2) {
          o << "(" << idx_names[1] << ", " << idx_names[0] << ")";
        } else if (start_ar_idx == 1) {
          o << "(" << idx_names[0] << ")";
        }
        o << " = " << vals << "__[pos__++]; " << EOL;

        // nested close
        for (size_t i = 0; i < limits.size(); ++i) {
          indentation--;                                    
          generate_indent(indentation, o);
          o << "}" << EOL;
        }        
      }
    }

  }
}
#endif

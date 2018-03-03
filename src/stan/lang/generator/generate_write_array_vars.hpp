#ifndef STAN_LANG_GENERATOR_GENERATE_WRITE_ARRAY_VARS_HPP
#define STAN_LANG_GENERATOR_GENERATE_WRITE_ARRAY_VARS_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/generate_validate_context_size.hpp>
#include <stan/lang/generator/generate_validate_positive.hpp>
#include <iostream>
#include <ostream>
#include <vector>
#include <string>

      // // writes parameters
      // /// this is same indexing logic as everywhere else....
      // /// see generate_data_block_var_inits!!!!
      // generate_write_array_xform_param_vars(prog.parameter_decl_)
      //   //      write_array_vars_visgen vis_writer(2, o);
      //   //      for (size_t i = 0; i < prog.parameter_decl_.size(); ++i)
      //   //        boost::apply_visitor(vis_writer, prog.parameter_decl_[i].decl_);
      //   //      o << EOL;



namespace stan {
  namespace lang {

    /**
     * Generate write array vars stmt.
     *
     * Output order is last-index major, (i.e., dump data format),
     * which corresponds to column-major order for matrices
     * represented as two-dimensional arrays.  As a result, the first
     * indices change fastest.  Therefore loops must be constructed:
     * (col) (row) (array-dim-N) ... (array-dim-1)
     *
     * @param[in] vs data block variable declarations
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void generate_write_array_vars(const std::vector<block_var_decl>& vs,
                                   int indent, std::ostream& o) {
      std::cout << "generate_write_array_vars" << std::endl;

      for (size_t i = 0; i < vs.size(); ++i) {
        std::cout << "var: " << vs[i].name() << std::endl;

        // generate_indent(indent, o);
        // o << "current_statement_begin__ = " <<  vs[i].begin_line_ << ";"
        //   << EOL;

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
        std::vector<expression> limits;
        int start_ar_idx = (!is_nil(arg2)) ? 2 : ((!is_nil(arg1)) ? 1 : 0);
        if (start_ar_idx == 2) {
          idx_names.push_back("j_2__");
          limits.push_back(arg2);
        }
        if (start_ar_idx > 0) {
          idx_names.push_back("j_1__");
          limits.push_back(arg1);
        }
        for (size_t k = ar_lens.size(); k > 0; --k) {
          std::stringstream ss_name;
          ss_name << "k_" << k - 1 << "__";
          idx_names.push_back(ss_name.str());
          limits.push_back(ar_lens[k - 1]);
        }

        // nested for stmts open
        for (size_t k = 0; k < limits.size(); ++k) {
          generate_indent(indent + k, o);
          o << "for (size_t " << idx_names[k] << " = 0; "
            << idx_names[k] << " < ";
          generate_expression(limits[k], NOT_USER_FACING, o);
          o << "; ++" << idx_names[k] << ") {" << EOL;
        }               
        // innermost assign - update pos__
        generate_indent(indent + limits.size(), o);
        o << "vars__.push_back(" << var_name;
        // array idxs
        for (size_t k = start_ar_idx; k < limits.size(); ++k)
          o << "[" << idx_names[k] << "]";
        if (start_ar_idx == 2) {
          o << "(" << idx_names[1] << ", " << idx_names[0] << ")";
        } else if (start_ar_idx == 1) {
          o << "(" << idx_names[0] << ")";
        }
        o << ");" << EOL;

        // nested close
        for (size_t k = limits.size(); k > 0; --k) {
          generate_indent(indent + k - 1, o);
          o << "}" << EOL;
        }        
        o << EOL;
      }
      o << EOL;
    }

  }
}
#endif

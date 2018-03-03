#ifndef STAN_LANG_GENERATOR_GENERATE_VALIDATE_VAR_DECL_HPP
#define STAN_LANG_GENERATOR_GENERATE_VALIDATE_VAR_DECL_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_expression.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Generate the openings of a sequence of zero or more for loops
     * corresponding to the specified dimension sizes with the
     * specified indentation level writing to the specified stream.
     *
     * @param[in] dims dimension sizes
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void write_dims_for_loop_begin(const std::vector<expression>& dims,
                                   int indent, std::ostream& o) {
      for (size_t i = 0; i < dims.size(); ++i) {
        generate_indent(indent+i, o);
        o << "for (int k"  << i << "__ = 0;" << " k" << i << "__ < ";
        generate_expression(dims[i].expr_, NOT_USER_FACING, o);
        o << ";" << " ++k" << i << "__) {" << EOL;
      }
    }

    /**
     * Generate the close `}` for a sequence of zero or more for loops
     * with the specified indentation level writing to the specified stream.
     *
     * @param[in] dims_size dimension sizes
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void write_dims_for_loop_end(size_t dims_size,
                                 int indent, std::ostream& o) {
      for (size_t i = 0; i < dims_size; ++i) {
        generate_indent(indent + dims_size - i - 1, o);
        o << "}" << EOL;
      }
    }
    
    /**
     * Generate the loop variable and indexes for the specified
     * variable name and number of dimensions with the
     * specified indentation level writing to the specified stream.
     *
     * @param[in] name name of variable
     * @param[in] dims_size number of dimensions of variable
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void write_dims_for_loop_var(const std::string& name, size_t dims_size,
                                 int indent, std::ostream& o) {
      o << name;
      for (size_t i = 0; i < dims_size; ++i)
        o << "[k" << i << "__]";
    }

    /**
     * Generate code to validate the specified variable declaration
     * using the specified indentation level and stream.
     * Checks any defined bounds or constraints on specialized types.
     * NOTE:  bounded / specialized types are mutually exclusive
     *
     * @param[in] decl variable declaration
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void generate_validate_var_decl(const block_var_decl decl,
                                    int indent, std::ostream& o) {

      std::string var_name(decl.name());
      std::vector<expression> ar_lens(decl.type().array_lens());
      // unfold array type to get array element info
      block_var_type btype = (decl.type());
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
#endif

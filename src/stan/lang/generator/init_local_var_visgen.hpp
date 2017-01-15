#ifndef STAN_LANG_GENERATOR_INIT_LOCAL_VAR_VISGEN_HPP
#define STAN_LANG_GENERATOR_INIT_LOCAL_VAR_VISGEN_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/generate_void_statement.hpp>
#include <stan/lang/generator/has_lb.hpp>
#include <stan/lang/generator/has_lub.hpp>
#include <stan/lang/generator/has_ub.hpp>
#include <stan/lang/generator/to_string.hpp>
#include <stan/lang/generator/visgen.hpp>
#include <ostream>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    void generate_expression(const expression& e, std::ostream& o);

    /**
     * Visitor to initialize local variables.
     */
    struct init_local_var_visgen : public visgen {
      /**
       * Flag indicating if variables need to be declared.
       */
      const bool declare_vars_;

      /**
       * Flag indicating if generation is in a variable context.
       */
      const bool is_var_context_;

      /**
       * Construct a visitor for initializing local variables with
       * flags indicating whether variables need to be declared and if
       * generation is in a variable context, writing to the specified
       * stream.
       *
       * @param[in] declare_vars true if variables need to be declared
       * @param[in] is_var_context true if generation is in a
       * variable context
       * @param[in,out] o stream for generating
       */
      explicit init_local_var_visgen(bool declare_vars, bool is_var_context,
                                     std::ostream& o)
        : visgen(o), declare_vars_(declare_vars),
          is_var_context_(is_var_context) {  }

      void generate_initialize_array(const std::string& var_type,
                                 const std::string& read_type,
                                 const std::vector<expression>& read_args,
                                 const std::string& name,
                                 const std::vector<expression>& dims) const {
        if (declare_vars_) {
          o_ << INDENT2;
          for (size_t i = 0; i < dims.size(); ++i) o_ << "vector<";
          o_ << var_type;
          for (size_t i = 0; i < dims.size(); ++i) o_ << "> ";
          if (dims.size() == 0) o_ << " ";
          o_ << name << ";" << EOL;
        }

        if (dims.size() == 0) {
          generate_void_statement(name, 2, o_);
          o_ << INDENT2 << "if (jacobian__)" << EOL;

          // w Jacobian
          generate_indent(3, o_);
          o_ << name << " = in__." << read_type  << "_constrain(";
          for (size_t j = 0; j < read_args.size(); ++j) {
            if (j > 0) o_ << ",";
            generate_expression(read_args[j], o_);
          }
          if (read_args.size() > 0)
            o_ << ",";
          o_ << "lp__";
          o_ << ");" << EOL;

          o_ << INDENT2 << "else" << EOL;

          // w/o Jacobian
          generate_indent(3, o_);
          o_ << name << " = in__." << read_type  << "_constrain(";
          for (size_t j = 0; j < read_args.size(); ++j) {
            if (j > 0) o_ << ",";
            generate_expression(read_args[j], o_);
          }
          o_ << ");" << EOL;

        } else {
          // dims > 0
          std::string name_dims(name);
          for (size_t i = 0; i < dims.size(); ++i) {
            generate_indent(i + 2, o_);
            o_ << "size_t dim_"  << name << "_" << i << "__ = ";
            generate_expression(dims[i], o_);
            o_ << ";" << EOL;

            if (i < dims.size() - 1) {
              generate_indent(i + 2, o_);
              o_ << name_dims << ".resize(dim" << "_"
                 << name << "_" << i << "__);"
                 << EOL;
              name_dims.append("[k_").append(to_string(i)).append("__]");
            }

            generate_indent(i + 2, o_);
            if (i == dims.size() - 1) {
              o_ << name_dims << ".reserve(dim_" << name
                 << "_" << i << "__);" << EOL;
              generate_indent(i + 2, o_);
            }

            o_ << "for (size_t k_" << i << "__ = 0;"
               << " k_" << i << "__ < dim_" << name << "_" << i << "__;"
               << " ++k_" << i << "__) {" << EOL;

            // if on the last loop, push read element into array
            if (i == dims.size() - 1) {
              generate_indent(i + 3, o_);
              o_ << "if (jacobian__)" << EOL;

              // w Jacobian
              generate_indent(i + 4, o_);
              o_ << name_dims << ".push_back(in__."
                 << read_type << "_constrain(";
              for (size_t j = 0; j < read_args.size(); ++j) {
                if (j > 0) o_ << ",";
                generate_expression(read_args[j], o_);
              }
              if (read_args.size() > 0)
                o_ << ",";
              o_ << "lp__";
              o_ << "));" << EOL;

              generate_indent(i + 3, o_);
              o_ << "else" << EOL;

              // w/o Jacobian
              generate_indent(i + 4, o_);
              o_ << name_dims << ".push_back(in__."
                 << read_type << "_constrain(";
              for (size_t j = 0; j < read_args.size(); ++j) {
                if (j > 0) o_ << ",";
                generate_expression(read_args[j], o_);
              }
              o_ << "));" << EOL;
            }
          }

          for (size_t i = dims.size(); i > 0; --i) {
            generate_indent(i + 1, o_);
            o_ << "}" << EOL;
          }
        }
        o_ << EOL;
      }

      template <typename D>
      void generate_initialize_array_bounded(const D& x,
                            const std::string& base_type,
                            const std::string& read_fun_prefix,
                            const std::vector<expression>& dim_args) const {
        std::vector<expression> read_args;
        std::string read_fun(read_fun_prefix);
        if (has_lub(x)) {
          read_fun += "_lub";
          read_args.push_back(x.range_.low_);
          read_args.push_back(x.range_.high_);
        } else if (has_lb(x)) {
          read_fun += "_lb";
          read_args.push_back(x.range_.low_);
        } else if (has_ub(x)) {
          read_fun += "_ub";
          read_args.push_back(x.range_.high_);
        }
        for (size_t i = 0; i < dim_args.size(); ++i)
          read_args.push_back(dim_args[i]);
        generate_initialize_array(base_type, read_fun, read_args,
                                    x.name_, x.dims_);
      }

      void operator()(const nil& /*x*/) const { }

      void operator()(const int_var_decl& x) const {
        generate_initialize_array("int", "integer", EMPTY_EXP_VECTOR,
                                  x.name_, x.dims_);
      }

      void operator()(const double_var_decl& x) const {
        std::vector<expression> read_args;
        generate_initialize_array_bounded(x,
                                          is_var_context_ ? "T__" : "double",
                                          "scalar", read_args);
      }

      void operator()(const vector_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.M_);
        generate_initialize_array_bounded(x,
                                  is_var_context_
                                  ? "Eigen::Matrix<T__,Eigen::Dynamic,1> "
                                  : "vector_d",
                                  "vector", read_args);
      }

      void operator()(const row_vector_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.N_);
        generate_initialize_array_bounded(x,
                                      is_var_context_
                                      ? "Eigen::Matrix<T__,1,Eigen::Dynamic> "
                                      : "row_vector_d",
                                      "row_vector", read_args);
      }

      void operator()(const matrix_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.M_);
        read_args.push_back(x.N_);
        generate_initialize_array_bounded(x,
                          is_var_context_
                          ? "Eigen::Matrix<T__,Eigen::Dynamic,Eigen::Dynamic> "
                          : "matrix_d",
                          "matrix", read_args);
      }

      void operator()(const unit_vector_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array(is_var_context_
                                  ? "Eigen::Matrix<T__,Eigen::Dynamic,1> "
                                  : "vector_d",
                                  "unit_vector", read_args, x.name_, x.dims_);
      }

      void operator()(const simplex_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array(is_var_context_
                                  ? "Eigen::Matrix<T__,Eigen::Dynamic,1> "
                                  : "vector_d",
                                  "simplex", read_args, x.name_, x.dims_);
      }

      void operator()(const ordered_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array(is_var_context_
                                  ? "Eigen::Matrix<T__,Eigen::Dynamic,1> "
                                  : "vector_d",
                                  "ordered", read_args, x.name_, x.dims_);
      }

      void operator()(const positive_ordered_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array(is_var_context_
                                  ? "Eigen::Matrix<T__,Eigen::Dynamic,1> "
                                  : "vector_d",
                                  "positive_ordered",
                                  read_args, x.name_, x.dims_);
      }

      void operator()(const cholesky_factor_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.M_);
        read_args.push_back(x.N_);
        generate_initialize_array(is_var_context_
                                  ? "Eigen::Matrix"
                                    "<T__,Eigen::Dynamic,Eigen::Dynamic> "
                                  : "matrix_d",
                                    "cholesky_factor",
                                  read_args, x.name_, x.dims_);
      }

      void operator()(const cholesky_corr_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array(is_var_context_
                          ? "Eigen::Matrix<T__,Eigen::Dynamic,Eigen::Dynamic> "
                          : "matrix_d",
                          "cholesky_corr", read_args, x.name_, x.dims_);
      }

      void operator()(const cov_matrix_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array(is_var_context_
                          ? "Eigen::Matrix<T__,Eigen::Dynamic,Eigen::Dynamic> "
                          : "matrix_d",
                          "cov_matrix", read_args, x.name_, x.dims_);
      }

      void operator()(const corr_matrix_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array(is_var_context_
                    ? "Eigen::Matrix<T__,Eigen::Dynamic,Eigen::Dynamic> "
                    : "matrix_d",
                    "corr_matrix", read_args, x.name_, x.dims_);
      }
    };

  }
}
#endif

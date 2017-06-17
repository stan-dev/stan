#ifndef STAN_LANG_GENERATOR_LOCAL_VAR_DECL_VISGEN_HPP
#define STAN_LANG_GENERATOR_LOCAL_VAR_DECL_VISGEN_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/generate_eigen_index_expression.hpp>
#include <stan/lang/generator/generate_validate_positive.hpp>
#include <stan/lang/generator/generate_void_statement.hpp>
#include <stan/lang/generator/visgen.hpp>
#include <ostream>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    void generate_expression(const expression& e, std::ostream& o);

    /*
     * Visitor for generating local variable declarations.
     */
    struct local_var_decl_visgen : public visgen {
      /**
       * true if generation is in a variable context.
       */
      bool is_var_context_;

      /**
       * true if generating a variable with the type of a function return.
       */
      bool is_fun_return_;

      /**
       * Construct a visitor to generate local variable declarations
       * with the specified indentation level and flags indicating if
       * the local variable is declared within a variable context or a
       * function with return type, and writing to the specified
       * stream.
       *
       * @param[in] indent indentation level
       * @param[in] is_var_context true if in variable context
       * @param[in] is_fun_return true if type should be function
       * return type
       * @param[in,out] o stream for generating
       */
      local_var_decl_visgen(int indent, bool is_var_context,
                            bool is_fun_return, std::ostream& o)
        : visgen(indent, o), is_var_context_(is_var_context),
          is_fun_return_(is_fun_return) {  }

      /**
       * Generate the type for the specified basic type and number of
       * array dimensions.
       *
       * @param[in] type generated basic type
       * @param[in] num_dims number of array dimensions
       */
      void generate_type(const std::string& type, size_t num_dims) const {
        for (size_t i = 0; i < num_dims; ++i)
          o_ << "vector<";
        o_ << type;
        for (size_t i = 0; i < num_dims; ++i) {
          if (i > 0) o_ << " ";
          o_ << ">";
        }
      }

      /**
       * Generate the initialize arguments for the specified type,
       * arguments for the constructor, and dimension sizes and number
       * of dimensions.
       *
       * @param[in] type basic type (int, var, or double)
       * @param[in] ctor_args arguments for basic type constructor
       * @param[in] dims dimension sizes
       * @param[in] dim number of dimensions
       */
      void generate_init_args(const std::string& type,
                              const std::vector<expression>& ctor_args,
                              const std::vector<expression>& dims,
                              size_t dim) const {
        if (dim < dims.size()) {  // more dims left
          o_ << '(';  // open(1)
          generate_expression(dims[dim], o_);
          if ((dim + 1 < dims.size()) ||  ctor_args.size() > 0) {
            o_ << ", (";  // open(2)
            generate_type(type, dims.size() - dim - 1);
            generate_init_args(type, ctor_args, dims, dim + 1);
            o_ << ')';  // close(2)
          } else if (type == "var") {
            o_ << ", DUMMY_VAR__";
          } else if (type == "int") {
            o_ << ", 0";
          } else if (type == "double") {
            o_ << ", 0.0";
          } else {
            // shouldn't hit this
          }
          o_ << ')';  // close(1)
        } else {
          if (ctor_args.size() == 0) {  // scalar int or real
            if (type == "int") {
              o_ << "(0)";
            } else if (type == "double") {
              o_ << "(0.0)";
            } else if (type == "var") {
              o_ << "(DUMMY_VAR__)";
            } else {
              // shouldn't hit this, either
            }
          } else if (ctor_args.size() == 1) {  // vector
            o_ << '(';
            generate_eigen_index_expression(ctor_args[0], o_);
            o_ << ')';
          } else if (ctor_args.size() > 1) {  // matrix
            o_ << '(';
            generate_eigen_index_expression(ctor_args[0], o_);
            o_ << ',';
            generate_eigen_index_expression(ctor_args[1], o_);
            o_ << ')';
          }
        }
      }

      /**
       * Generate an array declaration for the specified type,
       * arguments fo the basic type constructor, variable name,
       * dimension sizes, and optional definition.
       *
       * @param[in] type generated variable type
       * @param[in] ctor_args argument for basic type constructor
       * @param[in] name variable name
       * @param[in] dims dimension sizes
       * @param[in] definition value for initializing variable (optional)
       */
      void declare_array(const std::string& type,
                         const std::vector<expression>& ctor_args,
                         const std::string& name,
                         const std::vector<expression>& dims,
                         const expression& definition = expression()) const {
        // check array dimensions
        for (size_t i = 0; i < dims.size(); ++i)
          generate_validate_positive(name, dims[i], indent_, o_);
        // require double parens to counter "most vexing parse" problem
        generate_indent(indent_, o_);
        generate_type(type, dims.size());
        o_ << ' '  << name;
        generate_init_args(type, ctor_args, dims, 0);
        o_ << ";" << EOL;
        if (dims.size() == 0) {
          generate_void_statement(name, indent_, o_);
          o_ << EOL;
        }
        if (type == "Eigen::Matrix<T__, Eigen::Dynamic, Eigen::Dynamic> "
            || type == "Eigen::Matrix<T__, 1, Eigen::Dynamic> "
            || type == "Eigen::Matrix<T__, Eigen::Dynamic, 1> ") {
          generate_indent(indent_, o_);
          o_ << "stan::math::fill(" << name << ", DUMMY_VAR__);" << EOL;
        }
      }

      void operator()(const nil& /*x*/) const { }

      void operator()(const int_var_decl& x) const {
        std::vector<expression> ctor_args;
        declare_array("int", ctor_args, x.name_, x.dims_);
      }

      void operator()(const double_var_decl& x) const {
        std::vector<expression> ctor_args;
        declare_array(is_fun_return_
                      ? "fun_scalar_t__"
                      : (is_var_context_ ? "T__" : "double"),
                      ctor_args, x.name_, x.dims_);
      }

      void operator()(const vector_var_decl& x) const {
        std::vector<expression> ctor_args;
        generate_validate_positive(x.name_, x.M_, indent_, o_);
        ctor_args.push_back(x.M_);
        declare_array(is_fun_return_
                      ? "Eigen::Matrix<fun_scalar_t__,Eigen::Dynamic,1> "
                      : (is_var_context_
                         ? "Eigen::Matrix<T__,Eigen::Dynamic,1> " : "vector_d"),
                      ctor_args, x.name_, x.dims_);
      }

      void operator()(const row_vector_var_decl& x) const {
        std::vector<expression> ctor_args;
        generate_validate_positive(x.name_, x.N_, indent_, o_);
        ctor_args.push_back(x.N_);
        declare_array(is_fun_return_
                      ? "Eigen::Matrix<fun_scalar_t__,1,Eigen::Dynamic> "
                      : (is_var_context_
                         ? "Eigen::Matrix<T__,1,Eigen::Dynamic> "
                         : "row_vector_d"),
                      ctor_args, x.name_, x.dims_);
      }

      void operator()(const matrix_var_decl& x) const {
        std::vector<expression> ctor_args;
        generate_validate_positive(x.name_, x.M_, indent_, o_);
        generate_validate_positive(x.name_, x.N_, indent_, o_);
        ctor_args.push_back(x.M_);
        ctor_args.push_back(x.N_);
        declare_array(is_fun_return_
                      ? "Eigen::Matrix<fun_scalar_t__,"
                      "Eigen::Dynamic,Eigen::Dynamic> "
                      : (is_var_context_
                         ? "Eigen::Matrix<T__,Eigen::Dynamic,Eigen::Dynamic> "
                         : "matrix_d"),
                      ctor_args, x.name_, x.dims_);
      }

      void operator()(const unit_vector_var_decl& x) const {
        std::vector<expression> ctor_args;
        generate_validate_positive(x.name_, x.K_, indent_, o_);
        ctor_args.push_back(x.K_);
        declare_array(is_fun_return_
                      ? "Eigen::Matrix<fun_scalar_t__,Eigen::Dynamic,1> "
                      : (is_var_context_
                         ? "Eigen::Matrix<T__,Eigen::Dynamic,1> " : "vector_d"),
                      ctor_args, x.name_, x.dims_);
      }

      void operator()(const simplex_var_decl& x) const {
        std::vector<expression> ctor_args;
        generate_validate_positive(x.name_, x.K_, indent_, o_);
        ctor_args.push_back(x.K_);
        declare_array(is_fun_return_
                      ? "Eigen::Matrix<fun_scalar_t__,Eigen::Dynamic,1> "
                      : (is_var_context_
                         ? "Eigen::Matrix<T__,Eigen::Dynamic,1> " : "vector_d"),
                      ctor_args, x.name_, x.dims_);
      }

      void operator()(const ordered_var_decl& x) const {
        std::vector<expression> ctor_args;
        generate_validate_positive(x.name_, x.K_, indent_, o_);
        ctor_args.push_back(x.K_);
        declare_array(is_fun_return_
                      ? "Eigen::Matrix<fun_scalar_t__,Eigen::Dynamic,1> "
                      : (is_var_context_
                         ? "Eigen::Matrix<T__,Eigen::Dynamic,1> " : "vector_d"),
                      ctor_args, x.name_, x.dims_);
      }

      void operator()(const positive_ordered_var_decl& x) const {
        std::vector<expression> ctor_args;
        generate_validate_positive(x.name_, x.K_, indent_, o_);
        ctor_args.push_back(x.K_);
        declare_array(is_fun_return_
                      ? "Eigen::Matrix<fun_scalar_t__,Eigen::Dynamic,1> "
                      : (is_var_context_
                         ? "Eigen::Matrix<T__,Eigen::Dynamic,1> " : "vector_d"),
                      ctor_args, x.name_, x.dims_);
      }

      void operator()(const cholesky_factor_var_decl& x) const {
        std::vector<expression> ctor_args;
        generate_validate_positive(x.name_, x.M_, indent_, o_);
        generate_validate_positive(x.name_, x.N_, indent_, o_);
        ctor_args.push_back(x.M_);
        ctor_args.push_back(x.N_);
        declare_array(is_fun_return_
                      ? "Eigen::Matrix<fun_scalar_t__,"
                      "Eigen::Dynamic,Eigen::Dynamic> "
                      : (is_var_context_
                         ? "Eigen::Matrix<T__,Eigen::Dynamic,Eigen::Dynamic> "
                         : "matrix_d"),
                      ctor_args, x.name_, x.dims_);
      }

      void operator()(const cholesky_corr_var_decl& x) const {
        std::vector<expression> ctor_args;
        generate_validate_positive(x.name_, x.K_, indent_, o_);
        ctor_args.push_back(x.K_);
        ctor_args.push_back(x.K_);
        declare_array(is_var_context_
                      ? "Eigen::Matrix<T__,Eigen::Dynamic,Eigen::Dynamic> "
                      : "matrix_d",
                      ctor_args, x.name_, x.dims_);
      }

      void operator()(const cov_matrix_var_decl& x) const {
        std::vector<expression> ctor_args;
        generate_validate_positive(x.name_, x.K_, indent_, o_);
        ctor_args.push_back(x.K_);
        ctor_args.push_back(x.K_);
        declare_array(is_fun_return_
                      ? "Eigen::Matrix<fun_scalar_t__,"
                      "Eigen::Dynamic,Eigen::Dynamic> "
                      : (is_var_context_
                         ? "Eigen::Matrix<T__,Eigen::Dynamic,Eigen::Dynamic> "
                         : "matrix_d"),
                      ctor_args, x.name_, x.dims_);
      }

      void operator()(const corr_matrix_var_decl& x) const {
        std::vector<expression> ctor_args;
        generate_validate_positive(x.name_, x.K_, indent_, o_);
        ctor_args.push_back(x.K_);
        ctor_args.push_back(x.K_);
        declare_array(is_fun_return_
                      ? "Eigen::Matrix<fun_scalar_t__,"
                      "Eigen::Dynamic,Eigen::Dynamic> "
                      : (is_var_context_
                         ? "Eigen::Matrix<T__,Eigen::Dynamic,Eigen::Dynamic> "
                         : "matrix_d"),
                      ctor_args, x.name_, x.dims_);
      }
    };

  }
}
#endif

#ifndef __STAN__GM__GENERATOR_HPP__
#define __STAN__GM__GENERATOR_HPP__

#include <boost/variant/apply_visitor.hpp>

#include <cstddef>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <stan/maths/matrix.hpp>

#include <stan/version.hpp>
#include <stan/gm/ast.hpp>

namespace stan {

  namespace gm {

    const std::string EOL("\n");
    const std::string EOL2("\n\n");
    const std::string INDENT("    ");
    const std::string INDENT2("        ");
    const std::string INDENT3("            ");

    template <typename T>
    std::string to_string(T i) {
      std::stringstream ss;
      ss << i;
      return ss.str();
    }

    void generate_indent(size_t indent, std::ostream& o) {
      for (size_t k = 0; k < indent; ++k)
        o << INDENT;
    }

    /** generic visitor with output for extension */
    struct visgen {
      typedef void result_type;
      std::ostream& o_;
      visgen(std::ostream& o) : o_(o) { }
    };


    void generate_indexed_expr(const std::string& expr,
                               const std::vector<expression> indexes, 
                               base_expr_type base_type, // may have more dims
                               size_t e_num_dims, // array dims
                               std::ostream& o) {
      // FIXME: add more get_base1 functions and fold nested calls into API
      // up to a given size, then default to this behavior
      size_t ai_size = indexes.size();
      if (ai_size == 0) {
        // no indexes
        o << expr;
        return;
      }
      if (ai_size <= (e_num_dims + 1) || base_type != MATRIX_T) {
        for (size_t n = 0; n < ai_size; ++n)
          o << "get_base1(";
        o << expr;
        for (size_t n = 0; n < ai_size; ++n) {
          o << ',';
          generate_expression(indexes[n],o);
          o << ',' << '"' << expr << '"' << ',' << (n+1) << ')';
        }
      } else { 
        for (size_t n = 0; n < ai_size - 1; ++n)
          o << "get_base1(";
        o << expr;
        for (size_t n = 0; n < ai_size - 2; ++n) {
          o << ',';
          generate_expression(indexes[n],o);
          o << ',' << '"' << expr << '"' << ',' << (n+1) << ')';
        }
        o << ',';
        generate_expression(indexes[ai_size - 2U],o);
        o << ',';
        generate_expression(indexes[ai_size - 1U],o);
        o << ',' << '"' << expr << '"' << ',' << (ai_size-1U) << ')';
      }
    }

    struct expression_visgen : public visgen {
      expression_visgen(std::ostream& o) : visgen(o) {  }
      void operator()(nil const& x) const { 
        o_ << "nil";
      }
      void operator()(const int_literal& n) const { o_ << n.val_; }
      void operator()(const double_literal& x) const { o_ << x.val_; }
      void operator()(const variable& v) const { o_ << v.name_; }
      void operator()(int n) const { o_ << n; }
      void operator()(double x) const { o_ << x; }
      void operator()(const std::string& x) const { o_ << x; } // identifiers
      void operator()(const index_op& x) const {
        std::stringstream expr_o;
        generate_expression(x.expr_,expr_o);
        std::string expr_string = expr_o.str();
        std::vector<expression> indexes; 
        size_t e_num_dims = x.expr_.expression_type().num_dims_;
        base_expr_type base_type = x.expr_.expression_type().base_type_;
        for (size_t i = 0; i < x.dimss_.size(); ++i)
          for (size_t j = 0; j < x.dimss_[i].size(); ++j) 
            indexes.push_back(x.dimss_[i][j]); // wasteful copy, could use refs
        generate_indexed_expr(expr_string,indexes,base_type,e_num_dims,o_);
      }
      void operator()(const fun& fx) const { 
        o_ << fx.name_ << '(';
        for (size_t i = 0; i < fx.args_.size(); ++i) {
          if (i > 0) o_ << ',';
          boost::apply_visitor(*this, fx.args_[i].expr_);
        }
        o_ << ')';
      }
      void operator()(const binary_op& expr) const {
        o_ << '('; 
        boost::apply_visitor(*this, expr.left.expr_);
        o_ << ' ' << expr.op << ' ';
        boost::apply_visitor(*this, expr.right.expr_);
        o_ << ')';
      }
      void operator()(const unary_op& expr) const {
        o_ << expr.op << '(';
        boost::apply_visitor(*this, expr.subject.expr_);
        o_ << ')';
      }
    };

    void generate_expression(const expression& e, std::ostream& o) {
      expression_visgen vis(o);
      boost::apply_visitor(vis, e.expr_);
    }

    void generate_using(const std::string& type, std::ostream& o) {
      o << "using " << type << ";" << EOL;
    }

    void generate_using_namespace(const std::string& ns, std::ostream& o) {
      o << "using namespace " << ns << ";" << EOL;
    }


    void generate_usings(std::ostream& o) {
      generate_using("std::vector",o);
      generate_using("std::string",o);
      generate_using("std::stringstream",o);
      generate_using("stan::agrad::var",o);
      generate_using("stan::mcmc::prob_grad_ad",o);
      generate_using("stan::maths::get_base1",o);
      generate_using("stan::io::dump",o);
      generate_using("std::istream",o);
      generate_using_namespace("stan::maths",o);
      generate_using_namespace("stan::prob",o);
      o << EOL;
    }

    void generate_typedef(const std::string& type, 
                          const std::string& abbrev, 
                          std::ostream& o) {
      o << "typedef" << " " << type << " " << abbrev << ";" << EOL;
    }

    void generate_typedefs(std::ostream& o) {
      generate_typedef("Eigen::Matrix<double,Eigen::Dynamic,1>","vector_d",o);
      generate_typedef("Eigen::Matrix<double,1,Eigen::Dynamic>","row_vector_d",o);
      generate_typedef("Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>","matrix_d",o);

      generate_typedef("Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,1>","vector_v",o);
      generate_typedef("Eigen::Matrix<stan::agrad::var,1,Eigen::Dynamic>","row_vector_v",o);
      generate_typedef("Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,Eigen::Dynamic>","matrix_v",o);
      // moved to include
      o << EOL;
    }

    void generate_include(const std::string& lib_name, std::ostream& o) {
      o << "#include" << " " << "<" << lib_name << ">" << EOL;
    }
   
    void generate_includes(std::ostream& o) {
      generate_include("cassert",o);
      generate_include("cmath",o);
      generate_include("cstddef",o);
      generate_include("vector",o);
      generate_include("fstream",o);
      generate_include("iostream",o);
      generate_include("stdexcept",o);
      generate_include("sstream",o);
      generate_include("utility",o);
      generate_include("boost/exception/all.hpp",o);
      generate_include("stan/agrad/agrad.hpp",o);
      generate_include("stan/agrad/special_functions.hpp",o);
      generate_include("stan/agrad/matrix.hpp",o);
      generate_include("stan/gm/command.hpp",o);
      generate_include("stan/io/cmd_line.hpp",o);
      generate_include("stan/io/dump.hpp",o);
      generate_include("stan/io/reader.hpp",o);
      generate_include("stan/io/writer.hpp",o);
      generate_include("stan/io/csv_writer.hpp",o);
      generate_include("stan/maths/matrix.hpp",o);
      generate_include("stan/maths/special_functions.hpp",o);
      generate_include("stan/mcmc/hmc.hpp",o);
      generate_include("stan/mcmc/sampler.hpp",o);
      generate_include("stan/mcmc/prob_grad_ad.hpp",o);
      generate_include("stan/prob/distributions.hpp",o);
      o << EOL;
    }

    void generate_version_comment(std::ostream& o) {
      o << "// Code generated by Stan version "
        << stan::MAJOR_VERSION  << "." << stan::MINOR_VERSION << EOL2;
    }

    void generate_class_decl(const std::string& model_name,
                        std::ostream& o) {
      o << "class " << model_name << " : public prob_grad_ad {" << EOL;
    }

    void generate_end_class_decl(std::ostream& o) {
      o << "}; // model" << EOL2;
    }

    void generate_type(const std::string& base_type,
                       const std::vector<expression>& dims,
                       size_t end,
                       std::ostream& o) {
      for (size_t i = 0; i < end; ++i) o << "std::vector<";
      o << base_type;
      for (size_t i = 0; i < end; ++i) {
        if (i > 0) o << ' ';
        o << '>';
      } 
    }

    void generate_initializer(std::ostream& o,
                              const std::string& base_type,
                              const std::vector<expression>& dims,
                              const expression& type_arg1 = expression(),
                              const expression& type_arg2 = expression()) {
      for (size_t i = 0; i < dims.size(); ++i) {
        o << '(';
        generate_expression(dims[i].expr_,o);
        o << ',';
        generate_type(base_type,dims,dims.size()- i - 1,o);
      }

      o << '(';
      if (!is_nil(type_arg1)) {
        generate_expression(type_arg1.expr_,o);
        if (!is_nil(type_arg2)) {
          o << ',';
          generate_expression(type_arg2.expr_,o);
        }
      } else if (!is_nil(type_arg2.expr_)) {
        generate_expression(type_arg2.expr_,o);
      } else {
        o << '0';
      }
      o << ')';

      for (size_t i = 0; i < dims.size(); ++i)
        o << ')';
      o << ';' << EOL;
    }

    void generate_initialization(std::ostream& o,
                                 size_t indent,
                                 const std::string& var_name,
                                 const std::string& base_type,
                                 const std::vector<expression>& dims,
                                 const expression& type_arg1 = expression(),
                                 const expression& type_arg2 = expression()) {
      generate_indent(indent,o);
      o << var_name << " = ";
      generate_type(base_type,dims,dims.size(),o);
      generate_initializer(o,base_type,dims,type_arg1,type_arg2);
    }


    struct var_resizing_visgen : public visgen {
      var_resizing_visgen(std::ostream& o) 
        : visgen(o) {
      }
      void operator()(nil const& x) const { } // dummy
      void operator()(int_var_decl const& x) const {
        generate_initialization(o_,2U,x.name_,"int",x.dims_);
      }
      void operator()(double_var_decl const& x) const {
        generate_initialization(o_,2U,x.name_,"double",x.dims_);
      }
      void operator()(vector_var_decl const& x) const {
        generate_initialization(o_,2U,x.name_,"vector_d",x.dims_,x.M_);
      }
      void operator()(row_vector_var_decl const& x) const {
        generate_initialization(o_,2U,x.name_,"row_vector_d",x.dims_,x.N_);
      }
      void operator()(simplex_var_decl const& x) const {
        generate_initialization(o_,2U,x.name_,"vector_d",x.dims_,x.K_);
      }
      void operator()(pos_ordered_var_decl const& x) const {
        generate_initialization(o_,2U,x.name_,"vector_d",x.dims_,x.K_);
      }
      void operator()(matrix_var_decl const& x) const {
        generate_initialization(o_,2U,x.name_,"matrix_d",x.dims_,x.M_,x.N_);
      }
      void operator()(cov_matrix_var_decl const& x) const {
        generate_initialization(o_,2U,x.name_,"matrix_d",x.dims_,x.K_,x.K_);
      }
      void operator()(corr_matrix_var_decl const& x) const {
        generate_initialization(o_,2U,x.name_,"matrix_d",x.dims_,x.K_,x.K_);
      }
    };

    void generate_var_resizing(const std::vector<var_decl>& vs,
                               std::ostream& o) {
      var_resizing_visgen vis(o);
      for (size_t i = 0; i < vs.size(); ++i)
        boost::apply_visitor(vis, vs[i].decl_);
    }

    const std::vector<expression> EMPTY_EXP_VECTOR(0);

    struct init_local_var_visgen : public visgen {
      const bool declare_vars_;
      init_local_var_visgen(bool declare_vars,
                             std::ostream& o)
        : visgen(o),
          declare_vars_(declare_vars) {
      }
      void operator()(const nil& x) const { }
      void operator()(const int_var_decl& x) const {
        generate_initialize_array("int","integer",EMPTY_EXP_VECTOR,x.name_,x.dims_);
      }      
      void operator()(const double_var_decl& x) const {
        if (!is_nil(x.range_.low_.expr_)) {
          if (!is_nil(x.range_.high_.expr_)) {
            std::vector<expression> read_args;
            read_args.push_back(x.range_.low_);
            read_args.push_back(x.range_.high_);
            generate_initialize_array("var","scalar_lub",read_args,x.name_,x.dims_);
          } else {
            std::vector<expression> read_args;
            read_args.push_back(x.range_.low_);
            generate_initialize_array("var","scalar_lb",read_args,x.name_,x.dims_);
          }
        } else {
          if (!is_nil(x.range_.high_.expr_)) {
            std::vector<expression> read_args;
            read_args.push_back(x.range_.high_);
            generate_initialize_array("var","scalar_ub",read_args,x.name_,x.dims_);
          } else {
            generate_initialize_array("var","scalar",EMPTY_EXP_VECTOR,x.name_,x.dims_);
          }
        }
      }
      void operator()(const vector_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.M_);
        generate_initialize_array("vector_v","vector",read_args,x.name_,x.dims_);
      }
      void operator()(const row_vector_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.N_);
        generate_initialize_array("row_vector_v","row_vector",read_args,x.name_,x.dims_);
      }
      void operator()(const matrix_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.M_);
        read_args.push_back(x.N_);
        generate_initialize_array("matrix_v","matrix",read_args,x.name_,x.dims_);
      }
      void operator()(const simplex_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array("vector_v","simplex",read_args,x.name_,x.dims_);
      }
      void operator()(const pos_ordered_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array("vector_v","pos_ordered",read_args,x.name_,x.dims_);
      }
      void operator()(const cov_matrix_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array("matrix_v","cov_matrix",read_args,x.name_,x.dims_);
      }
      void operator()(const corr_matrix_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array("matrix_v","corr_matrix",read_args,x.name_,x.dims_);
      }
      void generate_initialize_array(const std::string& var_type,
                                     const std::string& read_type,
                                     const std::vector<expression>& read_args,
                                     const std::string& name,
                                     const std::vector<expression>& dims) 
        const {

        if (dims.size() == 0) {
          generate_indent(2,o_);
          if (declare_vars_) o_ << var_type << " ";
          o_ << name << " = in__." << read_type  << "_constrain(";
          for (size_t j = 0; j < read_args.size(); ++j) {
            if (j > 0) o_ << ",";
            generate_expression(read_args[j],o_);
          }
          if (read_args.size() > 0)
            o_ << ",";
          o_ << "lp__";
          o_ << ");" << EOL;
          return;
        }
        if (declare_vars_) {
          o_ << INDENT2;
          for (size_t i = 0; i < dims.size(); ++i) o_ << "vector<";
          o_ << var_type;
          for (size_t i = 0; i < dims.size(); ++i) o_ << "> ";
          o_ << name << ";" << EOL;
        }
        std::string name_dims(name);
        for (size_t i = 0; i < dims.size(); ++i) {
          generate_indent(i + 2, o_);
          o_ << "size_t dim_"  << name << "_" << i << " = ";
          generate_expression(dims[i],o_);
          o_ << ";" << EOL;
          if (i < dims.size() - 1) {  
            generate_indent(i + 2, o_);
            o_ << name_dims << ".resize(dim" << "_" << name << "_" << i << ");" 
               << EOL;
            name_dims.append("[k_").append(to_string(i)).append("]");
          }
          generate_indent(i + 2, o_);
          o_ << "for (size_t k_" << i << " = 0;"
             << " k_" << i << " < dim_" << name << "_" << i << ";"
             << " ++k_" << i << ") {" << EOL;
          if (i == dims.size() - 1) {
            generate_indent(i + 3, o_);
            o_ << name_dims << ".push_back(in__." << read_type << "_constrain(";
            for (size_t j = 0; j < read_args.size(); ++j) {
              if (j > 0) o_ << ",";
              generate_expression(read_args[j],o_);
            }
            if (read_args.size() > 0)
              o_ << ",";
            o_ << "lp__";
            o_ << "));" << EOL;
          }
        }
        for (size_t i = dims.size(); i > 0; --i) {
          generate_indent(i + 1, o_);
          o_ << "}" << EOL;
        }
      }
    };

    void generate_local_var_inits(std::vector<var_decl> vs,
                                  bool declare_vars,
                                  std::ostream& o) {
      o << INDENT2 << "stan::io::reader<var> in__(params_r__,params_i__);" << EOL2;
      init_local_var_visgen vis(declare_vars,o);
      for (size_t i = 0; i < vs.size(); ++i)
        boost::apply_visitor(vis, vs[i].decl_);
    }




    void generate_public_decl(std::ostream& o) {
      o << "public:" << EOL;
    }

    void generate_private_decl(std::ostream& o) {
      o << "private:" << EOL;
    }
   

    struct validate_var_decl_visgen : public visgen {
      int indents_;
      validate_var_decl_visgen(int indents,
                               std::ostream& o)
        : visgen(o),
          indents_(indents) {
      }
      void generate_begin_for_dims(const std::vector<expression>& dims) 
        const {

        for (size_t i = 0; i < dims.size(); ++i) {
          generate_indent(indents_+i,o_);
          o_ << "for (size_t k" << i << "__ = 0;"
             << " k" << i << "__ < ";
          generate_expression(dims[i].expr_,o_);
          o_ << ";";
          o_ << " ++k" << i << "__) {" << EOL;
        }
      }
      void generate_end_for_dims(size_t dims_size) const {
        for (size_t i = 0; i < dims_size; ++i) {
          generate_indent(indents_ + dims_size - i - 1, o_);
          o_ << "}" << EOL;
        }
      }

      void generate_loop_var(const std::string& name,
                             size_t dims_size) const {
        o_ << name;
        for (size_t i = 0; i < dims_size; ++i)
          o_ << "[k" << i << "__]";
      }
      void operator()(nil const& x) const { }
      template <typename T>
      void basic_validate(T const& x) const {
        if (!(x.range_.has_low() || x.range_.has_high()))
          return; // unconstrained
        generate_begin_for_dims(x.dims_);
        if (x.range_.has_low()) {
          generate_indent(indents_ + x.dims_.size(),o_);
          o_ << "assert(stan::prob::lb_validate(";
          generate_loop_var(x.name_,x.dims_.size());
          o_ << ",";
          generate_expression(x.range_.low_.expr_,o_);
          o_ << "));" << EOL;
        }
        if (x.range_.has_high()) {
          generate_indent(indents_ + x.dims_.size(),o_);
          o_ << "assert(stan::prob::ub_validate(";
          generate_loop_var(x.name_,x.dims_.size());
          o_ << ", ";
          generate_expression(x.range_.high_.expr_,o_);
          o_ << "));" << EOL;
        }
        generate_end_for_dims(x.dims_.size());
      }
      void operator()(int_var_decl const& x) const {
        basic_validate(x);
      }
      void operator()(double_var_decl const& x) const {
        basic_validate(x);
      }
      void operator()(vector_var_decl const& x) const {
        // vector always unconstrained
      }
      void operator()(row_vector_var_decl const& x) const {
        // row vector always unconstrained
      }
      void operator()(matrix_var_decl const& x) const {
        // matrix always unconstrained
      }
      template <typename T>
      void nonbasic_validate(const T& x,
                             const std::string& type_name) const {
        generate_begin_for_dims(x.dims_);
        generate_indent(indents_ + x.dims_.size(),o_);
        o_ << "assert(stan::prob::" << type_name << "_validate(";
        generate_loop_var(x.name_,x.dims_.size());
        o_ << "));" << EOL;
        generate_end_for_dims(x.dims_.size());
      }
      void operator()(simplex_var_decl const& x) const {
        nonbasic_validate(x,"simplex");
      }
      void operator()(pos_ordered_var_decl const& x) const {
        nonbasic_validate(x,"pos_ordered");
      }
      void operator()(corr_matrix_var_decl const& x) const {
        nonbasic_validate(x,"corr_matrix");
      }
      void operator()(cov_matrix_var_decl const& x) const {
        nonbasic_validate(x,"cov_matrix");
      }
    };


    void generate_validate_var_decl(const var_decl& decl,
                                     int indent,
                                     std::ostream& o) {
      validate_var_decl_visgen vis(indent,o);
      boost::apply_visitor(vis,decl.decl_);
    }

    void generate_validate_var_decls(const std::vector<var_decl> decls,
                                     int indent,
                                     std::ostream& o) {
      for (size_t i = 0; i < decls.size(); ++i)
        generate_validate_var_decl(decls[i],indent,o);
    }

    // see _var_decl_visgen cut & paste
    struct member_var_decl_visgen : public visgen {
      int indents_;
      member_var_decl_visgen(int indents,
                             std::ostream& o)
        : visgen(o),
          indents_(indents) {
      }
      void operator()(nil const& x) const { }
      void operator()(int_var_decl const& x) const {
        declare_array("int",x.name_,x.dims_.size());
      }
      void operator()(double_var_decl const& x) const {
        declare_array("double",x.name_,x.dims_.size());
      }
      void operator()(simplex_var_decl const& x) const {
        declare_array(("vector_d"), x.name_, x.dims_.size());
      }
      void operator()(pos_ordered_var_decl const& x) const {
        declare_array(("vector_d"), x.name_, x.dims_.size());
      }
      void operator()(cov_matrix_var_decl const& x) const {
        declare_array(("matrix_d"), x.name_, x.dims_.size());
      }
      void operator()(corr_matrix_var_decl const& x) const {
        declare_array(("matrix_d"), x.name_, x.dims_.size());
      }
      void operator()(vector_var_decl const& x) const {
        declare_array(("vector_d"), x.name_, x.dims_.size());
      }
      void operator()(row_vector_var_decl const& x) const {
        declare_array(("row_vector_d"), x.name_, x.dims_.size());
      }
      void operator()(matrix_var_decl const& x) const {
        declare_array(("matrix_d"), x.name_, x.dims_.size());
      }
      void declare_array(std::string const& type, std::string const& name, 
                         size_t size) const {
        for (int i = 0; i < indents_; ++i)
          o_ << INDENT;
        for (size_t i = 0; i < size; ++i) {
          o_ << "vector<";
        }
        o_ << type;
        if (size > 0) {
          o_ << ">";
        }
        for (size_t i = 1; i < size; ++i) {
          o_ << " >";
        }
        o_ << " " << name << ";" << EOL;
      }
    };

    void generate_member_var_decls(const std::vector<var_decl>& vs,
                                   int indent,
                                   std::ostream& o) {
      member_var_decl_visgen vis(indent,o);
      for (size_t i = 0; i < vs.size(); ++i)
        boost::apply_visitor(vis,vs[i].decl_);
    }

    // see member_var_decl_visgen cut & paste
    struct local_var_decl_visgen : public visgen {
      int indents_;
      bool is_var_;
      local_var_decl_visgen(int indents,
                            bool is_var,
                            std::ostream& o)
        : visgen(o),
          indents_(indents),
          is_var_(is_var) {
      }
      void operator()(nil const& x) const { }
      void operator()(int_var_decl const& x) const {
        std::vector<expression> ctor_args;
        declare_array("int",ctor_args,x.name_,x.dims_);
      }
      void operator()(double_var_decl const& x) const {
        std::vector<expression> ctor_args;
        declare_array(is_var_ ? "var" : "double",
                      ctor_args,x.name_,x.dims_);
      }
      void operator()(vector_var_decl const& x) const {
        std::vector<expression> ctor_args;
        ctor_args.push_back(x.M_);
        declare_array(is_var_ ? "vector_v" : "vector_d",
                      ctor_args, x.name_, x.dims_);
      }
      void operator()(row_vector_var_decl const& x) const {
        std::vector<expression> ctor_args;
        ctor_args.push_back(x.N_);
        declare_array(is_var_ ? "row_vector_v" : "row_vector_d", 
                      ctor_args, x.name_, x.dims_);
      }
      void operator()(matrix_var_decl const& x) const {
        std::vector<expression> ctor_args;
        ctor_args.push_back(x.M_);
        ctor_args.push_back(x.N_);
        declare_array(is_var_ ? "matrix_v" : "matrix_d", 
                      ctor_args, x.name_, x.dims_);
      }
      void operator()(simplex_var_decl const& x) const {
        std::vector<expression> ctor_args;
        ctor_args.push_back(x.K_);
        declare_array(is_var_ ? "vector_v" : "vector_d", 
                      ctor_args, x.name_, x.dims_);
      }
      void operator()(pos_ordered_var_decl const& x) const {
        std::vector<expression> ctor_args;
        ctor_args.push_back(x.K_);
        declare_array(is_var_ ? "vector_v" : "vector_d", 
                      ctor_args, x.name_, x.dims_);
      }
      void operator()(cov_matrix_var_decl const& x) const {
        std::vector<expression> ctor_args;
        ctor_args.push_back(x.K_);
        ctor_args.push_back(x.K_);
        declare_array(is_var_ ? "matrix_v" : "matrix_d", 
                      ctor_args, x.name_, x.dims_);
      }
      void operator()(corr_matrix_var_decl const& x) const {
        std::vector<expression> ctor_args;
        ctor_args.push_back(x.K_);
        ctor_args.push_back(x.K_);
        declare_array(is_var_ ? "matrix_v" : "matrix_d", 
                      ctor_args, x.name_, x.dims_);
      }
      void generate_type(const std::string& type,
                         size_t num_dims) const {
        for (size_t i = 0; i < num_dims; ++i)
          o_ << "vector<";
        o_ << type;
        for (size_t i = 0; i < num_dims; ++i) {
          if (i > 0) o_ << " ";
          o_ << ">";
        }
      }
      // var_decl     -> type[0] name init_args[0] ;
      // init_args[k] -> ctor_args  if no dims left
      // init_args[k] -> ( dim[k] , ( type[k+1] init_args[k+1] ) )   
      void generate_init_args(const std::string& type,
                              const std::vector<expression>& ctor_args,
                              const std::vector<expression>& dims,
                              size_t dim) const {
        if (dim < dims.size()) { // more dims left
          o_ << '('; // open(1)
          generate_expression(dims[dim],o_);
          if ((dim + 1 < dims.size()) ||  ctor_args.size() > 0) {
            o_ << ", ("; // open(2)
            generate_type(type,dims.size() - dim - 1);
            generate_init_args(type,ctor_args,dims,dim + 1);
            o_ << ')'; // close(2)
          }
          o_ << ')'; // close(1)
        } else {
          if (ctor_args.size() == 1) {// vector
            o_ << '(';
            generate_expression(ctor_args[0],o_);
            o_ << ')';
          }
          if (ctor_args.size() > 1) { // matrix
            o_ << '(';
            generate_expression(ctor_args[0],o_);
            o_ << ',';
            generate_expression(ctor_args[1],o_);
            o_ << ')';
          }
        }
      }
      void declare_array(const std::string& type, 
                         const std::vector<expression>& ctor_args,
                         const std::string& name, 
                         const std::vector<expression>& dims) const {

        // require double parens to counter "most vexing parse" problem

        generate_indent(indents_,o_);
        generate_type(type,dims.size());
        o_ << ' '  << name;
        generate_init_args(type,ctor_args,dims,0);
        o_ << ';' << EOL;
      }
    };

    void generate_local_var_decls(const std::vector<var_decl>& vs,
                                  int indent,
                                  std::ostream& o,
                                  bool is_var) {
      local_var_decl_visgen vis(indent,is_var,o);
      for (size_t i = 0; i < vs.size(); ++i)
        boost::apply_visitor(vis,vs[i].decl_);
    }


    void generate_start_namespace(std::string name,
                                   std::ostream& o) {
      o << "namespace " << name << "_namespace {" << EOL2;
    }

     void generate_end_namespace(std::ostream& o) {
       o << "} // namespace" << EOL2;
     }

     void generate_comment(std::string const& msg, int indent, 
                           std::ostream& o) {
       generate_indent(indent,o);
       o << "// " << msg        << EOL;
     }

    void generate_statement(statement const& s, int indent, std::ostream& o,
                            bool include_sampling, bool is_var);

    struct statement_visgen : public visgen {
      size_t indent_;
      bool include_sampling_;
      bool is_var_;
      statement_visgen(size_t indent, 
                       bool include_sampling,
                       bool is_var,
                       std::ostream& o)
        : visgen(o),
          indent_(indent),
          include_sampling_(include_sampling),
          is_var_(is_var) {
      }
      void operator()(nil const& x) const { 
      }
      void operator()(assignment const& x) const {
        generate_indent(indent_,o_);
        generate_indexed_expr(x.var_dims_.name_,
                              x.var_dims_.dims_,
                              x.var_type_.base_type_,
                              x.var_type_.dims_.size(),
                              o_);
        o_ << " = ";
        generate_expression(x.expr_,o_);
        o_ << ";" << EOL;
      }
      void operator()(sample const& x) const {
        if (!include_sampling_) return;
        generate_indent(indent_,o_);
        // FOO_log<true> is the log FOO distribution up to a proportion
        o_ << "lp__ += stan::prob::" << x.dist_.family_ << "_log<true>(";
        generate_expression(x.expr_,o_);
        for (size_t i = 0; i < x.dist_.args_.size(); ++i) {
          o_ << ", ";
          generate_expression(x.dist_.args_[i],o_);
        }
        o_ << ");" << EOL;
        if (x.truncation_.has_low() && x.truncation_.has_high()) {
          generate_indent(indent_,o_);
          o_ << "lp__ += log(";
          o_ << x.dist_.family_ << "_p(";
          generate_expression(x.truncation_.high_.expr_,o_);
          for (size_t i = 0; i < x.dist_.args_.size(); ++i) {
            o_ << ", ";
            generate_expression(x.dist_.args_[i],o_);
          }
          o_ << ") - " << x.dist_.family_ << "_p(";
          generate_expression(x.truncation_.low_.expr_,o_);
          for (size_t i = 0; i < x.dist_.args_.size(); ++i) {
            o_ << ", ";
            generate_expression(x.dist_.args_[i],o_);
          }
          o_ << "));" << EOL;
        } else if (!x.truncation_.has_low() && x.truncation_.has_high()) {
          generate_indent(indent_,o_);
          o_ << "lp__ += log(";
          o_ << x.dist_.family_ << "_p(";
          generate_expression(x.truncation_.high_.expr_,o_);
          for (size_t i = 0; i < x.dist_.args_.size(); ++i) {
            o_ << ", ";
            generate_expression(x.dist_.args_[i],o_);
          }
          o_ << "));" << EOL;
        } else if (x.truncation_.has_low() && !x.truncation_.has_high()) {
          generate_indent(indent_,o_);
          o_ << "lp__ += log(1.0 - "; // FIXME: use log1m()
          o_ << x.dist_.family_ << "_p(";
          generate_expression(x.truncation_.low_.expr_,o_);
          for (size_t i = 0; i < x.dist_.args_.size(); ++i) {
            o_ << ", ";
            generate_expression(x.dist_.args_[i],o_);
          }
          o_ << "));" << EOL;
        }
      }
      void operator()(const statements& x) const {
        bool has_local_vars = x.local_decl_.size() > 0;
        size_t indent = has_local_vars ? (indent_ + 1) : indent_;
        if (has_local_vars) {
          generate_indent(indent_,o_);
          o_ << "{" << EOL;  // need brackets for scope
          generate_local_var_decls(x.local_decl_,indent,o_,is_var_);
        }
                                 
        for (size_t i = 0; i < x.statements_.size(); ++i)
          generate_statement(x.statements_[i],indent,o_,include_sampling_,is_var_);

        if (has_local_vars) {
          generate_indent(indent_,o_);
          o_ << "}" << EOL;
        }
      }
      void operator()(const for_statement& x) const {
        generate_indent(indent_,o_);
        o_ << "for (int " << x.variable_ << " = ";
        generate_expression(x.range_.low_,o_);
        o_ << "; " << x.variable_ << " <= ";
        generate_expression(x.range_.high_,o_);
        o_ << "; ++" << x.variable_ << ") {" << EOL;
        generate_statement(x.statement_, indent_ + 1, o_, include_sampling_,is_var_);
        generate_indent(indent_,o_);
        o_ << "}" << EOL;
      }
      void operator()(const no_op_statement& x) const {
        // called no_op for a reason
      }
    };

    void generate_statement(const statement& s,
                            int indent,
                            std::ostream& o,
                            bool include_sampling,
                            bool is_var) {
      statement_visgen vis(indent,include_sampling,is_var,o);
      boost::apply_visitor(vis,s.statement_);
    }

    void generate_statements(const std::vector<statement>& ss,
                             int indent,
                             std::ostream& o,
                             bool include_sampling,
                             bool is_var) {
      statement_visgen vis(indent,include_sampling,is_var,o);
      for (size_t i = 0; i < ss.size(); ++i)
        boost::apply_visitor(vis,ss[i].statement_);
    }


    void generate_log_prob(program const& p,
                           std::ostream& o) {
      o << EOL;
      o << INDENT << "var log_prob(vector<var>& params_r__," << EOL;
      o << INDENT << "             vector<int>& params_i__) {" << EOL2;
      o << INDENT2 << "var lp__(0.0);" << EOL;

      generate_comment("model parameters",2,o);
      generate_local_var_inits(p.parameter_decl_,true,o);
      o << EOL;

      static bool is_var = true;
      generate_comment("transformed parameters",2,o);
      generate_local_var_decls(p.derived_decl_.first,2,o,is_var);
      o << EOL;
      static bool include_sampling = true;
      generate_statements(p.derived_decl_.second,2,o,include_sampling,is_var);
      o << EOL;

      generate_comment("model body",2,o);
      generate_statement(p.statement_,2,o,include_sampling,is_var);
      o << EOL;
      o << INDENT2 << "return lp__;" << EOL2;
      o << INDENT << "} // log_prob()" << EOL2;
    }

    struct dump_member_var_visgen : public visgen {
      var_resizing_visgen var_resizer_;
      dump_member_var_visgen(std::ostream& o) 
        : visgen(o),
          var_resizer_(var_resizing_visgen(o)) {
      }
      void operator()(nil const& x) const { } // dummy
      void operator()(int_var_decl const& x) const {
        std::vector<expression> dims = x.dims_;
        var_resizer_(x);
        o_ << INDENT2 << "if (!context__.contains_i(\"" << x.name_ << "\"))" << EOL;
        o_ << INDENT3 << "throw std::runtime_error(\"variable " << x.name_ <<" not found.\");" << EOL;
        o_ << INDENT2 << "vals_i__ = context__.vals_i(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        size_t indentation = 1;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation,o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim],o_);
          o_ << ";" << EOL;
          generate_indent(indentation,o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" 
             << dim << "__ < " << x.name_ << "_limit_" << dim 
             << "__; ++i_" << dim << "__) {" << EOL;
        }
        generate_indent(indentation+1,o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << " = vals_i__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 1 - dim,o_);
          o_ << "}" << EOL;
        }
      }
      // minor changes to int_var_decl
      void operator()(double_var_decl const& x) const {
        std::vector<expression> dims = x.dims_;
        var_resizer_(x);
        o_ << INDENT2 << "if(!context__.contains_r(\"" << x.name_ << "\"))" << EOL;
        o_ << INDENT3 << "throw std::runtime_error(\"variable " << x.name_ <<" not found.\");" << EOL;
        o_ << INDENT2 << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        size_t indentation = 1;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation,o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim],o_);
          o_ << ";" << EOL;
          generate_indent(indentation,o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < " << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {" << EOL;
        }
        generate_indent(indentation+1,o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 1 - dim,o_);
          o_ << "}" << EOL;
        }
      }
      // extra outer loop around double_var_decl
      void operator()(vector_var_decl const& x) const {
        std::vector<expression> dims = x.dims_;
        var_resizer_(x);
        o_ << INDENT2 << "if(!context__.contains_r(\"" << x.name_ << "\"))" << EOL;
        o_ << INDENT3 << "throw std::runtime_error(\"variable " << x.name_ <<" not found.\");" << EOL;
        o_ << INDENT2 << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_i_vec_lim__ = ";
        generate_expression(x.M_,o_);
        o_ << ";" << EOL;
        o_ << INDENT2 << "for (size_t " << "i_vec__ = 0; " << "i_vec__ < " << x.name_ << "_i_vec_lim__; ++i_vec__) {" << EOL;
        size_t indentation = 2;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation,o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim],o_);
          o_ << ";" << EOL;
          generate_indent(indentation,o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < " << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {" << EOL;
        }
        generate_indent(indentation+1,o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "[i_vec__]";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim,o_);
          o_ << "}" << EOL;
        }
        o_ << INDENT2 << "}" << EOL;
      }
      // change variable name from vector_var_decl
      void operator()(row_vector_var_decl const& x) const {
        std::vector<expression> dims = x.dims_;
        var_resizer_(x);
        o_ << INDENT2 << "if(!context__.contains_r(\"" << x.name_ << "\"))" << EOL;
        o_ << INDENT3 << "throw std::runtime_error(\"variable " << x.name_ <<" not found.\");" << EOL;
        o_ << INDENT2 << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_i_vec_lim__ = ";
        generate_expression(x.N_,o_);
        o_ << ";" << EOL;
        o_ << INDENT2 << "for (size_t " << "i_vec__ = 0; " << "i_vec__ < " << x.name_ << "_i_vec_lim__; ++i_vec__) {" << EOL;
        size_t indentation = 2;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation,o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim],o_);
          o_ << ";" << EOL;
          generate_indent(indentation,o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < " << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {" << EOL;
        }
        generate_indent(indentation+1,o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "[i_vec__]";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim,o_);
          o_ << "}" << EOL;
        }
        o_ << INDENT2 << "}" << EOL;
      }
      // diff name of dims from vector
      void operator()(simplex_var_decl const& x) const {
        std::vector<expression> dims = x.dims_;
        var_resizer_(x);
        o_ << INDENT2 << "if(!context__.contains_r(\"" << x.name_ << "\"))" << EOL;
        o_ << INDENT3 << "throw std::runtime_error(\"variable " << x.name_ <<" not found.\");" << EOL;
        o_ << INDENT2 << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_i_vec_lim__ = ";
        generate_expression(x.K_,o_);
        o_ << ";" << EOL;
        o_ << INDENT2 << "for (size_t " << "i_vec__ = 0; " << "i_vec__ < " << x.name_ << "_i_vec_lim__; ++i_vec__) {" << EOL;
        size_t indentation = 2;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation,o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim],o_);
          o_ << ";" << EOL;
          generate_indent(indentation,o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < " << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {" << EOL;
        }
        generate_indent(indentation+1,o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "[i_vec__]";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim,o_);
          o_ << "}" << EOL;
        }
        o_ << INDENT2 << "}" << EOL;
      }
      // same as simplex
      void operator()(pos_ordered_var_decl const& x) const {
        std::vector<expression> dims = x.dims_;
        var_resizer_(x);
        o_ << INDENT2 << "if(!context__.contains_r(\"" << x.name_ << "\"))" << EOL;
        o_ << INDENT3 << "throw std::runtime_error(\"variable " << x.name_ <<" not found.\");" << EOL;
        o_ << INDENT2 << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_i_vec_lim__ = ";
        generate_expression(x.K_,o_);
        o_ << ";" << EOL;
        o_ << INDENT2 << "for (size_t " << "i_vec__ = 0; " << "i_vec__ < " << x.name_ << "_i_vec_lim__; ++i_vec__) {" << EOL;
        size_t indentation = 2;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation,o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim],o_);
          o_ << ";" << EOL;
          generate_indent(indentation,o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < " << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {" << EOL;
        }
        generate_indent(indentation+1,o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "[i_vec__]";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim,o_);
          o_ << "}" << EOL;
        }
        o_ << INDENT2 << "}" << EOL;
      }
      // extra loop and different accessor vs. vector
      void operator()(matrix_var_decl const& x) const {
        std::vector<expression> dims = x.dims_;
        var_resizer_(x);
        o_ << INDENT2 << "if(!context__.contains_r(\"" << x.name_ << "\"))" << EOL;
        o_ << INDENT3 << "throw std::runtime_error(\"variable " << x.name_ <<" not found.\");" << EOL;
        o_ << INDENT2 << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_m_mat_lim__ = ";
        generate_expression(x.M_,o_);
        o_ << ";" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_n_mat_lim__ = ";
        generate_expression(x.N_,o_);
        o_ << ";" << EOL;
        o_ << INDENT2 << "for (size_t " << "n_mat__ = 0; " << "n_mat__ < " << x.name_ << "_n_mat_lim__; ++n_mat__) {" << EOL;
        o_ << INDENT3 << "for (size_t " << "m_mat__ = 0; " << "m_mat__ < " << x.name_ << "_m_mat_lim__; ++m_mat__) {" << EOL;
        size_t indentation = 3;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation,o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim],o_);
          o_ << ";" << EOL;
          generate_indent(indentation,o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < " << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {" << EOL;
        }
        generate_indent(indentation+1,o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "(m_mat__,n_mat__)";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim,o_);
          o_ << "}" << EOL;
        }
        o_ << INDENT3 << "}" << EOL;
        o_ << INDENT2 << "}" << EOL;
      }
      void operator()(corr_matrix_var_decl const& x) const {
        std::vector<expression> dims = x.dims_;
        var_resizer_(x);
        o_ << INDENT2 << "if(!context__.contains_r(\"" << x.name_ << "\"))" << EOL;
        o_ << INDENT3 << "throw std::runtime_error(\"variable " << x.name_ <<" not found.\");" << EOL;
        o_ << INDENT2 << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_k_mat_lim__ = ";
        generate_expression(x.K_,o_);
        o_ << ";" << EOL;
        o_ << INDENT2 << "for (size_t " << "n_mat__ = 0; " << "n_mat__ < " << x.name_ << "_k_mat_lim__; ++n_mat__) {" << EOL;
        o_ << INDENT3 << "for (size_t " << "m_mat__ = 0; " << "m_mat__ < " << x.name_ << "_k_mat_lim__; ++m_mat__) {" << EOL;
        size_t indentation = 3;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation,o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim],o_);
          o_ << ";" << EOL;
          generate_indent(indentation,o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < " << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {" << EOL;
        }
        generate_indent(indentation+1,o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "(m_mat__,n_mat__)";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim,o_);
          o_ << "}" << EOL;
        }
        o_ << INDENT3 << "}" << EOL;
        o_ << INDENT2 << "}" << EOL;
      }
      void operator()(cov_matrix_var_decl const& x) const {
        std::vector<expression> dims = x.dims_;
        var_resizer_(x);
        o_ << INDENT2 << "if(!context__.contains_r(\"" << x.name_ << "\"))" << EOL;
        o_ << INDENT3 << "throw std::runtime_error(\"variable " << x.name_ <<" not found.\");" << EOL;
        o_ << INDENT2 << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_k_mat_lim__ = ";
        generate_expression(x.K_,o_);
        o_ << ";" << EOL;
        o_ << INDENT2 << "for (size_t " << "n_mat__ = 0; " << "n_mat__ < " << x.name_ << "_k_mat_lim__; ++n_mat__) {" << EOL;
        o_ << INDENT3 << "for (size_t " << "m_mat__ = 0; " << "m_mat__ < " << x.name_ << "_k_mat_lim__; ++m_mat__) {" << EOL;
        size_t indentation = 3;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation,o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim],o_);
          o_ << ";" << EOL;
          generate_indent(indentation,o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < " << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {" << EOL;
        }
        generate_indent(indentation+1,o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "(m_mat__,n_mat__)";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim,o_);
          o_ << "}" << EOL;
        }
        o_ << INDENT3 << "}" << EOL;
        o_ << INDENT2 << "}" << EOL;
      }
    };

    void generate_member_var_inits(const std::vector<var_decl>& vs,
                                   std::ostream& o) {
      dump_member_var_visgen vis(o);
      for (size_t i = 0; i < vs.size(); ++i)
        boost::apply_visitor(vis, vs[i].decl_);
    }

    void generate_constructor(const program& prog,
                              const std::string& model_name,
                              std::ostream& o) {
      o << INDENT << model_name << "(stan::io::var_context& context__)" << EOL;
      o << INDENT2 << ": prob_grad_ad::prob_grad_ad(0) {" << EOL; // resize 0 with var_resizing
      o << INDENT2 << "size_t pos__;" << EOL;
      o << INDENT2 << "std::vector<int> vals_i__;" << EOL;
      o << INDENT2 << "std::vector<double> vals_r__;" << EOL;

      generate_member_var_inits(prog.data_decl_,o);

      generate_comment("validate data",2,o);
      generate_validate_var_decls(prog.data_decl_,2,o);

      generate_var_resizing(prog.derived_data_decl_.first, o);
      o << EOL;
      static bool include_sampling = false;
      static bool is_var = false;
      for (size_t i = 0; i < prog.derived_data_decl_.second.size(); ++i)
        generate_statement(prog.derived_data_decl_.second[i],2,o,include_sampling,is_var);
      
      generate_comment("validate transformed data",2,o);
      generate_validate_var_decls(prog.derived_data_decl_.first,2,o);

      o << EOL << INDENT2 << "set_param_ranges();" << EOL;
      o << INDENT << "} // dump ctor" << EOL;
    }

    struct generate_init_visgen : public visgen {
      generate_init_visgen(std::ostream& o) 
        : visgen(o) {
      }
      void operator()(nil const& x) const { } // dummy
      void operator()(int_var_decl const& x) const {
        generate_check_int(x.name_,x.dims_.size());
        generate_declaration(x.name_,"int",x.dims_);
        generate_buffer_loop("i",x.name_, x.dims_);
        generate_write_loop("integer(",x.name_,x.dims_);
      }
      void operator()(double_var_decl const& x) const {
        generate_check_double(x.name_,x.dims_.size());
        generate_declaration(x.name_,"double",x.dims_);
        generate_buffer_loop("r",x.name_,x.dims_);
        bool has_lower_bound = !is_nil(x.range_.low_.expr_);
        bool has_upper_bound = !is_nil(x.range_.high_.expr_);
        std::stringstream ss;
        if (has_lower_bound && has_upper_bound) {
          ss << "scalar_lub_unconstrain(";
          generate_expression(x.range_.low_.expr_,ss);
          ss << ',';
          generate_expression(x.range_.high_.expr_,ss);
          ss << ',';
        } else if (has_lower_bound && !has_upper_bound) {
          ss << "scalar_lb_unconstrain(";
          generate_expression(x.range_.low_.expr_,ss);
          ss << ',';
        } else if ((!has_lower_bound) && has_upper_bound) {
          ss << "scalar_ub_unconstrain(";
          generate_expression(x.range_.high_.expr_,ss);
          ss << ',';
        } else if ((!has_lower_bound) && (!has_upper_bound)) {
          ss << "scalar_unconstrain(";
        }
        generate_write_loop(ss.str(),x.name_,x.dims_);
      }
      void operator()(vector_var_decl const& x) const {
        generate_check_double(x.name_,x.dims_.size() + 1);
        generate_declaration(x.name_,"vector_d",x.dims_,x.M_);
        generate_buffer_loop("r",x.name_,x.dims_,x.M_);
        generate_write_loop("vector_unconstrain(",x.name_,x.dims_);
      }
      void operator()(row_vector_var_decl const& x) const {
        generate_check_double(x.name_,x.dims_.size() + 1);
        generate_declaration(x.name_,"row_vector_d",x.dims_,x.N_);
        generate_buffer_loop("r",x.name_,x.dims_,x.N_);
        generate_write_loop("row_vector_unconstrain(",x.name_,x.dims_);
      }
      void operator()(simplex_var_decl const& x) const {
        generate_check_double(x.name_,x.dims_.size() + 1);
        generate_declaration(x.name_,"vector_d",x.dims_,x.K_);
        generate_buffer_loop("r",x.name_,x.dims_,x.K_);
        generate_write_loop("simplex_unconstrain(",x.name_,x.dims_);
      }
      void operator()(pos_ordered_var_decl const& x) const {
        generate_check_double(x.name_,x.dims_.size() + 1);
        generate_declaration(x.name_,"vector_d",x.dims_,x.K_);
        generate_buffer_loop("r",x.name_,x.dims_,x.K_);
        generate_write_loop("pos_ordered_unconstrain(",x.name_,x.dims_);
      }
      void operator()(matrix_var_decl const& x) const {
        generate_check_double(x.name_,x.dims_.size() + 2);
        generate_declaration(x.name_,"matrix_d",x.dims_,x.M_,x.N_);
        generate_buffer_loop("r",x.name_,x.dims_,x.M_,x.N_);
        generate_write_loop("matrix_unconstrain(",x.name_,x.dims_);
      }
      void operator()(cov_matrix_var_decl const& x) const {
        generate_check_double(x.name_,x.dims_.size() + 2);
        generate_declaration(x.name_,"matrix_d",x.dims_,x.K_,x.K_);
        generate_buffer_loop("r",x.name_,x.dims_,x.K_,x.K_);
        generate_write_loop("cov_matrix_unconstrain(",x.name_,x.dims_);
      }
      void operator()(corr_matrix_var_decl const& x) const {
        generate_check_double(x.name_,x.dims_.size() + 2);
        generate_declaration(x.name_,"matrix_d",x.dims_,x.K_,x.K_);
        generate_buffer_loop("r",x.name_,x.dims_,x.K_,x.K_);
        generate_write_loop("corr_matrix_unconstrain(",x.name_,x.dims_);
      }
      void generate_write_loop(const std::string& write_method_name,
                               const std::string& var_name,
                               const std::vector<expression>& dims) const {
        generate_dims_loop_fwd(dims);
        o_ << "writer__." << write_method_name;
        generate_name_dims(var_name,dims.size());
        o_ << ");" << EOL;
      }
      void generate_name_dims(const std::string name, 
                              size_t num_dims) const {
        o_ << name;
        for (size_t i = 0; i < num_dims; ++i)
          o_ << "[i" << i << "__]";
      }
      void generate_declaration(const std::string& name,
                                const std::string& base_type,
                                const std::vector<expression>& dims,
                                const expression& type_arg1 = expression(),
                                const expression& type_arg2 = expression()) const {
        o_ << INDENT2;
        generate_type(base_type,dims,dims.size(),o_);
        o_ << ' ' << name;

        generate_initializer(o_,base_type,dims,type_arg1,type_arg2);
      }
      void generate_indent_num_dims(size_t base_indent,
                                    const std::vector<expression>& dims, 
                                    const expression& dim1,
                                    const expression& dim2) const {
        generate_indent(dims.size() + base_indent,o_);
        if (!is_nil(dim1)) o_ << INDENT;
        if (!is_nil(dim2)) o_ << INDENT;
      }
      void generate_buffer_loop(const std::string& base_type,
                                const std::string& name,
                                const std::vector<expression>& dims, 
                                const expression& dim1 = expression(),
                                const expression& dim2 = expression(), 
                                int indent = 2U) const {
        size_t size = dims.size();
        bool is_matrix = !is_nil(dim1) && !is_nil(dim2);
        bool is_vector = !is_nil(dim1) && is_nil(dim2);
        int extra_indent = is_matrix ? 2U : is_vector ? 1U : 0U;
        if (is_matrix) {
          generate_indent(indent,o_);
          o_ << "for (size_t j2__ = 0U; j2__ < ";
          generate_expression(dim2.expr_,o_);
          o_ << "; ++j2__)" << EOL;

          generate_indent(indent+1,o_);
          o_ << "for (size_t j1__ = 0U; j1__ < ";
          generate_expression(dim1.expr_,o_);
          o_ << "; ++j1__)" << EOL;
        } else if (is_vector) {
          generate_indent(indent,o_);
          o_ << "for (size_t j1__ = 0U; j1__ < ";
          generate_expression(dim1.expr_,o_);
          o_ << "; ++j1__)" << EOL;
        }
        for (size_t i = 0; i < size; ++i) {
          size_t idx = size - i - 1;
          generate_indent(i + indent + extra_indent, o_);
          o_ << "for (size_t i" << idx << "__ = 0U; i" << idx << "__ < ";
          generate_expression(dims[idx].expr_,o_);
          o_ << "; ++i" << idx << "__)" << EOL;
        }
        generate_indent_num_dims(2U,dims,dim1,dim2);
        o_ << name; 
        for (size_t i = 0; i < dims.size(); ++i)
          o_ << "[i" << i << "__]";
        if (is_matrix) 
          o_ << "(j1__,j2__)";
        else if (is_vector)
          o_ << "(j1__)";
        o_ << " = vals_" << base_type << "__[pos__++];" << EOL;
      }
      void generate_dims_loop_fwd(const std::vector<expression>& dims, 
                                  int indent = 2U) const {
        size_t size = dims.size();
        for (size_t i = 0; i < size; ++i) {
          generate_indent(i + indent, o_);
          o_ << "for (size_t i" << i << "__ = 0U; i" << i << "__ < ";
          generate_expression(dims[i].expr_,o_);
          o_ << "; ++i" << i << "__)" << EOL;
        }
        generate_indent(2U + dims.size(),o_);
      }
      void generate_check_int(const std::string& name, size_t n) const {
        o_ << EOL << INDENT2
           << "if (!(var_context__.contains_i(\"" << name << "\")))"
           << EOL << INDENT3
           << "throw std::runtime_error(\"variable " << name << " missing\");" << EOL;
        o_ << INDENT2
           << "if (var_context__.dims_i(\"" << name << "\").size() != " << n << ")"
           << EOL << INDENT3
           << "throw std::runtime_error(\"require " 
           << n << " dimensionss for variable " 
           << name << "\");" << EOL;
        o_ << INDENT2 << "vals_i__ = var_context__.vals_i(\"" << name << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0U;" << EOL;
      }
      void generate_check_double(const std::string& name, size_t n) const {
        o_ << EOL << INDENT2
           << "if (!(var_context__.contains_r(\"" << name << "\")))"
           << EOL << INDENT3
           << "throw std::runtime_error(\"variable " << name << " missing\");" << EOL;
        o_ << INDENT2
           << "if (var_context__.dims_r(\"" << name << "\").size() != " << n << ")"
           << EOL << INDENT3
           << "throw std::runtime_error(\"require " 
           << n << " dimensions for variable " 
           << name << "\");" << EOL;
        o_ << INDENT2 << "vals_r__ = var_context__.vals_r(\"" << name << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0U;" << EOL;
      }
    };
    

    void generate_init_method(const std::vector<var_decl>& vs,
                              std::ostream& o) {
      o << EOL;
      o << INDENT << "void transform_inits(const stan::io::var_context& var_context__," << EOL;
      o << INDENT << "                     std::vector<int>& params_i__," << EOL;
      o << INDENT << "                     std::vector<double>& params_r__) {" << EOL;
      o << INDENT2 << "params_r__.clear();" << EOL;
      o << INDENT2 << "params_i__.clear();" << EOL;
      o << INDENT2 << "stan::io::writer<double> writer__(params_r__,params_i__);" << EOL;
      o << INDENT2 << "size_t pos__;" << EOL;
      o << INDENT2 << "std::vector<double> vals_r__;" << EOL;
      o << INDENT2 << "std::vector<int> vals_i__;" << EOL;
      o << EOL;
      generate_init_visgen vis(o);
      for (size_t i = 0; i < vs.size(); ++i)
        boost::apply_visitor(vis, vs[i].decl_);
      o << INDENT << "}" << EOL;
    }

    // see write_csv_visgen for similar structure
    struct write_csv_header_visgen : public visgen {
      write_csv_header_visgen(std::ostream& o)
        : visgen(o) {
      }
      void operator()(const nil& x) const  { }
      void operator()(const int_var_decl& x) const {
        generate_csv_header_array(EMPTY_EXP_VECTOR,x.name_,x.dims_);
      }
      void operator()(const double_var_decl& x) const {
        generate_csv_header_array(EMPTY_EXP_VECTOR,x.name_,x.dims_);
      }
      void operator()(const vector_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.M_);
        generate_csv_header_array(matrix_args,x.name_,x.dims_);
      }
      void operator()(const row_vector_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.N_);
        generate_csv_header_array(matrix_args,x.name_,x.dims_);
      }
      void operator()(const matrix_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.M_);
        matrix_args.push_back(x.N_);
        generate_csv_header_array(matrix_args,x.name_,x.dims_);
      }
      void operator()(const simplex_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        generate_csv_header_array(matrix_args,x.name_,x.dims_);
      }
      void operator()(const pos_ordered_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        generate_csv_header_array(matrix_args,x.name_,x.dims_);
      }
      void operator()(const cov_matrix_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        matrix_args.push_back(x.K_);
        generate_csv_header_array(matrix_args,x.name_,x.dims_);
      }
      void operator()(const corr_matrix_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        matrix_args.push_back(x.K_);
        generate_csv_header_array(matrix_args,x.name_,x.dims_);
      }
      void 
      generate_csv_header_array(const std::vector<expression>& matrix_dims, 
                                const std::string& name,
                                const std::vector<expression>& dims) const {

        // begin for loop dims
        std::vector<expression> combo_dims(dims);
        for (size_t i = 0; i < matrix_dims.size(); ++i)
          combo_dims.push_back(matrix_dims[i]);

        for (size_t i = 0; i < combo_dims.size(); ++i) {
          generate_indent(2 + i,o_);
          o_ << "for (size_t k_" << i << "__ = 1;"
             << " k_" << i << "__ <= ";
          generate_expression(combo_dims[i].expr_,o_);
          o_ << "; ++k_" << i << "__) {" << EOL; // begin (1)
        }

        // variable + indices
        generate_indent(2 + combo_dims.size(),o_);
        o_ << "writer__.comma();" << EOL;  // only writes comma after first call

        generate_indent(2 + combo_dims.size(),o_);
        o_ << "o__ << \"" << name << '"';
        for (size_t i = 0; i < combo_dims.size(); ++i)
          o_ << " << '.' << k_" << i << "__";
        o_ << ';' << EOL;

        // end for loop dims
        for (size_t i = 0; i < combo_dims.size(); ++i) {
          generate_indent(1 + combo_dims.size() - i,o_);
          o_ << "}" << EOL; // end (1)
        }
      }
    };


    void generate_write_csv_header_method(const program& prog,
                                          std::ostream& o) {
      write_csv_header_visgen vis(o);
      o << EOL << INDENT << "void write_csv_header(std::ostream& o__) {" << EOL;
      o << INDENT2 << "stan::io::csv_writer writer__(o__);" << EOL;
      // parameters
      for (size_t i = 0; i < prog.parameter_decl_.size(); ++i) {
        boost::apply_visitor(vis,prog.parameter_decl_[i].decl_);
      }
      // transformed parameters
      for (size_t i = 0; i < prog.derived_decl_.first.size(); ++i) {
        boost::apply_visitor(vis,prog.derived_decl_.first[i].decl_);
      }
      // generated quantities
      for (size_t i = 0; i < prog.generated_decl_.first.size(); ++i) {
        boost::apply_visitor(vis,prog.generated_decl_.first[i].decl_);
      }
      o << INDENT2 << "writer__.newline();" << EOL;
      o << INDENT << "}" << EOL2;
    }

    // see init_member_var_visgen for cut & paste
    struct write_csv_visgen : public visgen {
      write_csv_visgen(std::ostream& o)
        : visgen(o) {
      }
      void operator()(const nil& x) const { }
      void operator()(const int_var_decl& x) const {
        generate_initialize_array("int","integer",EMPTY_EXP_VECTOR,
                                  x.name_,x.dims_);
      }      
      void operator()(const double_var_decl& x) const {
        if (!is_nil(x.range_.low_.expr_)) {
          if (!is_nil(x.range_.high_.expr_)) {
            std::vector<expression> read_args;
            read_args.push_back(x.range_.low_);
            read_args.push_back(x.range_.high_);
            generate_initialize_array("double","scalar_lub",read_args,
                                      x.name_,x.dims_);
          } else {
            std::vector<expression> read_args;
            read_args.push_back(x.range_.low_);
            generate_initialize_array("double","scalar_lb",read_args,x.name_,x.dims_);
          }
        } else {
          if (!is_nil(x.range_.high_.expr_)) {
            std::vector<expression> read_args;
            read_args.push_back(x.range_.high_);
            generate_initialize_array("double","scalar_ub",read_args,x.name_,x.dims_);
          } else {
            generate_initialize_array("double","scalar",EMPTY_EXP_VECTOR,x.name_,x.dims_);
          }
        }
      }
      void operator()(const vector_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.M_);
        generate_initialize_array("vector_d","vector",read_args,x.name_,x.dims_);
      }
      void operator()(const row_vector_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.N_);
        generate_initialize_array("row_vector_d","row_vector",read_args,x.name_,x.dims_);
      }
      void operator()(const matrix_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.M_);
        read_args.push_back(x.N_);
        generate_initialize_array("matrix_d","matrix",read_args,x.name_,x.dims_);
      }
      void operator()(const simplex_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array("vector_d","simplex",read_args,x.name_,x.dims_);
      }
      void operator()(const pos_ordered_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array("vector_d","pos_ordered",read_args,x.name_,x.dims_);
      }
      void operator()(const cov_matrix_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array("matrix_d","cov_matrix",read_args,x.name_,x.dims_);
      }
      void operator()(const corr_matrix_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array("matrix_d","corr_matrix",read_args,x.name_,x.dims_);
      }
      void generate_initialize_array(const std::string& var_type,
                                     const std::string& read_type,
                                     const std::vector<expression>& read_args,
                                     const std::string& name,
                                     const std::vector<expression>& dims) const {
        if (dims.size() == 0) {
          generate_indent(2,o_);
          o_ << var_type << " ";
          o_ << name << " = in__." << read_type  << "_constrain(";
          for (size_t j = 0; j < read_args.size(); ++j) {
            if (j > 0) o_ << ",";
            generate_expression(read_args[j],o_);
          }
          o_ << ");" << EOL;
          o_ << INDENT2 << "writer__.write(" << name << ");" << EOL;
          return;
        }
        o_ << INDENT2;
        for (size_t i = 0; i < dims.size(); ++i) o_ << "vector<";
        o_ << var_type;
        for (size_t i = 0; i < dims.size(); ++i) o_ << "> ";
        o_ << name << ";" << EOL;
        std::string name_dims(name);
        for (size_t i = 0; i < dims.size(); ++i) {
          generate_indent(i + 2, o_);
          o_ << "size_t dim_"  << name << "_" << i << " = ";
          generate_expression(dims[i],o_);
          o_ << ";" << EOL;
          if (i < dims.size() - 1) {  
            generate_indent(i + 2, o_);
            o_ << name_dims << ".resize(dim" << "_" << name << "_" << i << ");" 
               << EOL;
            name_dims.append("[k_").append(to_string(i)).append("]");
          }
          generate_indent(i + 2, o_);
          o_ << "for (size_t k_" << i << " = 0;"
             << " k_" << i << " < dim_" << name << "_" << i << ";"
             << " ++k_" << i << ") {" << EOL;
          if (i == dims.size() - 1) {
            generate_indent(i + 3, o_);
            o_ << name_dims << ".push_back(in__." << read_type << "_constrain(";
            for (size_t j = 0; j < read_args.size(); ++j) {
              if (j > 0) o_ << ",";
              generate_expression(read_args[j],o_);
            }
            o_ << "));" << EOL;
          }
        }
        generate_indent(dims.size() + 2, o_);
        o_ << "writer__.write(" << name;
        if (dims.size() > 0) {
          o_ << '[';
          for (size_t i = 0; i < dims.size(); ++i) {
            if (i > 0) o_ << "][";
            o_ << "k_" << i;
          }
          o_ << ']';
        }
        o_ << ");" << EOL;
        
        for (size_t i = dims.size(); i > 0; --i) {
          generate_indent(i + 1, o_);
          o_ << "}" << EOL;
        }
      }
    };

    struct write_csv_vars_visgen : public visgen {
      write_csv_vars_visgen(std::ostream& o)
        : visgen(o) {
      }
      void operator()(const nil& x) const { }
      // FIXME: template these out
      void operator()(const int_var_decl& x) const {
        write_array(x.name_,x.dims_);
      }
      void operator()(const double_var_decl& x) const {
        write_array(x.name_,x.dims_);
      }
      void operator()(const vector_var_decl& x) const {
        write_array(x.name_,x.dims_);
      }
      void operator()(const row_vector_var_decl& x) const {
        write_array(x.name_,x.dims_);
      }
      void operator()(const matrix_var_decl& x) const {
        write_array(x.name_,x.dims_);
      }
      void operator()(const simplex_var_decl& x) const {
        write_array(x.name_,x.dims_);
      }
      void operator()(const pos_ordered_var_decl& x) const {
        write_array(x.name_,x.dims_);
      }
      void operator()(const cov_matrix_var_decl& x) const {
        write_array(x.name_,x.dims_);
      }
      void operator()(const corr_matrix_var_decl& x) const {
        write_array(x.name_,x.dims_);
      }
      void write_array(const std::string& name,
                       const std::vector<expression>& dims) const {
        if (dims.size() == 0) {
          o_ << INDENT2 << "writer__.write(" << name << ");" << EOL;
          return;
        }
        for (size_t i = 0; i < dims.size(); ++i) {
          generate_indent(i + 2, o_);
          o_ << "for (size_t k_" << i << " = 0;"
             << " k_" << i << " < ";
          generate_expression(dims[i],o_);
          o_ << "; ++k_" << i << ") {" << EOL;
        }

        generate_indent(dims.size() + 2, o_);
        o_ << "writer__.write(" << name;
        if (dims.size() > 0) {
          o_ << '[';
          for (size_t i = 0; i < dims.size(); ++i) {
            if (i > 0) o_ << "][";
            o_ << "k_" << i;
          }
          o_ << ']';
        }
        o_ << ");" << EOL;
        
        for (size_t i = dims.size(); i > 0; --i) {
          generate_indent(i + 1, o_);
          o_ << "}" << EOL;
        }
      }
    };

    void generate_write_csv_method(const program& prog,
                                   std::ostream& o) {
      o << INDENT << "void write_csv(std::vector<double>& params_r__," << EOL;
      o << INDENT << "               std::vector<int>& params_i__," << EOL;
      o << INDENT << "               std::ostream& o__) {" << EOL;
      o << INDENT2 << "stan::io::reader<double> in__(params_r__,params_i__);" << EOL;
      o << INDENT2 << "stan::io::csv_writer writer__(o__);" << EOL;

      // declares, reads, and writes parameters
      generate_comment("read-transform, write parameters",2,o);
      write_csv_visgen vis(o);
      for (size_t i = 0; i < prog.parameter_decl_.size(); ++i)
        boost::apply_visitor(vis,prog.parameter_decl_[i].decl_);

      write_csv_vars_visgen vis_writer(o);

      // transformed parameters guaranteed to satisfy constraints

      o << EOL;
      generate_comment("declare and define transformed parameters",2,o);
      static bool is_var = false;
      generate_local_var_decls(prog.derived_decl_.first,2,o,is_var); 
      o << EOL;
      static bool include_sampling = false;
      generate_statements(prog.derived_decl_.second,2,o,include_sampling,is_var); 
      o << EOL;

      generate_comment("validate transformed parameters",2,o);
      generate_validate_var_decls(prog.derived_decl_.first,2,o);
      o << EOL;

      generate_comment("write transformed parameters",2,o);
      for (size_t i = 0; i < prog.derived_decl_.first.size(); ++i)
        boost::apply_visitor(vis_writer, prog.derived_decl_.first[i].decl_);
      o << EOL;

      generate_comment("declare and define generated quantities",2,o);
      generate_local_var_decls(prog.generated_decl_.first,2,o,is_var); 
      o << EOL;
      generate_statements(prog.generated_decl_.second,2,o,include_sampling,is_var); 
      o << EOL;

      generate_comment("validate generated quantities",2,o);
      generate_validate_var_decls(prog.generated_decl_.first,2,o);
      o << EOL;

      generate_comment("write generated quantities",2,o);
      for (size_t i = 0; i < prog.generated_decl_.first.size(); ++i)
        boost::apply_visitor(vis_writer, prog.generated_decl_.first[i].decl_);
      if (prog.generated_decl_.first.size() > 0)
        o << EOL;

      o << INDENT2 << "writer__.newline();" << EOL;
      o << INDENT << "}" << EOL2;
    }
    
    // know all data is set and range expressions only depend on data
    struct set_param_ranges_visgen : public visgen {
      set_param_ranges_visgen(std::ostream& o)
        : visgen(o) {
      }
      void operator()(const nil& x) const { }
      void operator()(const int_var_decl& x) const {
        generate_increment_i(x.dims_);
        // for loop for ranges
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_indent(i + 2, o_);
          o_ << "for (size_t i_" << i << "__ = 0; ";
          o_ << "i_" << i << "__ < ";
          generate_expression(x.dims_[i],o_);
          o_ << "; ++i_" << i << "__) {" << EOL;
        }
        // add range
        generate_indent(x.dims_.size() + 2,o_);
        o_ << "param_ranges_i__.push_back(std::pair<int,int>(";
        generate_expression(x.range_.low_,o_);
        o_ << ", ";
        generate_expression(x.range_.high_,o_);
        o_ << "));" << EOL;
        // close for loop
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_indent(x.dims_.size() + 1 - i, o_);
          o_ << "}" << EOL;
        }
      }
      void operator()(const double_var_decl& x) const {
        generate_increment(x.dims_);
      }
      void operator()(const vector_var_decl& x) const {
        generate_increment(x.M_,x.dims_);
      }
      void operator()(const row_vector_var_decl& x) const {
        generate_increment(x.N_,x.dims_);
      }
      void operator()(const matrix_var_decl& x) const {
        generate_increment(x.M_,x.N_,x.dims_);
      }
      void operator()(const simplex_var_decl& x) const {
        // only K-1 vals
        o_ << INDENT2 << "num_params_r__ += (";
        generate_expression(x.K_,o_);
        o_ << " - 1)";
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          o_ << " * ";
          generate_expression(x.dims_[i],o_);
        }
        o_ << ";" << EOL;
      }
      void operator()(const pos_ordered_var_decl& x) const {
        generate_increment(x.K_,x.dims_);
      }
      void operator()(const cov_matrix_var_decl& x) const {
        // (K * (K - 1))/2 + K  ?? define fun(K) = ??
        o_ << INDENT2 << "num_params_r__ += ((";
        generate_expression(x.K_,o_);
        o_ << " * (";
        generate_expression(x.K_,o_);
        o_ << " - 1)) / 2 + ";
        generate_expression(x.K_,o_);
        o_ << ")";
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          o_ << " * ";
          generate_expression(x.dims_[i],o_);
        }
        o_ << ";" << EOL;
      }
      void operator()(const corr_matrix_var_decl& x) const {
        o_ << INDENT2 << "num_params_r__ += ((";
        generate_expression(x.K_,o_);
        o_ << " * (";
        generate_expression(x.K_,o_);
        o_ << " - 1)) / 2)";
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          o_ << " * ";
          generate_expression(x.dims_[i],o_);
        }
        o_ << ";" << EOL;
      }
      // cut-and-paste from next for r
      void generate_increment_i(std::vector<expression> dims) const {
        if (dims.size() == 0) { 
          o_ << INDENT2 << "++num_params_i__;" << EOL;
          return;
        }
        o_ << INDENT2 << "num_params_r__ += ";
        for (size_t i = 0; i < dims.size(); ++i) {
          if (i > 0) o_ << " * ";
          generate_expression(dims[i],o_);
        }
        o_ << ";" << EOL;
      }
      void generate_increment(std::vector<expression> dims) const {
        if (dims.size() == 0) { 
          o_ << INDENT2 << "++num_params_r__;" << EOL;
          return;
        }
        o_ << INDENT2 << "num_params_r__ += ";
        for (size_t i = 0; i < dims.size(); ++i) {
          if (i > 0) o_ << " * ";
          generate_expression(dims[i],o_);
        }
        o_ << ";" << EOL;
      }
      void generate_increment(expression K, 
                              std::vector<expression> dims) const {
        o_ << INDENT2 << "num_params_r__ += ";
        generate_expression(K,o_);
        for (size_t i = 0; i < dims.size(); ++i) {
          o_ << " * ";
          generate_expression(dims[i],o_);
        }
        o_ << ";" << EOL;

      }
      void generate_increment(expression M, expression N, 
                              std::vector<expression> dims) const {
        o_ << INDENT2 << "num_params_r__ += ";
        generate_expression(M,o_);
        o_ << " * ";
        generate_expression(N,o_);
        for (size_t i = 0; i < dims.size(); ++i) {
          o_ << " * ";
          generate_expression(dims[i],o_);
        }
        o_ << ";" << EOL;
      }
    };

    void generate_set_param_ranges(const std::vector<var_decl>& var_decls,
                                   std::ostream& o) {
      o << EOL;
      o << INDENT << "void set_param_ranges() {" << EOL;
      o << INDENT2 << "num_params_r__ = 0U;" << EOL;
      o << INDENT2 << "param_ranges_i__.clear();" << EOL;
      set_param_ranges_visgen vis(o);
      for (size_t i = 0; i < var_decls.size(); ++i)
        boost::apply_visitor(vis,var_decls[i].decl_);
      o << INDENT << "}" << EOL;
    }
   
    void generate_main(const std::string& model_name,
                       std::ostream& out) {
      out << "int main(int argc, const char* argv[]) {" << EOL;
      out << INDENT << "try {" << EOL;
      out << INDENT2 << "stan::gm::nuts_command<" << model_name << "_namespace::" << model_name << ">(argc,argv);" << EOL;
      out << INDENT << "} catch (std::exception& e) {" << EOL;
      out << INDENT2 << "std::cerr << std::endl << \"Exception: \" << e.what() << std::endl;" << EOL;
      out << INDENT2 << "std::cerr << \"Diagnostic information: \" << std::endl << boost::diagnostic_information(e) << std::endl;" << EOL;
      out << INDENT2 << "return -1;" << EOL;
      out << INDENT << "}" << EOL;

      out << "}" << EOL2;
    }

    void generate_cpp(const program& prog, 
                      const std::string& model_name,
                      std::ostream& out) {
      generate_version_comment(out);
      generate_includes(out);
      generate_start_namespace(model_name,out);
      generate_usings(out);
      generate_typedefs(out);
      generate_class_decl(model_name,out);
      generate_private_decl(out);
      generate_member_var_decls(prog.data_decl_,1,out);
      generate_member_var_decls(prog.derived_data_decl_.first,1,out);
      generate_public_decl(out);
      generate_constructor(prog,model_name,out);
      generate_set_param_ranges(prog.parameter_decl_,out);
      generate_init_method(prog.parameter_decl_,out);
      generate_log_prob(prog,out);
      // FIXME: put back
      generate_write_csv_header_method(prog,out);
      generate_write_csv_method(prog,out);
      generate_end_class_decl(out);
      generate_end_namespace(out);
      generate_main(model_name,out);
    }

  }
  
}

#endif

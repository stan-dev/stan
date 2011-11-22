#ifndef __STAN__GM__GENERATOR_HPP__
#define __STAN__GM__GENERATOR_HPP__

#include <boost/lexical_cast.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_numeric.hpp>
#include <boost/spirit/include/classic_position_iterator.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_fusion.hpp>
#include <boost/spirit/include/phoenix_object.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_function.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/spirit/include/support_multi_pass.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <boost/variant/get.hpp>
#include <boost/variant/recursive_variant.hpp>

#include <iomanip>
#include <iostream>
#include <fstream>
#include <istream>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <stan/stan.hpp>
#include <stan/io/dump.hpp>


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

    void generate_indent(unsigned int indent, std::ostream& o) {
      for (unsigned int k = 0; k < indent; ++k)
	o << INDENT;
    }

    /** generic visitor with output for extension */
    struct visgen {
      typedef void result_type;
      std::ostream& o_;
      visgen(std::ostream& o) : o_(o) { }
    };

    struct expression_visgen : public visgen {
      expression_visgen(std::ostream& o) : visgen(o) {  }
      void operator()(nil const& x) const { 
	o_ << "nil";
      }
      void operator()(const int_literal& n) const { o_ << n.val_; }
      void operator()(const double_literal& x) const { o_ << x.val_; }
      void operator()(const identifier& v) const { o_ << v.name_; }
      void operator()(int n) const { o_ << n; }
      void operator()(double x) const { o_ << x; }
      void operator()(const std::string& x) const { o_ << x; } // identifiers
      void operator()(const index_op& x) const {
	boost::apply_visitor(*this, x.expr_.expr_); 
	for (unsigned int i = 0; i < x.dimss_.size(); ++i) {
	  std::vector<expression> indexes = x.dimss_[i];
	  o_ << '[';
	  for (unsigned j = 0; j < indexes.size(); ++j) {
	    if (j > 0) o_ << "][";
	    boost::apply_visitor(*this,indexes[j].expr_);
	    o_ << " - 1";
	  }
	  o_ << ']';
	}
      }
      void operator()(const fun& fx) const { 
	o_ << fx.name_ << '(';
	for (unsigned int i = 0; i < fx.args_.size(); ++i) {
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
      generate_using("stan::io::dump",o);
      generate_using("std::istream",o);
      generate_using_namespace("stan::maths",o);
      o << EOL;
    }

    void generate_typedef(const std::string& type, 
			  const std::string& abbrev, 
			  std::ostream& o) {
      o << "typedef" << " " << type << " " << abbrev << ";" << EOL;
    }

    void generate_typedefs(std::ostream& o) {
      generate_typedef("Eigen::Matrix<double,1,Eigen::Dynamic>","vector_d",o);
      generate_typedef("Eigen::Matrix<double,Eigen::Dynamic,1>","row_vector_d",o);
      generate_typedef("Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>","matrix_d",o);

      generate_typedef("Eigen::Matrix<stan::agrad::var,1,Eigen::Dynamic>","vector_v",o);
      generate_typedef("Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,1>","row_vector_v",o);
      generate_typedef("Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,Eigen::Dynamic>","matrix_v",o);
      // moved to include
      o << EOL;
    }

    void generate_include(const std::string& lib_name, std::ostream& o) {
      o << "#include" << " " << "<" << lib_name << ">" << EOL;
    }
   
    void generate_includes(std::ostream& o) {
      generate_include("cmath",o);
      generate_include("vector",o);
      generate_include("fstream",o);
      generate_include("iostream",o);
      generate_include("stdexcept",o);
      generate_include("sstream",o);
      generate_include("boost/exception/all.hpp",o);
      generate_include("Eigen/Dense",o);
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
	<< stan::MAJOR_VERSION 	<< "." << stan::MINOR_VERSION << EOL2;
    }

    void generate_class_decl(const std::string& model_name,
			std::ostream& o) {
      o << "class " << model_name << " : public prob_grad_ad {"	<< EOL;
    }

    void generate_end_class_decl(std::ostream& o) {
      o << "}; // model" << EOL2;
    }

    void generate_type(const std::string& base_type,
		       const std::vector<expression>& dims,
		       unsigned int end,
		       std::ostream& o) {
      for (unsigned int i = 0; i < end; ++i) o << "std::vector<";
      o << base_type;
      for (unsigned int i = 0; i < end; ++i) {
	if (i > 0) o << ' ';
	o << '>';
      }	
    }

    void generate_initializer(std::ostream& o,
			      const std::string& base_type,
			      const std::vector<expression>& dims,
			      const expression& type_arg1 = expression(),
			      const expression& type_arg2 = expression()) {
      for (unsigned int i = 0; i < dims.size(); ++i) {
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

      for (unsigned int i = 0; i < dims.size(); ++i)
	o << ')';
      o << ';' << EOL;
    }

    void generate_initialization(std::ostream& o,
				 unsigned int indent,
				 const std::string& var_name,
				 const std::string& base_type,
				 const std::string& type_suffix,
				 const std::vector<expression>& dims,
				 const expression& type_arg1 = expression(),
				 const expression& type_arg2 = expression()) {
      std::stringstream base_type_suffix(base_type);
      base_type_suffix << type_suffix;
      generate_indent(indent,o);
      o << var_name << " = ";
      generate_type(base_type,dims,dims.size(),o);
      generate_initializer(o,base_type_suffix.str(),dims,type_arg1,type_arg2);
    }


    struct var_resizing_visgen : public visgen {
      const std::string type_suffix_;
      var_resizing_visgen(std::ostream& o, const std::string& type_suffix) 
	: visgen(o),
	  type_suffix_(type_suffix) {
      }
      void operator()(nil const& x) const { } // dummy
      void operator()(int_var_decl const& x) const {
	generate_initialization(o_,2U,x.name_,"int","",x.dims_);
      }
      void operator()(double_var_decl const& x) const {
	generate_initialization(o_,2U,x.name_,"double","",x.dims_);
      }
      void operator()(vector_var_decl const& x) const {
	generate_initialization(o_,2U,x.name_,"vector_",type_suffix_,x.dims_,x.M_);
      }
      void operator()(row_vector_var_decl const& x) const {
	generate_initialization(o_,2U,x.name_,"row_vector_",type_suffix_,x.dims_,x.N_);
      }
      void operator()(simplex_var_decl const& x) const {
	generate_initialization(o_,2U,x.name_,"vector_",type_suffix_,x.dims_,x.K_);
      }
      void operator()(pos_ordered_var_decl const& x) const {
	generate_initialization(o_,2U,x.name_,"vector_",type_suffix_,x.dims_,x.K_);
      }
      void operator()(matrix_var_decl const& x) const {
	generate_initialization(o_,2U,x.name_,"matrix_",type_suffix_,x.dims_,x.M_,x.N_);
      }
      void operator()(cov_matrix_var_decl const& x) const {
	generate_initialization(o_,2U,x.name_,"matrix",type_suffix_,x.dims_,x.K_,x.K_);
      }
      void operator()(corr_matrix_var_decl const& x) const {
	generate_initialization(o_,2U,x.name_,"matrix",type_suffix_,x.dims_,x.K_,x.K_);
      }
    };

    void generate_var_resizing(const std::string& type_suffix,
			       const std::vector<var_decl>& vs,
			       std::ostream& o) {
      var_resizing_visgen vis(o,type_suffix);
      for (unsigned int i = 0; i < vs.size(); ++i)
	boost::apply_visitor(vis, vs[i].decl_);
    }

    const std::vector<expression> EMPTY_EXP_VECTOR(0);

    // see init_local_var_visgen for cut & paste
    struct init_member_var_visgen : public visgen {
      const bool declare_vars_;
      init_member_var_visgen(bool declare_vars,
			     std::ostream& o)
	: visgen(o),
	  declare_vars_(declare_vars) {
      }
      void operator()(const nil& x) const { }
      void operator()(const int_var_decl& x) const {
	generate_initialize_array("integer",EMPTY_EXP_VECTOR,x.name_,x.dims_);
      }      
      void operator()(const double_var_decl& x) const {
	if (!is_nil(x.range_.low_.expr_)) {
	  if (!is_nil(x.range_.high_.expr_)) {
	    std::vector<expression> read_args;
	    read_args.push_back(x.range_.low_);
	    read_args.push_back(x.range_.high_);
	    generate_initialize_array("double_lub",read_args,x.name_,x.dims_);
	  } else {
	    std::vector<expression> read_args;
	    read_args.push_back(x.range_.low_);
	    generate_initialize_array("double_lb",read_args,x.name_,x.dims_);
	  }
	} else {
	  if (!is_nil(x.range_.high_.expr_)) {
	    std::vector<expression> read_args;
	    read_args.push_back(x.range_.high_);
	    generate_initialize_array("double_ub",read_args,x.name_,x.dims_);
	  } else {
	    generate_initialize_array("double",EMPTY_EXP_VECTOR,x.name_,x.dims_);
	  }
	}
      }
      void operator()(const vector_var_decl& x) const {
	std::vector<expression> read_args;
	read_args.push_back(x.M_);
	generate_initialize_array("vector_d",read_args,x.name_,x.dims_);
      }
      void operator()(const row_vector_var_decl& x) const {
	std::vector<expression> read_args;
	read_args.push_back(x.N_);
	generate_initialize_array("row_vector",read_args,x.name_,x.dims_);
      }
      void operator()(const matrix_var_decl& x) const {
	std::vector<expression> read_args;
	read_args.push_back(x.N_);
	generate_initialize_array("matrix",read_args,x.name_,x.dims_);
      }
      void operator()(const simplex_var_decl& x) const {
	std::vector<expression> read_args;
	read_args.push_back(x.K_);
	generate_initialize_array("simplex",read_args,x.name_,x.dims_);
      }
      void operator()(const pos_ordered_var_decl& x) const {
	std::vector<expression> read_args;
	read_args.push_back(x.K_);
	generate_initialize_array("pos_ordered",read_args,x.name_,x.dims_);
      }
      void operator()(const cov_matrix_var_decl& x) const {
	std::vector<expression> read_args;
	read_args.push_back(x.K_);
	generate_initialize_array("cov_matrix",read_args,x.name_,x.dims_);
      }
      void operator()(const corr_matrix_var_decl& x) const {
	std::vector<expression> read_args;
	read_args.push_back(x.K_);
	generate_initialize_array("corr_matrix",read_args,x.name_,x.dims_);
      }
      void generate_initialize_array(std::string const& type,
				     const std::vector<expression>& read_args,
				     const std::string& name,
				     const std::vector<expression>& dims) const {
	if (dims.size() > 0) {
	  // if == 0, partial eval expressions inside read
	  for (unsigned int j = 0; j < read_args.size(); ++j) {
	    generate_indent(2,o_);
	    o_ << "unsigned int " << "read_arg_" << name << "_" << j << " = ";
	    generate_expression(read_args[j],o_);
	    o_ << ";" << EOL;
	  }
	}
	if (dims.size() == 0) {
	  generate_indent(2,o_);
	  if (declare_vars_) o_ << type << " ";
	  o_ << name << " = in.next_" << type  << "(";
	  for (unsigned int j = 0; j < read_args.size(); ++j) {
	    if (j > 0) o_ << ",";
	    generate_expression(read_args[j],o_);
	  }
	  o_ << ");" << EOL;
	  return;
	}
	if (declare_vars_) {
	  o_ << INDENT2;
	  for (unsigned int i = 0; i < dims.size(); ++i) o_ << "vector<";
	  o_ << type;
	  for (unsigned int i = 0; i < dims.size(); ++i) o_ << "> ";
	  o_ << name << ";" << EOL;
	}
	std::string name_dims(name);
	for (unsigned int i = 0; i < dims.size(); ++i) {
	  generate_indent(i + 2, o_);
	  o_ << "unsigned int dim_"  << name << "_" << i << " = ";
	  generate_expression(dims[i],o_);
	  o_ << ";" << EOL;
	  if (i < dims.size() - 1) {  
	    generate_indent(i + 2, o_);
	    o_ << name_dims << ".resize(dim" << "_" << name << "_" << i << ");" 
	       << EOL;
	    name_dims.append("[k_").append(to_string(i)).append("]");
	  }
	  generate_indent(i + 2, o_);
	  o_ << "for (unsigned int k_" << i << " = 0;"
	     << " k_" << i << " < dim_" << name << "_" << i << ";"
	     << " ++k_" << i << ") {" << EOL;
	  if (i == dims.size() - 1) {
	    generate_indent(i + 3, o_);
	    o_ << name_dims << ".push_back(in.next_" << type << "(";
	    for (unsigned int j = 0; j < read_args.size(); ++j) {
	      if (j > 0) o_ << ",";
	      o_ << "read_arg_" << name << "_" << j;
	    }
	    o_ << "));" << EOL;
	  }
	}
	for (unsigned int i = dims.size(); i > 0; --i) {
	  generate_indent(i + 1, o_);
	  o_ << "}" << EOL;
	}
      }
    };

    void generate_member_var_inits(const std::vector<var_decl>& vs,
				   bool declare_vars,
				   std::ostream& o) {
      init_member_var_visgen vis(declare_vars,o);
      for (unsigned int i = 0; i < vs.size(); ++i)
	boost::apply_visitor(vis, vs[i].decl_);
    }

    // see init_member_var_visgen for cut & paste
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
	  for (unsigned int j = 0; j < read_args.size(); ++j) {
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
	  for (unsigned int i = 0; i < dims.size(); ++i) o_ << "vector<";
	  o_ << var_type;
	  for (unsigned int i = 0; i < dims.size(); ++i) o_ << "> ";
	  o_ << name << ";" << EOL;
	}
	std::string name_dims(name);
	for (unsigned int i = 0; i < dims.size(); ++i) {
	  generate_indent(i + 2, o_);
	  o_ << "unsigned int dim_"  << name << "_" << i << " = ";
	  generate_expression(dims[i],o_);
	  o_ << ";" << EOL;
	  if (i < dims.size() - 1) {  
	    generate_indent(i + 2, o_);
	    o_ << name_dims << ".resize(dim" << "_" << name << "_" << i << ");" 
	       << EOL;
	    name_dims.append("[k_").append(to_string(i)).append("]");
	  }
	  generate_indent(i + 2, o_);
	  o_ << "for (unsigned int k_" << i << " = 0;"
	     << " k_" << i << " < dim_" << name << "_" << i << ";"
	     << " ++k_" << i << ") {" << EOL;
	  if (i == dims.size() - 1) {
	    generate_indent(i + 3, o_);
	    o_ << name_dims << ".push_back(in__." << read_type << "_constrain(";
	    for (unsigned int j = 0; j < read_args.size(); ++j) {
	      if (j > 0) o_ << ",";
	      generate_expression(read_args[j],o_);
	    }
	    if (read_args.size() > 0)
	      o_ << ",";
	    o_ << "lp__";
	    o_ << "));" << EOL;
	  }
	}
	for (unsigned int i = dims.size(); i > 0; --i) {
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
      for (unsigned int i = 0; i < vs.size(); ++i)
	boost::apply_visitor(vis, vs[i].decl_);
    }



    void generate_constructor(const program& p,
			      const std::string& model_name,
			      std::ostream& o) {
      o << INDENT << model_name	<< "(vector<double> data_r__, vector<int> data_i__)"
	<< EOL;
      o << INDENT2 << ": prob_grad_ad::prob_grad_ad(0) {" << EOL;
      o << INDENT2 << "stan::io::reader<double> in(data_r__,data_i__);" << EOL;
      generate_member_var_inits(p.data_decl_,false,o);

      o << INDENT << "} // vector ctor" << EOL2;
    }

    void generate_public_decl(std::ostream& o) {
      o << "public:" << EOL;
    }

    void generate_private_decl(std::ostream& o) {
      o << "private:" << EOL;
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
			 unsigned int size) const {
	for (int i = 0; i < indents_; ++i)
	  o_ << INDENT;
	for (unsigned int i = 0; i < size; ++i) {
	  o_ << "vector<";
	}
	o_ << type;
	if (size > 0) {
	  o_ << ">";
	}
	for (unsigned int i = 1; i < size; ++i) {
	  o_ << " >";
	}
	o_ << " " << name << ";" << EOL;
      }
    };

    void generate_member_var_decls(std::vector<var_decl> const& vs,
				   int indent,
				   std::ostream& o) {
      member_var_decl_visgen vis(indent,o);
      for (unsigned int i = 0; i < vs.size(); ++i)
	boost::apply_visitor(vis,vs[i].decl_);
    }

    // see member_var_decl_visgen cut & paste
    struct local_var_decl_visgen : public visgen {
      int indents_;
      local_var_decl_visgen(int indents,
			     std::ostream& o)
	: visgen(o),
	  indents_(indents) {
      }
      void operator()(nil const& x) const { }
      void operator()(int_var_decl const& x) const {
	declare_array("int",x.name_,x.dims_);
      }
      void operator()(double_var_decl const& x) const {
	declare_array("var",x.name_,x.dims_);
      }
      void operator()(vector_var_decl const& x) const {
	declare_array("vector_v", x.name_, x.dims_);
      }
      void operator()(row_vector_var_decl const& x) const {
	declare_array("row_vector_v", x.name_, x.dims_);
      }
      void operator()(matrix_var_decl const& x) const {
	declare_array("matrix_v", x.name_, x.dims_);
      }
      void operator()(simplex_var_decl const& x) const {
	declare_array("vector_v", x.name_, x.dims_);
      }
      void operator()(pos_ordered_var_decl const& x) const {
	declare_array("vector_v", x.name_, x.dims_);
      }
      void operator()(cov_matrix_var_decl const& x) const {
	declare_array("matrix_v", x.name_, x.dims_);
      }
      void operator()(corr_matrix_var_decl const& x) const {
	declare_array("matrix_v", x.name_, x.dims_);
      }
      void declare_array(const std::string& type, 
			 const std::string& name, 
			 const std::vector<expression>& dims) const {
	for (int i = 0; i < indents_; ++i)
	  o_ << INDENT;
	for (unsigned int i = 0; i < dims.size(); ++i) {
	  o_ << "vector<";
	}
	o_ << type;
	for (unsigned int i = 0; i < dims.size(); ++i) {
	  if (i > 0) o_ << " ";
	  o_ << ">";
	}
	o_ << " "  << name;
	if (dims.size() > 0)
	  o_ << "(";
	for (unsigned int i = 0; i < dims.size(); ++i) {
	  if (i > 0)
	    o_ << ", vector<" << type << ">(";
	  generate_expression(dims[i].expr_,o_);
	}
	for (unsigned int i = 0; i < dims.size(); ++i)
	  o_ << ")";
	o_ << ";" << EOL;
      }
    };

    void generate_local_var_decls(std::vector<var_decl> const& vs,
				  int indent,
				  std::ostream& o) {
       local_var_decl_visgen vis(indent,o);
       for (unsigned int i = 0; i < vs.size(); ++i)
	 boost::apply_visitor(vis,vs[i].decl_);
     }


     void generate_start_namespace(std::string name,
				   std::ostream& o) {
       o << "namespace "	<< name << "_namespace {" << EOL2;
     }

     void generate_end_namespace(std::ostream& o) {
       o << "} // namespace" << EOL2;
     }

     void generate_comment(std::string const& msg, int indent, 
			   std::ostream& o) {
       generate_indent(indent,o);
       o << "// " << msg	<< EOL;
     }

     void generate_var(var const& x, std::ostream& o) {
       o << x.name_;
       if (x.dims_.size() == 0) return;
       o << '[';
       for (unsigned int i = 0; i < x.dims_.size(); ++i) {
	 if (i > 0) o << "][";
	 generate_expression(x.dims_[i],o);
	o << " - 1";
      }
      o << ']';
    }

    void generate_statement(statement const& s, int indent, std::ostream& o);

    struct statement_visgen : public visgen {
      unsigned int indent_;
      statement_visgen(unsigned int indent, std::ostream& o)
	: visgen(o),
	  indent_(indent) {
      }
      void operator()(nil const& x) const { }
      void operator()(assignment const& x) const {
	generate_indent(indent_,o_);
	generate_var(x.var_,o_);
	o_ << " = ";
	generate_expression(x.expr_,o_);
	o_ << ";" << EOL;
      }
      void operator()(sample const& x) const {
	generate_indent(indent_,o_);
	o_ << "lp__ += stan::prob::" << x.dist_.family_ << "_log(";
	generate_var(x.v_,o_);
	for (unsigned int i = 0; i < x.dist_.args_.size(); ++i) {
	  o_ << ", ";
	  generate_expression(x.dist_.args_[i],o_);
	}
	o_ << ");" << EOL;
      }
      void operator()(statements const& x) const {
	for (unsigned int i = 0; i < x.statements_.size(); ++i)
	  generate_statement(x.statements_[i],indent_,o_);
      }
      void operator()(for_statement const& x) const {
	generate_indent(indent_,o_);
	o_ << "for (int " << x.variable_ << " = ";
	generate_expression(x.range_.low_,o_);
	o_ << "; " << x.variable_ << " <= ";
	generate_expression(x.range_.high_,o_);
	o_ << "; ++" << x.variable_ << ") {" << EOL;
	generate_statement(x.statement_, indent_ + 1, o_);
	generate_indent(indent_,o_);
	o_ << "}" << EOL;
      }
    };

    void generate_statement(statement const& s,
			    int indent,
			    std::ostream& o) {
      statement_visgen vis(indent,o);
      boost::apply_visitor(vis, s.statement_);
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

      generate_comment("derived variables",2,o);
      generate_local_var_decls(p.derived_decl_.first,2,o);
      o << EOL;
      generate_statement(p.derived_decl_.second,2,o);
      o << EOL;

      generate_comment("model body",2,o);
      generate_statement(p.statement_,2,o);
      o << EOL;
      o << INDENT2 << "return lp__;" << EOL2;
      o << INDENT << "} // log_prob()" << EOL2;
    }

    struct dump_member_var_visgen : public visgen {
      var_resizing_visgen var_resizer_;
      dump_member_var_visgen(std::ostream& o) 
	: visgen(o),
	  var_resizer_(var_resizing_visgen(o,"d")) {
      }
      void operator()(nil const& x) const { } // dummy
      void operator()(int_var_decl const& x) const {
	std::vector<expression> dims = x.dims_;
	var_resizer_(x);
	o_ << INDENT2 << "if (!context__.contains_i(\"" << x.name_ << "\"))" << EOL;
	o_ << INDENT3 << "throw std::runtime_error(\"variable " << x.name_ <<" not found.\");" << EOL;
	o_ << INDENT2 << "vals_i__ = context__.vals_i(\"" << x.name_ << "\");" << EOL;
	o_ << INDENT2 << "pos__ = 0;" << EOL;
	unsigned int indentation = 1;
	for (unsigned int dim_up = 0U; dim_up < dims.size(); ++dim_up) {
	  unsigned int dim = dims.size() - dim_up - 1U;
	  ++indentation;
	  generate_indent(indentation,o_);
	  o_ << "unsigned int " << x.name_ << "_limit_" << dim << "__ = ";
	  generate_expression(dims[dim],o_);
	  o_ << ";" << EOL;
	  generate_indent(indentation,o_);
	  o_ << "for (unsigned int i_" << dim << "__ = 0; i_" 
	     << dim << "__ < " << x.name_ << "_limit_" << dim 
	     << "__; ++i_" << dim << "__) {" << EOL;
	}
	generate_indent(indentation+1,o_);
	o_ << x.name_;
	for (unsigned int dim = 0; dim < dims.size(); ++dim)
	  o_ << "[i_" << dim << "__]";
	o_ << " = vals_i__[pos__++];" << EOL;
	for (unsigned int dim = 0; dim < dims.size(); ++dim) {
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
	unsigned int indentation = 1;
	for (unsigned int dim_up = 0U; dim_up < dims.size(); ++dim_up) {
	  unsigned int dim = dims.size() - dim_up - 1U;
	  ++indentation;
	  generate_indent(indentation,o_);
	  o_ << "unsigned int " << x.name_ << "_limit_" << dim << "__ = ";
	  generate_expression(dims[dim],o_);
	  o_ << ";" << EOL;
	  generate_indent(indentation,o_);
	  o_ << "for (unsigned int i_" << dim << "__ = 0; i_" << dim << "__ < " << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {" << EOL;
	}
	generate_indent(indentation+1,o_);
	o_ << x.name_;
	for (unsigned int dim = 0; dim < dims.size(); ++dim)
	  o_ << "[i_" << dim << "__]";
	o_ << " = vals_r__[pos__++];" << EOL;
	for (unsigned int dim = 0; dim < dims.size(); ++dim) {
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
	o_ << INDENT2 << "unsigned int " << x.name_ << "_i_vec_lim__ = ";
	generate_expression(x.M_,o_);
	o_ << ";" << EOL;
	o_ << INDENT2 << "for (unsigned int " << "i_vec__ = 0; " << "i_vec__ < " << x.name_ << "_i_vec_lim__; ++i_vec__) {" << EOL;
	unsigned int indentation = 2;
	for (unsigned int dim_up = 0U; dim_up < dims.size(); ++dim_up) {
	  unsigned int dim = dims.size() - dim_up - 1U;
	  ++indentation;
	  generate_indent(indentation,o_);
	  o_ << "unsigned int " << x.name_ << "_limit_" << dim << "__ = ";
	  generate_expression(dims[dim],o_);
	  o_ << ";" << EOL;
	  generate_indent(indentation,o_);
	  o_ << "for (unsigned int i_" << dim << "__ = 0; i_" << dim << "__ < " << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {" << EOL;
	}
	generate_indent(indentation+1,o_);
	o_ << x.name_;
	for (unsigned int dim = 0; dim < dims.size(); ++dim)
	  o_ << "[i_" << dim << "__]";
	o_ << "[i_vec__]";
	o_ << " = vals_r__[pos__++];" << EOL;
	for (unsigned int dim = 0; dim < dims.size(); ++dim) {
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
	o_ << INDENT2 << "unsigned int " << x.name_ << "_i_vec_lim__ = ";
	generate_expression(x.N_,o_);
	o_ << ";" << EOL;
	o_ << INDENT2 << "for (unsigned int " << "i_vec__ = 0; " << "i_vec__ < " << x.name_ << "_i_vec_lim__; ++i_vec__) {" << EOL;
	unsigned int indentation = 2;
	for (unsigned int dim_up = 0U; dim_up < dims.size(); ++dim_up) {
	  unsigned int dim = dims.size() - dim_up - 1U;
	  ++indentation;
	  generate_indent(indentation,o_);
	  o_ << "unsigned int " << x.name_ << "_limit_" << dim << "__ = ";
	  generate_expression(dims[dim],o_);
	  o_ << ";" << EOL;
	  generate_indent(indentation,o_);
	  o_ << "for (unsigned int i_" << dim << "__ = 0; i_" << dim << "__ < " << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {" << EOL;
	}
	generate_indent(indentation+1,o_);
	o_ << x.name_;
	for (unsigned int dim = 0; dim < dims.size(); ++dim)
	  o_ << "[i_" << dim << "__]";
	o_ << "[i_vec__]";
	o_ << " = vals_r__[pos__++];" << EOL;
	for (unsigned int dim = 0; dim < dims.size(); ++dim) {
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
	o_ << INDENT2 << "unsigned int " << x.name_ << "_i_vec_lim__ = ";
	generate_expression(x.K_,o_);
	o_ << ";" << EOL;
	o_ << INDENT2 << "for (unsigned int " << "i_vec__ = 0; " << "i_vec__ < " << x.name_ << "_i_vec_lim__; ++i_vec__) {" << EOL;
	unsigned int indentation = 2;
	for (unsigned int dim_up = 0U; dim_up < dims.size(); ++dim_up) {
	  unsigned int dim = dims.size() - dim_up - 1U;
	  ++indentation;
	  generate_indent(indentation,o_);
	  o_ << "unsigned int " << x.name_ << "_limit_" << dim << "__ = ";
	  generate_expression(dims[dim],o_);
	  o_ << ";" << EOL;
	  generate_indent(indentation,o_);
	  o_ << "for (unsigned int i_" << dim << "__ = 0; i_" << dim << "__ < " << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {" << EOL;
	}
	generate_indent(indentation+1,o_);
	o_ << x.name_;
	for (unsigned int dim = 0; dim < dims.size(); ++dim)
	  o_ << "[i_" << dim << "__]";
	o_ << "[i_vec__]";
	o_ << " = vals_r__[pos__++];" << EOL;
	for (unsigned int dim = 0; dim < dims.size(); ++dim) {
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
	o_ << INDENT2 << "unsigned int " << x.name_ << "_i_vec_lim__ = ";
	generate_expression(x.K_,o_);
	o_ << ";" << EOL;
	o_ << INDENT2 << "for (unsigned int " << "i_vec__ = 0; " << "i_vec__ < " << x.name_ << "_i_vec_lim__; ++i_vec__) {" << EOL;
	unsigned int indentation = 2;
	for (unsigned int dim_up = 0U; dim_up < dims.size(); ++dim_up) {
	  unsigned int dim = dims.size() - dim_up - 1U;
	  ++indentation;
	  generate_indent(indentation,o_);
	  o_ << "unsigned int " << x.name_ << "_limit_" << dim << "__ = ";
	  generate_expression(dims[dim],o_);
	  o_ << ";" << EOL;
	  generate_indent(indentation,o_);
	  o_ << "for (unsigned int i_" << dim << "__ = 0; i_" << dim << "__ < " << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {" << EOL;
	}
	generate_indent(indentation+1,o_);
	o_ << x.name_;
	for (unsigned int dim = 0; dim < dims.size(); ++dim)
	  o_ << "[i_" << dim << "__]";
	o_ << "[i_vec__]";
	o_ << " = vals_r__[pos__++];" << EOL;
	for (unsigned int dim = 0; dim < dims.size(); ++dim) {
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
	o_ << INDENT2 << "unsigned int " << x.name_ << "_m_mat_lim__ = ";
	generate_expression(x.M_,o_);
	o_ << ";" << EOL;
	o_ << INDENT2 << "unsigned int " << x.name_ << "_n_mat_lim__ = ";
	generate_expression(x.N_,o_);
	o_ << ";" << EOL;
	o_ << INDENT2 << "for (unsigned int " << "n_mat__ = 0; " << "n_mat__ < " << x.name_ << "_n_mat_lim__; ++n_mat__) {" << EOL;
	o_ << INDENT3 << "for (unsigned int " << "m_mat__ = 0; " << "m_mat__ < " << x.name_ << "_m_mat_lim__; ++m_mat__) {" << EOL;
	unsigned int indentation = 3;
	for (unsigned int dim_up = 0U; dim_up < dims.size(); ++dim_up) {
	  unsigned int dim = dims.size() - dim_up - 1U;
	  ++indentation;
	  generate_indent(indentation,o_);
	  o_ << "unsigned int " << x.name_ << "_limit_" << dim << "__ = ";
	  generate_expression(dims[dim],o_);
	  o_ << ";" << EOL;
	  generate_indent(indentation,o_);
	  o_ << "for (unsigned int i_" << dim << "__ = 0; i_" << dim << "__ < " << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {" << EOL;
	}
	generate_indent(indentation+1,o_);
	o_ << x.name_;
	for (unsigned int dim = 0; dim < dims.size(); ++dim)
	  o_ << "[i_" << dim << "__]";
	o_ << "(m_mat__,n_mat__)";
	o_ << " = vals_r__[pos__++];" << EOL;
	for (unsigned int dim = 0; dim < dims.size(); ++dim) {
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
	o_ << INDENT2 << "unsigned int " << x.name_ << "_k_mat_lim__ = ";
	generate_expression(x.K_,o_);
	o_ << ";" << EOL;
	o_ << INDENT2 << "for (unsigned int " << "n_mat__ = 0; " << "n_mat__ < " << x.name_ << "_k_mat_lim__; ++n_mat__) {" << EOL;
	o_ << INDENT3 << "for (unsigned int " << "m_mat__ = 0; " << "m_mat__ < " << x.name_ << "_k_mat_lim__; ++m_mat__) {" << EOL;
	unsigned int indentation = 3;
	for (unsigned int dim_up = 0U; dim_up < dims.size(); ++dim_up) {
	  unsigned int dim = dims.size() - dim_up - 1U;
	  ++indentation;
	  generate_indent(indentation,o_);
	  o_ << "unsigned int " << x.name_ << "_limit_" << dim << "__ = ";
	  generate_expression(dims[dim],o_);
	  o_ << ";" << EOL;
	  generate_indent(indentation,o_);
	  o_ << "for (unsigned int i_" << dim << "__ = 0; i_" << dim << "__ < " << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {" << EOL;
	}
	generate_indent(indentation+1,o_);
	o_ << x.name_;
	for (unsigned int dim = 0; dim < dims.size(); ++dim)
	  o_ << "[i_" << dim << "__]";
	o_ << "(m_mat__,n_mat__)";
	o_ << " = vals_r__[pos__++];" << EOL;
	for (unsigned int dim = 0; dim < dims.size(); ++dim) {
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
	o_ << INDENT2 << "unsigned int " << x.name_ << "_k_mat_lim__ = ";
	generate_expression(x.K_,o_);
	o_ << ";" << EOL;
	o_ << INDENT2 << "for (unsigned int " << "n_mat__ = 0; " << "n_mat__ < " << x.name_ << "_k_mat_lim__; ++n_mat__) {" << EOL;
	o_ << INDENT3 << "for (unsigned int " << "m_mat__ = 0; " << "m_mat__ < " << x.name_ << "_k_mat_lim__; ++m_mat__) {" << EOL;
	unsigned int indentation = 3;
	for (unsigned int dim_up = 0U; dim_up < dims.size(); ++dim_up) {
	  unsigned int dim = dims.size() - dim_up - 1U;
	  ++indentation;
	  generate_indent(indentation,o_);
	  o_ << "unsigned int " << x.name_ << "_limit_" << dim << "__ = ";
	  generate_expression(dims[dim],o_);
	  o_ << ";" << EOL;
	  generate_indent(indentation,o_);
	  o_ << "for (unsigned int i_" << dim << "__ = 0; i_" << dim << "__ < " << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {" << EOL;
	}
	generate_indent(indentation+1,o_);
	o_ << x.name_;
	for (unsigned int dim = 0; dim < dims.size(); ++dim)
	  o_ << "[i_" << dim << "__]";
	o_ << "(m_mat__,n_mat__)";
	o_ << " = vals_r__[pos__++];" << EOL;
	for (unsigned int dim = 0; dim < dims.size(); ++dim) {
	  generate_indent(dims.size() + 2 - dim,o_);
	  o_ << "}" << EOL;
	}
	o_ << INDENT3 << "}" << EOL;
	o_ << INDENT2 << "}" << EOL;
      }
    };

    void generate_dump_member_var_inits(const std::vector<var_decl>& vs,
					std::ostream& o) {
      dump_member_var_visgen vis(o);
      for (unsigned int i = 0; i < vs.size(); ++i)
	boost::apply_visitor(vis, vs[i].decl_);
    }

    void generate_dump_constructor(const program& prog,
				   const std::string& model_name,
				   std::ostream& o) {
      o << INDENT << model_name << "(stan::io::var_context& context__)" << EOL;
      o << INDENT2 << ": prob_grad_ad::prob_grad_ad(0) {" << EOL; // FIXME: need size of params_r here
      o << INDENT2 << "unsigned int pos__;" << EOL;
      o << INDENT2 << "std::vector<int> vals_i__;" << EOL;
      o << INDENT2 << "std::vector<double> vals_r__;" << EOL;

      generate_dump_member_var_inits(prog.data_decl_,o);

      generate_var_resizing("d",prog.derived_data_decl_.first, o);
      o << EOL;
      for (unsigned int i = 0; i < prog.derived_data_decl_.second.size(); ++i)
	generate_statement(prog.derived_data_decl_.second[i],2,o);

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
			      unsigned int num_dims) const {
	o_ << name;
	for (unsigned int i = 0; i < num_dims; ++i)
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
      void generate_indent_num_dims(unsigned int base_indent,
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
	unsigned int size = dims.size();
	bool is_matrix = !is_nil(dim1) && !is_nil(dim2);
	bool is_vector = !is_nil(dim1) && is_nil(dim2);
	int extra_indent = is_matrix ? 2U : is_vector ? 1U : 0U;
	if (is_matrix) {
	  generate_indent(indent,o_);
	  o_ << "for (unsigned int j2__ = 0U; j2__ < ";
	  generate_expression(dim2.expr_,o_);
	  o_ << "; ++j2__)" << EOL;

	  generate_indent(indent+1,o_);
	  o_ << "for (unsigned int j1__ = 0U; j1__ < ";
	  generate_expression(dim1.expr_,o_);
	  o_ << "; ++j1__)" << EOL;
	} else if (is_vector) {
	  generate_indent(indent,o_);
	  o_ << "for (unsigned int j1__ = 0U; j1__ < ";
	  generate_expression(dim1.expr_,o_);
	  o_ << "; ++j1__)" << EOL;
	}
	for (unsigned int i = 0; i < size; ++i) {
	  unsigned int idx = size - i - 1;
	  generate_indent(i + indent + extra_indent, o_);
	  o_ << "for (unsigned int i" << idx << "__ = 0U; i" << idx << "__ < ";
	  generate_expression(dims[idx].expr_,o_);
	  o_ << "; ++i" << idx << "__)" << EOL;
	}
	generate_indent_num_dims(2U,dims,dim1,dim2);
	o_ << name; 
	for (unsigned int i = 0; i < dims.size(); ++i)
	  o_ << "[i" << i << "__]";
	if (is_matrix) 
	  o_ << "(j1__,j2__)";
	else if (is_vector)
	  o_ << "(j1__)";
	o_ << " = vals_" << base_type << "__[pos__++];" << EOL;
      }
      void generate_dims_loop_fwd(const std::vector<expression>& dims, 
				  int indent = 2U) const {
	unsigned int size = dims.size();
	for (unsigned int i = 0; i < size; ++i) {
	  generate_indent(i + indent, o_);
	  o_ << "for (unsigned int i" << i << "__ = 0U; i" << i << "__ < ";
	  generate_expression(dims[i].expr_,o_);
	  o_ << "; ++i" << i << "__)" << EOL;
	}
	generate_indent(2U + dims.size(),o_);
      }
      void generate_check_int(const std::string& name, unsigned int n) const {
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
      void generate_check_double(const std::string& name, unsigned int n) const {
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
      o << INDENT2 << "unsigned int pos__;" << EOL;
      o << INDENT2 << "std::vector<double> vals_r__;" << EOL;
      o << INDENT2 << "std::vector<int> vals_i__;" << EOL;
      o << EOL;
      generate_init_visgen vis(o);
      for (unsigned int i = 0; i < vs.size(); ++i)
	boost::apply_visitor(vis, vs[i].decl_);
      o << INDENT << "}" << EOL;
    }


    // see write_csv_visgen for similar structure
    struct write_csv_header_visgen : public visgen {
      write_csv_header_visgen(std::ostream& o)
	: visgen(o) {
      }
      void operator()(const nil& x) const { }
      void operator()(const int_var_decl& x) const {
	generate_csv_header_array(EMPTY_EXP_VECTOR,x.name_,x.dims_);
      }
      void operator()(const double_var_decl& x) const {
	generate_csv_header_array(EMPTY_EXP_VECTOR,x.name_,x.dims_);
      }
      void operator()(const vector_var_decl& x) const {
	std::vector<expression> read_args;
	read_args.push_back(x.M_);
	generate_csv_header_array(read_args,x.name_,x.dims_);
      }
      void operator()(const row_vector_var_decl& x) const {
	std::vector<expression> read_args;
	read_args.push_back(x.N_);
	generate_csv_header_array(read_args,x.name_,x.dims_);
      }
      void operator()(const matrix_var_decl& x) const {
	std::vector<expression> read_args;
	read_args.push_back(x.M_);
	read_args.push_back(x.N_);
	generate_csv_header_array(read_args,x.name_,x.dims_);
      }
      void operator()(const simplex_var_decl& x) const {
	std::vector<expression> read_args;
	read_args.push_back(x.K_);
	generate_csv_header_array(read_args,x.name_,x.dims_);
      }
      void operator()(const pos_ordered_var_decl& x) const {
	std::vector<expression> read_args;
	read_args.push_back(x.K_);
	generate_csv_header_array(read_args,x.name_,x.dims_);
      }
      void operator()(const cov_matrix_var_decl& x) const {
	std::vector<expression> read_args;
	read_args.push_back(x.K_);
	generate_csv_header_array(read_args,x.name_,x.dims_);
      }
      void operator()(const corr_matrix_var_decl& x) const {
	std::vector<expression> read_args;
	read_args.push_back(x.K_);
	generate_csv_header_array(read_args,x.name_,x.dims_);
      }
      void generate_csv_header_array(const std::vector<expression>& arg_dims, 
				     const std::string& name,
				     const std::vector<expression>& dims) const {
	unsigned int size = dims.size() + arg_dims.size();
	for (unsigned int i = 0; i < size; ++i) {
	  o_ << INDENT2 << "unsigned int bound_" << name << '_' << i << " = ";
	  generate_expression(dims[i],o_);
	  o_ << ';' << EOL;
	}
	for (unsigned int i = 0; i < size; ++i) {
	  generate_indent(2 + i,o_);
	  o_ << "for (unsigned int i_" << name << '_' << i << " = 0; ";
	  o_ << "i_" << name << '_' << i << " < bound_" << name << '_' << i << "; ";
	  o_ << "++i_" << name << '_' << i << ") {" << EOL; 
	}
	generate_indent(2 + size, o_);
	o_ << "stringstream ss_" << name << "__;" << EOL;
	generate_indent(2 + size, o_);
	o_ << "ss_" << name << "__" << " << " << '"' << name << '[' << '"';
	for (unsigned int i = 0; i < size; ++i)
	  o_ << " << i_" << name << '_' << i << " << ','";
	o_ << " << ']';" << EOL;
	generate_indent(2 + size, o_);
	o_ << "writer__.write(ss_" << name << "__.str());" << EOL;
	for (unsigned int i = 0; i < size; ++i) {
	  generate_indent(1 + size - i, o_);
	  o_ << '}' << EOL;
	}
      }
    };

    void generate_write_csv_header_method(const std::vector<var_decl>& var_decls,
					  std::ostream& o) {
      o << INDENT << "void write_csv_header(std::ostream& o__) {" << EOL;
      o << INDENT2 << "stan::io::csv_writer writer__(o__);" << EOL;
      write_csv_header_visgen vis(o);
      for (unsigned int i = 0; i < var_decls.size(); ++i)
	boost::apply_visitor(vis,var_decls[i].decl_);
      o << INDENT << "}" << EOL2;
    }

 // see init_member_var_visgen for cut & paste
    struct write_csv_visgen : public visgen {
      write_csv_visgen(std::ostream& o)
	: visgen(o) {
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
	    generate_initialize_array("double","scalar_lub",read_args,x.name_,x.dims_);
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
				     const std::vector<expression>& dims) 
	const {

	if (dims.size() == 0) {
	  generate_indent(2,o_);
	  o_ << var_type << " ";
	  o_ << name << " = in__." << read_type  << "_constrain(";
	  for (unsigned int j = 0; j < read_args.size(); ++j) {
	    if (j > 0) o_ << ",";
	    generate_expression(read_args[j],o_);
	  }
	  o_ << ");" << EOL;
	  o_ << INDENT2 << "writer__.write(" << name << ");" << EOL;
	  return;
	}
	o_ << INDENT2;
	for (unsigned int i = 0; i < dims.size(); ++i) o_ << "vector<";
	o_ << var_type;
	for (unsigned int i = 0; i < dims.size(); ++i) o_ << "> ";
	o_ << name << ";" << EOL;
	std::string name_dims(name);
	for (unsigned int i = 0; i < dims.size(); ++i) {
	  generate_indent(i + 2, o_);
	  o_ << "unsigned int dim_"  << name << "_" << i << " = ";
	  generate_expression(dims[i],o_);
	  o_ << ";" << EOL;
	  if (i < dims.size() - 1) {  
	    generate_indent(i + 2, o_);
	    o_ << name_dims << ".resize(dim" << "_" << name << "_" << i << ");" 
	       << EOL;
	    name_dims.append("[k_").append(to_string(i)).append("]");
	  }
	  generate_indent(i + 2, o_);
	  o_ << "for (unsigned int k_" << i << " = 0;"
	     << " k_" << i << " < dim_" << name << "_" << i << ";"
	     << " ++k_" << i << ") {" << EOL;
	  if (i == dims.size() - 1) {
	    generate_indent(i + 3, o_);
	    o_ << name_dims << ".push_back(in__." << read_type << "_constrain(";
	    for (unsigned int j = 0; j < read_args.size(); ++j) {
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
	  for (unsigned int i = 0; i < dims.size(); ++i) {
	    if (i > 0) o_ << "][";
	    o_ << "k_" << i;
	  }
	  o_ << ']';
	}
	o_ << ");" << EOL;
	
	for (unsigned int i = dims.size(); i > 0; --i) {
	  generate_indent(i + 1, o_);
	  o_ << "}" << EOL;
	}
      }
    };


    void generate_write_csv_method(const std::vector<var_decl>& var_decls,
				   std::ostream& o) {
      o << INDENT << "void write_csv(std::vector<double>& params_r__," << EOL;
      o << INDENT << "               std::vector<int>& params_i__," << EOL;
      o << INDENT << "               std::ostream& o__) {" << EOL;
      o << INDENT2 << "stan::io::reader<double> in__(params_r__,params_i__);" << EOL;
      o << INDENT2 << "stan::io::csv_writer writer__(o__);" << EOL;
      write_csv_visgen vis(o);
      for (unsigned int i = 0; i < var_decls.size(); ++i)
	boost::apply_visitor(vis,var_decls[i].decl_);
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
	// o_ << INDENT2 << "if (0 != ";
	// generate_expression(x.range_.low_,o_);
	// o_ << ")" << EOL;
	// o_ << INDENT3 << "throw std::runtime_error(\"param_ranges error\");";
	// for (int i = 0; i < x.dims_.size(); ++i) {
	//   generate_indent(i + 2, o_);
	//   o_ << "for (unsigned int i_" << i << "__ = 0; ";
	//   o_ << "i_" << i << "__ < ";
	//   generate_expression(x.dims_[i],o_);
	//   o_ << "; ++i_" << i << "__) {" << EOL;
	// }

	// generate_indent(x.dims_.size() + 2,o_);
	// o_ << "param_ranges_i__.push_back(";
	// generate_expression(x.range_.high_,o_);
	// o_ << ");" << EOL;

	// for (int i = 0; i < x.dims_.size(); ++i) {
	//   generate_indent(x.dims_.size() + 1 - i, o_);
	//   o_ << "}" << EOL;
	// }
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
	for (unsigned int i = 0; i < x.dims_.size(); ++i) {
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
	for (unsigned int i = 0; i < x.dims_.size(); ++i) {
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
	for (unsigned int i = 0; i < x.dims_.size(); ++i) {
	  o_ << " * ";
	  generate_expression(x.dims_[i],o_);
	}
	o_ << ";" << EOL;
      }
      void generate_increment(std::vector<expression> dims) const {
	if (dims.size() == 0) { 
	  o_ << INDENT2 << "++num_params_r__;" << EOL;
	  return;
	}
	o_ << INDENT2 << "num_params_r__ += ";
	for (unsigned int i = 0; i < dims.size(); ++i) {
	  if (i > 0) o_ << " * ";
	  generate_expression(dims[i],o_);
	}
	o_ << ";" << EOL;
      }
      void generate_increment(expression K, 
			      std::vector<expression> dims) const {
	o_ << INDENT2 << "num_params_r__ += ";
	generate_expression(K,o_);
	for (unsigned int i = 0; i < dims.size(); ++i) {
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
	for (unsigned int i = 0; i < dims.size(); ++i) {
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
      for (unsigned int i = 0; i < var_decls.size(); ++i)
	boost::apply_visitor(vis,var_decls[i].decl_);
      o << INDENT << "}" << EOL;
    }
   
    void generate_main(const std::string& model_name,
		       std::ostream& out) {
      out << "int main(int argc__, const char* argv__[]) {" << EOL;
      out << INDENT << "try {" << EOL;
      out << INDENT2 << "stan::io::cmd_line cmd__(argc__,argv__);" << EOL;
      out << INDENT2 << "std::string data_file_path__;" << EOL;
      out << INDENT2 << "cmd__.val(\"data_file\",data_file_path__);" << EOL;
      out << INDENT2 << "std::fstream data_file__(data_file_path__.c_str(),std::fstream::in);" << EOL;
      out << INDENT2 << "stan::io::dump dump__(data_file__);" << EOL;
      
      out << INDENT2 << model_name << "_namespace::" << model_name << " model__(dump__);" << EOL;

      out << INDENT2 << "data_file__.close();" << EOL;
      out << INDENT2 << "stan::gm::nuts_command(cmd__,model__);" << EOL;

      out << INDENT << "} catch (std::exception& e) {" << EOL;
      out << INDENT2 << "std::cerr << std::endl << \"Exception caught: \" << e.what() << std::endl;" << EOL;
      out << INDENT2 << "std::cerr << \"Diagnostic informtion: \" << std::endl << boost::diagnostic_information(e) << std::endl;" << EOL;
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
      // FIXME: generate or delete
      // generate_constructor(prog,model_name,out);
      generate_dump_constructor(prog,model_name,out);
      generate_set_param_ranges(prog.parameter_decl_,out);
      generate_init_method(prog.parameter_decl_,out);
      generate_log_prob(prog,out);
      generate_write_csv_method(prog.parameter_decl_,out);
      // FIXME: generate or delete
      // generate_write_csv_header_method(prog.parameter_decl_,out);
      generate_end_class_decl(out);
      generate_end_namespace(out);
      generate_main(model_name,out);
    }

  }
  
}

#endif

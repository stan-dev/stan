#ifndef STAN__GM__AST_HPP
#define STAN__GM__AST_HPP

#include <map>
#include <string>
#include <vector>
#include <map>
#include <set>

#include <boost/variant/recursive_variant.hpp>

namespace stan {

  namespace gm {

    /** Placeholder struct for boost::variant default ctors 
     */
    struct nil { };

    // components of abstract syntax tree 
    struct array_literal;
    struct assignment;
    struct binary_op;
    struct conditional_statement;
    struct distribution;
    struct double_var_decl;
    struct double_literal;
    struct expression;
    struct for_statement;
    struct fun;
    struct function_decl_def;
    struct function_decl_defs;
    struct identifier;
    struct increment_log_prob_statement;
    struct index_op;
    struct int_literal;
    struct inv_var_decl;
    struct matrix_var_decl;
    struct no_op_statement;
    struct ordered_var_decl;
    struct positive_ordered_var_decl;
    struct print_statement;
    struct program;
    struct range;
    struct reject_statement;
    struct return_statement;
    struct row_vector_var_decl;
    struct sample;
    struct simplex_var_decl;
    struct integrate_ode;
    struct unit_vector_var_decl;
    struct statement;
    struct statements;
    struct unary_op;
    struct variable;
    struct variable_dims;
    struct var_decl;
    struct var_type;
    struct vector_var_decl;
    struct while_statement;

    // forward declarable enum hack (can't fwd-decl enum)
    typedef int base_expr_type;
    const int VOID_T = 0;
    const int INT_T = 1;
    const int DOUBLE_T = 2;
    const int VECTOR_T = 3;
    const int ROW_VECTOR_T = 4;
    const int MATRIX_T = 5;
    const int ILL_FORMED_T = 6;

    std::ostream& write_base_expr_type(std::ostream& o, base_expr_type type);

    struct expr_type {
      base_expr_type base_type_;
      size_t num_dims_;
      expr_type();
      expr_type(const base_expr_type base_type); 
      expr_type(const base_expr_type base_type,
                size_t num_dims); 
      bool operator==(const expr_type& et) const;
      bool operator!=(const expr_type& et) const;
      bool operator<(const expr_type& et) const;
      bool operator<=(const expr_type& et) const;
      bool operator>(const expr_type& et) const;
      bool operator>=(const expr_type& et) const;
      bool is_primitive() const;
      bool is_primitive_int() const;
      bool is_primitive_double() const;
      bool is_ill_formed() const;
      bool is_void() const;
      base_expr_type type() const;
      size_t num_dims() const;
    };


    std::ostream& operator<<(std::ostream& o, const expr_type& et);

    expr_type promote_primitive(const expr_type& et);

    expr_type promote_primitive(const expr_type& et1,
                                const expr_type& et2);

    typedef std::pair<expr_type, std::vector<expr_type> > function_signature_t;

    class function_signatures {
    public:
      static function_signatures& instance();
      static void reset_sigs();
      void set_user_defined(const std::pair<std::string,function_signature_t>&
                            name_sig);
      bool is_user_defined(const std::pair<std::string,function_signature_t>&
                           name_sig);
      void add(const std::string& name,
               const expr_type& result_type,
               const std::vector<expr_type>& arg_types);
      void add(const std::string& name,
               const expr_type& result_type);
      void add(const std::string& name,
               const expr_type& result_type,
               const expr_type& arg_type);
      void add(const std::string& name,
               const expr_type& result_type,
               const expr_type& arg_type1,
               const expr_type& arg_type2);
      void add(const std::string& name,
               const expr_type& result_type,
               const expr_type& arg_type1,
               const expr_type& arg_type2,
               const expr_type& arg_type3);
      void add(const std::string& name,
               const expr_type& result_type,
               const expr_type& arg_type1,
               const expr_type& arg_type2,
               const expr_type& arg_type3,
               const expr_type& arg_type4);
      void add(const std::string& name,
               const expr_type& result_type,
               const expr_type& arg_type1,
               const expr_type& arg_type2,
               const expr_type& arg_type3,
               const expr_type& arg_type4,
               const expr_type& arg_type5);
      void add(const std::string& name,
               const expr_type& result_type,
               const expr_type& arg_type1,
               const expr_type& arg_type2,
               const expr_type& arg_type3,
               const expr_type& arg_type4,
               const expr_type& arg_type5,
               const expr_type& arg_type6);
      void add(const std::string& name,
               const expr_type& result_type,
               const expr_type& arg_type1,
               const expr_type& arg_type2,
               const expr_type& arg_type3,
               const expr_type& arg_type4,
               const expr_type& arg_type5,
               const expr_type& arg_type6,
               const expr_type& arg_type7);
      void add_nullary(const::std::string& name);
      void add_unary(const::std::string& name);
      void add_binary(const::std::string& name);
      void add_ternary(const::std::string& name);
      void add_quaternary(const::std::string& name);
      int num_promotions(const std::vector<expr_type>& call_args,
                         const std::vector<expr_type>& sig_args);
      expr_type get_result_type(const std::string& name,
                                const std::vector<expr_type>& args,
                                std::ostream& error_msgs);
      int get_signature_matches(const std::string& name,
                                const std::vector<expr_type>& args,
                                function_signature_t& signature);
      bool is_defined(const std::string& name, 
                      const function_signature_t& sig);
      std::set<std::string> key_set() const;
    private:
      function_signatures(); 
      function_signatures(const function_signatures& fs);
      std::map<std::string, std::vector<function_signature_t> > sigs_map_;
      std::set<std::pair<std::string,function_signature_t> > user_defined_set_;
      static function_signatures* sigs_;  // init below outside of class
    };
    
    struct statements {
      std::vector<var_decl> local_decl_;
      std::vector<statement> statements_;
      statements();
      statements(const std::vector<var_decl>& local_decl,
                 const std::vector<statement>& stmts);
    };


    struct distribution {
      std::string family_;
      std::vector<expression> args_;
    };

    struct expression_type_vis : public boost::static_visitor<expr_type> {
      expr_type operator()(const nil& e) const;
      expr_type operator()(const int_literal& e) const;
      expr_type operator()(const double_literal& e) const;
      expr_type operator()(const array_literal& e) const;
      expr_type operator()(const variable& e) const;
      expr_type operator()(const fun& e) const;
      expr_type operator()(const integrate_ode& e) const;
      expr_type operator()(const index_op& e) const;
      expr_type operator()(const binary_op& e) const;
      expr_type operator()(const unary_op& e) const;
      // template <typename T> expr_type operator()(const T& e) const;
    };


    struct expression;

    struct expression {
      typedef boost::variant<boost::recursive_wrapper<nil>, 
                             boost::recursive_wrapper<int_literal>,
                             boost::recursive_wrapper<double_literal>,
                             boost::recursive_wrapper<array_literal>,
                             boost::recursive_wrapper<variable>,
                             boost::recursive_wrapper<integrate_ode>,
                             boost::recursive_wrapper<fun>,
                             boost::recursive_wrapper<index_op>,
                             boost::recursive_wrapper<binary_op>,
                             boost::recursive_wrapper<unary_op> > 
      expression_t;

      expression();
      expression(const expression& e);

      // template <typename Expr> expression(const Expr& expr);
      expression(const nil& expr);
      expression(const int_literal& expr);
      expression(const double_literal& expr);
      expression(const array_literal& expr);
      expression(const variable& expr);
      expression(const fun& expr);
      expression(const integrate_ode& expr);
      expression(const index_op& expr);
      expression(const binary_op& expr);
      expression(const unary_op& expr);
      expression(const expression_t& expr_);

      expr_type expression_type() const; 

      expression& operator+=(const expression& rhs);
      expression& operator-=(const expression& rhs);
      expression& operator*=(const expression& rhs);
      expression& operator/=(const expression& rhs);

      expression_t expr_;
    };

   
    struct printable {
      typedef boost::variant<boost::recursive_wrapper<std::string>,
                             boost::recursive_wrapper<expression> >
      printable_t;

    printable();
    printable(const expression& expression);
    printable(const std::string& msg);
    printable(const printable_t& printable);
    printable(const printable& printable);

    printable_t printable_;
    };

    struct is_nil_op : public boost::static_visitor<bool> {
      bool operator()(const nil& x) const;
      bool operator()(const int_literal& x) const;
      bool operator()(const double_literal& x) const;
      bool operator()(const array_literal& x) const;
      bool operator()(const variable& x) const;
      bool operator()(const integrate_ode& x) const;
      bool operator()(const fun& x) const;
      bool operator()(const index_op& x) const;
      bool operator()(const binary_op& x) const;
      bool operator()(const unary_op& x) const;

      // template <typename T>
      // bool operator()(const T& x) const;
    };

    bool is_nil(const expression& e);

    struct variable_dims {
      std::string name_;
      std::vector<expression> dims_;
      variable_dims();
      variable_dims(std::string const& name,
                    std::vector<expression> const& dims); 
    };


    struct int_literal {
      int val_;
      expr_type type_;
      int_literal();
      int_literal(int val);
      int_literal(const int_literal& il);
      int_literal& operator=(const int_literal& il);
    };


    struct double_literal {
      double val_;
      expr_type type_;
      double_literal();
      double_literal(double val);
      double_literal& operator=(const double_literal& dl);
    };

    struct array_literal {
      std::vector<expression> args_;
      expr_type type_;
      array_literal();
      array_literal(const std::vector<expression>& args);
      array_literal& operator=(const array_literal& al);
    };

    struct variable {
      std::string name_;
      expr_type type_;
      variable();
      variable(std::string name);
      void set_type(const base_expr_type& base_type, 
                    size_t num_dims);
    };

    struct integrate_ode {
      std::string system_function_name_;
      expression y0_;    // initial state
      expression t0_;    // initial time
      expression ts_;    // solution times
      expression theta_; // params
      expression x_;     // data
      expression x_int_;     // integer data
      integrate_ode();
      integrate_ode(const std::string& system_function_name,
                const expression& y0,
                const expression& t0,
                const expression& ts,
                const expression& theta,
                const expression& x,
                const expression& x_int);
    };

    struct fun {
      std::string name_;
      std::vector<expression> args_;
      expr_type type_;
      fun();
      fun(std::string const& name,
          std::vector<expression> const& args); 
      void infer_type();  // FIXME: is this used anywhere?
    };

    size_t total_dims(const std::vector<std::vector<expression> >& dimss);

    expr_type infer_type_indexing(const base_expr_type& expr_base_type,
                                  size_t num_expr_dims,
                                  size_t num_index_dims);

    expr_type infer_type_indexing(const expression& expr,
                                  size_t num_index_dims);


    struct index_op {
      expression expr_;
      std::vector<std::vector<expression> > dimss_;
      expr_type type_;
      index_op();
      // vec of vec for e.g., e[1,2][3][4,5,6]
      index_op(const expression& expr,
               const std::vector<std::vector<expression> >& dimss);
      void infer_type();
    };


    struct binary_op {
      std::string op;
      expression left;
      expression right;
      expr_type type_;
      binary_op();
      binary_op(const expression& left,
                const std::string& op,
                const expression& right);
    };

    struct unary_op {
      char op;
      expression subject;
      expr_type type_;
      unary_op(char op,
               expression const& subject);
    };

    struct range {
      expression low_;
      expression high_;
      range();
      range(expression const& low,
            expression const& high);
      bool has_low() const;
      bool has_high() const;
    };

    typedef int var_origin;
    const int model_name_origin = 0;
    const int data_origin = 1;
    const int transformed_data_origin = 2;
    const int parameter_origin = 3;
    const int transformed_parameter_origin = 4;
    const int derived_origin = 5;
    const int local_origin = 6;
    const int function_argument_origin = 7;
    const int function_argument_origin_lp = 8;
    const int function_argument_origin_rng = 9;
    const int void_function_argument_origin = 10;
    const int void_function_argument_origin_lp = 11;
    const int void_function_argument_origin_rng = 12;

    void print_var_origin(std::ostream& o, const var_origin& vo);

    struct base_var_decl {
      std::string name_;
      std::vector<expression> dims_;
      base_expr_type base_type_;
      base_var_decl();
      base_var_decl(const base_expr_type& base_type); 
      base_var_decl(const std::string& name,
                    const std::vector<expression>& dims,
                    const base_expr_type& base_type);
    };

    struct variable_map {
      typedef std::pair<base_var_decl,var_origin> range_t;

      bool exists(const std::string& name) const;
      base_var_decl get(const std::string& name) const;
      base_expr_type get_base_type(const std::string& name) const;
      size_t get_num_dims(const std::string& name) const;
      var_origin get_origin(const std::string& name) const;
      void add(const std::string& name,
               const base_var_decl& base_decl,
               const var_origin& vo);
      void remove(const std::string& name);

      std::map<std::string, range_t> map_;
    };

    struct int_var_decl : public base_var_decl {
      range range_;
      int_var_decl(); 
      int_var_decl(range const& range,
                   std::string const& name,
                   std::vector<expression> const& dims); 
    };


    struct double_var_decl : public base_var_decl {
      range range_;
      double_var_decl();
      double_var_decl(range const& range,
                      std::string const& name,
                      std::vector<expression> const& dims);
    };

    struct unit_vector_var_decl : public base_var_decl {
      expression K_;
      unit_vector_var_decl();
      unit_vector_var_decl(expression const& K,
                           std::string const& name,
                           std::vector<expression> const& dims);
    };

    struct simplex_var_decl : public base_var_decl {
      expression K_;
      simplex_var_decl();
      simplex_var_decl(expression const& K,
                       std::string const& name,
                       std::vector<expression> const& dims);
    };

    struct ordered_var_decl : public base_var_decl {
      expression K_;
      ordered_var_decl();
      ordered_var_decl(expression const& K,
                       std::string const& name,
                       std::vector<expression> const& dims);
    };

    struct positive_ordered_var_decl : public base_var_decl {
      expression K_;
      positive_ordered_var_decl();
      positive_ordered_var_decl(expression const& K,
                                std::string const& name,
                                std::vector<expression> const& dims);
    };

    struct vector_var_decl : public base_var_decl {
      range range_;
      expression M_;
      vector_var_decl();
      vector_var_decl(range const& range, 
                      expression const& M,
                      std::string const& name,
                      std::vector<expression> const& dims);
    };

    struct row_vector_var_decl : public base_var_decl {
      range range_;
      expression N_;
      row_vector_var_decl();
      row_vector_var_decl(range const& range, 
                          expression const& N,
                          std::string const& name,
                          std::vector<expression> const& dims);
    };

    struct matrix_var_decl : public base_var_decl {
      range range_;
      expression M_;
      expression N_;
      matrix_var_decl();
      matrix_var_decl(range const& range, 
                      expression const& M,
                      expression const& N,
                      std::string const& name,
                      std::vector<expression> const& dims);
    };

    struct cholesky_factor_var_decl : public base_var_decl {
      expression M_;
      expression N_;
      cholesky_factor_var_decl();
      cholesky_factor_var_decl(expression const& M,
                               expression const& N,
                               std::string const& name,
                               std::vector<expression> const& dims);
    };

    struct cholesky_corr_var_decl : public base_var_decl {
      expression K_;
      cholesky_corr_var_decl();
      cholesky_corr_var_decl(const expression& K,
                             const std::string& name,
                             const std::vector<expression>& dims);
    };

    struct cov_matrix_var_decl : public base_var_decl {
      expression K_;
      cov_matrix_var_decl();
      cov_matrix_var_decl(expression const& K,
                          std::string const& name,
                          std::vector<expression> const& dims);
    };


    struct corr_matrix_var_decl : public base_var_decl {
      expression K_;
      corr_matrix_var_decl();
      corr_matrix_var_decl(expression const& K,
                           std::string const& name,
                           std::vector<expression> const& dims);
    };

    struct name_vis : public boost::static_visitor<std::string> {
      name_vis();
      std::string operator()(const nil& x) const;
      std::string operator()(const int_var_decl& x) const;
      std::string operator()(const double_var_decl& x) const;
      std::string operator()(const vector_var_decl& x) const;
      std::string operator()(const row_vector_var_decl& x) const;
      std::string operator()(const matrix_var_decl& x) const;
      std::string operator()(const simplex_var_decl& x) const;
      std::string operator()(const unit_vector_var_decl& x) const;
      std::string operator()(const ordered_var_decl& x) const;
      std::string operator()(const positive_ordered_var_decl& x) const;
      std::string operator()(const cholesky_factor_var_decl& x) const;
      std::string operator()(const cholesky_corr_var_decl& x) const;
      std::string operator()(const cov_matrix_var_decl& x) const;
      std::string operator()(const corr_matrix_var_decl& x) const;
    };




    struct var_decl {
      typedef boost::variant<boost::recursive_wrapper<nil>,
                             boost::recursive_wrapper<int_var_decl>,
                             boost::recursive_wrapper<double_var_decl>,
                             boost::recursive_wrapper<vector_var_decl>,
                             boost::recursive_wrapper<row_vector_var_decl>,
                             boost::recursive_wrapper<matrix_var_decl>,
                             boost::recursive_wrapper<simplex_var_decl>,
                             boost::recursive_wrapper<unit_vector_var_decl>,
                             boost::recursive_wrapper<ordered_var_decl>,
                             boost::recursive_wrapper<positive_ordered_var_decl>,
                             boost::recursive_wrapper<cholesky_factor_var_decl>,
                             boost::recursive_wrapper<cholesky_corr_var_decl>,
                             boost::recursive_wrapper<cov_matrix_var_decl>,
                             boost::recursive_wrapper<corr_matrix_var_decl> >
      var_decl_t;

      var_decl_t decl_;

      var_decl();

      // template <typename Decl>
      // var_decl(Decl const& decl);
      var_decl(const var_decl_t& decl);
      var_decl(const nil& decl);
      var_decl(const int_var_decl& decl);
      var_decl(const double_var_decl& decl);
      var_decl(const vector_var_decl& decl);
      var_decl(const row_vector_var_decl& decl);
      var_decl(const matrix_var_decl& decl);
      var_decl(const simplex_var_decl& decl);
      var_decl(const unit_vector_var_decl& decl);
      var_decl(const ordered_var_decl& decl);
      var_decl(const positive_ordered_var_decl& decl);
      var_decl(const cholesky_factor_var_decl& decl);
      var_decl(const cholesky_corr_var_decl& decl);
      var_decl(const cov_matrix_var_decl& decl);
      var_decl(const corr_matrix_var_decl& decl);

      std::string name() const;
    };

    struct statement {
      typedef boost::variant<boost::recursive_wrapper<nil>,
                             boost::recursive_wrapper<assignment>,
                             boost::recursive_wrapper<sample>,
                             boost::recursive_wrapper<increment_log_prob_statement>,
                             boost::recursive_wrapper<expression>, // dummy now
                             boost::recursive_wrapper<statements>,
                             boost::recursive_wrapper<for_statement>,
                             boost::recursive_wrapper<conditional_statement>,
                             boost::recursive_wrapper<while_statement>,
                             boost::recursive_wrapper<print_statement>,
                             boost::recursive_wrapper<reject_statement>,
                             boost::recursive_wrapper<return_statement>,
                             boost::recursive_wrapper<no_op_statement> >
      statement_t;
    
      statement_t statement_;

      statement();
      statement(const statement_t& st);
      statement(const nil& st);
      statement(const assignment& st);
      statement(const sample& st);
      statement(const increment_log_prob_statement& st);
      statement(const expression& st);
      statement(const statements& st);
      statement(const for_statement& st);
      statement(const conditional_statement& st);
      statement(const while_statement& st);
      statement(const print_statement& st);
      statement(const reject_statement& st);
      statement(const no_op_statement& st);
      statement(const return_statement& st);

      bool is_no_op_statement() const;
    };

    struct is_no_op_statement_vis : public boost::static_visitor<bool> {
      bool operator()(const nil& st) const;
      bool operator()(const assignment& st) const;
      bool operator()(const sample& st) const;
      bool operator()(const increment_log_prob_statement& t) const;
      bool operator()(const expression& st) const;
      bool operator()(const statements& st) const;
      bool operator()(const for_statement& st) const;
      bool operator()(const conditional_statement& st) const;
      bool operator()(const while_statement& st) const;
      bool operator()(const print_statement& st) const;
      bool operator()(const reject_statement& st) const;
      bool operator()(const no_op_statement& st) const;
      bool operator()(const return_statement& st) const;
    };
    

    struct returns_type_vis : public boost::static_visitor<bool> {
      expr_type return_type_;
      std::ostream& error_msgs_;
      returns_type_vis(const expr_type& return_type,
                       std::ostream& error_msgs);
      bool operator()(const nil& st) const;
      bool operator()(const assignment& st) const;
      bool operator()(const sample& st) const;
      bool operator()(const increment_log_prob_statement& t) const;
      bool operator()(const expression& st) const;
      bool operator()(const statements& st) const;
      bool operator()(const for_statement& st) const;
      bool operator()(const conditional_statement& st) const;
      bool operator()(const while_statement& st) const;
      bool operator()(const print_statement& st) const;
      bool operator()(const reject_statement& st) const;
      bool operator()(const no_op_statement& st) const;
      bool operator()(const return_statement& st) const;
    };

    bool returns_type(const expr_type& return_type,
                      const statement& statement,
                      std::ostream& error_msgs);

    struct increment_log_prob_statement {
      expression log_prob_;
      increment_log_prob_statement();
      increment_log_prob_statement(const expression& log_prob);
    };

    struct for_statement {
      std::string variable_;
      range range_;
      statement statement_;
      for_statement();
      for_statement(std::string& variable,
                    range& range,
                    statement& stmt);
    };
    
    // bodies may be 1 longer than conditions due to else
    struct conditional_statement {
      std::vector<expression> conditions_;
      std::vector<statement> bodies_; 
      conditional_statement();
      conditional_statement(const std::vector<expression>& conditions,
                            const std::vector<statement>& statements);
    };

    struct while_statement {
      expression condition_;
      statement body_;
      while_statement();
      while_statement(const expression& condition,
                      const statement& body);
    };

    struct print_statement {
      std::vector<printable> printables_;
      print_statement();
      print_statement(const std::vector<printable>& printables);
    };

    struct reject_statement {
      std::vector<printable> printables_;
      reject_statement();
      reject_statement(const std::vector<printable>& printables);
    };

    struct return_statement {
      expression return_value_;
      return_statement();
      return_statement(const expression& expr);
    };

    struct no_op_statement {
      // no op, no data
    };

    struct arg_decl {
      expr_type arg_type_;
      std::string name_;
      arg_decl();
      arg_decl(const expr_type& arg_type,
               const std::string& name);
      base_var_decl base_variable_declaration();
    };

    struct function_decl_def {
      function_decl_def();
      function_decl_def(const expr_type& return_type,
                        const std::string& name,
                        const std::vector<arg_decl>& arg_decls,
                        const statement& body);
      expr_type return_type_;
      std::string name_;
      std::vector<arg_decl> arg_decls_;
      statement body_;
    };

    struct function_decl_defs {
      function_decl_defs();
      function_decl_defs(const std::vector<function_decl_def>& decl_defs);
      std::vector<function_decl_def> decl_defs_;
    };


    struct program {
      std::vector<function_decl_def> function_decl_defs_;
      std::vector<var_decl> data_decl_;
      std::pair<std::vector<var_decl>,std::vector<statement> > 
        derived_data_decl_;
      std::vector<var_decl> parameter_decl_;
      std::pair<std::vector<var_decl>,std::vector<statement> > 
        derived_decl_;
      statement statement_;
      std::pair<std::vector<var_decl>,std::vector<statement> > generated_decl_;
      program();
      program(const std::vector<function_decl_def>& function_decl_defs,
              const std::vector<var_decl>& data_decl,
              const std::pair<std::vector<var_decl>,
                              std::vector<statement> >& derived_data_decl,
              const std::vector<var_decl>& parameter_decl,
              const std::pair<std::vector<var_decl>,
                              std::vector<statement> >& derived_decl,
              const statement& st,
              const std::pair<std::vector<var_decl>,
                              std::vector<statement> >& generated_decl);
    };

    struct sample {
      expression expr_;
      distribution dist_;
      range truncation_;
      sample();
      sample(expression& e,
             distribution& dist);
      bool is_ill_formed() const;
    };

    struct assignment {
      variable_dims var_dims_; // lhs_var[dim0,...,dimN-1] 
      expression expr_;        // = rhs
      base_var_decl var_type_; // type of lhs_var
      assignment();
      assignment(variable_dims& var_dims,
                 expression& expr);
    };
  
    // FIXME:  is this next dependency necessary?
    // from generator.hpp
    void generate_expression(const expression& e, std::ostream& o);

    bool has_rng_suffix(const std::string& s);
    bool has_lp_suffix(const std::string& s);
    bool is_user_defined(const std::string& name,
                         const std::vector<expression>& args);
    bool is_user_defined_prob_function(const std::string& name,
                                       const expression& variate,
                                       const std::vector<expression>& params);
    bool is_user_defined(const fun& fx);

    struct contains_var : public boost::static_visitor<bool> {
      const variable_map& var_map_;
      contains_var(const variable_map& var_map);
      bool operator()(const nil& e) const;
      bool operator()(const int_literal& e) const;
      bool operator()(const double_literal& e) const;
      bool operator()(const array_literal& e) const;
      bool operator()(const variable& e) const;
      bool operator()(const integrate_ode& e) const;
      bool operator()(const fun& e) const;
      bool operator()(const index_op& e) const;
      bool operator()(const binary_op& e) const;
      bool operator()(const unary_op& e) const;
    };

    bool has_var(const expression& e,
                 const variable_map& var_map);


    struct contains_nonparam_var : public boost::static_visitor<bool> {
      const variable_map& var_map_;
      contains_nonparam_var(const variable_map& var_map);
      bool operator()(const nil& e) const;
      bool operator()(const int_literal& e) const;
      bool operator()(const double_literal& e) const;
      bool operator()(const array_literal& e) const;
      bool operator()(const variable& e) const;
      bool operator()(const integrate_ode& e) const;
      bool operator()(const fun& e) const;
      bool operator()(const index_op& e) const;
      bool operator()(const binary_op& e) const;
      bool operator()(const unary_op& e) const;
    };

    bool has_non_param_var(const expression& e,
                           const variable_map& var_map);

    bool is_assignable(const expr_type& l_type,
                       const expr_type& r_type,
                       const std::string& failure_message,
                       std::ostream& error_msgs);


    bool ends_with(const std::string& suffix, 
                   const std::string& s);

  }
}
 
#endif

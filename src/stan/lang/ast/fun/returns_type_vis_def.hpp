#ifndef STAN_LANG_AST_FUN_RETURNS_TYPE_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_RETURNS_TYPE_VIS_DEF_HPP

#include <stan/lang/ast/expr_type.hpp>
#include <stan/lang/ast/fun/is_assignable.hpp>
#include <stan/lang/ast/fun/returns_type.hpp>
#include <stan/lang/ast/fun/returns_type_vis.hpp>
#include <stan/lang/ast/node/break_continue_statement.hpp>
#include <stan/lang/ast/node/conditional_statement.hpp>
#include <stan/lang/ast/node/for_statement.hpp>
#include <stan/lang/ast/node/for_array_statement.hpp>
#include <stan/lang/ast/node/for_matrix_statement.hpp>
#include <stan/lang/ast/node/statement.hpp>
#include <stan/lang/ast/node/statements.hpp>
#include <stan/lang/ast/node/return_statement.hpp>
#include <stan/lang/ast/node/while_statement.hpp>
#include <ostream>

namespace stan {
  namespace lang {


    returns_type_vis::returns_type_vis(const expr_type& return_type,
                                       std::ostream& error_msgs)
      : return_type_(return_type), error_msgs_(error_msgs) { }

    bool returns_type_vis::operator()(const nil& st) const {
      error_msgs_ << "Expecting return, found nil statement."
                  << std::endl;
      return false;
    }

    bool returns_type_vis::operator()(const assgn& st) const {
      error_msgs_ << "Expecting return, found assignment statement."
                  << std::endl;
      return false;
    }

    bool returns_type_vis::operator()(const sample& st) const {
      error_msgs_ << "Expecting return, found sampling statement."
                  << std::endl;
      return false;
    }

    bool returns_type_vis::operator()(const
                                      increment_log_prob_statement& t) const {
      error_msgs_ << "Expecting return, found increment_log_prob statement."
                  << std::endl;
      return false;
    }

    bool returns_type_vis::operator()(const expression& st) const  {
      error_msgs_ << "Expecting return, found increment_log_prob statement."
                  << std::endl;
      return false;
    }

    bool returns_type_vis::operator()(const print_statement& st) const  {
      error_msgs_ << "Expecting return, found print statement."
                 << std::endl;
      return false;
    }

    bool returns_type_vis::operator()(const reject_statement& st) const  {
      error_msgs_ << "Expecting return, found reject statement."
                  << std::endl;
      return false;
    }

    bool returns_type_vis::operator()(const no_op_statement& st) const  {
      error_msgs_ << "Expecting return, found no_op statement."
                  << std::endl;
      return false;
    }

    bool returns_type_vis::operator()(const statements& st) const  {
      // last statement in sequence must return type
      if (st.statements_.size() == 0) {
        error_msgs_ << ("Expecting return, found"
                        " statement sequence with empty body.")
                    << std::endl;
        return false;
      }
      return returns_type(return_type_, st.statements_.back(), error_msgs_);
    }

    bool returns_type_vis::operator()(const for_statement& st) const  {
      // body must end in appropriate return
      return returns_type(return_type_, st.statement_, error_msgs_);
    }

    bool returns_type_vis::operator()(const for_array_statement& st) const  {
      // body must end in appropriate return
      return returns_type(return_type_, st.statement_, error_msgs_);
    }

    bool returns_type_vis::operator()(const for_matrix_statement& st) const  {
      // body must end in appropriate return
      return returns_type(return_type_, st.statement_, error_msgs_);
    }

     bool returns_type_vis::operator()(const while_statement& st) const  {
      // body must end in appropriate return
      return returns_type(return_type_, st.body_, error_msgs_);
    }

    bool returns_type_vis::operator()(const break_continue_statement& st)
      const  {
      // break/continue OK only as end of nested loop in void return
      bool pass = (return_type_.is_void());
      if (!pass)
        error_msgs_ << "statement " << st.generate_
                    << " does not match return type";
      return pass;
    }

    bool returns_type_vis::operator()(const conditional_statement& st) const {
      // all condition bodies must end in appropriate return
      if (st.bodies_.size() != (st.conditions_.size() + 1)) {
        error_msgs_ << ("Expecting return, found conditional"
                        " without final else.")
                    << std::endl;
        return false;
      }
      for (size_t i = 0; i < st.bodies_.size(); ++i)
        if (!returns_type(return_type_, st.bodies_[i], error_msgs_))
          return false;
      return true;
    }

    bool returns_type_vis::operator()(const return_statement& st) const  {
      // return checked for type
      return return_type_.is_void()
        || is_assignable(return_type_, st.return_value_.expression_type(),
                         "Returned expression does not match return type",
                         error_msgs_);
    }

  }
}
#endif

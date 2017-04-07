#ifndef STAN_LANG_GENERATOR_IS_NUMBERED_STATEMENT_VIS_HPP
#define STAN_LANG_GENERATOR_IS_NUMBERED_STATEMENT_VIS_HPP

#include <stan/lang/ast.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <boost/variant/static_visitor.hpp>

namespace stan {
  namespace lang {

    /**
     * Visitor to return true if the statement is numbered for error
     * reporting.  All statements are numbered other than the nil
     * statement and sequences of statements, the former because there
     * is nothing to number and the latter because each statement
     * within the sequence is numbered.
     */
    struct is_numbered_statement_vis : public boost::static_visitor<bool> {
      /**
       * Construct a numbered statement visitor.
       */
      is_numbered_statement_vis() { }

      bool operator()(const nil& st) const { return false; }

      bool operator()(const assignment& st) const { return true; }

      bool operator()(const assgn& st) const { return true; }

      bool operator()(const sample& st) const { return true; }

      bool operator()(const increment_log_prob_statement& t) const {
        return true;
      }

      bool operator()(const expression& st) const  { return true; }

      bool operator()(const statements& st) const  { return false; }

      bool operator()(const for_statement& st) const  { return true; }

      bool operator()(const conditional_statement& st) const { return true; }

      bool operator()(const while_statement& st) const { return true; }

      bool operator()(const break_continue_statement& st) const {
        return true;
      }

      bool operator()(const print_statement& st) const { return true; }

      bool operator()(const reject_statement& st) const { return true; }

      bool operator()(const no_op_statement& st) const { return true; }

      bool operator()(const return_statement& st) const { return true; }
    };

  }
}
#endif

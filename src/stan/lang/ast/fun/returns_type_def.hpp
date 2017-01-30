#ifndef STAN_LANG_AST_FUN_RETURNS_TYPE_DEF_HPP
#define STAN_LANG_AST_FUN_RETURNS_TYPE_DEF_HPP

#include <stan/lang/ast/expr_type.hpp>
#include <stan/lang/ast/fun/returns_type.hpp>
#include <stan/lang/ast/fun/returns_type_vis.hpp>
#include <stan/lang/ast/node/statement.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <ostream>

namespace stan {
  namespace lang {

    bool returns_type(const expr_type& return_type, const statement& statement,
                      std::ostream& error_msgs) {
      if (return_type == VOID_T)
        return true;
      returns_type_vis vis(return_type, error_msgs);
      return boost::apply_visitor(vis, statement.statement_);
    }

  }
}
#endif

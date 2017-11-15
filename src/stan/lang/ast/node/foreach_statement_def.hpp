#ifndef STAN_LANG_AST_NODE_FOREACH_STATEMENT_DEF_HPP
#define STAN_LANG_AST_NODE_FOREACH_STATEMENT_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    foreach_statement::foreach_statement() { }

    foreach_statement::foreach_statement(const std::string& variable,
                                         const range& range,
                                         const statement& stmt)
      : variable_(variable), range_(range), statement_(stmt) { }

  }
}
#endif

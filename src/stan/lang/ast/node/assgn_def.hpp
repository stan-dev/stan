#ifndef STAN_LANG_AST_NODE_ASSGN_DEF_HPP
#define STAN_LANG_AST_NODE_ASSGN_DEF_HPP

#include <stan/lang/ast.hpp>
#include <vector>

namespace stan {
  namespace lang {

    assgn::assgn() { }

    assgn::assgn(const variable& lhs_var, const std::vector<idx>& idxs,
                 const expression& rhs)
      : lhs_var_(lhs_var), idxs_(idxs), rhs_(rhs) { }

    bool assgn::lhs_var_occurs_on_rhs() const {
      var_occurs_vis vis(lhs_var_);
      return boost::apply_visitor(vis, rhs_.expr_);
    }

  }
}
#endif

#ifndef STAN_LANG_AST_NODE_SAMPLE_DEF_HPP
#define STAN_LANG_AST_NODE_SAMPLE_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {


    sample::sample() { }

    sample::sample(expression& e, distribution& dist)
        : expr_(e), dist_(dist) { }

    bool sample::is_ill_formed() const {
        return expr_.expression_type().is_ill_formed()
          || (truncation_.has_low()
              && expr_.expression_type() != truncation_.low_.expression_type())
          || (truncation_.has_high()
               && expr_.expression_type()
                  != truncation_.high_.expression_type());
    }

    bool sample::is_discrete() const {
      return is_discrete_;
    }

  }
}
#endif

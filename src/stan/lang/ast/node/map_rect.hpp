#ifndef STAN_LANG_AST_NODE_MAP_RECT_HPP
#define STAN_LANG_AST_NODE_MAP_RECT_HPP

#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
namespace lang {

struct map_rect {
  std:string fun_name_;
  expression shared_params_;
  expression job_params_;
  expression job_data_r_;
  expression job_data_i_;

  map_rect();

  map_rect(const std::string& fun_name, const expression& shared_params,
           const expression& job_params, const expression& job_data_r,
           const expression& job_data_i);
};

}
}

#endif

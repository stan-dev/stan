#ifndef STAN_LANG_AST_NODE_MAP_RECT_DEF_HPP
#define STAN_LANG_AST_NODE_MAP_RECT_DEF_HPP

#include <stan/lang/ast/node/map_rect.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
namespace lang {

int map_rect::CALL_ID_ = 0;

stan::lang::map_rect::map_rect() : call_id_(CALL_ID_++) { }

stan::lang::map_rect::map_rect(const std::string& fun_name,
               const expression& shared_params,
               const expression& job_params, const expression& job_data_r,
               const expression& job_data_i)
    : call_id_(CALL_ID_++), fun_name_(fun_name), shared_params_(shared_params),
      job_params_(job_params), job_data_r_(job_data_r),
      job_data_i_(job_data_i) {
}

}
}
#endif

#include <stan/lang/ast/sigs/function_signatures_def.hpp>
#include <stan/lang/ast/base_expr_type_def.hpp>
#include <stan/lang/ast/expr_type_def.hpp>
#include <src/stan/lang/ast/sigs/function_arg_type_def.hpp>
#include <src/stan/lang/ast/fun/ends_with_def.hpp>
#include <src/stan/lang/ast/fun/write_base_expr_type_def.hpp>

#include <iostream>

int main() {
  stan::lang::function_signatures::instance().print_signatures(std::cout);
  return 0;
}

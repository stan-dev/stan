#ifndef STAN_LANG_GENERATOR_HPP
#define STAN_LANG_GENERATOR_HPP

// TODO(carpenter): move into AST
#include <stan/lang/generator/has_lb.hpp>
#include <stan/lang/generator/has_lub.hpp>
#include <stan/lang/generator/has_ub.hpp>
#include <stan/lang/generator/has_only_int_args.hpp>

// TODO(carpenter): move into general utilities
#include <stan/lang/generator/to_string.hpp>

// utilities
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/fun_scalar_type.hpp>

// visitor classes for tests
#include <stan/lang/generator/is_numbered_statement_vis.hpp>

// visitor classes for generation
#include <stan/lang/generator/constrained_param_names_visgen.hpp>
#include <stan/lang/generator/dump_member_var_visgen.hpp>
#include <stan/lang/generator/expression_visgen.hpp>
#include <stan/lang/generator/printable_visgen.hpp>
#include <stan/lang/generator/idx_visgen.hpp>
#include <stan/lang/generator/idx_user_visgen.hpp>
#include <stan/lang/generator/init_local_var_visgen.hpp>
#include <stan/lang/generator/init_vars_visgen.hpp>
#include <stan/lang/generator/init_visgen.hpp>
#include <stan/lang/generator/local_var_decl_visgen.hpp>
#include <stan/lang/generator/local_var_init_nan_visgen.hpp>
#include <stan/lang/generator/member_var_decl_visgen.hpp>
#include <stan/lang/generator/set_param_ranges_visgen.hpp>
#include <stan/lang/generator/statement_visgen.hpp>
#include <stan/lang/generator/unconstrained_param_names_visgen.hpp>
#include <stan/lang/generator/validate_var_decl_visgen.hpp>
#include <stan/lang/generator/validate_transformed_params_visgen.hpp>
#include <stan/lang/generator/var_resizing_visgen.hpp>
#include <stan/lang/generator/var_size_validating_visgen.hpp>
#include <stan/lang/generator/visgen.hpp>
#include <stan/lang/generator/write_array_visgen.hpp>
#include <stan/lang/generator/write_array_vars_visgen.hpp>
#include <stan/lang/generator/write_dims_visgen.hpp>
#include <stan/lang/generator/write_param_names_visgen.hpp>

// generation functions, starts from geneate_cpp
#include <stan/lang/generator/generate_arg_decl.hpp>
#include <stan/lang/generator/generate_array_var_type.hpp>
#include <stan/lang/generator/generate_array_builder_adds.hpp>
#include <stan/lang/generator/generate_bare_type.hpp>
#include <stan/lang/generator/generate_catch_throw_located.hpp>
#include <stan/lang/generator/generate_class_decl.hpp>
#include <stan/lang/generator/generate_class_decl_end.hpp>
#include <stan/lang/generator/generate_comment.hpp>
#include <stan/lang/generator/generate_constrained_param_names_method.hpp>
#include <stan/lang/generator/generate_constructor.hpp>
#include <stan/lang/generator/generate_cpp.hpp>
#include <stan/lang/generator/generate_destructor.hpp>
#include <stan/lang/generator/generate_dims_method.hpp>
#include <stan/lang/generator/generate_eigen_index_expression.hpp>
#include <stan/lang/generator/generate_expression.hpp>
#include <stan/lang/generator/generate_function.hpp>
#include <stan/lang/generator/generate_functions.hpp>
#include <stan/lang/generator/generate_function_arguments.hpp>
#include <stan/lang/generator/generate_function_body.hpp>
#include <stan/lang/generator/generate_function_functor.hpp>
#include <stan/lang/generator/generate_function_inline_return_type.hpp>
#include <stan/lang/generator/generate_function_template_parameters.hpp>
#include <stan/lang/generator/generate_functor_arguments.hpp>
#include <stan/lang/generator/generate_globals.hpp>
#include <stan/lang/generator/generate_idx.hpp>
#include <stan/lang/generator/generate_idxs.hpp>
#include <stan/lang/generator/generate_idxs_user.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/generate_include.hpp>
#include <stan/lang/generator/generate_includes.hpp>
#include <stan/lang/generator/generate_indexed_expr.hpp>
#include <stan/lang/generator/generate_indexed_expr_user.hpp>
#include <stan/lang/generator/generate_init_method.hpp>
#include <stan/lang/generator/generate_initializer.hpp>
#include <stan/lang/generator/generate_initialization.hpp>
#include <stan/lang/generator/generate_local_var_decls.hpp>
#include <stan/lang/generator/generate_local_var_inits.hpp>
#include <stan/lang/generator/generate_located_statement.hpp>
#include <stan/lang/generator/generate_located_statements.hpp>
#include <stan/lang/generator/generate_log_prob.hpp>
#include <stan/lang/generator/generate_member_var_decls.hpp>
#include <stan/lang/generator/generate_member_var_decls_all.hpp>
#include <stan/lang/generator/generate_member_var_inits.hpp>
#include <stan/lang/generator/generate_model_name_method.hpp>
#include <stan/lang/generator/generate_model_typedef.hpp>
#include <stan/lang/generator/generate_namespace_end.hpp>
#include <stan/lang/generator/generate_namespace_start.hpp>
#include <stan/lang/generator/generate_param_names_method.hpp>
#include <stan/lang/generator/generate_printable.hpp>
#include <stan/lang/generator/generate_private_decl.hpp>
#include <stan/lang/generator/generate_propto_default_function.hpp>
#include <stan/lang/generator/generate_propto_default_function_body.hpp>
#include <stan/lang/generator/generate_public_decl.hpp>
#include <stan/lang/generator/generate_quoted_expression.hpp>
#include <stan/lang/generator/generate_quoted_string.hpp>
#include <stan/lang/generator/generate_real_var_type.hpp>
#include <stan/lang/generator/generate_set_param_ranges.hpp>
#include <stan/lang/generator/generate_statement.hpp>
#include <stan/lang/generator/generate_type.hpp>
#include <stan/lang/generator/generate_typedef.hpp>
#include <stan/lang/generator/generate_typedefs.hpp>
#include <stan/lang/generator/generate_try.hpp>
#include <stan/lang/generator/generate_unconstrained_param_names_method.hpp>
#include <stan/lang/generator/generate_using.hpp>
#include <stan/lang/generator/generate_using_namespace.hpp>
#include <stan/lang/generator/generate_usings.hpp>
#include <stan/lang/generator/generate_validate_context_size.hpp>
#include <stan/lang/generator/generate_validate_positive.hpp>
#include <stan/lang/generator/generate_validate_transformed_params.hpp>
#include <stan/lang/generator/generate_validate_var_decl.hpp>
#include <stan/lang/generator/generate_validate_var_decls.hpp>
#include <stan/lang/generator/generate_var_resizing.hpp>
#include <stan/lang/generator/generate_version_comment.hpp>
#include <stan/lang/generator/generate_void_statement.hpp>
#include <stan/lang/generator/generate_write_array_method.hpp>

#endif

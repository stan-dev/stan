#ifndef STAN__LANG__PARSER__COMMON_ADAPTORS_DEF__HPP
#define STAN__LANG__PARSER__COMMON_ADAPTORS_DEF__HPP

#include <boost/fusion/include/adapt_struct.hpp>

#include <stan/lang/ast.hpp>

BOOST_FUSION_ADAPT_STRUCT(stan::lang::range,
                          (stan::lang::expression, low_)
                          (stan::lang::expression, high_) )


#endif

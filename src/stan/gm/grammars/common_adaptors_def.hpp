#ifndef __STAN__GM__PARSER__COMMON_ADAPTORS_DEF__HPP__
#define __STAN__GM__PARSER__COMMON_ADAPTORS_DEF__HPP__

#include <stan/gm/ast.hpp>


BOOST_FUSION_ADAPT_STRUCT(stan::gm::range,
                          (stan::gm::expression, low_)
                          (stan::gm::expression, high_) )


#endif

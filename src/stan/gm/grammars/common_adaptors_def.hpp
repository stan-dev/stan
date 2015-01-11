#ifndef STAN__GM__PARSER__COMMON_ADAPTORS_DEF__HPP
#define STAN__GM__PARSER__COMMON_ADAPTORS_DEF__HPP

#include <boost/fusion/include/adapt_struct.hpp>

#include <stan/gm/ast.hpp>

BOOST_FUSION_ADAPT_STRUCT(stan::gm::range,
                          (stan::gm::expression, low_)
                          (stan::gm::expression, high_) )


#endif

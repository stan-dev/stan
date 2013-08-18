#ifndef __STAN__GM__PARSER__PARSER__ITERATOR_TYPEDEFS__HPP__
#define __STAN__GM__PARSER__PARSER__ITERATOR_TYPEDEFS__HPP__

#include <boost/spirit/include/support_multi_pass.hpp>
//#include <iterator>
//#include <boost/spirit/include/support_line_pos_iterator.hpp>
#include <boost/spirit/include/version.hpp>
#include <boost/spirit/home/support/iterators/line_pos_iterator.hpp>


namespace stan {
  namespace gm {
      typedef std::string::const_iterator input_iterator;
      typedef boost::spirit::line_pos_iterator<input_iterator> lp_iterator;
  }
}
#endif

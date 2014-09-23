#ifndef STAN__MATH__ODE__UTIL_HPP__
#define STAN__MATH__ODE__UTIL_HPP__

#include <vector>
#include <ostream>
#include <stan/agrad/rev/var.hpp>
 
namespace stan {
  
  namespace math {

    template<class T>
    struct push_back_state_and_time {
      std::vector< std::vector<T> >& m_states;
      std::vector< T >& m_times;
      
      push_back_state_and_time(std::vector< std::vector<T> > &states,
                               std::vector< T > &times)
        : m_states( states ), 
          m_times( times ) { }
      
      void operator()(const std::vector<T> &x, T t) {
        m_states.push_back( x );
        m_times.push_back( t );
      }
      
      std::vector<std::vector<T> > get() {
        return m_states;
      }
      
      void print(std::ostream& out) {
        out << "time,x_0";
        for (size_t n = 1; n < m_states[0].size(); n++)
          out << ",x_" << n;
        out << std::endl;
        for (size_t n = 0; n < m_states.size(); n++) {
          out << m_times[n]
              << "," << m_states[n][0]
              << "," << m_states[n][1]
              << std::endl;
        }
      }
    };
    
  }
}
#endif

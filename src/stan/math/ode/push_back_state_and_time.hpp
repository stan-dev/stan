#ifndef STAN__MATH__ODE__UTIL_HPP__
#define STAN__MATH__ODE__UTIL_HPP__

#include <vector>
#include <ostream>
#include <stan/agrad/rev/var.hpp>
 
namespace stan {
  
  namespace math {
    
    /**
     * Structure used by stan::math::integrate_ode as an observer.
     *
     * (Not fully documenting because this will change before the pull
     * request is complete. This doesn't need to be templated any
     * more. This was first used for auto-diffing through integrator.)
     */
    template<class T>
    struct push_back_state_and_time {
      std::vector< std::vector<T> >& m_states;
      std::vector< T >& m_times;
      
      /**
       * Constructor. Passes in the vectors to record
       */
      push_back_state_and_time(std::vector< std::vector<T> > &states,
                               std::vector< T > &times)
        : m_states( states ), 
          m_times( times ) { }
      
      /**
       * Operator that observes values.
       * This is the only required method in this concept.
       */
      void operator()(const std::vector<T> &x, T t) {
        m_states.push_back( x );
        m_times.push_back( t );
      }
      
      /**
       * A getter. This is redundant.
       */
      std::vector<std::vector<T> > get() {
        return m_states;
      }
      
      /**
       * Print method.
       */
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

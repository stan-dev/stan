#ifndef STAN_LANG_AST_NODE_GENERALODEMODEL_CONTROL_HPP
#define STAN_LANG_AST_NODE_GENERALODEMODEL_CONTROL_HPP

#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    struct expression;

    /**
     * Structure for a Torsten generalOdeModel_* function with control
     * parameters for the integrator.
     */
    struct generalOdeModel_control {
      /**
       * The name of the function (* tells us which integrator is being
       * called).
       */
      std::string integration_function_name_;

      /**
       * Name of the ODE system.
       */
      std::string system_function_name_;

      /**
       * Number of compartments/equations (int)
       */
      expression nCmt_;

      /**
       * Time of events (array of real).
       */
      expression time_;

      /**
       * Amount at events (array of real)
       */
      expression amt_;

      /**
       * Rate at events (array of real).
       */
      expression rate_;

      /**
       * Interdose Interval (ii) at event (array of real).
       */
      expression ii_;

      /**
       * Event type (evid) (array of int).
       */
      expression evid_;

      /**
       * Compartment number at event (array of int).
       */
      expression cmt_;

      /**
       * Number of additional doses (array of int).
       */
      expression addl_;

      /**
       * Steady State boolean at event (array of int).
       */
      expression ss_;

      /**
       * ODE parameters (1D or 2D array of real).
       */
      expression pMatrix_;

      /**
       * Biovariability parameters (1D or 2D array of real).
       */
      expression biovar_;

      /**
       * lag time parameters (1D or 2D array of real).
       */
      expression tlag_;

      /**
       * Relative tolerance (real).
       */
      expression rel_tol_;

      /**
       * Absolute tolerance (real).
       */
      expression abs_tol_;

      /**
       * Maximum number of steps (integer).
       */
      expression max_num_steps_;

      /**
       * Construct a default ODE integrator object with control.
       */
      generalOdeModel_control();

      /**
       * Construt an ODE integrator with control parameter with the
       * specified values.
       *
       * @param integration_function_name name of integrator
       * @param f functor for base ordinary differential equation that 
       *   defines compartment model.
       * @param nCmt number of compartments in model
       * @param pMatrix parameters at each event
       * @param time times of events  
       * @param amt amount at each event
       * @param rate rate at each event
       * @param ii inter-dose interval at each event
       * @param evid event identity: 
       *                    (0) observation 
       *                    (1) dosing
       *                    (2) other 
       *                    (3) reset 
       *                    (4) reset AND dosing 
       * @param cmt compartment number at each event 
       * @param addl additional dosing at each event 
       * @param ss steady state approximation at each event (0: no, 1: yes)
       * @param rel_tol relative tolerance for the Boost ode solver 
       * @param abs_tol absolute tolerance for the Boost ode solver
       * @param max_num_steps maximal number of steps to take within 
       *   the Boost ode solver
       */
      generalOdeModel_control(const std::string& integration_function_name,
                              const std::string& system_function_name,
                              const expression& nCmt,
                              const expression& time,
                              const expression& amt,
                              const expression& rate,
                              const expression& ii,
                              const expression& evid,
                              const expression& cmt,
                              const expression& addl,
                              const expression& ss,
                              const expression& pMatrix,
                              const expression& biovar,
                              const expression& tlag,
                              const expression& rel_tol,
                              const expression& abs_tol,
                              const expression& max_steps);
    };

  }
}
#endif

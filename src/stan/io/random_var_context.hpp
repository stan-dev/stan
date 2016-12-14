#ifndef STAN_IO_RANDOM_VAR_CONTEXT_HPP
#define STAN_IO_RANDOM_VAR_CONTEXT_HPP

#include <stan/io/var_context.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <algorithm>
#include <limits>
#include <string>
#include <vector>

namespace stan {
  namespace io {

    /**
     * This is an implementation of a var_context that initializes
     * the unconstrained values randomly. This is used for initialization.
     */
    class random_var_context : public var_context {
    public:
      /**
       * Destructor.
       *
       */
      ~random_var_context() {}

      /**
       * Constructor.
       *
       * On construction, this var_context will generate random
       * numbers on the unconstrained scale for the model provided.
       * Once generated, the class is immutable.
       *
       * This class only generates values for the parameters in the
       * Stan program and does not generate values for transformed parameters
       * or generated quantities.
       *
       * @tparam Model a generated Stan program or a class that follows
       *   that template
       * @tparam RNG Random number generator type. Typically
       *   boost::ecuyer1988
       * @param model instantiated model to generate variables for
       * @param[in,out] rng pseudo-random number generator
       * @param[in] init_radius the unconstrained variables are uniform draws
       *   from -init_radius to init_radius.
       * @param[in] init_zero indicates whether all unconstrained variables
       *   should be initialized at 0. When init_zero is false, init_radius
       *   must be greater than 0.
       */
      template <class Model, class RNG>
      random_var_context(Model& model,
                         RNG& rng,
                         double init_radius,
                         bool init_zero)
        : unconstrained_params_(model.num_params_r()) {
        size_t num_unconstrained_ = model.num_params_r();
        model.get_param_names(names_);
        model.get_dims(dims_);

        // cutting names_ and dims_ down to just the constrained parameters
        std::vector<std::string> constrained_params_names;
        model.constrained_param_names(constrained_params_names, false, false);
        size_t keep = constrained_params_names.size();
        size_t index = 0;
        size_t num = 0;
        for (index = 0; index < dims_.size(); index++) {
          size_t size = 1;
          for (size_t n = 0; n < dims_[index].size(); n++)
            size *= dims_[index][n];
          num += size;
          if (num > keep)
            break;
        }
        dims_.erase(dims_.begin() + index, dims_.end());
        names_.erase(names_.begin() + index, names_.end());

        if (init_zero) {
          for (size_t n = 0; n < num_unconstrained_; n++)
            unconstrained_params_[n] = 0.0;
        } else {
          boost::random::uniform_real_distribution<double>
            unif(-init_radius, init_radius);
          for (size_t n = 0; n < num_unconstrained_; n++)
            unconstrained_params_[n] = unif(rng);
        }

        std::vector<double> constrained_params;
        std::vector<int> int_params;
        model.write_array(rng,
                          unconstrained_params_, int_params,
                          constrained_params,
                          false, false, 0);

        vals_r_.resize(dims_.size());
        std::vector<double>::iterator start = constrained_params.begin();

        for (size_t i = 0; i < dims_.size(); i++) {
          size_t size = 1;
          for (size_t j = 0; j < dims_[i].size(); j++)
            size *= dims_[i][j];
          vals_r_[i] = std::vector<double>(start,
                                           start + size);
          start += size;
        }
      }

      /**
       * Return <code>true</code> if the specified variable name is
       * defined. Will return <code>true</code> if the name matches
       * a parameter in the model.
       *
       * @param name Name of variable.
       * @return <code>true</code> if the name is a parameter in the
       * model.
       */
      bool contains_r(const std::string& name) const {
        return std::find(names_.begin(), names_.end(), name) != names_.end();
      }

      /**
       * Returns the values of the constrained variables.
       *
       * @param name Name of variable.
       *
       * @return the constrained values if the variable is in the
       *   var_context; an empty vector is returned otherwise
       */
      std::vector<double> vals_r(const std::string& name) const {
        size_t index
          = std::find(names_.begin(), names_.end(), name) - names_.begin();
        if (index >= names_.size()) {
          std::vector<double> empty_vals_r;
          return empty_vals_r;
        }
        return vals_r_[index];
      }

      /**
       * Returns the dimensions of the variable
       *
       * @param name Name of variable.
       * @return the dimensions of the variable if it exists; an empty vector
       *   is returned otherwise
       */
      std::vector<size_t> dims_r(const std::string& name) const {
        size_t index
          = std::find(names_.begin(), names_.end(), name) - names_.begin();
        if (index >= names_.size()) {
          std::vector<size_t> dims_r;
          return dims_r;
        }
        return dims_[index];
      }

      /**
       * Return <code>true</code> if the specified variable name has
       * integer values. Always returns <code>false</code>.
       *
       * @param name Name of variable.
       * @return false
       */
      bool contains_i(const std::string& name) const {
        return false;
      }

      /**
       * Returns an empty vector.
       *
       * @param name Name of variable.
       * @return empty vector
       */
      std::vector<int> vals_i(const std::string& name) const {
        std::vector<int> empty_vals_i;
        return empty_vals_i;
      }

      /**
       * Return the dimensions of the specified floating point variable.
       * Returns an empty vector.
       *
       * @param name Name of variable.
       * @return empty vector
       */
      std::vector<size_t> dims_i(const std::string& name) const {
        std::vector<size_t> empty_dims_i;
        return empty_dims_i;
      }

      /**
       * Return a list of the names of the floating point variables in
       * the context. This will return the names of the parameters in
       * the model.
       *
       * @param names Vector to store the list of names in.
       */
      void names_r(std::vector<std::string>& names) const {
        names = names_;
      }

      /**
       * Return a list of the names of the integer variables in
       * the context. This context has no variables.
       *
       * @param names Vector to store the list of names in.
       */
      void names_i(std::vector<std::string>& names) const {
        names.clear();
      }

      std::vector<double> get_unconstrained() const {
        return unconstrained_params_;
      }

    private:
      std::vector<std::string> names_;
      std::vector<std::vector<size_t> > dims_;
      std::vector<std::vector<double> > vals_r_;
      std::vector<double> unconstrained_params_;
    };

  }
}
#endif

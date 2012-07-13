
#ifndef __RSTAN__STAN_FIT_HPP__
#define __RSTAN__STAN_FIT_HPP__

#include <cmath>
#include <cstddef>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <stan/version.hpp>
#include <stan/io/cmd_line.hpp>
#include <stan/io/dump.hpp>
#include <stan/mcmc/adaptive_sampler.hpp>
#include <stan/mcmc/adaptive_hmc.hpp>
#include <stan/mcmc/hmc.hpp>
#include <stan/mcmc/nuts.hpp>
#include <stan/mcmc/nuts_diag.hpp>
#include <stan/model/prob_grad_ad.hpp>
#include <stan/model/prob_grad.hpp>
#include <stan/mcmc/sampler.hpp>

#include <rstan/io/rlist_ref_var_context.hpp> 
#include <rstan/io/r_ostream.hpp> 
#include <rstan/stan_args.hpp> 
#include <rstan/chains_for_R.hpp>
// #include <stan/mcmc/chains.hpp>

#include <Rcpp.h>
// #include <Rinternals.h>

// REF: stan/gm/command.hpp 


namespace rstan {

  namespace { 
    // Potentially this could be problematic for 
    // converting size_t vector to unsigned int. 
    // But given the limitation of R, the values
    // passed from R should be fine using unsigned int 
    // inside stan. 
    template <class T1, class T2> 
    void  T1v_to_T2v(const std::vector<T1>& v,
                     std::vector<T2>& v2) {
      v2.resize(0); 
      for (typename std::vector<T1>::const_iterator it = v.begin();
           it != v.end();
           ++it) { 
        v2.push_back(static_cast<T2>(*it));
      }     
    } 

    bool do_print(int refresh) {
      return refresh > 0;
    }
  
    bool do_print(int n, int refresh) {
      return do_print(refresh)
        && ((n + 1) % refresh == 0);
    }
  
    /*
    void write_comment(std::ostream& o) {
      o << "#" << std::endl;
    }
  
    template <typename M>
    void write_comment(std::ostream& o,
                       const M& msg) {
      o << "# " << msg << std::endl;
    }
  
    template <typename K, typename V>
    void write_comment_property(std::ostream& o,
                                const K& key,
                                const V& val) {
      o << "# " << key << "=" << val << std::endl;
    }
    */

    /** 
     * Print out a std::vector 
     */
    /**
    template <typename T>
    void printv(std::ostream& o, const std::vector<T>& v) {
      for (typename std::vector<T>::const_iterator it = v.begin(); it != v.end(); ++it)
        o << *it << std::endl;
    }
    */
    template <typename T>
    std::string to_string(T i) {
      std::stringstream ss;
      ss << i;
      return ss.str();
    }

    template <class Model>
    std::vector<std::string> get_param_names(Model& m) { 
      std::vector<std::string> names;
      m.get_param_names(names);
      return names; // copy for return
    }

    template <class Model>
    std::vector<std::vector<size_t> > get_param_dims(Model& m) {
      std::vector<std::vector<size_t> > dimss; 
      m.get_dims(dimss); 
      return dimss; 
    } 

    template <class Sampler, class Model, class RNG>
    void sample_from(Sampler& sampler,
                     bool epsilon_adapt,
                     int refresh,
                     int num_iterations,
                     int num_warmup,
                     int num_thin,
                     std::ostream& sample_file_stream,
                     bool sample_file_flag, 
                     std::vector<double>& params_r,
                     std::vector<int>& params_i,
                     Model& model,
                     stan::mcmc::chains<RNG>& chains,
                     unsigned int chain_id) {
  
      sampler.set_params(params_r,params_i);

     
      int it_print_width = std::ceil(std::log10(num_iterations));
      rstan::io::rcout << std::endl;
  
      // rstan::io::rcout << "in sample_from." << std::endl; 
      if (epsilon_adapt)
        sampler.adapt_on(); 
      else 
        sampler.adapt_off(); 
      std::vector<double> params_inr; 
      for (int m = 0; m < num_iterations; ++m) {
        if (do_print(m,refresh)) {
          rstan::io::rcout << "\rIteration: ";
          rstan::io::rcout << std::setw(it_print_width) << (m + 1)
                           << " / " << num_iterations;
          rstan::io::rcout << " [" << std::setw(3)
                           << static_cast<int>((100.0 * (m + 1))/num_iterations)
                           << "%] ";
          rstan::io::rcout << ((m < num_warmup) ? " (Adapting)" : " (Sampling)");
          rstan::io::rcout.flush();
        } 

        if (m == num_warmup && epsilon_adapt) 
          sampler.adapt_off();

        if ((m % num_thin) != 0) {
          sampler.next();
          continue;
        }

        stan::mcmc::sample sample = sampler.next();
        sample.params_r(params_r);
        sample.params_i(params_i);
        model.write_array(params_r,params_i,params_inr); 
        chains.add(chain_id - 1, params_inr); 
  
        // FIXME: use csv_writer arg to make comma optional?
        if (sample_file_flag) { 
          sample_file_stream << sample.log_prob() << ',';
          sampler.write_sampler_params(sample_file_stream);
          model.write_csv(params_r,params_i,sample_file_stream);
        }
      }
    }

    /**
     * @tparam Model 
     * @tparam RNG RNG for stan::mcmc::chains 
     */
    
    template <class Model, class RNG> 
    int sampler_command(const io::rlist_ref_var_context& data, 
                       const stan_args& args, 
                       Model& model, 
                       stan::mcmc::chains<RNG>& chains_) {

      bool sample_file_flag = args.get_sample_file_flag(); 
      std::string sample_file = args.get_sample_file(); 

      unsigned int num_iterations = args.get_iter(); 
      unsigned int num_warmup = args.get_warmup(); 
      unsigned int num_thin = args.get_thin(); 
      int leapfrog_steps = args.get_leapfrog_steps(); 

      unsigned int random_seed = args.get_random_seed();

      double epsilon = args.get_epsilon(); 
      bool epsilon_adapt = epsilon <= 0.0; 

      bool unit_mass_matrix = args.get_unit_mass_matrix();

      int max_treedepth = args.get_max_treedepth(); 

      double epsilon_pm = args.get_epsilon_pm(); 


      double delta = args.get_delta(); 

      double gamma = args.get_gamma(); 

      int refresh = args.get_refresh(); 

      unsigned int chain_id = args.get_chain_id(); 

      // FASTER, but no parallel guarantees:
      // typedef boost::mt19937 rng_t;
      // rng_t base_rng(static_cast<size_t>(seed_ + chain_id - 1);

      typedef boost::ecuyer1988 rng_t;
      rng_t base_rng(random_seed);
      // (2**50 = 1T samples, 1000 chains)
      static boost::uintmax_t DISCARD_STRIDE = 
        static_cast<boost::uintmax_t>(1) << 50;
      // rstan::io::rcout << "DISCARD_STRIDE=" << DISCARD_STRIDE << std::endl;

      base_rng.discard(DISCARD_STRIDE * (chain_id - 1));
      
      std::vector<int> params_i;
      std::vector<double> params_r;

      std::string init_val = args.get_init();
      // parameter initialization
      if (init_val == "0") {
          params_i = std::vector<int>(model.num_params_i(),0);
          params_r = std::vector<double>(model.num_params_r(),0.0);
      } else if (init_val == "user") {
          Rcpp::List init_lst(args.get_init_list()); 
          rstan::io::rlist_ref_var_context init_var_context(init_lst); 
          model.transform_inits(init_var_context,params_i,params_r);
      } else {
        init_val = "random"; 
        // init_rng generates uniformly from -2 to 2
        boost::random::uniform_real_distribution<double> 
          init_range_distribution(-2.0,2.0);
        boost::variate_generator<rng_t&, 
                       boost::random::uniform_real_distribution<double> >
          init_rng(base_rng,init_range_distribution);

        params_i = std::vector<int>(model.num_params_i(),0);
        params_r = std::vector<double>(model.num_params_r());

        for (size_t i = 0; i < params_r.size(); ++i)
          params_r[i] = init_rng();
      }

      std::fstream sample_stream; 
      bool append_samples = args.get_append_samples(); 
      if (sample_file_flag) {
        std::ios_base::openmode samples_append_mode
          = append_samples
          ? (std::fstream::out | std::fstream::app)
          : std::fstream::out;
        
        sample_stream.open(sample_file.c_str(), 
                           samples_append_mode);
        
        write_comment(sample_stream,"Samples Generated by Stan");
        write_comment_property(sample_stream,"stan_version_major",stan::MAJOR_VERSION);
        write_comment_property(sample_stream,"stan_version_minor",stan::MINOR_VERSION);
        write_comment_property(sample_stream,"stan_version_patch",stan::PATCH_VERSION);
        args.write_args_as_comment(sample_stream); 
      } 

      if (0 > leapfrog_steps && !unit_mass_matrix) {
        // NUTS II (with diagonal mass matrix estimation during warmup)
        stan::mcmc::nuts_diag<rng_t> nuts2_sampler(model, 
                                                   max_treedepth, epsilon, 
                                                   epsilon_pm, epsilon_adapt,
                                                   delta, gamma, 
                                                   base_rng);

        // cut & paste (see below) to enable sample-specific params
        if (sample_file_flag && !append_samples) {
          sample_stream << "lp__,"; // log probability first
          nuts2_sampler.write_sampler_param_names(sample_stream);
          model.write_csv_header(sample_stream);
        }

        sample_from(nuts2_sampler,epsilon_adapt,refresh,
                    num_iterations,num_warmup,num_thin,
                    sample_stream,sample_file_flag,params_r,params_i,
                    model,chains_,chain_id); 

  
      } else if (0 > leapfrog_steps && unit_mass_matrix) {
        // NUTS I (unit mass matrix)
        stan::mcmc::nuts<rng_t> nuts_sampler(model, 
                                             max_treedepth, epsilon, 
                                             epsilon_pm, epsilon_adapt,
                                             delta, gamma, 
                                             base_rng);

        // cut & paste (see below) to enable sample-specific params
        if (sample_file_flag && !append_samples) {
          sample_stream << "lp__,"; // log probability first
          nuts_sampler.write_sampler_param_names(sample_stream);
          model.write_csv_header(sample_stream);
        }

        sample_from(nuts_sampler,epsilon_adapt,refresh,
                    num_iterations,num_warmup,num_thin,
                    sample_stream,sample_file_flag,params_r,params_i,
                    model,chains_,chain_id); 
      } else {
        // Stardard HMC
        stan::mcmc::adaptive_hmc<rng_t> hmc_sampler(model,
                                                    leapfrog_steps,
                                                    epsilon, epsilon_pm, epsilon_adapt,
                                                    delta, gamma,
                                                    base_rng);

        // cut & paste (see above) to enable sample-specific params
        if (sample_file_flag && !append_samples) {
          sample_stream << "lp__,"; // log probability first
          hmc_sampler.write_sampler_param_names(sample_stream);
          model.write_csv_header(sample_stream);
        }

        sample_from(hmc_sampler,epsilon_adapt,refresh,
                    num_iterations,num_warmup,num_thin,
                    sample_stream,sample_file_flag,params_r,params_i,
                    model,chains_,chain_id);
      }
      
      if (sample_file_flag) {
        rstan::io::rcout << std::endl << "Samples of chain " 
                         << chain_id 
                         << " is written to file " << sample_file;

        sample_stream.close();
      }
      rstan::io::rcout << std::endl << std::endl; 
      return 0;
    }
  } 

  /**
   * <p> To implement a Rcpp class module for R's user interface with 
   * samplers in stan (HMC, NUTS-I, NUTS-II).  
   * Adapted from <code> stan/src/stan/gm/command.hpp</code>. 
   * 
   * @tparam Model The model translated from the Stan language.
   * @tparam RNG RNG for stan::mcmc::chains 
   *
   */

  template <class Model, class RNG> 
  class stan_fit {

  private:
    io::rlist_ref_var_context data_;
    std::vector<std::string>  names_;
    Model model_;
    unsigned int num_chains_; 
    std::map<unsigned int, stan_args> argss_; 
    // std::vector<stan_args> argss_;
    // stan::mcmc::chains<RNG> chains_; 
    chains_for_R<RNG> chains_; 

    std::vector<std::string> flatnames_; 

  private: 
  
    /**
     * Tell if a parameter name is an element of an array parameter. 
     * Note that it only supports full specified name; slicing 
     * is not supported. The test only tries to see if there 
     * are brackets. 
     */
  
    bool is_flatname(const std::string& name) {
      return name.find('[') != name.npos && name.find(']') != name.npos; 
    } 
 
    /**
     * Obtain the dimensions for parameters.
     *
     * @param names[in] Names of parameters of interest. Note here
     *  the name could be an element of an array parameter, 
     *  for example a[5], b[3,4] are legitimate names. 
     * @param names2[out] Names of those parameters that we 
     *  do have. This is just in case that we requets some
     *  parameter that are not in the model. 
     * @param dimms[out] Dimensions of the parameters requsted. 
     *  For scalars and element of array parameters, it is a 
     *  empty vector if size_t. 
     * 
     ** @tparam T the type of the dimensions. In Stan, size_t is 
     **  used.  But when communicating between R, Rcpp does
     **  not support wrap/as size_t (as of Thu Jul 12 14:33:32 EDT 2012), 
     **  in which case, unsigned int could be used. 
     */ 

    /*
    template <class T>
     */
    void param_dimss(const std::vector<std::string>& names,
                     std::vector<std::string>& names2,
                     std::vector<std::vector<size_t> >& dimss) {
      dimss.resize(0); 
      for (std::vector<std::string>::const_iterator it = names.begin();
           it != names.end(); 
           ++it) {
        if (is_flatname(*it)) { // an element of an array 
          size_t ts = std::distance(flatnames_.begin(),
                                    std::find(flatnames_.begin(), 
                                              flatnames_.end(), *it));       
          if (ts == flatnames_.size()) // not found 
            continue; 
          names2.push_back(*it); 
          dimss.push_back(std::vector<size_t>());
          continue;
        }
        size_t j = chains_.param_name_to_index(*it);
        dimss.push_back(chains_.param_dims(j)); 
        names2.push_back(*it); 
      }
    } 
   
    /**
     * Obtain the total indices for parameters.
     *
     * @param names[in] Names of parameters of interest. Note here
     *  the name could be an element of an array parameter, 
     *  for example a[5], b[3,4] are legitimate names. 
     * @param names2[out] Names of those parameters that we 
     *  do have. This is just in case that we request 
     *  parameters that are not in the model. 
     * @param indices Indices for the parameter requested. 
     *  The indices for each parameter is a vector, the min 
     *  length of which is 1. 
     */ 
    void param_total_indices(const std::vector<std::string>& names,
                             std::vector<std::string>& names2, 
                             std::vector<std::vector<size_t> >& indices) {
      names2.resize(0);
      indices.resize(0);
      for (std::vector<std::string>::const_iterator it = names.begin();
           it != names.end(); 
           ++it) {
        if (is_flatname(*it)) { // an element of an array  
          size_t ts = std::distance(flatnames_.begin(),
                                    std::find(flatnames_.begin(), 
                                              flatnames_.end(), *it));       
          if (ts == flatnames_.size()) // not found 
            continue; 
          names2.push_back(*it); 
          indices.push_back(std::vector<size_t>(1, ts)); 
          continue;
        }
        size_t j = chains_.param_name_to_index(*it);
        size_t j_size = chains_.param_size(j); 
        size_t j_start = chains_.param_start(j); 
        std::vector<size_t> j_idx; 
        for (size_t k = 0; k < j_size; k++) {
          j_idx.push_back(j_start + k); 
        } 
        names2.push_back(*it); 
        indices.push_back(j_idx); 
      }
    } 

    /* Obtain the indices and flatnames for a vector of parameter names. 
     * @param names[in] Names of parameters of interests 
     * @param indices[out] The indices for all parameters in the overall
     * samples. Note the index here is the index as in 
     * <code>stan::mcmc::chains::get_total_param_index</code>; 
     * but not the index in 
     * <code>stan::mcmc::chains::param_name_to_index</code>.  
    
     * @param flatnames[out] Flatnames for all the names. That is, 
     * if parameter a is of length 3, it would be added as 
     * a[1], a[2], a[3]. 
     *
     */
  
    void param_names_to_indices_and_flatnames(
      const std::vector<std::string>& names, 
      std::vector<size_t>& indices,
      std::vector<std::string>& flatnames) {
      indices.resize(0);
      flatnames.resize(0);

      for (std::vector<std::string>::const_iterator it = names.begin();
           it != names.end(); 
           ++it) {
        if (it -> find('[') != it -> npos && 
            it -> find(']') != it -> npos) { // already a flatname 
          size_t ts = std::distance(flatnames_.begin(),
                                    std::find(flatnames_.begin(), 
                                              flatnames_.end(), *it));       
          if (ts == flatnames_.size()) // not found 
            continue; 
          flatnames.push_back(*it); 
          indices.push_back(ts); 
          continue;
        } 

        size_t j = chains_.param_name_to_index(*it);
        std::vector<size_t> j_dims = chains_.param_dims(j); 

        std::vector<std::vector<size_t> > j_idx;  
        expand_indices(j_dims, j_idx); // col_major = false 

        for (std::vector<std::vector<size_t> >::const_iterator it = j_idx.begin(); 
             it != j_idx.end(); 
             ++it) { 
          size_t total_idx = chains_.get_total_param_index(j, *it); 
          indices.push_back(total_idx); 
          flatnames.push_back(flatnames_[total_idx]); 
        } 
      }
    }

    /**
     * Obtain kept samples by index from a chain for one parameter. 
     *
     * @param k Index of chain (starting from 0).
     * @param n Indexe of a parameter (starting from 0). 
     * @return A vector of samples in form of R's numeric vector. 
     *
     */
    SEXP get_chain_samples_0(size_t k, size_t n, bool keep_warmup = true) {
      std::vector<double> s; 
      if (keep_warmup) 
        chains_.get_samples(k, n, s);
      else
        chains_.get_kept_samples(k, n, s);
      return Rcpp::wrap(s);
    } 

    /**
     * Obtain kept samples by index from a chain. 
     *
     * @param k Index of chain (starting from 0).
     * @param ns Indexes of parameters (starting from 0). 
     * @return A vector of samples in form of R's numeric vector, in which
     *  samples for multiple parameters are concatenated. 
     *
     */
    SEXP get_chain_samples_0(size_t k, 
                             const std::vector<size_t>& ns, 
                             bool keep_warmup = true) {
      size_t num = keep_warmup
                   ? chains_.num_samples(k) 
                   : chains_.num_kept_samples(k);
      Rcpp::NumericVector nv(ns.size() * num); 
      Rcpp::NumericVector::iterator nv_it = nv.begin(); 
      for (std::vector<size_t>::const_iterator it = ns.begin(); 
           it != ns.end(); 
           ++it) {
        std::vector<double> s; 
        if (keep_warmup) 
          chains_.get_samples(k, *it, s);
        else
          chains_.get_kept_samples(k, *it, s);
        for (std::vector<double>::const_iterator sit = s.begin(); 
             sit != s.end(); 
             ++sit) {
          *nv_it++ = *sit; 
        }
      } 
      return Rcpp::wrap(nv);    
    } 

    /**
     * @param k The chain id, starting from 0.
     * @param n The parameter index 
     * @param probs Probabilities specifying quantiles of interest. 
     * @return An R vector of quantiles 
     */

    SEXP get_chain_quantiles_0(
      size_t k, size_t n, const std::vector<double>& probs) {

      std::vector<double> qois; 
      chains_.quantiles(k, n, probs, qois); 
      return Rcpp::wrap(qois); 
    } 

    /**
     * @param n The parameter index.
     * @param probs Probabilities specifying quantiles of interest.
     * @return An R vector of quantiles.  
     */

    SEXP get_quantiles_0(size_t n, const std::vector<double>& probs) {

      std::vector<double> qois; 
      chains_.quantiles(n, probs, qois); 
      return Rcpp::wrap(qois); 
    } 

    /**
     * Return the kept samples permuted for multiple parameters. 
     *
     * @param ns The total indices of parameters. 
     * @return An R vector of samples that are permuted. 
     *  The samples are concatenated into one vector. 
     */
    SEXP get_kept_samples_permuted_0(const std::vector<size_t>& ns) {
      size_t num = chains_.num_kept_samples(); 
      Rcpp::NumericVector nv(ns.size() * num); 
      Rcpp::NumericVector::iterator nv_it = nv.begin(); 
      for (std::vector<size_t>::const_iterator it = ns.begin(); 
           it != ns.end(); 
           ++it) {
        std::vector<double> s; 
        chains_.get_kept_samples_permuted(*it, s); 
        for (std::vector<double>::const_iterator sit = s.begin(); 
             sit != s.end(); 
             ++sit) {
          *nv_it++ = *sit; 
        }
      }
      return Rcpp::wrap(nv);
    }

    SEXP get_kept_samples_permuted_0(size_t n) { 
      std::vector<double> s; 
      chains_.get_kept_samples_permuted(n, s); 
      return Rcpp::wrap(s);
    } 

  public:

    /**
     * @param data The data for the model. From R's perspective, 
     *  it is a named list. 
     *
     * @param n_chains The number of chains. 
     *
     */ 

    stan_fit(SEXP data, SEXP n_chains) : // try : 
      data_(Rcpp::as<Rcpp::List>(data)), 
      names_(get_param_names(model_)), 
      model_(data_), 
      num_chains_(Rcpp::as<unsigned int>(n_chains)), 
      chains_(num_chains_, names_, get_param_dims(model_)) 
    {  

      std::vector<std::string> names = chains_.param_names();
      for (std::vector<std::string>::const_iterator it = names.begin();
           it != names.end(); 
           ++it) {
        size_t j = chains_.param_name_to_index(*it);
        std::vector<size_t> j_dims = chains_.param_dims(j); 
        std::vector<std::string> j_n;  
        get_col_major_names(*it, j_dims, j_n);
        flatnames_.insert(flatnames_.end(), j_n.begin(), j_n.end()); 
      }

      // argss_.resize(0); 
    }/* catch (std::exception& e) {
      rstan::io::rcerr << std::endl << "Exception: " 
                       << e.what() << std::endl;
      rstan::io::rcerr << "Diagnostic information: " << std::endl
                       << boost::diagnostic_information(e) << std::endl;
      throw; 
    } */ 
    // not really helpful of using try---catch though it could throw
    // exception in the ctor.


    /**
     * Get the arguments used for sampling a chain. 
     * 
     * @param chain_id The chain ID stariting from 1.
     * @return An R list providing the arguments used for the chain.
  
     */ 
  
    SEXP get_chain_stan_args(SEXP chain_id) {
      unsigned int k = Rcpp::as<unsigned int>(chain_id); 
      std::map<unsigned int, stan_args>::const_iterator it
        = argss_.find(k);  
      if (it != argss_.end()) 
        return (it -> second).stan_args_to_rlist(); 
      rstan::io::rcerr << "error: chain id " << chain_id 
                       << " not found." << std::endl;
      return R_NilValue;
    }
     
    /**
     * Get the arguments used for sampling all the chains. 
     * 
     * @return An R list providing the arguments used for all the chain.
     *  each element of the R list is an R list for one chain. 
  
     */ 

    SEXP get_stan_args() { 
      Rcpp::List lst; 
      std::vector<std::string> cnames(num_chains_, "chain."); 

      std::string cname("chain."); 
      for (std::map<unsigned int, stan_args>::const_iterator it = argss_.begin(); 
           it != argss_.end(); 
           ++it)   
        lst[cname + to_string(it -> first)] = (it -> second).stan_args_to_rlist(); 

      return lst;
    }
     

    /**
     * This function would be exposed (using Rcpp module, see
     * <code>rcpp_module_def_for_rstan.hpp</code>) to R to call 
     * methods defined here. 
     *
     *
     * @param args The arguments for nuts in form of R's list. 
     * @return TRUE if there is no exception; FALSE otherwise. 
     *
     */
    SEXP call_sampler(SEXP args) { 

      stan_args t(Rcpp::as<Rcpp::List>(args)); 
      // set the seeds to be the same for all chains
      if (!argss_.empty()) 
        t.set_random_seed((argss_.begin() -> second).get_random_seed()); 

      unsigned int c_id = t.get_chain_id(); 
      // rstan::io::rcout << "chain id = " << c_id << std::endl;
      if (c_id > num_chains_) { 
        rstan::io::rcerr << "chain id cannot be larger than # of chains"
                         << "; chain_id = " << c_id 
                         << ", num_chains = " << num_chains_  << "."
                         << std::endl;
        return Rcpp::wrap(false);
      } 
      if (argss_.count(c_id)) {
        rstan::io::rcerr << "chain of id " << c_id 
                         << " was sampled before." << std::endl;
        return Rcpp::wrap(false);
      } 
      argss_.insert(std::map<unsigned int, stan_args>::value_type(c_id, t));
     
      // assuming that the warmup are set all the same for 
      // all the chains or simply here we only use one
      if (1 == argss_.size())
        chains_.set_warmup(t.get_warmup() / t.get_thin()); 

      try {
        sampler_command(data_, t, model_, chains_); 
     
      } catch (std::exception& e) {
        rstan::io::rcerr << std::endl << "Exception: " << e.what() << std::endl;
        rstan::io::rcerr << "Diagnostic information: " << std::endl << boost::diagnostic_information(e) << std::endl;
        return Rcpp::wrap(false); 
      }
      return Rcpp::wrap(true);  
    } 


    /** 
     * Obtain samples by names from a chain.  
     * 
     * @param chain_id  The chain id starting from 1.
     * @param names The names of parameter of interests. 
     * @param keep_warmup Whether to keep the warmup samples. 
     * @param expand  See comments in <code>get_samples</code>.
     * 
     * @return An R list, each element of which includes the samples of one
     *  parameter. Note if a parameter is a vector or array, 
     *  when expand is TRUE, all the samples are concatenated so that when
     *  returned to R, after the dimension attribute is set, it is an array. Or
     *  when expand is FALSE, parameter are expanded with indices. 
     */

    SEXP get_chain_samples(SEXP chain_id, SEXP names, 
                           SEXP keep_warmup = Rcpp::wrap(true),
                           SEXP expand = Rcpp::wrap(false)) {
      size_t k = Rcpp::as<unsigned int>(chain_id) - 1;  // make it start from 0
      bool kw = Rcpp::as<bool>(keep_warmup); 
      bool ep = Rcpp::as<bool>(expand); 
      std::vector<SEXP> samples; 
      std::vector<std::string> names2; 
      if (!ep) { 
        std::vector<std::vector<size_t> > indices; 
        param_total_indices(Rcpp::as<std::vector<std::string> >(names), 
                          names2, indices); 
        for (size_t i = 0; i < names2.size(); i++) {
          samples.push_back(get_chain_samples_0(k, indices[i], kw)); 
        } 
      } else {
        std::vector<size_t> indices; 
        param_names_to_indices_and_flatnames(
          Rcpp::as<std::vector<std::string> >(names),
          indices, 
          names2); 
        for (std::vector<size_t>::const_iterator it = indices.begin(); 
             it != indices.end(); 
             ++it) {
          samples.push_back(get_chain_samples_0(k, *it, kw));
        }
      } 
      Rcpp::List lst(samples.begin(), samples.end());
      lst.names() = names2; 
      return Rcpp::wrap(lst);
    } 

    /**
     * Obtain the parameter's dimensions. Also an element of an
     * array parameter is allowed, but the returned dimension 
     * for such as parameter, i.e., alpha[1], is an empty vector. 
     *
     * @param names The names of parameters of interests, which 
     *  could be scalar, array, or an element of an array. 
     */
    SEXP get_param_dimss(SEXP names) {
      std::vector<std::string> names2; 
      std::vector<std::vector<size_t> > dimss; 
      param_dimss(Rcpp::as<std::vector<std::string> >(names), names2, dimss); 
      std::vector<SEXP>  dimss2; 
      for (std::vector<std::vector<size_t> >::const_iterator it = dimss.begin(); 
           it != dimss.end(); 
           ++it) {
        std::vector<unsigned int> v2; 
        T1v_to_T2v(*it, v2); 
        dimss2.push_back(Rcpp::wrap(v2)); 
        // Cast size_t to unsigned int, which potentially is problematic. 
        // But Rcpp (and/or? R) could not deal with size_t on windows. 
      }
      Rcpp::List lst(dimss2.begin(), dimss2.end()); 
      lst.names() = names2;
      return Rcpp::wrap(lst);
    } 

    /** 
     * Obtain samples for all the chains 
     * 
     * @param expand TRUE means that the samples for 
     *   an array parameter would be expanded as multiple
     *   parameters. So parameter alpha[5] defined in stan
     *   model would have vector in a list named from alpha[1]
     *   to alpha[5]. When expand is FALSE, it would only 
     *   appear as alpha, but in a form of multiple array. 
     * 
     * @return A list, each element of which are samples of a chain. 
     *  The element is also a list, each element of which is a vector 
     *  of samples for a parameter (or other quantity of interest). 
     *  In the case of array parameters, the vectors are concatenated. 
     *
     */
    SEXP get_samples(SEXP names, 
                     SEXP keep_warmup = Rcpp::wrap(true), 
                     SEXP expand = Rcpp::wrap(false)) {
      Rcpp::List lst(num_chains_); 
      std::vector<std::string> cnames(num_chains_, "chain."); 
      for (unsigned int i = 0; i < num_chains_; ++i) {
        lst[i] = get_chain_samples(Rcpp::wrap(i + 1), names, keep_warmup, expand);
        cnames[i] += to_string(i + 1);  
      }
      lst.names() = cnames; 
      return lst;
    } 


    /**
     * Get the quantiles of the samples from all chains. 
     *
     * @param names An R vector of parameter names 
     * @param probs An R vector of probabilities specifying
     *  quantiles of interest. 
     *
     */
    SEXP get_quantiles(SEXP names, SEXP probs) {
      
      std::vector<size_t> indices; 
      std::vector<std::string> flatnames; // names for the returned samples 
      param_names_to_indices_and_flatnames(
        Rcpp::as<std::vector<std::string> >(names),
        indices, 
        flatnames); 

     std::vector<double> ps = Rcpp::as<std::vector<double> >(probs); 
    
     std::vector<SEXP> quanss; 
     for (std::vector<size_t>::const_iterator it = indices.begin(); 
          it != indices.end(); 
          ++it) {
       quanss.push_back(get_quantiles_0(*it, ps)); 
         
     } 
     Rcpp::List lst(quanss.begin(), quanss.end());
     lst.names() = flatnames; 
     return Rcpp::wrap(lst);
    } 


    /**
     * Get the quantiles of the samples of one chain: 
     *
     * @param chain_id The chain id from R starting at 1 
     * @param names An R vector of parameter names 
     * @param probs An R vector of probabilities specifying
     *  quantiles of interest. 
     *
     */
    SEXP get_chain_quantiles(SEXP chain_id, SEXP names, SEXP probs) {
      
      size_t k = Rcpp::as<unsigned int>(chain_id) - 1;  // make it start from 0

      std::vector<size_t> indices; 
      std::vector<std::string> flatnames; // names for the returned samples 
      param_names_to_indices_and_flatnames(
        Rcpp::as<std::vector<std::string> >(names),
        indices, 
        flatnames); 

     std::vector<double> ps = Rcpp::as<std::vector<double> >(probs); 
    
     std::vector<SEXP> quanss; 
     for (std::vector<size_t>::const_iterator it = indices.begin(); 
          it != indices.end(); 
          ++it) {
       quanss.push_back(get_chain_quantiles_0(k, *it, ps)); 
         
     } 
     Rcpp::List lst(quanss.begin(), quanss.end());
     lst.names() = flatnames; 
     return Rcpp::wrap(lst);
    } 

    /**
     * Get the mean and standard deviation for samples of one chain
     * 
     * @param names An R vector of names specifying parameters of 
     *  interest
     * @param chain_id The chain id starting from 1. 
     * @return An R list, each element of which contains a vector for 
     *  a parameter. The fist element of the vector is mean and the 
     *  second is the SD. 
     */
    SEXP get_chain_mean_and_sd(SEXP chain_id, SEXP names) {
      size_t k = Rcpp::as<unsigned int>(chain_id) - 1;  // make it start from 0

      std::vector<size_t> indices; 
      std::vector<std::string> flatnames; 
      param_names_to_indices_and_flatnames(
        Rcpp::as<std::vector<std::string> >(names),
        indices, 
        flatnames); 

      std::vector<SEXP> mnsds;  //mean and sd's 
      for (std::vector<size_t>::const_iterator it = indices.begin(); 
           it != indices.end(); 
           ++it) {
        Rcpp::NumericVector v(2); 
        v[0] = chains_.mean(k, *it);
        v[1] = chains_.sd(k, *it);
        mnsds.push_back(Rcpp::wrap(v)); 
         
      } 
      Rcpp::List lst(mnsds.begin(), mnsds.end());
      lst.names() = flatnames; 
      return Rcpp::wrap(lst);
    } 

    /**
     * Get the mean and standard deviation for samples of all chains
     * 
     * @param names An R vector of names specifying parameters of 
     *  interest
     * @return An R list, each element of which contains a vector for 
     *  a parameter. The fist element of the vector is mean and the 
     *  second is the SD. 
     */
    SEXP get_mean_and_sd(SEXP names) {
      std::vector<size_t> indices; 
      std::vector<std::string> flatnames; 
      param_names_to_indices_and_flatnames(
        Rcpp::as<std::vector<std::string> >(names),
        indices, 
        flatnames); 

      std::vector<SEXP> mnsds;  
      for (std::vector<size_t>::const_iterator it = indices.begin(); 
           it != indices.end(); 
           ++it) {
        Rcpp::NumericVector v(2); 
        v[0] = chains_.mean(*it);
        // rstan::io::rcout << "v[0] = " << v[0] << std::endl;
        v[1] = chains_.sd(*it);
        // rstan::io::rcout << "v[1] = " << v[1] << std::endl;
        mnsds.push_back(Rcpp::wrap(v)); 
         
      } 
      Rcpp::List lst(mnsds.begin(), mnsds.end());
      lst.names() = flatnames; 
      return Rcpp::wrap(lst);
    } 

    /**
     * Get the effective sample size (ESS). 
     * 
     * @param names An R vector of paramemter names. 
     * @return The ESS for all the paramemters in form of an R list, every
     *  element of which is the ESS for a paramemter. 
     */
    SEXP get_ess(SEXP names) {
      std::vector<size_t> indices; 
      std::vector<std::string> flatnames; 
      param_names_to_indices_and_flatnames(
        Rcpp::as<std::vector<std::string> >(names),
        indices, 
        flatnames); 

      std::vector<SEXP> esss;  
      for (std::vector<size_t>::const_iterator it = indices.begin(); 
           it != indices.end(); 
           ++it) {
        esss.push_back(Rcpp::wrap(chains_.effective_sample_size(*it))); 
         
      } 
      Rcpp::List lst(esss.begin(), esss.end());
      lst.names() = flatnames; 
      return Rcpp::wrap(lst);
    } 

    /**
     * Get the split R hat (the split potential scale reduction)
     * 
     * @param names An R vector of paramemter names. 
     * @return The R hat's for all the paramemters in form of an R list, every
     *  element of which is the ESS for a paramemter. 
     */
    SEXP get_split_rhat(SEXP names) {
      std::vector<size_t> indices; 
      std::vector<std::string> flatnames; 
      param_names_to_indices_and_flatnames(
        Rcpp::as<std::vector<std::string> >(names),
        indices, 
        flatnames); 

      std::vector<SEXP> rhats;  
      for (std::vector<size_t>::const_iterator it = indices.begin(); 
           it != indices.end(); 
           ++it) {
        rhats.push_back(Rcpp::wrap(chains_.split_potential_scale_reduction(*it))); 
         
      } 
      Rcpp::List lst(rhats.begin(), rhats.end());
      lst.names() = flatnames; 
      return Rcpp::wrap(lst);
    } 

    /**
     * Get all the parameter names that are in the chains object 
     * @return An R vector of names of the parameters.
     */
    SEXP param_names() {
      std::vector<std::string> names(chains_.param_names());
      return Rcpp::wrap(names); 
    } 

    /**
     * Get all the parameter names in forms of `flat names`,
     * i.e., beta of length 3 would be beta[1], beta[2], beta[3]
     *
     * @return An R vector of names of the parameters.
     */

    SEXP param_flat_names() {
      return Rcpp::wrap(flatnames_); 
    } 

    /**
     * Return the warmup for the stored samples. 
     *
     * @return Number of warmup iterations. 
     */
    SEXP warmup() {
      return Rcpp::wrap(static_cast<unsigned int>(chains_.warmup())); 
    } 

    /**
     * Return the number of samples including warmup and kept samples
     * in the specified chain.
     *
     * @param k Markov chain index, starting from 1. 
     *  Note that in Stan, size_t is used, but Rcpp has some issue on some
     *  platforms, so here unsigned int is used. 
     * @return Number of samples in the specified chain.
     * @throw std::out_of_range If the identifier is greater than
     * or equal to the number of chains.
     */
    
    SEXP num_chain_samples(unsigned int k) {
      return Rcpp::wrap(static_cast<unsigned int>(chains_.num_samples(k - 1))); 
    } 

    SEXP num_chain_kept_samples(unsigned int k) {
      return Rcpp::wrap(static_cast<unsigned int>(chains_.num_kept_samples(k - 1))); 
    } 

    SEXP num_samples() { 
      return Rcpp::wrap(static_cast<unsigned int>(chains_.num_samples())); 
    } 

    SEXP num_kept_samples() {
      return Rcpp::wrap(static_cast<unsigned int>(chains_.num_kept_samples())); 
    } 

    /* Return the kept samples permuted for parameters.
     * 
     * @param names The names of parameters of interest. 
     * @return An R list, every element of which is an 
     *  R vector of samples that are permuted for one parameter.
     */
    SEXP get_kept_samples_permuted(SEXP names, SEXP expand) {
      std::vector<std::string> names2; 

      std::vector<SEXP> samples; 
      bool ep = Rcpp::as<bool>(expand); 
      if (!ep) { 
        std::vector<std::vector<size_t> > indices; 
        param_total_indices(Rcpp::as<std::vector<std::string> >(names), 
                            names2, indices); 

        for (size_t i = 0; i < names2.size(); i++) {
          samples.push_back(get_kept_samples_permuted_0(indices[i]));
        } 
      } else {
        std::vector<size_t> indices; 
        param_names_to_indices_and_flatnames(
          Rcpp::as<std::vector<std::string> >(names),
          indices, 
          names2); 
        for (std::vector<size_t>::const_iterator it = indices.begin(); 
             it != indices.end(); 
             ++it) {
          samples.push_back(get_kept_samples_permuted_0(*it)); 
        }
      } 
      Rcpp::List lst(samples.begin(), samples.end());
      lst.names() = names2; 
      return Rcpp::wrap(lst);
    } 
  };
} 

#endif 


/**
 * Part of the rstan package for an R interface to Stan 
 * Copyright (C) 2012 Columbia University
 * 
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
**/

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
//#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <stan/version.hpp>
//#include <stan/io/cmd_line.hpp>
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
// #include <rstan/chains_for_R.hpp>
// #include <stan/mcmc/chains.hpp>
#include <Rcpp.h>
// #include <Rinternals.h>

//http://cran.r-project.org/doc/manuals/R-exts.html#Allowing-interrupts
#include <R_ext/Utils.h>
// void R_CheckUserInterrupt(void);


// REF: stan/gm/command.hpp 

namespace rstan {

  namespace {
    template <class T> 
    T product(std::vector<T> dims) {
      T y = 1U;
      for (size_t i = 0; i < dims.size(); ++i)
        y *= dims[i];
      return y;
    }

    /**
     *  Get the parameter indexes for a vector(array) parameter.
     *  For example, we have parameter beta, which has 
     *  dimension [2,3]. Then this function gets 
     *  the indexes as (if col_major = false)
     *  [0,0], [0,1], [0,2] 
     *  [1,0], [1,1], [1,2] 
     *  or (if col_major = true) 
     *  [0,0], [1,0] 
     *  [0,1], [1,1] 
     *  [0,2], [121] 
     *
     *  @param dim[in] the dimension of parameter
     *  @param idx[out] for keeping all the indexes
     *
     *  <p> when idx is empty (size = 0), idx 
     *  would be inserted an empty vector. 
     * 
     *
     */
    
    template <class T>
    void expand_indices(std::vector<T> dim,
                        std::vector<std::vector<T> >& idx,
                        bool col_major = false) {
      size_t len = dim.size();
      idx.resize(0);
      size_t total = product(dim);
      std::vector<size_t> loopj;
      for (size_t i = 1; i <= len; ++i)
        loopj.push_back(len - i);
    
      if (col_major)
        for (size_t i = 0; i < len; ++i)
          loopj[i] = len - 1 - loopj[i];
    
      idx.push_back(std::vector<T>(len, 0));
      for (size_t i = 1; i < total; i++) {
        std::vector<T>  v(idx.back());
        for (size_t j = 0; j < len; ++j) {
          size_t k = loopj[j];
          if (v[k] < dim[k] - 1) {
            v[k] += 1;
            break;
          }
          v[k] = 0;
        }
        idx.push_back(v);
      }
    }

    /**
     * Get the names for an array of given dimensions 
     * in the way of column majored. 
     * For exmaple, if we know an array named `a`, with
     * dimensions of [2, 3, 4], the names then are (starting
     * from 0):
     * a[0,0,0]
     * a[1,0,0]
     * a[0,1,0]
     * a[1,1,0]
     * a[0,2,0]
     * a[1,2,0]
     * a[0,0,1]
     * a[1,0,1]
     * a[0,1,1]
     * a[1,1,1]
     * a[0,2,1]
     * a[1,2,1]
     * a[0,0,2]
     * a[1,0,2]
     * a[0,1,2]
     * a[1,1,2]
     * a[0,2,2]
     * a[1,2,2]
     * a[0,0,3]
     * a[1,0,3]
     * a[0,1,3]
     * a[1,1,3]
     * a[0,2,3]
     * a[1,2,3]
     *
     * @param name The name of the array variable 
     * @param dim The dimensions of the array 
     * @param fnames[out] Where the names would be pushed. 
     * @param first_is_one[true] Where to start for the first index: 0 or 1. 
     *
     */
    template <class T> void
    get_flatnames(const std::string& name,
                  const std::vector<T>& dim,
                  std::vector<std::string>& fnames,
                  bool col_major = true,
                  bool first_is_one = true) {

      fnames.clear(); 
      if (0 == dim.size()) {
        fnames.push_back(name);
        return;
      }

      std::vector<std::vector<T> > idx;
      expand_indices(dim, idx, col_major); 
      size_t first = first_is_one ? 1 : 0;
      for (typename std::vector<std::vector<T> >::const_iterator it = idx.begin();
           it != idx.end();
           ++it) {
        std::stringstream stri;
        stri << name << "[";

        size_t lenm1 = it -> size() - 1;
        for (size_t i = 0; i < lenm1; i++)
          stri << ((*it)[i] + first) << ",";
        stri << ((*it)[lenm1] + first) << "]";
        fnames.push_back(stri.str());
      }
    }

    // vectorize get_flatnames 
    template <class T> 
    void get_all_flatnames(const std::vector<std::string>& names, 
                           const std::vector<T>& dims, 
                           std::vector<std::string>& fnames, 
                           bool col_major = true) {
      fnames.clear(); 
      for (size_t i = 0; i < names.size(); ++i) {
        std::vector<std::string> i_names; 
        get_flatnames(names[i], dims[i], i_names, col_major, true); // col_major = true
        fnames.insert(fnames.end(), i_names.begin(), i_names.end());
      } 
    } 
  }

  namespace { 
    /**
     *@tparam T The type by which we use for dimensions. T could be say size_t
     * or unsigned int. This whole business (not using size_t) is due to that
     * Rcpp::wrap/as does not support size_t on some platforms and R could not
     * deal with 64bits integers. 
     *
     */ 
    template <class T> 
    size_t calc_num_params(const std::vector<T>& dim) {
      T num_params = 1;
      for (size_t i = 0;  i < dim.size(); ++i)
        num_params *= dim[i];
      return num_params;
    }

    template <class T> 
    void calc_starts(const std::vector<std::vector<T> >& dims,
                     std::vector<T>& starts) { 
      starts.resize(0); 
      starts.push_back(0); 
      for (size_t i = 1; i < dims.size(); ++i)
        starts.push_back(starts[i - 1] + calc_num_params(dims[i - 1]));
    }

    template <class T> 
    T calc_total_num_params(const std::vector<std::vector<T> >& dims) {
      T num_params = 0;
      for (size_t i = 0; i < dims.size(); ++i)
        num_params += calc_num_params(dims[i]);
      return num_params;
    }

    /**
     * Only calculate parameter with index in indexes. 
     */ 
    template <class T> 
    T calc_total_num_params(const std::vector<std::vector<T> >& dims, 
                            const std::vector<size_t> indexes) {
      T num_params = 0;
      for (size_t i = 0; i < indexes.size(); ++i)
        num_params += calc_num_params(dims[indexes[i]]);
      return num_params;
    }

    bool do_print(int refresh) {
      return refresh > 0;
    }
  
    bool do_print(int n, int refresh) {
      return do_print(refresh)
        && ((n + 1) % refresh == 0);
    }
  
    template <class Model>
    std::vector<std::string> get_param_names(Model& m) { 
      std::vector<std::string> names;
      m.get_param_names(names);
      return names; // copy for return
    }

    /**
     * Cast a size_t vector to an unsigned int vector. 
     * The reason is that first Rcpp::wrap/as does not 
     * support size_t on some platforms; second R 
     * could not deal with 64bits integers.  
     */ 

    std::vector<unsigned int> 
    sizet_to_uint(std::vector<size_t> v1) {
      std::vector<unsigned int> v2(v1.size());
      for (size_t i = 0;i < v1.size(); ++i) 
        v2[i] = static_cast<unsigned int>(v1[i]);
      return v2;
    } 

    template <class Model>
    std::vector<std::vector<unsigned int> > get_param_dims(Model& m) {
      std::vector<std::vector<size_t> > dims; 
      m.get_dims(dims); 

      std::vector<std::vector<unsigned int> > uintdims; 
      for (std::vector<std::vector<size_t> >::const_iterator it = dims.begin();
           it != dims.end(); 
           ++it) 
        uintdims.push_back(sizet_to_uint(*it)); 
      return uintdims; 
    } 

    template <class Sampler, class Model> 
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
                     std::vector<Rcpp::NumericVector>& chains, 
                     Rcpp::NumericVector& lp,
                     const size_t iter_save, 
                     const std::vector<size_t>& qoi_idx,
                     std::vector<Rcpp::NumericVector>& sampler_params, 
                     std::string& adaptation_info) { 
                         
      adaptation_info.clear(); 
      sampler.set_params(params_r,params_i);
      int it_print_width = std::ceil(std::log10(num_iterations));
      // rstan::io::rcout << std::endl;
  
      // rstan::io::rcout << "in sample_from." << std::endl; 
      if (epsilon_adapt)
        sampler.adapt_on(); 

      std::vector<double> params_inr; 
     
      unsigned ii = 0; // index for iterations saved to chains
      for (int m = 0; m < num_iterations; ++m, ++ii) {
        if (do_print(m,refresh)) {
          R_CheckUserInterrupt(); 
          rstan::io::rcout << "\rIteration: ";
          rstan::io::rcout << std::setw(it_print_width) << (m + 1)
                           << " / " << num_iterations;
          rstan::io::rcout << " [" << std::setw(3)
                           << static_cast<int>((100.0 * (m + 1))/num_iterations)
                           << "%] ";
          rstan::io::rcout << ((m < num_warmup) ? " (Adapting)" : " (Sampling)");
          rstan::io::rcout.flush();
        } 

        if (m == num_warmup && epsilon_adapt) {
          sampler.adapt_off();
          std::stringstream ss; 
          sampler.write_adaptation_params(ss); 
          adaptation_info.append(ss.str()); 
          if (sample_file_flag)  
            sample_file_stream << adaptation_info; 
        }

        if ((m % num_thin) != 0) {
          sampler.next();
          continue;
        }

        stan::mcmc::sample sample = sampler.next();
        sample.params_r(params_r);
        sample.params_i(params_i);
        model.write_array(params_r,params_i,params_inr); 

        double lp__ = sample.log_prob();
        lp[ii] = lp__; 
        if (sample_file_flag)
          sample_file_stream << lp__ << ","; 

        std::vector<double> ii_sampler_params; 
        sampler.get_sampler_params(ii_sampler_params); 
        for (size_t z = 0; z < ii_sampler_params.size(); z++) {
          sampler_params[z][ii] = ii_sampler_params[z];
          if (sample_file_flag)
            sample_file_stream << ii_sampler_params[z] << ",";
        } 
        
        for (size_t z = 0; z < qoi_idx.size(); ++z)  
          chains[z][ii] = params_inr[qoi_idx[z]]; 
          
  
        // FIXME: use csv_writer arg to make comma optional?
        if (sample_file_flag) { 
          // sampler.write_sampler_params(sample_file_stream);
          model.write_csv(params_r,params_i,sample_file_stream);
        }
      }
      if (refresh > 0) 
        rstan::io::rcout << std::endl << std::endl; 
      // rstan::io::rcout << "out of sample_from." << std::endl; 
    }

    /**
     * @tparam Model 
     * @param chains[out]: the object into which the samples for parameters
     *   of interest are written. 
     * @param initv[out]: the initial values used, or the first iteration. 
     * @iter_save: the number of iterations that would be save after 
     *   taking account of the thinning. 
     * @qoi_idx: the indexes for all parameters of interest.  
     */
    
    template <class Model> 
    int sampler_command(const io::rlist_ref_var_context& data, 
                        stan_args& args, 
                        Model& model, 
                        std::vector<Rcpp::NumericVector>& chains, 
                        Rcpp::NumericVector& lp, 
                        std::vector<double>& initv, 
                        unsigned int iter_save,
                        const std::vector<size_t>& qoi_idx,
                        std::vector<std::string>& sampler_param_names, 
                        std::vector<Rcpp::NumericVector>& sampler_params, 
                        std::string& adaptation_info) { 
      sampler_params.clear();
                            
      bool sample_file_flag = args.get_sample_file_flag(); 
      std::string sample_file = args.get_sample_file(); 
      unsigned int num_iterations = args.get_iter(); 
      unsigned int num_warmup = args.get_warmup(); 
      unsigned int num_thin = args.get_thin(); 
      int leapfrog_steps = args.get_leapfrog_steps(); 
      unsigned int random_seed = args.get_random_seed();
      double epsilon = args.get_epsilon(); 
      bool epsilon_adapt = (epsilon <= 0.0); 
      bool equal_step_sizes = args.get_equal_step_sizes();
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

      // keep a record of the initial values 
      model.write_array(params_r,params_i,initv); 
   
      /*
      for (size_t i = 0; i < params_r.size(); i++) 
        rstan::io::rcout << "params_r[" << i << "]=" << params_r[i] << std::endl; 

      for (size_t i = 0; i < params_i.size(); i++) 
        rstan::io::rcout << "params_i[" << i << "]=" << params_i[i] << std::endl; 
      */
      // reset the seed 
      base_rng.seed(random_seed); 
      base_rng.discard(DISCARD_STRIDE * (chain_id - 1));

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

      if (0 > leapfrog_steps && !equal_step_sizes) {
        // NUTS II (with diagonal mass matrix estimation during warmup)
        args.set_sampler("NUTS2"); 
        stan::mcmc::nuts_diag<rng_t> nuts2_sampler(model, 
                                                   max_treedepth, epsilon, 
                                                   epsilon_pm, epsilon_adapt,
                                                   delta, gamma, 
                                                   base_rng);

        nuts2_sampler.get_sampler_param_names(sampler_param_names);
        for (size_t i = 0; i < sampler_param_names.size(); i++) 
          sampler_params.push_back(Rcpp::NumericVector(iter_save));
            
        // cut & paste (see below) to enable sample-specific params
        if (sample_file_flag && !append_samples) {
          sample_stream << "lp__,"; // log probability first
          for (size_t i = 0; i < sampler_param_names.size(); i++)
            sample_stream << sampler_param_names[i] << ","; 
          model.write_csv_header(sample_stream);
        }
        nuts2_sampler.set_error_stream(rstan::io::rcerr); 

        sample_from(nuts2_sampler,epsilon_adapt,refresh,
                    num_iterations,num_warmup,num_thin,
                    sample_stream,sample_file_flag,params_r,params_i,
                    model,chains,lp,iter_save,qoi_idx,sampler_params,
                    adaptation_info); 

      } else if (0 > leapfrog_steps && equal_step_sizes) {
        // NUTS I (unit mass matrix)
        args.set_sampler("NUTS1"); 
        stan::mcmc::nuts<rng_t> nuts_sampler(model, 
                                             max_treedepth, epsilon, 
                                             epsilon_pm, epsilon_adapt,
                                             delta, gamma, 
                                             base_rng);

        nuts_sampler.get_sampler_param_names(sampler_param_names);
        for (size_t i = 0; i < sampler_param_names.size(); i++) 
          sampler_params.push_back(Rcpp::NumericVector(iter_save));
        // cut & paste (see below) to enable sample-specific params
        if (sample_file_flag && !append_samples) {
          sample_stream << "lp__,"; // log probability first
          nuts_sampler.write_sampler_param_names(sample_stream);
          model.write_csv_header(sample_stream);
        }
        nuts_sampler.set_error_stream(rstan::io::rcerr); 
        sample_from(nuts_sampler,epsilon_adapt,refresh,
                    num_iterations,num_warmup,num_thin,
                    sample_stream,sample_file_flag,params_r,params_i,
                    model,chains,lp,iter_save,qoi_idx,sampler_params,
                    adaptation_info); 
      } else {
        // Stardard HMC
        args.set_sampler("HMC"); 
        stan::mcmc::adaptive_hmc<rng_t> hmc_sampler(model,
                                                    leapfrog_steps,
                                                    epsilon, epsilon_pm, epsilon_adapt,
                                                    delta, gamma,
                                                    base_rng);

        hmc_sampler.get_sampler_param_names(sampler_param_names);
        for (size_t i = 0; i < sampler_param_names.size(); i++) 
          sampler_params.push_back(Rcpp::NumericVector(iter_save));
        // cut & paste (see above) to enable sample-specific params
        if (sample_file_flag && !append_samples) {
          sample_stream << "lp__,"; // log probability first
          hmc_sampler.write_sampler_param_names(sample_stream);
          model.write_csv_header(sample_stream);
        }
        hmc_sampler.set_error_stream(rstan::io::rcerr); 
        sample_from(hmc_sampler,epsilon_adapt,refresh,
                    num_iterations,num_warmup,num_thin,
                    sample_stream,sample_file_flag,params_r,params_i,
                    model,chains,lp,iter_save,qoi_idx,sampler_params,
                    adaptation_info); 
      }
      
      if (sample_file_flag) {
        rstan::io::rcout << "Sample of chain " 
                         << chain_id 
                         << " are written to file " << sample_file 
                         << std::endl;
        sample_stream.close();
      }
      return 0;
    }

  }

  template <class Model, class RNG> 
  class stan_fit {

  private:
    io::rlist_ref_var_context data_;
    Model model_;
    const std::vector<std::string> names_;
    const std::vector<std::vector<unsigned int> > dims_; 
    const unsigned int num_params_; 

    std::vector<std::string> names_oi_; // parameters of interest 
    std::vector<std::vector<unsigned int> > dims_oi_; 
    std::vector<size_t> names_oi_tidx_;  // the total indexes of names2.
    std::vector<unsigned int> starts_oi_;  
    unsigned int num_params2_;  // total number of POI's.   
    std::vector<std::string> fnames_oi_; 

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
  
    /*
     * Update the parameters we are interested for the model. 
     * As well, the dimensions vector for the parameters are 
     * updated. 
     */
    void update_param_oi0(const std::vector<std::string>& pnames) {
      names_oi_.clear(); 
      dims_oi_.clear(); 
      names_oi_tidx_.clear(); 

      std::vector<unsigned int> starts; 
      calc_starts(dims_, starts);
      for (std::vector<std::string>::const_iterator it = pnames.begin(); 
           it != pnames.end(); 
           ++it) { 
        size_t p = find_index(names_, *it); 
        if (p != names_.size()) {
          names_oi_.push_back(*it); 
          dims_oi_.push_back(dims_[p]); 
          size_t i_num = calc_num_params(dims_[p]); 
          size_t i_start = starts[p]; 
          for (size_t j = i_start; j < i_start + i_num; j++)
            names_oi_tidx_.push_back(j);
        } 
      }
      calc_starts(dims_oi_, starts_oi_);
      num_params2_ = names_oi_tidx_.size(); 
    } 

  public:
    SEXP update_param_oi(SEXP pars) {
      std::vector<std::string> pnames = 
        Rcpp::as<std::vector<std::string> >(pars);  
      update_param_oi0(pnames); 
      get_all_flatnames(names_oi_, dims_oi_, fnames_oi_, true); 
      return Rcpp::wrap(true); 
    } 

    stan_fit(SEXP data, SEXP pars) : 
      data_(Rcpp::as<Rcpp::List>(data)), 
      model_(data_),  
      names_(get_param_names(model_)), 
      dims_(get_param_dims(model_)), 
      num_params_(calc_total_num_params(dims_)) 
    {  
      update_param_oi(pars); 
    }

    stan_fit(SEXP data) : 
      data_(Rcpp::as<Rcpp::List>(data)), 
      model_(data_),
      names_(get_param_names(model_)), 
      dims_(get_param_dims(model_)), 
      num_params_(calc_total_num_params(dims_)), 
      names_oi_(names_), 
      dims_oi_(dims_),
      num_params2_(num_params_)  
    {
      for (size_t j = 0; j < num_params2_; j++) 
        names_oi_tidx_.push_back(j);
      calc_starts(dims_oi_, starts_oi_);
      get_all_flatnames(names_oi_, dims_oi_, fnames_oi_, true); 
    }             


    SEXP call_sampler(SEXP args_) { 
      BEGIN_RCPP; 
      Rcpp::List lst_args(args_); 
      stan_args args(lst_args); 
      unsigned int iter_save = args.get_iter_save(); 
      std::vector<Rcpp::NumericVector> chains; 
      std::vector<Rcpp::NumericVector> sampler_params;
      std::vector<std::string> sampler_param_names; 
      std::string adaptation_info; 
      std::vector<double> initv; 
      for (unsigned int i = 0; i < num_params2_; i++) 
        chains.push_back(Rcpp::NumericVector(iter_save)); 

      // the log posterior up to a constant 
      Rcpp::NumericVector lp(iter_save); 

      sampler_command(data_, args, model_, chains, lp, initv, iter_save, names_oi_tidx_, 
                      sampler_param_names, sampler_params, adaptation_info); 
      // let Rcpp handle the error dispatching. 
      /*
      try {
      } catch (std::exception& e) {
        rstan::io::rcerr << std::endl << "Exception: " << e.what() << std::endl;
        rstan::io::rcerr << "Diagnostic information: " << std::endl << boost::diagnostic_information(e) << std::endl;
        return R_NilValue; 
      }
      */
      // chains.attr("ncol") = Rcpp::wrap(num_params2_); 
      Rcpp::List lst(chains.begin(), chains.end()); 
      lst.attr("args") = args.stan_args_to_rlist(); 
      lst.attr("inits") = initv; 
      lst.attr("lp") = lp;
      lst.attr("adaptation_info") = adaptation_info;
      
      // sampler parameters such as treedepth
      Rcpp::List slst(sampler_params.begin(), sampler_params.end());
      slst.names() = sampler_param_names;
      lst.attr("sampler_params") = slst;
      
      lst.names() = fnames_oi_; 
      return lst; 
      END_RCPP; 
    } 

    SEXP param_names() const {
      BEGIN_RCPP; 
      return Rcpp::wrap(names_);
      END_RCPP; 
    } 

    SEXP param_names_oi() const {
      BEGIN_RCPP; 
      return Rcpp::wrap(names_oi_);
      END_RCPP; 
    } 

    /**
     * tidx (total indexes) 
     * the index is among those parameters of interest, not 
     * all the parameters. 
     */ 
    SEXP param_oi_tidx(SEXP pars) {
      BEGIN_RCPP; 
      std::vector<std::string> names = 
        Rcpp::as<std::vector<std::string> >(pars); 
      std::vector<std::string> names2; 
      std::vector<std::vector<unsigned int> > indexes; 
      for (std::vector<std::string>::const_iterator it = names.begin();
           it != names.end(); 
           ++it) {
        if (is_flatname(*it)) { // an element of an array  
          size_t ts = std::distance(fnames_oi_.begin(),
                                    std::find(fnames_oi_.begin(), 
                                              fnames_oi_.end(), *it));       
          if (ts == fnames_oi_.size()) // not found 
            continue; 
          names2.push_back(*it); 
          indexes.push_back(std::vector<unsigned int>(1, ts)); 
          continue;
        }
        size_t j = std::distance(names_oi_.begin(), 
                                 std::find(names_oi_.begin(),    
                                           names_oi_.end(), *it)); 
        if (j == names_oi_.size()) // nout found 
          continue; 
        unsigned int j_size = calc_num_params(dims_oi_[j]); 
        unsigned int j_start = starts_oi_[j]; 
        std::vector<unsigned int> j_idx; 
        for (unsigned int k = 0; k < j_size; k++) {
          j_idx.push_back(j_start + k); 
        } 
        names2.push_back(*it); 
        indexes.push_back(j_idx); 
      }
      Rcpp::List lst = Rcpp::wrap(indexes); 
      lst.names() = names2; 
      return lst; 
      END_RCPP;
    } 


    SEXP param_dims() const {
      BEGIN_RCPP; 
      Rcpp::List lst = Rcpp::wrap(dims_); 
      lst.names() = names_; 
      return lst; 
      END_RCPP;
    } 

    SEXP param_dims_oi() const {
      BEGIN_RCPP; 
      Rcpp::List lst = Rcpp::wrap(dims_oi_); 
      lst.names() = names_oi_; 
      return lst; 
      END_RCPP;
    } 
    
    SEXP param_fnames_oi() const {
      BEGIN_RCPP; 
      std::vector<std::string> fnames; 
      get_all_flatnames(names_oi_, dims_oi_, fnames, true); 
      return Rcpp::wrap(fnames); 
      END_RCPP;
    } 

    /*
     * Obtain a permutation of size n. 
     * see <code>permutation</code> in <code>mcmc::chains.hpp</code>. 
     *
     * @param nsc_ Size of permutation to create; the starting point of the
     *  sequence, typically 0 or 1; and the chain id. 
     *  If the starting point is not specified, defaults to 1. 
     *  If the chain id is not specified, defaults to 1. 
     * @return A permutation of length n starting from 's'.
     */ 
    SEXP permutation(SEXP nsc_) { 
      static time_t seed = std::time(0);  //defaults to time-based init 
      boost::uintmax_t DISCARD_STRIDE = 1 << 31; 
      Rcpp::IntegerVector nsc(nsc_); 
      unsigned int n = nsc[0]; 
      unsigned int s = 1;
      unsigned int cid = 1; 
      // rstan::io::rcout << "stride = " << DISCARD_STRIDE << std::endl; 
      // rstan::io::rcout << "nsc.size() = " << nsc.size() << std::endl; 
      // rstan::io::rcout << "nsc[0]=" << nsc[0] 
      //                  << "nsc[1]=" << nsc[1] 
      //                  << "nsc[2]=" << nsc[2] << std::endl; 
      switch (nsc.size()) {
        case 3:  cid = nsc[2]; s = nsc[1]; break;
        case 2:  s = nsc[1]; break;
      } 
      // rstan::io::rcout << "cid = " << cid << std::endl; 
      RNG rng(seed); 
      rng.discard(DISCARD_STRIDE * (cid - 1));
      Rcpp::IntegerVector x(n); 
      for (size_t i = 0; i < n; ++i)
        x[i] = i + s;
      if (n < 2) return x; 
      for (size_t i = n; --i != 0; ) {
        boost::random::uniform_int_distribution<size_t> uid(0, i);
        size_t j = uid(rng);
        unsigned int temp = x[i];
        x[i] = x[j];
        x[j] = temp;
      } 
      return x; 
    } 
  };
} 

#endif 

/*
 * compile to check syntax error
 */
/*
STAN= ../../../../../ 
RCPPINC=`Rscript -e "cat(system.file('include', package='Rcpp'))"`
RINC=`Rscript -e "cat(R.home('include'))"` 
g++ -I${RINC} -I"${STAN}/lib/boost_1.51.0" -I"${STAN}/lib/eigen_3.1.1"  -I"${STAN}/src" -I"${RCPPINC}" -I"../" stan_fit.hpp 
*/


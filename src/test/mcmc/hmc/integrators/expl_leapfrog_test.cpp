#include <stan/mcmc/hmc/integrators/expl_leapfrog.hpp>
#include <gtest/gtest.h>

#include <stan/mcmc/hmc/hamiltonians/unit_e_metric.hpp>
#include <stan/mcmc/hmc/hamiltonians/unit_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/diag_e_metric.hpp>
#include <stan/mcmc/hmc/hamiltonians/diag_e_point.hpp>
#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG

//************************************************************
// Generated model from src/models/command.stan
#include <stan/model/model_header.hpp>

namespace command_model_namespace {

  using std::vector;
  using std::string;
  using std::stringstream;
  using stan::agrad::var;
  using stan::model::prob_grad_ad;
  using stan::math::get_base1;
  using stan::math::stan_print;
  using stan::io::dump;
  using std::istream;
  using namespace stan::math;
  using namespace stan::prob;
  using namespace stan::agrad;

  typedef Eigen::Matrix<double,Eigen::Dynamic,1> vector_d;
  typedef Eigen::Matrix<double,1,Eigen::Dynamic> row_vector_d;
  typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> matrix_d;

  class command_model : public prob_grad_ad {
  private:
    double mu;
  public:
    command_model(stan::io::var_context& context__,
                  std::ostream* pstream__ = 0)
      : prob_grad_ad::prob_grad_ad(0) {
      static const char* function__ = "command_model_namespace::command_model(%1%)";
      (void) function__; // dummy call to supress warning
      size_t pos__;
      (void) pos__; // dummy call to supress warning
      std::vector<int> vals_i__;
      std::vector<double> vals_r__;
      context__.validate_dims("data initialization", "mu", "double", context__.to_vec());
      mu = double(0);
      vals_r__ = context__.vals_r("mu");
      pos__ = 0;
      mu = vals_r__[pos__++];
      // validate data

      // validate transformed data

      set_param_ranges();
    } // dump ctor

    void set_param_ranges() {
      num_params_r__ = 0U;
      param_ranges_i__.clear();
      ++num_params_r__;
    }

    void transform_inits(const stan::io::var_context& context__,
                         std::vector<int>& params_i__,
                         std::vector<double>& params_r__) {
      stan::io::writer<double> writer__(params_r__,params_i__);
      size_t pos__;
      std::vector<double> vals_r__;
      std::vector<int> vals_i__;


      if (!(context__.contains_r("y")))
        throw std::runtime_error("variable y missing");
      vals_r__ = context__.vals_r("y");
      pos__ = 0U;
      context__.validate_dims("initialization", "y", "double", context__.to_vec());
      double y(0);
      y = vals_r__[pos__++];
      writer__.scalar_unconstrain(y);
      params_r__ = writer__.data_r();
      params_i__ = writer__.data_i();
    }

    var log_prob(vector<var>& params_r__,
                 vector<int>& params_i__,
                 std::ostream* pstream__ = 0) {
      return log_prob_poly<true,var>(params_r__,params_i__,pstream__);
    }

    template <bool propto__, typename T__>
    T__ log_prob_poly(vector<T__>& params_r__,
                      vector<int>& params_i__,
                      std::ostream* pstream__ = 0) {

      T__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
      (void) DUMMY_VAR__;  // suppress unused var warning

      T__ lp__(0.0);

      // model parameters
      stan::io::reader<T__> in__(params_r__,params_i__);

      T__ y = in__.scalar_constrain(lp__);
      (void) y;  // supress unused variable warning

      // transformed parameters

      // initialized transformed params to avoid seg fault on val access
        

      // validate transformed parameters

      const char* function__ = "validate transformed params %1%";
      (void) function__; // dummy to suppress unused var warning
      // model body
      lp__ += stan::prob::normal_log<true>(y, mu, 1);

      return lp__;

    } // log_prob(...var...)


    void get_param_names(std::vector<std::string>& names__) {
      names__.resize(0);
      names__.push_back("y");
    }


    void get_dims(std::vector<std::vector<size_t> >& dimss__) {
      dimss__.resize(0);
      std::vector<size_t> dims__;
      dims__.resize(0);
      dimss__.push_back(dims__);
    }

    template <typename RNG>
    void write_array(RNG& base_rng__,
                     std::vector<double>& params_r__,
                     std::vector<int>& params_i__,
                     std::vector<double>& vars__,
                     std::ostream* pstream__ = 0) {
      vars__.resize(0);
      stan::io::reader<double> in__(params_r__,params_i__);
      static const char* function__ = "command_model_namespace::write_array(%1%)";
      (void) function__; // dummy call to supress warning
      // read-transform, write parameters
      double y = in__.scalar_constrain();
      vars__.push_back(y);

      // declare and define transformed parameters
      double lp__ = 0.0;
      (void) lp__; // dummy call to supress warning


      // validate transformed parameters

      // write transformed parameters

      // declare and define generated quantities


      // validate generated quantities

      // write generated quantities
    }


    void write_csv_header(std::ostream& o__) {
      stan::io::csv_writer writer__(o__);
      writer__.comma();
      o__ << "y";
      writer__.newline();
    }

    template <typename RNG>
    void write_csv(RNG& base_rng__,
                   std::vector<double>& params_r__,
                   std::vector<int>& params_i__,
                   std::ostream& o__,
                   std::ostream* pstream__ = 0) {
      stan::io::reader<double> in__(params_r__,params_i__);
      stan::io::csv_writer writer__(o__);
      static const char* function__ = "command_model_namespace::write_csv(%1%)";
      (void) function__; // dummy call to supress warning
      // read-transform, write parameters
      double y = in__.scalar_constrain();
      writer__.write(y);

      // declare, define and validate transformed parameters
      double lp__ = 0.0;
      (void) lp__; // dummy call to supress warning



      // write transformed parameters

      // declare and define generated quantities


      // validate generated quantities

      // write generated quantities
      writer__.newline();
    }

    std::string model_name() {
      return "command_model";
    }


    void constrained_param_names(std::vector<std::string>& param_names__) {
      std::stringstream param_name_stream__;
      param_name_stream__.str(std::string());
      param_name_stream__ << "y";
      param_names__.push_back(param_name_stream__.str());
    }

  }; // model

} // namespace
//************************************************************


class McmcHmcIntegratorsExplLeapfrog : public testing::Test {
public:
  
  void SetUp() {
    // setup hamiltonian
    std::fstream data_stream("src/test/models/command1.data.R",
                             std::fstream::in);
    stan::io::dump data_var_context(data_stream);
    data_stream.close();

    model = new command_model_namespace::command_model(data_var_context);
  }
  
  void TearDown() {
    delete(model);
  }  
  
  typedef boost::ecuyer1988 rng_t;
  
  // integrator under test
  stan::mcmc::expl_leapfrog<
    stan::mcmc::unit_e_metric<command_model_namespace::command_model,rng_t>, 
    stan::mcmc::unit_e_point> unit_e_integrator;

  stan::mcmc::expl_leapfrog<
    stan::mcmc::diag_e_metric<command_model_namespace::command_model,rng_t>, 
    stan::mcmc::diag_e_point> diag_e_integrator;
  
  // model
  command_model_namespace::command_model *model;
};



TEST_F(McmcHmcIntegratorsExplLeapfrog, begin_update_p) {
  // setup z
  stan::mcmc::unit_e_point z(1,0);
  z.V    =  1.99974742955684;
  z.q[0] =  1.99987371079118;
  z.p(0) = -1.58612292129732;
  z.g(0) =  1.99987371079118;
  EXPECT_NEAR(z.V,     1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q[0],  1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0), -1.58612292129732, 1e-15);
  EXPECT_NEAR(z.g(0),  1.99987371079118, 1e-15);
  
  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_model_namespace::command_model,
    rng_t> hamiltonian(*model, &std::cout);

  // setup epsilon
  double epsilon = 0.1;

  unit_e_integrator.begin_update_p(z, hamiltonian, 0.5 * epsilon);
  EXPECT_NEAR(z.V,     1.99974742955684, 5e-14);
  EXPECT_NEAR(z.q[0],  1.99987371079118, 5e-14);
  EXPECT_NEAR(z.p(0), -1.68611660683688, 5e-14);
  EXPECT_NEAR(z.g(0),  1.99987371079118, 5e-14);
}

TEST_F(McmcHmcIntegratorsExplLeapfrog, update_q) {
  // setup z
  stan::mcmc::unit_e_point z(1,0);
  z.V    =  1.99974742955684;
  z.q[0] =  1.99987371079118;
  z.p(0) = -1.68611660683688;
  z.g(0) =  1.99987371079118;
  EXPECT_NEAR(z.V,     1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q[0],  1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0), -1.68611660683688, 1e-15);
  EXPECT_NEAR(z.g(0),  1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_model_namespace::command_model,
    rng_t> hamiltonian(*model, &std::cout);

  // setup epsilon
  double epsilon = 0.1;

  unit_e_integrator.update_q(z, hamiltonian, epsilon);
  EXPECT_NEAR(z.V,     1.99974742955684, 5e-14);
  EXPECT_NEAR(z.q[0],  1.83126205010749, 5e-14);
  EXPECT_NEAR(z.p(0), -1.68611660683688, 5e-14);
  EXPECT_NEAR(z.g(0),  1.99987371079118, 5e-14);
}

TEST_F(McmcHmcIntegratorsExplLeapfrog, end_update_p) {
  // setup z
  stan::mcmc::unit_e_point z(1,0);
  z.V    =  1.39887860643153;
  z.q[0] =  1.67264975797776;
  z.p(0) = -1.68611660683688;
  z.g(0) =  1.67264975797776;
  EXPECT_NEAR(z.V,     1.39887860643153, 1e-15);
  EXPECT_NEAR(z.q[0],  1.67264975797776, 1e-15);
  EXPECT_NEAR(z.p(0), -1.68611660683688, 1e-15);
  EXPECT_NEAR(z.g(0),  1.67264975797776, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_model_namespace::command_model,
    rng_t> hamiltonian(*model, &std::cout);

  // setup epsilon
  double epsilon = 0.1;

  unit_e_integrator.end_update_p(z, hamiltonian, 0.5 * epsilon);
  EXPECT_NEAR(z.V,     1.39887860643153, 5e-14);
  EXPECT_NEAR(z.q[0],  1.67264975797776, 5e-14);
  EXPECT_NEAR(z.p(0), -1.76974909473577, 5e-14);
  EXPECT_NEAR(z.g(0),  1.67264975797776, 5e-14);
}

TEST_F(McmcHmcIntegratorsExplLeapfrog, evolve_1) {
  // setup z
  stan::mcmc::unit_e_point z(1,0);
  z.V    =  1.99974742955684;
  z.q[0] =  1.99987371079118;
  z.p(0) = -1.58612292129732;
  z.g(0) =  1.99987371079118;
  EXPECT_NEAR(z.V,     1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q[0],  1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0), -1.58612292129732, 1e-15);
  EXPECT_NEAR(z.g(0),  1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_model_namespace::command_model,
    rng_t> hamiltonian(*model, &std::cout);

  // setup epsilon
  double epsilon = 0.1;

  unit_e_integrator.evolve(z, hamiltonian, epsilon);
  EXPECT_NEAR(z.V,     1.67676034808195, 5e-14);
  EXPECT_NEAR(z.q[0],  1.83126205010749, 5e-14);
  EXPECT_NEAR(z.p(0), -1.77767970934226, 5e-14);
  EXPECT_NEAR(z.g(0),  1.83126205010749, 5e-14);
}

TEST_F(McmcHmcIntegratorsExplLeapfrog, evolve_2) {
  // setup z
  stan::mcmc::unit_e_point z(1,0);
  z.V    =  1.99974742955684;
  z.q[0] =  1.99987371079118;
  z.p(0) =  0.531143888645192;
  z.g(0) =  1.99987371079118;
  EXPECT_NEAR(z.V,     1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q[0],  1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0),  0.531143888645192, 1e-15);
  EXPECT_NEAR(z.g(0),  1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_model_namespace::command_model,
    rng_t> hamiltonian(*model, &std::cout);

  // setup epsilon
  double epsilon = 0.2;

  unit_e_integrator.evolve(z, hamiltonian, epsilon);
  EXPECT_NEAR(z.V,     2.13439496506688, 5e-14);
  EXPECT_NEAR(z.q[0],  2.06610501430439, 5e-14);
  EXPECT_NEAR(z.p(0),  0.124546016135635, 5e-14);
  EXPECT_NEAR(z.g(0),  2.06610501430439, 5e-14);
}

TEST_F(McmcHmcIntegratorsExplLeapfrog, evolve_3) {
  // setup z
  stan::mcmc::unit_e_point z(1,0);
  z.V    =  1.99974742955684;
  z.q[0] =  1.99987371079118;
  z.p(0) =  0.531143888645192;
  z.g(0) =  1.99987371079118;
  EXPECT_NEAR(z.V,     1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q[0],  1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0),  0.531143888645192, 1e-15);
  EXPECT_NEAR(z.g(0),  1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_model_namespace::command_model,
    rng_t> hamiltonian(*model, &std::cout);

  // setup epsilon
  double epsilon = 0.2;

  unit_e_integrator.evolve(z, hamiltonian, epsilon);
  EXPECT_NEAR(z.V,     2.13439496506688, 5e-14);
  EXPECT_NEAR(z.q[0],  2.06610501430439, 5e-14);
  EXPECT_NEAR(z.p(0),  0.124546016135635, 5e-14);
  EXPECT_NEAR(z.g(0),  2.06610501430439, 5e-14);
}

TEST_F(McmcHmcIntegratorsExplLeapfrog, evolve_4) {
  // setup z
  stan::mcmc::unit_e_point z(1,0);
  z.V    =  1.99974742955684;
  z.q[0] =  1.99987371079118;
  z.p(0) = -1.01150787313287;
  z.g(0) =  1.99987371079118;
  EXPECT_NEAR(z.V,     1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q[0],  1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0), -1.01150787313287, 1e-15);
  EXPECT_NEAR(z.g(0),  1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_model_namespace::command_model,
    rng_t> hamiltonian(*model, &std::cout);

  // setup epsilon
  double epsilon = 0.4;

  unit_e_integrator.evolve(z, hamiltonian, epsilon);
  EXPECT_NEAR(z.V,     1.03001529319458, 5e-14);
  EXPECT_NEAR(z.q[0],  1.43528066467474, 5e-14);
  EXPECT_NEAR(z.p(0), -1.69853874822605, 5e-14);
  EXPECT_NEAR(z.g(0),  1.43528066467474, 5e-14);
}

TEST_F(McmcHmcIntegratorsExplLeapfrog, evolve_5) {
  // setup z
  stan::mcmc::unit_e_point z(1,0);
  z.V    =  1.99974742955684;
  z.q[0] =  1.99987371079118;
  z.p(0) = -0.141638464197442;
  z.g(0) =  1.99987371079118;
  EXPECT_NEAR(z.V,     1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q[0],  1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0), -0.141638464197442, 1e-15);
  EXPECT_NEAR(z.g(0),  1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_model_namespace::command_model,
    rng_t> hamiltonian(*model, &std::cout);

  // setup epsilon
  double epsilon = 0.8;

  unit_e_integrator.evolve(z, hamiltonian, epsilon);
  EXPECT_NEAR(z.V,     0.777009958583946, 5e-14);
  EXPECT_NEAR(z.q[0],  1.24660335198005, 5e-14);
  EXPECT_NEAR(z.p(0), -1.44022928930593, 5e-14);
  EXPECT_NEAR(z.g(0),  1.24660335198005, 5e-14);
}
TEST_F(McmcHmcIntegratorsExplLeapfrog, evolve_6) {
  // setup z
  stan::mcmc::unit_e_point z(1,0);
  z.V    =  1.99974742955684;
  z.q[0] =  1.99987371079118;
  z.p(0) =  0.0942249427134016;
  z.g(0) =  1.99987371079118;
  EXPECT_NEAR(z.V,     1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q[0],  1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0),  0.0942249427134016, 1e-15);
  EXPECT_NEAR(z.g(0),  1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_model_namespace::command_model,
    rng_t> hamiltonian(*model, &std::cout);

  // setup epsilon
  double epsilon = 1.6;

  unit_e_integrator.evolve(z, hamiltonian, epsilon);
  EXPECT_NEAR(z.V,     0.0837242558054816, 5e-14);
  EXPECT_NEAR(z.q[0], -0.409204730680088, 5e-14);
  EXPECT_NEAR(z.p(0), -1.17831024137547, 5e-14);
  EXPECT_NEAR(z.g(0), -0.409204730680088, 5e-14);
}

TEST_F(McmcHmcIntegratorsExplLeapfrog, evolve_7) {
  // setup z
  stan::mcmc::unit_e_point z(1,0);
  z.V    =  1.99974742955684;
  z.q[0] =  1.99987371079118;
  z.p(0) =  1.01936184962275;
  z.g(0) =  1.99987371079118;
  EXPECT_NEAR(z.V,     1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q[0],  1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0),  1.01936184962275, 1e-15);
  EXPECT_NEAR(z.g(0),  1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_model_namespace::command_model,
    rng_t> hamiltonian(*model, &std::cout);

  // setup epsilon
  double epsilon = 3.2;

  unit_e_integrator.evolve(z, hamiltonian, epsilon);
  EXPECT_NEAR(z.V,     12.3878614837537, 5e-13);
  EXPECT_NEAR(z.q[0],  -4.97752176966686, 5e-14);
  EXPECT_NEAR(z.p(0),  5.78359874382383, 5e-14);
  EXPECT_NEAR(z.g(0),  -4.97752176966686, 5e-14);
}

TEST_F(McmcHmcIntegratorsExplLeapfrog, evolve_8) {
  // setup z
  stan::mcmc::unit_e_point z(1,0);
  z.V    =  1.99974742955684;
  z.q[0] =  1.99987371079118;
  z.p(0) = -2.73131279771964;
  z.g(0) =  1.99987371079118;
  EXPECT_NEAR(z.V,     1.99974742955684, 1e-15);
  EXPECT_NEAR(z.q[0],  1.99987371079118, 1e-15);
  EXPECT_NEAR(z.p(0), -2.73131279771964, 1e-15);
  EXPECT_NEAR(z.g(0),  1.99987371079118, 1e-15);

  // setup hamiltonian
  stan::mcmc::unit_e_metric<command_model_namespace::command_model,
    rng_t> hamiltonian(*model, &std::cout);

  // setup epsilon
  double epsilon = -1;

  unit_e_integrator.evolve(z, hamiltonian, epsilon);
  EXPECT_NEAR(z.V,     6.96111198693627, 5e-14);
  EXPECT_NEAR(z.q[0],  3.73124965311523, 5e-14);
  EXPECT_NEAR(z.p(0),  0.134248884233563, 5e-14);
  EXPECT_NEAR(z.g(0),  3.73124965311523, 5e-14);
}

TEST_F(McmcHmcIntegratorsExplLeapfrog, evolve_9) {
  // setup z
  stan::mcmc::diag_e_point z(1,0);
  z.V    =  0.807684865121721;
  z.q[0] =  1.27097196280777;
  z.p(0) = -0.159996782671291;
  z.g(0) =  1.27097196280777;
  z.mInv(0) = 0.733184698671436;
  EXPECT_NEAR(z.V,     0.807684865121721, 1e-15);
  EXPECT_NEAR(z.q[0],  1.27097196280777, 1e-15);
  EXPECT_NEAR(z.p(0), -0.159996782671291, 1e-15);
  EXPECT_NEAR(z.g(0),  1.27097196280777, 1e-15);

  // setup hamiltonian
  stan::mcmc::diag_e_metric<command_model_namespace::command_model,
    rng_t> hamiltonian(*model, &std::cout);

  // setup epsilon
  double epsilon = 2.40769920051673;

  diag_e_integrator.evolve(z, hamiltonian, epsilon);
  EXPECT_NEAR(z.V,     1.46626604258356, 5e-14);
  EXPECT_NEAR(z.q[0], -1.71246374711032, 5e-14);
  EXPECT_NEAR(z.p(0),  0.371492925378682, 5e-14);
  EXPECT_NEAR(z.g(0), -1.71246374711032, 5e-14);
}

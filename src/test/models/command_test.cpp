/** 
 * This test tries out different command line options for a 
 * generated model.
 *
 */
#include <gtest/gtest.h>
#include <test/models/utility.hpp>

#include <stdexcept>
#include <stan/mcmc/chains.hpp>
#include <bitset>

using std::stringstream;
using std::vector;
using std::string;
using std::pair;
using std::make_pair;
using std::bitset;

using testing::Combine;
using testing::Range;

enum cl_options {
  append_samples, // should be first option for testing purposes
  data,
  init,
  seed,
  chain_id,
  iter,
  warmup_opt,
  thin,
  leapfrog_steps,
  max_treedepth,
  epsilon,
  epsilon_pm,
  equal_step_sizes,
  delta,
  gamma_opt,
  options_count   // should be last. will hold the number of tested options
};


class ModelCommand : public ::testing::TestWithParam<int> {
public:
  static string data_file_base;
  static string model_path;
  static vector<string> expected_help_options;
  static vector<pair<string, string> > expected_output;
  static vector<string> option_name;
  static vector<pair<string, string> > command_changes;
  static vector<pair<string, string> > output_changes;
  
  void static SetUpTestCase() {
    model_path.append("models");
    model_path.append(1, get_path_separator());
    model_path.append("command");

    data_file_base.append("src");
    data_file_base.append(1, get_path_separator());
    data_file_base.append("test");
    data_file_base.append(1, get_path_separator());
    data_file_base.append(model_path);

    expected_help_options.push_back("help");
    expected_help_options.push_back("data");
    expected_help_options.push_back("init");
    expected_help_options.push_back("samples");
    expected_help_options.push_back("append_samples");
    expected_help_options.push_back("seed");
    expected_help_options.push_back("chain_id");
    expected_help_options.push_back("iter");
    expected_help_options.push_back("warmup");
    expected_help_options.push_back("thin");
    expected_help_options.push_back("refresh");
    expected_help_options.push_back("leapfrog_steps");
    expected_help_options.push_back("max_treedepth");
    expected_help_options.push_back("epsilon");
    expected_help_options.push_back("epsilon_pm");
    expected_help_options.push_back("equal_step_sizes");
    expected_help_options.push_back("delta");
    expected_help_options.push_back("gamma");
    expected_help_options.push_back("save_warmup");
    expected_help_options.push_back("test_grad");
    expected_help_options.push_back("point_estimate");
    expected_help_options.push_back("point_estimate_newton\n");
    expected_help_options.push_back("point_estimate_bfgs\n");
    expected_help_options.push_back("nondiag_mass");
    expected_help_options.push_back("cov_matrix");

    expected_output.push_back(make_pair("data","(specified model requires no data)"));
    expected_output.push_back(make_pair("init", "random initialization"));
    expected_output.push_back(make_pair("init tries", "1"));
    expected_output.push_back(make_pair("samples", model_path+".csv"));
    expected_output.push_back(make_pair("append_samples", "0"));
    expected_output.push_back(make_pair("save_warmup", "0"));
    expected_output.push_back(make_pair("seed", ""));
    expected_output.push_back(make_pair("chain_id", "1 (default)"));
    expected_output.push_back(make_pair("iter", "2000"));
    expected_output.push_back(make_pair("warmup", "1000"));
    expected_output.push_back(make_pair("thin", "1"));
    expected_output.push_back(make_pair("equal_step_sizes", "0"));
    expected_output.push_back(make_pair("nondiag_mass", "0"));
    expected_output.push_back(make_pair("leapfrog_steps", "-1"));
    expected_output.push_back(make_pair("max_treedepth", "10"));
    expected_output.push_back(make_pair("epsilon", "-1"));
    expected_output.push_back(make_pair("epsilon_pm", "0"));
    expected_output.push_back(make_pair("delta", "0.5"));
    expected_output.push_back(make_pair("gamma", "0.05"));
    
    option_name.resize(options_count);    
    command_changes.resize(options_count);
    output_changes.resize(options_count);

    option_name[append_samples] = "append_samples";
    command_changes[append_samples] = make_pair("",
                                                " --append_samples");
    output_changes [append_samples] = make_pair("",
                                                "1");

    option_name[data] = "data";
    command_changes[data] = make_pair(" --data="+data_file_base+"1.data.R",
                                      " --data="+data_file_base+"2.data.R");
    output_changes [data] = make_pair(data_file_base+"1.data.R",
                                      data_file_base+"2.data.R");

    option_name[init] = "init";
    command_changes[init] = make_pair("",
                                      " --init=" + data_file_base + ".init.R");
    output_changes [init] = make_pair("",
                                      data_file_base + ".init.R");


    option_name[seed] = "seed";
    command_changes[seed] = make_pair("",
                                      " --seed=100");
    output_changes [seed] = make_pair("",
                                      "100 (user specified)");


    option_name[chain_id] = "chain_id";
    command_changes[chain_id] = make_pair("",
                                          " --chain_id=2");
    output_changes [chain_id] = make_pair("",
                                          "2 (user specified)");
    
    option_name[iter] = "iter";
    command_changes[iter] = make_pair("",
                                      " --iter=100");
    output_changes [iter] = make_pair("",
                                      "100");

    option_name[warmup_opt] = "warmup";
    command_changes[warmup_opt] = make_pair("",
                                        " --warmup=60");
    output_changes [warmup_opt] = make_pair("",
                                        "60");

    option_name[thin] = "thin";
    command_changes[thin] = make_pair("",
                                        " --thin=3");
    output_changes [thin] = make_pair("",
                                        "3");



    option_name[leapfrog_steps] = "leapfrog_steps";
    command_changes[leapfrog_steps] = make_pair("",
                                                " --leapfrog_steps=1");
    output_changes [leapfrog_steps] = make_pair("",
                                                "1");
    

    option_name[max_treedepth] = "max_treedepth";
    command_changes[max_treedepth] = make_pair("",
                                               " --max_treedepth=2");
    output_changes [max_treedepth] = make_pair("",
                                               "2");
    
    option_name[epsilon] = "epsilon";
    command_changes[epsilon] = make_pair("",
                                         " --epsilon=1.5");
    output_changes [epsilon] = make_pair("",
                                         "1.5");
    

    option_name[epsilon_pm] = "epsilon_pm";
    command_changes[epsilon_pm] = make_pair("",
                                            " --epsilon_pm=0.5");
    output_changes [epsilon_pm] = make_pair("",
                                            "0.5");
    
    option_name[equal_step_sizes] = "equal_step_sizes";
    command_changes[equal_step_sizes] = make_pair("",
                                                  " --equal_step_sizes");
    output_changes [equal_step_sizes] = make_pair("",
                                                  "1");

    option_name[delta] = "delta";
    command_changes[delta] = make_pair("",
                                       " --delta=0.75");
    output_changes [delta] = make_pair("",
                                       "0.75");
    
    option_name[gamma_opt] = "gamma";
    command_changes[gamma_opt] = make_pair("",
                                       " --gamma=0.025");
    output_changes [gamma_opt] = make_pair("",
                                       "0.025");

    //for (int i = 0; i < options_count; i++) {
    //  std::cout << "\t" << i << ": " << option_name[i] << std::endl;
    //}
  }

  void check_output(const string& command_output,
                    const vector<pair<string, string> >& changed_options) {
    vector<pair<string, string> > expected_output(this->expected_output);
    for (size_t i = 0; i < changed_options.size(); i++) {
      for (size_t j = 0; j < expected_output.size(); j++) {
        if (expected_output[j].first == changed_options[i].first) {
          expected_output[j].second = changed_options[i].second;
          break;
        }
      }
    }
    for (size_t i = 0; i < changed_options.size(); i++) {
      if (changed_options[i].first == "init") {
        size_t j;
        for (j = 0; j < expected_output.size(); j++) {
          if (expected_output[j].first == "init tries") {
            break;
          }
        }
        expected_output.erase(expected_output.begin() + j);
      }
    }
    
    
    vector<pair<string, string> > output = parse_command_output(command_output);
    ASSERT_EQ(expected_output.size(), output.size());
    for (size_t i = 0; i < expected_output.size(); i++) {
      EXPECT_EQ(expected_output[i].first, output[i].first) <<
        "Order of output should match";
      if (expected_output[i].first == "seed" && expected_output[i].second == "") {
        // when seed is default, check to see that it is randomly generated
        if (boost::algorithm::ends_with(output[i].second, "(randomly generated)"))
          SUCCEED();
        else
          ADD_FAILURE() 
            << "'" << output[i].first 
            << "' is not randomly generated: " << output[i].second;
      } else {
        EXPECT_EQ(expected_output[i].second, output[i].second)
          << "Option '" << expected_output[i].first << "' returned unexpected value";
      }
        
    }

  }
  
  void check_output(const string& command_output) {
    vector<pair<string, string> > changed_options;
    check_output(command_output, changed_options);
  }
  
  string get_command(const bitset<options_count> options, 
                     vector<pair<string, string> > &changed_options) {
    stringstream command;
    command << model_path;
    command << " --samples=" + model_path + ".csv";

    //for (int i = options_count-1; i > -1; i--) {
    for (int i = 0; i < options_count; i++) {
      string output_option;
      if (!options[i]) {
        command << command_changes[i].first;
        output_option = output_changes[i].first;
      } else {
        command << command_changes[i].second;
        output_option = output_changes[i].second;
      }
      if (output_option != "") {
        changed_options.push_back(make_pair(option_name[i], 
                                            output_option));
      }
    }
    if (!options[warmup_opt]) {
      int num_iter = options[iter] ? 100 : 2000;
      int num_warmup = options[warmup_opt] ? 60 : num_iter/2;
      stringstream warmup_stream;
      warmup_stream << num_iter - num_warmup;
      changed_options.push_back(make_pair("warmup",
                                          warmup_stream.str()));
    }
    
    return command.str();
  }
};

vector<string> ModelCommand::expected_help_options;
string ModelCommand::model_path;
string ModelCommand::data_file_base;
vector<pair<string, string> > ModelCommand::expected_output;
vector<pair<string, string> > ModelCommand::command_changes;
vector<pair<string, string> > ModelCommand::output_changes;
vector<string> ModelCommand::option_name;


TEST_F(ModelCommand, HelpOptionsMatch) {
  using std::string;
  using std::vector;
  string help_command = model_path;
  help_command.append(" --help");

  vector<string> help_options = 
    parse_help_options(run_command(help_command));

  ASSERT_EQ(expected_help_options.size(), help_options.size());
  for (size_t i = 0; i < expected_help_options.size(); i++) {
    EXPECT_EQ(expected_help_options[i], help_options[i]);
  }
}

void test_sampled_mean(const bitset<options_count>& options, stan::mcmc::chains<>& c) {
  double expected_mean = (options[data])*100.0; // 1: mean = 0, 2: mean = 100
  EXPECT_NEAR(expected_mean, c.mean("y"), 50)
    << "Test that data file is being used";
}

void test_number_of_samples(const bitset<options_count>& options, stan::mcmc::chains<>& c) {
  int num_iter = options[iter] ? 100 : 2000;
  int num_warmup = options[warmup_opt] ? 60 : num_iter/2;
  size_t expected_num_samples = num_iter - num_warmup;
  if (options[thin]) {
    expected_num_samples = ceil(expected_num_samples / 3.0);
  }
  if (options[append_samples]) {
    EXPECT_EQ(2*expected_num_samples, c.num_samples())
      << "Test number of samples when appending samples";
  } else {
    EXPECT_EQ(expected_num_samples, c.num_samples())
      << "Test number of samples when not appending samples";
  }
}

/*void test_specific_sample_values(const bitset<options_count>& options, stan::mcmc::chains<>& c) {
  if (options[iter] || 
      options[leapfrog_steps] || 
      options[epsilon] ||
      options[epsilon_pm] ||
      options[delta] ||
      options[gamma_opt] ||
      options[thin] ||
      options[append_samples] ||
      !options[seed])
    return;
  // seed / chain_id test
  if (!options[append_samples] 
      && !options[warmup_opt]) {
    double expected_first_y;
    
    // options[data] = true
    // -> --data=src/test/models/command1.data.R
    // options[data] = false
    // -> --data=src/test/modles/command2.data.R
    // options[init] = true
    // -> --init=src/test/models/command.init.R
    // options[init = false
    // -> no init (defaults to random init
    // All with --seed=100
    
    if (options[data]) {
      expected_first_y = options[init] ? 99.4208 : 100.727;
    } else { 
      expected_first_y = options[init] ? -0.0852457 : 0.3504832;
    }
    
    Eigen::VectorXd sampled_y;
    sampled_y = c.samples(0, "y");
    if (options[chain_id]) {
      if (expected_first_y == sampled_y(0)) {
        ADD_FAILURE()
          << "chain_id is not default. "
          << "sampled_y[0] should not be drawn from the same seed";
      } else {
        SUCCEED()
          << "chain_id is the default. "
          << "The samples are not drawn from the same seed";
      }
    } else {
      EXPECT_NEAR(expected_first_y, sampled_y(0), 1e-3)
        << "Test for first sample when chain_id == 1";
    }
  }
  }*/

TEST_P(ModelCommand, OptionsTest) {
  bitset<options_count> options(1 << GetParam());
  vector<pair<string, string> > changed_options;
  
  std::string command = get_command(options, changed_options);
  SCOPED_TRACE(command);
  /*std::cout << command << std::endl;
  for (int i = 0; i < changed_options.size(); i++) {
    std::cout << i << ": " << changed_options[i].first << ", " << changed_options[i].second << std::endl;
  }
  std::cout << "---" << std::endl;*/

  // check_output
  check_output(run_command(command), changed_options);

  // test sampled values
  vector<string> names;
  vector<vector<size_t> > dimss;
  size_t skip;
  if (options[leapfrog_steps])
    skip = options[epsilon] && !options[epsilon_pm] ? 1U : 2U;
  else {
    skip = options[epsilon] && !options[epsilon_pm] ? 2U : 3U;
    /*if (options[epsilon] && options[epsilon_pm])
      skip = 3U;
    else if (!options[epsilon] && options[epsilon_pm])
      skip = 3U;
    else if (options[epsilon] && !options[epsilon_pm])
      skip = 2U;
    else if (!options[epsilon] && !options[epsilon_pm])
    skip = 3U;*/
  }
  //std::cout << "options[leapfrog_steps]: " << options[leapfrog_steps] << std::endl;
  //std::cout << "options[epsilon]:        " << options[epsilon] << std::endl;
  //std::cout << "options[epsilon_pm]:     " << options[epsilon_pm] << std::endl;
  //std::cout << "skip:                    " << skip << std::endl;

  std::ifstream ifstream;
  std::string file = model_path+".csv";
  ifstream.open(file.c_str());
  stan::io::stan_csv stan_csv = stan::io::stan_csv_reader::parse(ifstream);
  ifstream.close();

  stan::mcmc::chains<> c(stan_csv);

  test_sampled_mean(options, c);
  test_number_of_samples(options, c);
  //test_specific_sample_values(options, c);
}
INSTANTIATE_TEST_CASE_P(,
                        ModelCommand,
                        Range(-1, int(options_count)));

#include <gtest/gtest.h>

#include <stan/gm/arguments/argument_probe.hpp>
#include <stan/gm/arguments/arg_id.hpp>
#include <stan/gm/arguments/arg_data.hpp>
#include <stan/gm/arguments/arg_init.hpp>
#include <stan/gm/arguments/arg_random.hpp>
#include <stan/gm/arguments/arg_output.hpp>

#include <test/models/utility.hpp>

#include <vector>
#include <sstream>

TEST(StanGmArgumentsConfiguration, Test) {
  
  // Prepare model
  std::vector<std::string> model_path;
  model_path.push_back("src");
  model_path.push_back("test");
  model_path.push_back("gm");
  model_path.push_back("arguments");
  model_path.push_back("test_model");
  
  // Prepare arguments
  std::stringstream s;
  
  std::vector<stan::gm::argument*> valid_arguments;
  //valid_arguments.push_back(new stan::gm::arg_id());
  //valid_arguments.push_back(new stan::gm::arg_data());
  //valid_arguments.push_back(new stan::gm::arg_init());
  //valid_arguments.push_back(new stan::gm::arg_random());
  //valid_arguments.push_back(new stan::gm::arg_output());
  
  stan::gm::argument_probe probe(valid_arguments);
  probe.probe_args(s);
  
  // Check argument consistency
  bool expected_success;
  
  std::string l1;
  std::stringstream expected_output;
  std::stringstream output;
  
  while (s.good()) {
  
    std::getline(s, l1);
    if (!s.good()) continue;
    
    if      (l1 == "good") expected_success = true;
    else if (l1 == "bad") expected_success = false;
    else if (l1 != "") expected_output << l1 << std::endl;
    else {
      
      int n_output = 0;
      
      std::string l2;
      std::string argument("");
      
      while (expected_output.good()) {
        
        std::getline(expected_output, l2);
        if (!expected_output.good()) continue;
        
        l2.erase(0, 1);
        
        if (l2.find("(Default)") != std::string::npos)
          l2 = l2.substr(0, l2.find("(Default)"));
        
        l2.erase(std::remove(l2.begin(), l2.end(), ' '), l2.end());
        
        argument += " " + l2;
        ++n_output;
      
      }
      
      if (argument.length() == 0) continue;
      
      unsigned int cursor = argument.find("=");
      while (cursor < argument.length()) {
        
        unsigned int end = argument.find(" ", cursor) - 1;
        
        std::string value = " " + argument.substr(cursor + 1, end - cursor);

        if (argument.find(value, end) != std::string::npos)
          argument.erase(argument.find(value, end), value.length());

        cursor = argument.find("=", end);
        
      }
      
      expected_output.clear();
      expected_output.seekg(std::ios_base::beg);
      
      std::string command = convert_model_path(model_path) + argument;
      std::string command_output;
      long time;
      
      try {
        command_output = run_command(command, time);
      } catch(...) {
        ADD_FAILURE() << "Failed running command: " << command;
      }
      
      if (expected_success == false) {
                
        unsigned int c1 = command_output.find("is not");
        command_output.erase(0, c1);
        
        unsigned int c2 = command_output.find(" \"");
        unsigned int c3 = command_output.find("Failed to parse");
        command_output.replace(c2, c3 - c2, "\n");
        
        expected_output.str(std::string());

        expected_output << "is not a valid value for" << std::endl;
        expected_output << "Failed to parse arguments, terminating Stan" << std::endl;
        
        n_output = 2;
        
      }

      output.str(command_output);
      
      for (int i = 0; i < n_output; ++i) {
        
        std::string expected_line;
        std::getline(expected_output, expected_line);
        
        std::string actual_line;
        std::getline(output, actual_line);
        
        EXPECT_EQ(expected_line, actual_line);
        
      }
      
      expected_output.clear();
      expected_output.seekg(std::ios_base::beg);
      expected_output.str(std::string());
      
    }
    
  }
  
  for (int i = 0; i < valid_arguments.size(); ++i)
    delete valid_arguments.at(i);
  
}

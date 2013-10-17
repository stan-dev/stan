#include <gtest/gtest.h>
#include <stan/gm/error_codes.hpp>
#include <stan/gm/arguments/argument_probe.hpp>
#include <stan/gm/arguments/arg_method.hpp>
#include <stan/gm/arguments/arg_id.hpp>
#include <stan/gm/arguments/arg_data.hpp>
#include <stan/gm/arguments/arg_init.hpp>
#include <stan/gm/arguments/arg_random.hpp>
#include <stan/gm/arguments/arg_output.hpp>

#include <test/models/utility.hpp>

#include <vector>
#include <sstream>

void clean_line(std::string& line) {
  line.erase(0, 1);
  if (line.find("(Default)") != std::string::npos)
    line = line.substr(0, line.find("(Default)"));
  line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
}

void remove_duplicates(std::string& argument) {
  unsigned int cursor = argument.find("=");
  while (cursor < argument.length()) {
    unsigned int end = argument.find(" ", cursor) - 1;
    std::string value = " " + argument.substr(cursor + 1, end - cursor);

    if (value.length() == 1) ++end;
    else if (argument.find(value, end) != std::string::npos)
      argument.erase(argument.find(value, end), value.length());
    cursor = argument.find("=", end);
  }
}

TEST(StanGmArgumentsConfiguration, TestMethod) {
  
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
  valid_arguments.push_back(new stan::gm::arg_method());
  
  stan::gm::argument_probe probe(valid_arguments);
  probe.probe_args(s);
  
  // Check argument consistency
  bool expected_success = false;
  
  std::string l1;
  std::stringstream expected_output;
  std::stringstream output;
  
  while (s.good()) {
    std::getline(s, l1);
    if (!s.good()) 
      continue;
    
    if (l1 == "good") 
      expected_success = true;
    else if (l1 == "bad") 
      expected_success = false;
    else if (l1 != "") 
      expected_output << l1 << std::endl;
    else {
      int n_output = 0;
      
      std::string l2;
      std::string argument("");
      
      while (expected_output.good()) {
        std::getline(expected_output, l2);
        if (!expected_output.good()) 
          continue;
        clean_line(l2);
        argument += " " + l2;
        ++n_output;
      }
      
      if (argument.length() == 0) 
        continue;
      
      remove_duplicates(argument);

      std::string command = convert_model_path(model_path) + argument;

      SCOPED_TRACE(command);

      run_command_output out = run_command(command);

      expected_output.clear();
      expected_output.seekg(std::ios_base::beg);
      
      if (expected_success == false) {
        unsigned int c1 = out.output.find("is not");
        out.output.erase(0, c1);

        unsigned int c2 = out.output.find(" \"");

        unsigned int c3 = out.output.find("Failed to parse");

        if (c3 != c2)
          out.output.replace(c2, c3 - c2, "\n");
        
        expected_output.str(std::string());

        expected_output << "is not a valid value for" << std::endl;
        expected_output << "Failed to parse arguments, terminating Stan" << std::endl;
        n_output = 2;
        EXPECT_EQ(int(stan::gm::error_codes::USAGE), out.err_code);
      } else {
        EXPECT_EQ(int(stan::gm::error_codes::OK), out.err_code);
      }
      
      output.clear();
      output.seekg(std::ios_base::beg);
      output.str(out.output);
      
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
  
  for (size_t i = 0; i < valid_arguments.size(); ++i)
    delete valid_arguments.at(i);
}

TEST(StanGmArgumentsConfiguration, TestIdWithMethod) {
  
  // Prepare model
  std::vector<std::string> model_path;
  model_path.push_back("src");
  model_path.push_back("test");
  model_path.push_back("gm");
  model_path.push_back("arguments");
  model_path.push_back("test_model");
  
  // Prepare arguments
  std::stringstream method_output;
  stan::gm::arg_method method;
  method.print(&method_output, 0, '\0');
  
  std::string l0;
  std::string method_argument("");
  int n_method_output = 0;
  
  while (method_output.good()) {
    std::getline(method_output, l0);
    if (!method_output.good()) continue;
    clean_line(l0);
    method_argument += " " + l0;
    ++n_method_output;
  }
      
  remove_duplicates(method_argument);

  method_output.clear();
  method_output.seekg(std::ios_base::beg);

  std::stringstream s;
  std::vector<stan::gm::argument*> valid_arguments;
  valid_arguments.push_back(new stan::gm::arg_id());
  
  stan::gm::argument_probe probe(valid_arguments);
  probe.probe_args(s);
  
  // Check argument consistency
  bool expected_success = false;
  
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
        clean_line(l2);
        argument += " " + l2;
        ++n_output;
      }
      
      if (argument.length() == 0) continue;
      remove_duplicates(argument);
      argument = method_argument + argument;
      
      std::string command = convert_model_path(model_path) + argument;
      run_command_output out = run_command(command);
      
      expected_output.clear();
      expected_output.seekg(std::ios_base::beg);
      expected_output.str( method_output.str() + expected_output.str() );
      
      if (expected_success == false) {
        
        unsigned int c1 = out.output.find("is not");
        out.output.erase(0, c1);
        unsigned int c2 = out.output.find(" \"");
        unsigned int c3 = out.output.find("Failed to parse");
        out.output.replace(c2, c3 - c2, "\n");
        
        expected_output.str(std::string());
        expected_output << "is not a valid value for" << std::endl;
        expected_output << "Failed to parse arguments, terminating Stan" << std::endl;
        n_output = 2;
        
      }
    
      output.str(out.output);
      
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
  
  for (size_t i = 0; i < valid_arguments.size(); ++i)
    delete valid_arguments.at(i);

}


TEST(StanGmArgumentsConfiguration, TestIdWithoutMethod) {
  
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
  valid_arguments.push_back(new stan::gm::arg_id());
  
  stan::gm::argument_probe probe(valid_arguments);
  probe.probe_args(s);
  
  // Check argument consistency
  bool expected_success = false;
  
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
        clean_line(l2);
        argument += " " + l2;
        ++n_output;
      }
      
      if (argument.length() == 0) continue;
      
      remove_duplicates(argument);
      
      std::string command = convert_model_path(model_path) + argument;
      run_command_output out = run_command(command);
      
      expected_output.clear();
      expected_output.seekg(std::ios_base::beg);
      
      if (expected_success == false) {
        
        unsigned int c1 = out.output.find("is not");
        out.output.erase(0, c1);
        unsigned int c2 = out.output.find(" \"");
        unsigned int c3 = out.output.find("Failed to parse");
        out.output.replace(c2, c3 - c2, "\n");
        
        expected_output.str(std::string());
        expected_output << "is not a valid value for" << std::endl;
        expected_output << "Failed to parse arguments, terminating Stan" << std::endl;
        n_output = 2;
        
      }
      else {
        expected_output.str(std::string());
        expected_output << "A method must be specified!" << std::endl;
        expected_output << "Failed to parse arguments, terminating Stan" << std::endl;
        n_output = 2;
      }
    
      output.str(out.output);
      
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
  
  for (size_t i = 0; i < valid_arguments.size(); ++i)
    delete valid_arguments.at(i);
  
}

TEST(StanGmArgumentsConfiguration, TestDataWithMethod) {
  
  // Prepare model
  std::vector<std::string> model_path;
  model_path.push_back("src");
  model_path.push_back("test");
  model_path.push_back("gm");
  model_path.push_back("arguments");
  model_path.push_back("test_model");
  
  // Prepare arguments
  std::stringstream method_output;
  stan::gm::arg_method method;
  method.print(&method_output, 0, '\0');
  
  std::string l0;
  std::string method_argument("");
  int n_method_output = 0;
  
  while (method_output.good()) {
    std::getline(method_output, l0);
    if (!method_output.good()) continue;
    clean_line(l0);
    method_argument += " " + l0;
    ++n_method_output;
  }
  
  remove_duplicates(method_argument);
  
  method_output.clear();
  method_output.seekg(std::ios_base::beg);
  
  std::stringstream s;
  std::vector<stan::gm::argument*> valid_arguments;
  valid_arguments.push_back(new stan::gm::arg_data());
  
  stan::gm::argument_probe probe(valid_arguments);
  probe.probe_args(s);
  
  // Check argument consistency
  bool expected_success = false;
  
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
        clean_line(l2);
        argument += " " + l2;
        ++n_output;
      }
      
      if (argument.length() == 0) continue;
      remove_duplicates(argument);
      argument = method_argument + argument;
      
      std::string command = convert_model_path(model_path) + argument;
      run_command_output out = run_command(command);

      expected_output.clear();
      expected_output.seekg(std::ios_base::beg);
      expected_output.str( method_output.str() + expected_output.str() );
      
      if (expected_success == false) {
        
        unsigned int c1 = out.output.find("is not");
        out.output.erase(0, c1);
        unsigned int c2 = out.output.find(" \"");
        unsigned int c3 = out.output.find("Failed to parse");
        out.output.replace(c2, c3 - c2, "\n");
        
        expected_output.str(std::string());
        expected_output << "is not a valid value for" << std::endl;
        expected_output << "Failed to parse arguments, terminating Stan" << std::endl;
        n_output = 2;
        
      }
      
      output.str(out.output);
      
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
  
  for (size_t i = 0; i < valid_arguments.size(); ++i)
    delete valid_arguments.at(i);
  
}


TEST(StanGmArgumentsConfiguration, TestDataWithoutMethod) {
  
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
  valid_arguments.push_back(new stan::gm::arg_data());
  
  stan::gm::argument_probe probe(valid_arguments);
  probe.probe_args(s);
  
  // Check argument consistency
  bool expected_success = false;
  
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
        clean_line(l2);
        argument += " " + l2;
        ++n_output;
      }
      
      if (argument.length() == 0) continue;
      
      remove_duplicates(argument);
      
      std::string command = convert_model_path(model_path) + argument;
      run_command_output out = run_command(command);
      
      expected_output.clear();
      expected_output.seekg(std::ios_base::beg);
      
      if (expected_success == false) {
        
        unsigned int c1 = out.output.find("is not");
        out.output.erase(0, c1);
        unsigned int c2 = out.output.find(" \"");
        unsigned int c3 = out.output.find("Failed to parse");
        out.output.replace(c2, c3 - c2, "\n");
        
        expected_output.str(std::string());
        expected_output << "is not a valid value for" << std::endl;
        expected_output << "Failed to parse arguments, terminating Stan" << std::endl;
        n_output = 2;
        
      }
      else {
        expected_output.str(std::string());
        expected_output << "A method must be specified!" << std::endl;
        expected_output << "Failed to parse arguments, terminating Stan" << std::endl;
        n_output = 2;
      }
      
      output.str(out.output);
      
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
  
  for (size_t i = 0; i < valid_arguments.size(); ++i)
    delete valid_arguments.at(i);
  
}

TEST(StanGmArgumentsConfiguration, TestInitWithMethod) {
  
  // Prepare model
  std::vector<std::string> model_path;
  model_path.push_back("src");
  model_path.push_back("test");
  model_path.push_back("gm");
  model_path.push_back("arguments");
  model_path.push_back("test_model");
  
  // Prepare arguments
  std::stringstream method_output;
  stan::gm::arg_method method;
  method.print(&method_output, 0, '\0');
  
  std::string l0;
  std::string method_argument("");
  int n_method_output = 0;
  
  while (method_output.good()) {
    std::getline(method_output, l0);
    if (!method_output.good()) continue;
    clean_line(l0);
    method_argument += " " + l0;
    ++n_method_output;
  }
  
  remove_duplicates(method_argument);
  
  method_output.clear();
  method_output.seekg(std::ios_base::beg);
  
  std::stringstream s;
  std::vector<stan::gm::argument*> valid_arguments;
  valid_arguments.push_back(new stan::gm::arg_init());
  
  stan::gm::argument_probe probe(valid_arguments);
  probe.probe_args(s);
  
  // Check argument consistency
  bool expected_success = false;
  
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
        clean_line(l2);
        argument += " " + l2;
        ++n_output;
      }
      
      if (argument.length() == 0) continue;
      remove_duplicates(argument);
      argument = method_argument + argument;
      
      std::string command = convert_model_path(model_path) + argument;
      run_command_output out = run_command(command);
      
      expected_output.clear();
      expected_output.seekg(std::ios_base::beg);
      expected_output.str( method_output.str() + expected_output.str() );
      
      if (expected_success == false) {
        
        unsigned int c1 = out.output.find("is not");
        out.output.erase(0, c1);
        unsigned int c2 = out.output.find(" \"");
        unsigned int c3 = out.output.find("Failed to parse");
        out.output.replace(c2, c3 - c2, "\n");
        
        expected_output.str(std::string());
        expected_output << "is not a valid value for" << std::endl;
        expected_output << "Failed to parse arguments, terminating Stan" << std::endl;
        n_output = 2;
        
      }
      
      output.str(out.output);
      
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
  
  for (size_t i = 0; i < valid_arguments.size(); ++i)
    delete valid_arguments.at(i);
  
}


TEST(StanGmArgumentsConfiguration, TestInitWithoutMethod) {
  
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
  valid_arguments.push_back(new stan::gm::arg_init());
  
  stan::gm::argument_probe probe(valid_arguments);
  probe.probe_args(s);
  
  // Check argument consistency
  bool expected_success = false;
  
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
        clean_line(l2);
        argument += " " + l2;
        ++n_output;
      }
      
      if (argument.length() == 0) continue;
      
      remove_duplicates(argument);
      
      std::string command = convert_model_path(model_path) + argument;
      run_command_output out = run_command(command);
      
      expected_output.clear();
      expected_output.seekg(std::ios_base::beg);
      
      if (expected_success == false) {
        
        unsigned int c1 = out.output.find("is not");
        out.output.erase(0, c1);
        unsigned int c2 = out.output.find(" \"");
        unsigned int c3 = out.output.find("Failed to parse");
        out.output.replace(c2, c3 - c2, "\n");
        
        expected_output.str(std::string());
        expected_output << "is not a valid value for" << std::endl;
        expected_output << "Failed to parse arguments, terminating Stan" << std::endl;
        n_output = 2;
        
      }
      else {
        expected_output.str(std::string());
        expected_output << "A method must be specified!" << std::endl;
        expected_output << "Failed to parse arguments, terminating Stan" << std::endl;
        n_output = 2;
      }
      
      output.str(out.output);
      
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
  
  for (size_t i = 0; i < valid_arguments.size(); ++i)
    delete valid_arguments.at(i);
  
}

TEST(StanGmArgumentsConfiguration, TestRandomWithMethod) {
  
  // Prepare model
  std::vector<std::string> model_path;
  model_path.push_back("src");
  model_path.push_back("test");
  model_path.push_back("gm");
  model_path.push_back("arguments");
  model_path.push_back("test_model");
  
  // Prepare arguments
  std::stringstream method_output;
  stan::gm::arg_method method;
  method.print(&method_output, 0, '\0');
  
  std::string l0;
  std::string method_argument("");
  int n_method_output = 0;
  
  while (method_output.good()) {
    std::getline(method_output, l0);
    if (!method_output.good()) continue;
    clean_line(l0);
    method_argument += " " + l0;
    ++n_method_output;
  }
  
  remove_duplicates(method_argument);
  
  method_output.clear();
  method_output.seekg(std::ios_base::beg);
  
  std::stringstream s;
  std::vector<stan::gm::argument*> valid_arguments;
  valid_arguments.push_back(new stan::gm::arg_random());
  
  stan::gm::argument_probe probe(valid_arguments);
  probe.probe_args(s);
  
  // Check argument consistency
  bool expected_success = false;
  
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
        clean_line(l2);
        argument += " " + l2;
        ++n_output;
      }
      
      if (argument.length() == 0) continue;
      remove_duplicates(argument);
      argument = method_argument + argument;
      
      std::string command = convert_model_path(model_path) + argument;
      run_command_output out = run_command(command);
      
      expected_output.clear();
      expected_output.seekg(std::ios_base::beg);
      expected_output.str( method_output.str() + expected_output.str() );
      
      if (expected_success == false) {
        
        unsigned int c1 = out.output.find("is not");
        out.output.erase(0, c1);
        unsigned int c2 = out.output.find(" \"");
        unsigned int c3 = out.output.find("Failed to parse");
        out.output.replace(c2, c3 - c2, "\n");
        
        expected_output.str(std::string());
        expected_output << "is not a valid value for" << std::endl;
        expected_output << "Failed to parse arguments, terminating Stan" << std::endl;
        n_output = 2;
        
      }
      
      output.str(out.output);
      
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
  
  for (size_t i = 0; i < valid_arguments.size(); ++i)
    delete valid_arguments.at(i);
  
}


TEST(StanGmArgumentsConfiguration, TestRandomWithoutMethod) {
  
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
  valid_arguments.push_back(new stan::gm::arg_random());
  
  stan::gm::argument_probe probe(valid_arguments);
  probe.probe_args(s);
  
  // Check argument consistency
  bool expected_success = false;
  
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
        clean_line(l2);
        argument += " " + l2;
        ++n_output;
      }
      
      if (argument.length() == 0) continue;
      
      remove_duplicates(argument);
      
      std::string command = convert_model_path(model_path) + argument;
      run_command_output out = run_command(command);
      
      expected_output.clear();
      expected_output.seekg(std::ios_base::beg);
      
      if (expected_success == false) {
        
        unsigned int c1 = out.output.find("is not");
        out.output.erase(0, c1);
        unsigned int c2 = out.output.find(" \"");
        unsigned int c3 = out.output.find("Failed to parse");
        out.output.replace(c2, c3 - c2, "\n");
        
        expected_output.str(std::string());
        expected_output << "is not a valid value for" << std::endl;
        expected_output << "Failed to parse arguments, terminating Stan" << std::endl;
        n_output = 2;
        
      }
      else {
        expected_output.str(std::string());
        expected_output << "A method must be specified!" << std::endl;
        expected_output << "Failed to parse arguments, terminating Stan" << std::endl;
        n_output = 2;
      }
      
      output.str(out.output);
      
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
  
  for (size_t i = 0; i < valid_arguments.size(); ++i)
    delete valid_arguments.at(i);
  
}

TEST(StanGmArgumentsConfiguration, TestOutputWithMethod) {
  
  // Prepare model
  std::vector<std::string> model_path;
  model_path.push_back("src");
  model_path.push_back("test");
  model_path.push_back("gm");
  model_path.push_back("arguments");
  model_path.push_back("test_model");
  
  // Prepare arguments
  std::stringstream method_output;
  stan::gm::arg_method method;
  method.print(&method_output, 0, '\0');
  
  std::string l0;
  std::string method_argument("");
  int n_method_output = 0;
  
  while (method_output.good()) {
    std::getline(method_output, l0);
    if (!method_output.good()) continue;
    clean_line(l0);
    method_argument += " " + l0;
    ++n_method_output;
  }
  
  remove_duplicates(method_argument);
  
  method_output.clear();
  method_output.seekg(std::ios_base::beg);
  
  std::stringstream s;
  std::vector<stan::gm::argument*> valid_arguments;
  valid_arguments.push_back(new stan::gm::arg_output());
  
  stan::gm::argument_probe probe(valid_arguments);
  probe.probe_args(s);
  
  // Check argument consistency
  bool expected_success = false;
  
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
        clean_line(l2);
        argument += " " + l2;
        ++n_output;
      }
      
      if (argument.length() == 0) continue;
      remove_duplicates(argument);
      argument = method_argument + argument;
      
      std::string command = convert_model_path(model_path) + argument;
      run_command_output out = run_command(command);
      
      expected_output.clear();
      expected_output.seekg(std::ios_base::beg);
      expected_output.str( method_output.str() + expected_output.str() );
      
      if (expected_success == false) {
        
        unsigned int c1 = out.output.find("is not");
        out.output.erase(0, c1);
        unsigned int c2 = out.output.find(" \"");
        unsigned int c3 = out.output.find("Failed to parse");
        out.output.replace(c2, c3 - c2, "\n");
        
        expected_output.str(std::string());
        expected_output << "is not a valid value for" << std::endl;
        expected_output << "Failed to parse arguments, terminating Stan" << std::endl;
        n_output = 2;
        
      }
      
      output.str(out.output);
      
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
  
  for (size_t i = 0; i < valid_arguments.size(); ++i)
    delete valid_arguments.at(i);
  
}


TEST(StanGmArgumentsConfiguration, TestOutputWithoutMethod) {
  
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
  valid_arguments.push_back(new stan::gm::arg_output());
  
  stan::gm::argument_probe probe(valid_arguments);
  probe.probe_args(s);
  
  // Check argument consistency
  bool expected_success = false;
  
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
        clean_line(l2);
        argument += " " + l2;
        ++n_output;
      }
      
      if (argument.length() == 0) continue;
      
      remove_duplicates(argument);
      
      std::string command = convert_model_path(model_path) + argument;
      run_command_output out = run_command(command);
      
      expected_output.clear();
      expected_output.seekg(std::ios_base::beg);
      
      if (expected_success == false) {
        
        unsigned int c1 = out.output.find("is not");
        out.output.erase(0, c1);
        unsigned int c2 = out.output.find(" \"");
        unsigned int c3 = out.output.find("Failed to parse");
        out.output.replace(c2, c3 - c2, "\n");
        
        expected_output.str(std::string());
        expected_output << "is not a valid value for" << std::endl;
        expected_output << "Failed to parse arguments, terminating Stan" << std::endl;
        n_output = 2;
        
      }
      else {
        expected_output.str(std::string());
        expected_output << "A method must be specified!" << std::endl;
        expected_output << "Failed to parse arguments, terminating Stan" << std::endl;
        n_output = 2;
      }
      
      output.str(out.output);
      
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
  
  for (size_t i = 0; i < valid_arguments.size(); ++i)
    delete valid_arguments.at(i);
  
}

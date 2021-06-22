#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

#include <exception>
#include <string>
#include <iostream>

namespace utils{

  // Basic wrapper of std::exception...
  class Exception : std::exception{
  private:
      std::string e;
  public:
      Exception(const std::string s) : e(s){
        std::cout << "ERROR: " << e << "\n";
        throw;
      }
  private:
    virtual const char* what(void) {
      return e.c_str();
    }
  };

} // utils

#endif //EXCEPTIONS_H

#ifndef UTILS_H
#define UTILS_H

#include <exception>
#include <iostream>

namespace snail {
namespace utils {

/// π
const double PI = 3.1415926535;

/// radius of earth (km)
const double R = 6371.0;

/// conversion from degrees to radians
const double toRad = PI / 180.0;

/// Basic wrapper of std::exception.
class Exception : std::exception {
private:
  std::string e;

public:
  Exception(const std::string s) : e(s) {
    std::cout << "ERROR: " << e << "\n";
    throw std::runtime_error(s);
  }

private:
  virtual const char *what(void) { return e.c_str(); }
};

} // namespace utils
} // namespace snail
#endif // UTILS_H

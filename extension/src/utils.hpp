#ifndef UTILS_H
#define UTILS_H

#include <exception>
#include <iostream>
#include <limits>
#include <cmath>

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

/// Compare float/double almost equal
/// From example code at
/// https://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
template <class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
almost_equal(T x, T y, int ulp) {
  // the machine epsilon has to be scaled to the magnitude of the values used
  // and multiplied by the desired precision in ULPs (units in the last place)
  return std::fabs(x - y) <=
             std::numeric_limits<T>::epsilon() * std::fabs(x + y) * ulp
         // unless the result is subnormal
         || std::fabs(x - y) < std::numeric_limits<T>::min();
}

} // namespace utils
} // namespace snail
#endif // UTILS_H

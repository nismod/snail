#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "exceptions.hpp"

namespace utils {

/// π
const double PI = 3.1415926535;

/// radius of earth (km)
const double R = 6371.0;

/// conversion from degrees to radians
const double toRad = PI / 180.0;

/// Determine if a file exists
inline bool exists(const std::string &name) {
  std::ifstream f(name.c_str());
  return f.good();
}

/// Determine a MapInfo file exists (both parts).
inline void mifExists(const std::string &name) {
  // Take a local copy, for purpose of DRY.
  std::string _name = name;
  if (_name.find(".mif") != std::string::npos ||
      _name.find(".mid") != std::string::npos)
    _name = _name.substr(0, _name.length() - 4);

  // Test that both parts of the MIF file exist.
  if (!exists(_name + ".mif"))
    Exception("File missing: " + _name + ".mif");

  if (!exists(_name + ".mid"))
    Exception("File missing: " + _name + ".mid");
}

/// Make a string lower-case.
inline std::string lower_case(const std::string s) {
  std::string l_s = s;
  std::transform(l_s.begin(), l_s.end(), l_s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return l_s;
}

/// Read a line of delimited text into vector of strings,
/// respecting any strings that may have delimiters in quotes.
inline std::vector<std::string> readLine(const std::string line,
                                         const char delim = ',',
                                         const bool stripQuotes = false) {
  std::stringstream ss(line);

  // Create some space to store the words in the incoming line.
  std::vector<std::string> words;
  std::string word;

  // Parse the incoming line, respecting delims as requested.
  while (std::getline(ss, word, delim)) {
    if (word.size() > 0) {
      // If needed, strip the quotes from a word.
      if (stripQuotes)
        word.erase(remove(word.begin(), word.end(), '\"'), word.end());
      // Stick it on the tab.
      words.push_back(word);
    }
  }

  // Loop through the words, and check for any unhandled open-quotes which
  // indicates delimters were included between quotes.
  std::vector<std::string> fixedWords;
  // Flag to indicate we need to append neighboring words.
  bool append = false;
  for (std::size_t i = 0; i < words.size(); i++) {
    // Count the number of open-quotes in the string.
    int numSQ = 0, numDQ = 0;
    // Look for quotes...
    for (auto c : words.at(i)) {
      if (c == '\'')
        numSQ++;
      if (c == '\"')
        numDQ++;
    }

    // Append words if odd-numbers of any type of quote characters.
    if (append) {
      fixedWords.at(fixedWords.size() - 1) =
          fixedWords.at(fixedWords.size() - 1) + "," + words.at(i);
    } else {
      fixedWords.push_back(words.at(i));
    }

    // Update the append flag f or the next go around.
    if (numDQ % 2 != 0 || numSQ % 2 != 0) {
      append = !append;
    }
  }

  return fixedWords;
}

/// Write a data buffer to disk (NOTE: this is done to
/// keep the memory footprint down for large study areas).
template <typename T>
inline void writeBuffer(const std::vector<T> &data, const std::string f) {
  std::ofstream b(f, std::ios::out | std::ios::binary);
  b.write((char *)&data[0], data.size() * sizeof(T));
  b.close();
}
} // namespace utils

#endif // UTILS_H

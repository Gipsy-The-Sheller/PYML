#ifndef PYML_CPP_VECTORIZE_HPP
#define PYML_CPP_VECTORIZE_HPP

#include <concepts>
#include <vector>

namespace VecMnpUtils {

template <typename T>
concept Vectorable = std::same_as<T, float> || std::same_as<T, double>;

template <Vectorable Precision>
std::vector<Precision> Multiplier(const Precision* queue1,
                                  const Precision* queue2,
                                  int qlength);

extern template std::vector<float> Multiplier<float>(
    const float*, const float*, int);
extern template std::vector<double> Multiplier<double>(
    const double*, const double*, int);

}  // namespace VecMnpUtils

#endif  // PYML_CPP_VECTORIZE_HPP

#include "likelihood.hpp"

template struct PYML::Node<double>;
template struct PYML::BinaryTree<double>;
template struct PYML::SubstMatrix<float>;
template struct PYML::SubstMatrix<double>;
template class PYML::LikelihoodCalculator<float>;
template class PYML::LikelihoodCalculator<double>;

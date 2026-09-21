// #include <pybind11/pybind11.h>
// #include <pybind11/embed.h>

#include "likelihood.hpp"
#include "vectorize.hpp"
#include <vector>
#include <queue>
#include <tuple>
#include <concepts>
#include <type_traits>

template<typename T>
concept Vectorable = std::same_as<T, float> || std::same_as<T, double>;

template <typename FEATURE> struct Node {
    int parent_idx;
    int cleft_idx;
    int cright_idx;
    FEATURE features;

    bool IsTip() {
        return (!cleft_idx && !cright_idx);
    }

    bool IsRoot() {
        return !parent_idx;
    }
};

template <typename FEATURE> struct BinaryTree {

    int TAXA_COUNT;
    int NODE_COUNT;
    int BRANCH_COUNT;

    bool isNewick(char* newick) {
        //TODO
    }

    void NewickParser(char* newick) {
        //TODO
    }

    std::vector<Node<FEATURE>> Nodes;
    
    // BinaryTree(int taxa_count, char* newick): TAXA_COUNT(taxa_count) {
    //     NODE_COUNT = 2 * TAXA_COUNT - 1;
    //     BRANCH_COUNT = 2 * TAXA_COUNT - 2;
    //     if (!isNewick(newick)) {

    //     }
    // }
};

template <Vectorable PRECISION> struct SubstMatrix {
    int states;
    std::vector<PRECISION> UMatrix;
    std::vector<PRECISION> AMatrix;
    
    std::vector<PRECISION> PMatrix (double t) {
        // P = exp (Qt); EVD method.
        auto A_square = VecMnpUtils::Multiplier(/*TODO*/);
    }
    
    void InitWithQPi (std::vector<PRECISION> QMatrix, std::vector<PRECISION> Pis) {
        // TODO
        // EVD
    }
};

template <Vectorable PRECISION> struct SingleLikelihoodVector {
    std::vector<PRECISION> vec;
};

template <Vectorable PRECISION> struct LikelihoodVectors {
    PRECISION* data;

    std::tuple<std::vector<PRECISION>> PackAllVectors () {
        // return all vectors for pruning
    }

    LikelihoodVectors(int nstates, int ncols, PRECISION* cachestart) {
        data = cachestart;
    }
}

template <Vectorable PRECISION> struct LikelihoodCalculator {

    BinaryTree<int> topology;
    SubstMatrix<PRECISION> model;

    template <Vectorable PRECISION> LogLikPruner ();


    LikelihoodCalculator(BinaryTree<*LikelihoodVectors<PRECISION>> _topology, SubstMatrix _model, MSA taxastates, void* cachestart = nullptr): 
        topology(_topology),
        model(_model) {
        // init taxa states by order
    }
};

Calculable (){
    //TODO
}

GetChildrenPtr (){
    //TODO
}

template <Vectorable PRECISION> LikelihoodCalculator::LogLikPruner () {
    auto root = topology.Nodes[1];
    std::queue<Node> q = {&root};
    // rule: left first, then right
    while(!q.empty()) {
        auto currentptr = q.top();
        if (!Calculable(current)) {
            auto cleftptr = GetChildrenPtr(current.cleft_idx);
            auto crightptr = GetChildrenPtr(current.cright_idx);
            if(!crightptr->IsTip() || !Calculable(crightptr)) q.push(crightptr);
            if(!cleftptr->IsTip() || !Calculable(cleftptr)) q.push(cleftptr);
        } else {
            auto left_vecs = ;
            auto right_vecs = ;
            auto current_vec = VecMnpUtils::Multiplier(/*TODO*/);
        }
    }
}
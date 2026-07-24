#ifndef GPRAT_GPHYPERPARAMETERS_HPP
#define GPRAT_GPHYPERPARAMETERS_HPP

#pragma once

#include "gprat/detail/config.hpp"

#include <memory>
#include <string>

GPRAT_NS_BEGIN

/**
 * @brief Hyperparameters for the Adam optimizer
 */
struct AdamParams
{
    /**
     * @brief Learning rate is step size per iteration
     */
    double learning_rate;

    /**
     * @brief Beta1 is the exponential decay rate for the first moment estimates
     */
    double beta1;

    /**
     * @brief Beta2 is the exponential decay rate for the second moment estimates
     */
    double beta2;

    /**
     * @brief Epsilon is a small constant to prevent division by zero
     */
    double epsilon;

    /**
     * @brief Number of optimization iterations
     */
    std::size_t opt_iter;

    /**
     * @brief Initialize hyperparameters
     *
     * @param lr learning rate
     * @param b1 beta1
     * @param b2 beta2
     * @param eps epsilon
     * @param opt_i number of optimization iterationsgp op
     */
    AdamParams(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8, std::size_t opt_i = 0);

    /**
     * @brief Returns a string representation of the hyperparameters
     */
    std::string repr() const;
};

template <class Archive>
void save_construct_data(Archive &ar, const AdamParams *v, const unsigned int)
{
    ar << v->learning_rate;
    ar << v->beta1;
    ar << v->beta2;
    ar << v->epsilon;
    ar << v->opt_iter;
}

template <class Archive>
void load_construct_data(Archive &ar, AdamParams *v, const unsigned int)
{
    double learning_rate, beta1, beta2, epsilon;
    std::size_t opt_iter;
    ar >> learning_rate;
    ar >> beta1;
    ar >> beta2;
    ar >> epsilon;
    ar >> opt_iter;

    new (v) AdamParams(learning_rate, beta1, beta2, epsilon, opt_iter);
}

GPRAT_NS_END

#endif

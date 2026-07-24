#ifndef GPRAT_GPKERNELS_HPP
#define GPRAT_GPKERNELS_HPP

#pragma once

#include "gprat/detail/config.hpp"

#include <cstddef>
#include <memory>
#include <vector>

GPRAT_NS_BEGIN

/**
 * @brief Squared Exponential Kernel Parameters
 */
struct SEKParams
{
    /**
     * @brief Lengthscale: variance of training output
     *
     * Sometimes denoted with index 0.
     */
    double lengthscale;

    /**
     * @brief Vertical Lengthscale: standard deviation of training input
     *
     * Sometimes denoted with index 1.
     */
    double vertical_lengthscale;

    /**
     * @brief Noise Variance: small value
     *
     * Sometimes denoted with index 2.
     */
    double noise_variance;

    std::vector<double> m_T;

    std::vector<double> w_T;

    /**
     * @brief Construct a new SEKParams object
     *
     * @param in_lengthscale Lengthscale: variance of training output
     * @param in_vertical_lengthscale Vertical Lengthscale: standard deviation
     * of training input
     * @param in_noise_variance Noise Variance: small value
     */
    SEKParams(double in_lengthscale, double in_vertical_lengthscale, double in_noise_variance);

    /**
     * @brief Return the number of parameters
     */
    std::size_t size();

    /**
     * @brief Sets the parameter at the given index
     *
     * Index 0: lengthscale
     * Index 1: vertical_lengthscale
     * Index 2: noise_variance
     *
     * @param index Index of the parameter to set
     * @param value Value to set
     */
    void set_param(std::size_t index, double value);

    /**
     * @brief Returns the parameter at the given index
     *
     * Index 0: lengthscale
     * Index 1: vertical_lengthscale
     * Index 2: noise_variance
     *
     * @param index Index of the parameter to get
     */
    const double &get_param(std::size_t index) const;
};

template <class Archive>
void save_construct_data(Archive &ar, const SEKParams *v, const unsigned int)
{
    ar << v->lengthscale;
    ar << v->vertical_lengthscale;
    ar << v->noise_variance;
}

template <class Archive>
void load_construct_data(Archive &ar, SEKParams *v, const unsigned int)
{
    double lengthscale, vertical_lengthscale, noise_variance;
    ar >> lengthscale;
    ar >> vertical_lengthscale;
    ar >> noise_variance;

    new (v) SEKParams(lengthscale, vertical_lengthscale, noise_variance);
}

template <typename Archive>
void serialize(Archive &ar, SEKParams &pt, const unsigned int)
{
    ar & pt.m_T & pt.w_T;
}

GPRAT_NS_END

#endif

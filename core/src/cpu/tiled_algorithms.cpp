#include "gprat/cpu/tiled_algorithms.hpp"

#include "gprat/cpu/adapter_cblas_fp64.hpp"
#include "gprat/cpu/gp_algorithms.hpp"
#include "gprat/cpu/gp_optimizer.hpp"

GPRAT_NS_BEGIN

namespace cpu
{

namespace impl
{

void update_parameters(
    const AdamParams &adam_params,
    SEKParams &sek_params,
    std::size_t N,
    std::size_t n_tiles,
    std::size_t iter,
    std::size_t param_idx,
    double trace,
    double dot,
    bool jitter,
    double factor)
{
    // Compute gradient = trace + dot
    double gradient = factor * compute_gradient(trace, dot, N, n_tiles);

    ////////////////////////////////////
    // PART 2: Update parameter
    // Update moments
    // m_T = beta1 * m_T-1 + (1 - beta1) * g_T
    sek_params.m_T[param_idx] = update_first_moment(gradient, sek_params.m_T[param_idx], adam_params.beta1);
    // w_T = beta2 + w_T-1 + (1 - beta2) * g_T^2
    sek_params.w_T[param_idx] = update_second_moment(gradient, sek_params.w_T[param_idx], adam_params.beta2);

    // Transform hyperparameter to unconstrained form
    double unconstrained_param = to_unconstrained(sek_params.get_param(param_idx), jitter);
    // Adam step update with unconstrained parameter
    // compute beta_t inside
    double updated_param =
        adam_step(unconstrained_param, adam_params, sek_params.m_T[param_idx], sek_params.w_T[param_idx], iter);
    // Transform hyperparameter back to constrained form
    sek_params.set_param(param_idx, to_constrained(updated_param, jitter));
}

}  // namespace impl

}  // end of namespace cpu

GPRAT_NS_END

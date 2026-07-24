#pragma once

#include "gprat/gprat.hpp"

#include <boost/json.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief Resolves the data/ directory to use: the GPRAT_ROOT environment variable if set,
 *        otherwise `fallback`. Shared so every test binary that reads data/data_1024/... agrees
 *        on the same directory when GPRAT_ROOT is set, regardless of each binary's own default.
 */
inline std::string get_data_directory(const std::string &fallback)
{
    const char *env_root = std::getenv("GPRAT_ROOT");
    if (env_root)
    {
        return env_root;
    }
    return fallback;
}

// Struct containing all results we'd like to compare
struct gprat_results
{
    std::vector<std::vector<double>> cholesky;
    std::vector<double> losses;
    std::vector<std::vector<double>> sum;
    std::vector<std::vector<double>> full;
    std::vector<double> pred;
};

// The following two functions are for JSON (de-)serialization
inline void tag_invoke(boost::json::value_from_tag, boost::json::value &jv, const gprat_results &results)
{
    jv = {
        { "cholesky", boost::json::value_from(results.cholesky) },
        { "losses", boost::json::value_from(results.losses) },
        { "sum", boost::json::value_from(results.sum) },
        { "full", boost::json::value_from(results.full) },
        { "pred", boost::json::value_from(results.pred) },
    };
}

// This helper function deduces the type and assigns the value with the matching key
template <typename T>
BOOST_FORCEINLINE void extract(const boost::json::object &obj, T &t, std::string_view key)
{
    t = boost::json::value_to<T>(obj.at(key));
}

inline gprat_results tag_invoke(boost::json::value_to_tag<gprat_results>, const boost::json::value &jv)
{
    gprat_results results;
    const auto &obj = jv.as_object();
    extract(obj, results.cholesky, "cholesky");
    extract(obj, results.losses, "losses");
    extract(obj, results.sum, "sum");
    extract(obj, results.full, "full");
    extract(obj, results.pred, "pred");
    return results;
}

template <typename T>
std::vector<T> to_vector(const gprat::const_tile_data<T> &data)
{
    return { data.begin(), data.end() };
}

template <typename T>
std::vector<std::vector<T>> to_vector(const std::vector<gprat::const_tile_data<T>> &data)
{
    std::vector<std::vector<T>> out;
    out.reserve(data.size());
    for (const auto &row : data)
    {
        out.emplace_back(to_vector<T>(row));
    }
    return out;
}

template <typename T>
std::vector<std::vector<T>> to_vector(const std::vector<gprat::mutable_tile_data<T>> &data)
{
    std::vector<std::vector<T>> out;
    out.reserve(data.size());
    for (const auto &row : data)
    {
        out.emplace_back(to_vector<T>(row));
    }
    return out;
}

/**
 * @brief Tries to load expected results from `filename`. If the file does not exist, writes
 *        `fallback_results` to it and returns false. Returns true when results are loaded.
 */
inline bool load_or_create_expected_results(
    const std::string &filename, const gprat_results &fallback_results, gprat_results &results)
{
    {
        std::ifstream ifs(filename);
        if (!ifs.fail())
        {
            try
            {
                using iterator_type = std::istreambuf_iterator<char>;
                const std::string content(iterator_type{ ifs }, iterator_type{});
                results = boost::json::value_to<gprat_results>(boost::json::parse(content));
            }
            catch (const std::exception &e)
            {
                std::cerr << "Failed to parse baseline " << filename << ": " << e.what()
                          << " — overwriting with current results.\n";
                results = gprat_results{};
            }

            // Stale if any field present in the current run is absent or has a different outer
            // size in the baseline (e.g. CPU baseline loaded by the GPU test, or n_tiles changed).
            const bool stale =
                (!fallback_results.cholesky.empty()
                 && (results.cholesky.empty() || results.cholesky.size() != fallback_results.cholesky.size()))
                || (!fallback_results.losses.empty() && results.losses.size() != fallback_results.losses.size())
                || (!fallback_results.sum.empty()
                    && (results.sum.empty() || results.sum.size() != fallback_results.sum.size()))
                || (!fallback_results.full.empty()
                    && (results.full.empty() || results.full.size() != fallback_results.full.size()))
                || (!fallback_results.pred.empty() && results.pred.size() != fallback_results.pred.size());
            if (!stale)
            {
                return true;
            }

            std::cerr << "Baseline in " << filename << " is incomplete or mismatched"
                      << " — overwriting with current results.\n";
        }
    }

    std::ofstream fout(filename);
    fout << boost::json::serialize(boost::json::value_from(fallback_results));
    return false;
}

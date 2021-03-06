#include "types.h"

std::size_t ceil_div(std::size_t x, std::size_t y)
{
    return (x + y - 1) / y;
}

template <class ElementType, class BinaryFn>
ElementType reduce_vector(const ElementType* V, std::size_t n, BinaryFn f, ElementType zero)
{
    unsigned T = get_num_threads();
    struct reduction_partial_result_t
    {
        alignas(hardware_destructive_interface_size) ElementType value;

        explicit reduction_partial_result_t(ElementType value)
        {
            value = value;
        }
    };
    static auto reduction_partial_results =
            std::vector<reduction_partial_result_t>(T, reduction_partial_result_t(zero));

    constexpr std::size_t k = 2;
    auto thread_proc = [=](unsigned t)
    {
        auto K = ceil_div(n, k);
        std::size_t Mt = K / T, it1;

        if (t < (K % T))
        {
            it1 = ++Mt * t;
        }
        else
        {
            it1 = (K % T) * Mt + t;
        }
        it1 *= k;
        std::size_t mt = Mt * k;
        std::size_t it2 = it1 + mt;

        ElementType accum = zero;
        for (std::size_t i = it1; i < it2; ++i)
        {
            accum = f(accum, V[i]);
        }

        reduction_partial_results[t].value = accum;
    };

    auto thread_proc_2_ = [=](unsigned t, std::size_t s)
    {
        if (((t % (s * k)) == 0) && (t + s < T))
        {
            reduction_partial_results[t].value =
                    f(reduction_partial_results[t].value, reduction_partial_results[t + s].value);
        }
    };

    std::vector<std::thread> threads;
    for (unsigned t = 1; t < T; ++t) {
        threads.emplace_back(thread_proc, t);
    }

    thread_proc(0);

    for (auto& thread:threads) {
        thread.join();
    }

    std::size_t s = 1;
    while (s < T)
    {
        for (unsigned t = 1; t < T; ++t)
        {
            threads[t-1] = std::thread(thread_proc_2_, t, s);
        }
        thread_proc_2_(0, s);
        s *= k;

        for (auto& thread:threads)
        {
            thread.join();
        }
    }

    return reduction_partial_results[0].value;
}


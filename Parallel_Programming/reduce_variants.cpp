#include "OMP_realizations.cpp"


//------------------------------------------------------

class barrier
{
    const unsigned m_T_max;
    unsigned m_T;
    bool barrier_id = false;
    std::condition_variable cv;
    std::mutex mtx;
public:
    barrier(unsigned T) :m_T_max(T), m_T(T) {}
    void arrive_and_wait()
    {
        std::unique_lock<std::mutex> lock(mtx);
        bool my_barrier_id = barrier_id;
        if (--m_T > 0)
        {
            while (my_barrier_id == barrier_id)
                cv.wait(lock);
        }
        else
        {
            cv.notify_all();
            m_T = m_T_max;
            barrier_id = !my_barrier_id;
        }
    }
};

namespace {


    //------------------------------------------------------


    template <class ElementType, class binary_fn>
    ElementType reduce_par(ElementType* V, unsigned count, binary_fn f, ElementType zero)
    {
        unsigned j = 1;
        constexpr unsigned k = 2;
        std::vector<std::thread> threads;
        unsigned T =numOfThreads;
        for (unsigned t = 1; t < T; t++)
            threads.emplace_back(std::thread{});
        while (count > j)
        {
            auto thread_fn = [k, count, j, T, zero, V, f](unsigned t)
            {
                for (unsigned i = t * j * k; i < count; i += T * j * k)
                {
                    ElementType other = zero;
                    if (i + j < count)
                        other = V[i + j];
                    V[i] = f(V[i], other);
                }
            };
            for (unsigned t = 1; t < T; t++)
                threads[t - 1] = std::thread(thread_fn, t);
            thread_fn(0);
            for (auto& thread : threads)
                thread.join();
            j *= k;
        }
        return V[0];
    }

    template <class binary_fn, class unary_fn, class ElementType>
    auto reduce_par_2(binary_fn f, unary_fn get, ElementType x0, ElementType xn, ElementType step, ElementType zero)
    {
        struct element_t
        {
            alignas(std::hardware_destructive_interference_size) ElementType value;
        };

        unsigned T = numOfThreads;
        static std::vector<element_t> reduction_buffer(std::thread::hardware_concurrency(), element_t{ 0.0 });
        std::vector<std::thread> threads;

        barrier bar{ T };

        auto thread_proc = [f, get, x0, xn, step, zero, T, &bar](unsigned t)
        {
            std::size_t count = (std::size_t)ElementType((xn - x0) / step);
            std::size_t nt = count / T, it0 = nt * t;
            ElementType my_result = zero;

            if (nt < (count % T))
                ++nt;
            else
                it0 += count % T;

            std::size_t it1 = it0 + nt;
            ElementType x = x0 + step * it0;

            for (std::size_t i = it0; i < it1; ++i, x += step)
                my_result = f(my_result, get(x));

            reduction_buffer[t].value = my_result;
            bar.arrive_and_wait();

            for (std::size_t reduction_distance = 1u, reduction_next = 2; reduction_distance < T; reduction_distance = reduction_next, reduction_next += reduction_next)
            {
                if (t + reduction_distance < T && (t & reduction_next - 1) == 0)
                    reduction_buffer[t].value = f(reduction_buffer[t].value, reduction_buffer[t + reduction_distance].value);

                bar.arrive_and_wait();
            }
        };

        for (unsigned t = 1; t < T; ++t)
            threads.emplace_back(thread_proc, t);

        thread_proc(0);

        for (auto& thread : threads)
            thread.join();

        return reduction_buffer[0].value;
    }


    //------------------------------------------------------
}
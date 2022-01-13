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

    ElementType reduce_par_custom_barrier(ElementType* V, size_t count, binary_fn f, ElementType zero)
    {
        unsigned j = 1;
        constexpr unsigned k = 2;
        vector<thread> threads;
        unsigned T = numOfThreads;
        barrier bar{ T };

        for (unsigned t = 0; t < T; t++)
            threads.emplace_back(thread{});

        while (count > j)
        {
            auto thread_fn = [k, count, j, T, zero, V, &bar, f](unsigned t)
            {
                for (unsigned i = t * j * k; i < count; i += T * j * k)
                {
                    ElementType other = zero;
                    if (i + j < count)
                        other = V[i + j];
                    V[i] = f(V[i], other);
                }
                bar.arrive_and_wait(); //самодельный барьер
            };

            for (unsigned t = 0; t < T; t++)
                threads[t] = thread(thread_fn, t);

            thread_fn(0);

            for (auto& thread : threads)
                thread.join();

            j *= k;
        }
        return V[0];
    }


    //------------------------------------------------------

    template <class ElementType, class binary_fn>

    ElementType reduce_par(ElementType* V, size_t count, binary_fn f, ElementType zero)
    {
        unsigned j = 1;
        constexpr unsigned k = 2;
        vector<thread> threads;
        unsigned T = numOfThreads;

        for (unsigned t = 0; t < T; t++)
            threads.emplace_back(thread{});

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

            for (unsigned t = 0; t < T; t++)
                threads[t] = thread(thread_fn, t);

            thread_fn(0);

            for (auto& thread : threads)
                thread.join();

            j *= k;
        }
        return V[0];
    }


    //------------------------------------------------------


    template <class binary_fn, class unary_fn, class ElementType>
        requires (
    std::is_invocable_v<binary_fn, ElementType, ElementType>&& std::is_convertible_v<std::invoke_result_t<binary_fn, ElementType, ElementType>, ElementType>&&
        std::is_invocable_v<unary_fn, ElementType>&& std::is_convertible_v<std::invoke_result_t<unary_fn, ElementType>, ElementType>
        )

        auto reduce_parallel(binary_fn f, unary_fn get, ElementType x0, ElementType xn, ElementType step, ElementType zero)
    {
        struct element_t
        {
            alignas(std::hardware_destructive_interference_size) ElementType value;
        };

        unsigned T = numOfThreads;
        static vector<element_t> reduction_buffer(T, element_t{ 0.0 });
        vector<thread> threads(T);

        auto thread_proc = [f, get, x0, xn, step, zero, T](unsigned t)
        {
            unsigned count = (unsigned)ElementType((xn - x0) / step);
            unsigned nt = count / T;
            unsigned it0 = nt * t;
            ElementType my_result = zero;

            if (nt < (count % T))
                ++nt;
            else
                it0 += count % T;

            unsigned it1 = it0 + nt;
            ElementType x = x0 + step * it0;

            for (unsigned i = it0; i < it1; ++i, x += step)
                my_result = f(my_result, get(x));

            reduction_buffer[t].value = my_result;
        };

        for (unsigned t = 0; t < T; t++)
            threads[t] = thread(thread_proc, t);

        thread_proc(0);

        for (auto& thread : threads) {
            std::cout << threads.size() << std::endl;
            thread.join();
        }

        return reduce_type;
    }
}
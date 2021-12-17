#include "OMP_realizations.cpp"

#define numOfTypes 5

//------------------------------------------------------------------


namespace {
    using std::vector;
    using std::thread;

    //---------------Вычисление интеграла-------------------

    //------------------------------------------------------

    template <class binary_fn, class unary_fn, class ElementType>
        requires (
    std::is_invocable_v<binary_fn, ElementType, ElementType>&& std::is_convertible_v<std::invoke_result_t<binary_fn, ElementType, ElementType>, ElementType>&&
        std::is_invocable_v<unary_fn, ElementType>&& std::is_convertible_v<std::invoke_result_t<unary_fn, ElementType>, ElementType>
        )


        auto reduce_par_2(binary_fn f, unary_fn get, ElementType x0, ElementType xn, ElementType step, ElementType zero)
    {
        struct element_t
        {
            alignas(std::hardware_destructive_interference_size) ElementType value;
        };

        unsigned T = thread::hardware_concurrency();
        static vector<element_t> reduction_buffer(T, element_t{ 0.0 });
        vector<thread> threads;

        auto thread_proc = [f, get, x0, xn, step, zero, T](unsigned t)
        {
            unsigned count = ElementType((xn - x0) / step);
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

        for (unsigned t = 1; t < T; ++t)
            threads.emplace_back(thread_proc, t);

        thread_proc(0);

        for (auto& thread : threads)
            thread.join();

        return reduce_par(reduction_buffer.data(), reduction_buffer.size(),
            [f, zero](const element_t& x, const element_t& y) {return element_t{ f(x.value, y.value) }; }, element_t{ zero }).value;
    }

    template <class ElementType, class binary_fn>


    ElementType reduce_par(ElementType* V, unsigned count, binary_fn f, ElementType zero)
    {
        unsigned j = 1;
        constexpr unsigned k = 2;
        vector<thread> threads;
        unsigned T = thread::hardware_concurrency();

        for (unsigned t = 1; t < T; t++)
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

            for (unsigned t = 1; t < T; t++)
                threads[t - 1] = thread(thread_fn, t);

            thread_fn(0);

            for (auto& thread : threads)
                thread.join();

            j *= k;
        }
        return V[0];
    }

    //------------------------------------------------------

    //------------------>integrate_cpp_reduce
    double integrate_cpp_reduce(double a, double b, f_t f)
    {
        double dx = (b - a) / ndx;
        return reduce_par_2([f, dx](double x, double y) {return x + y; }, f, a, b, dx, 0.0) * dx;
    }


    //------------------>integrate_cpp_base
    double integrate_cpp_base(double a, double b, f_t f) {
        double dx = (b - a) / ndx;
        double res = 0;

        //Получаем количество потоков
        unsigned T = thread::hardware_concurrency();
        vector<double> results(T);

        //Лямбда, содержащая функции, которые затем будут выполнены каждым из потоков
        auto thread_proc = [=, &results](unsigned t) {

            results[t] = 0;
            for (unsigned i = t; i < ndx; i += T)
                results[t] += f(dx * i + a);
        };

        //Создаем массив исполняющих потоков
        vector<thread> threads;

        for (unsigned t = 1; t < T; ++t)
            threads.emplace_back(thread_proc, t);

        thread_proc(0);

        //Инициализируем выполнение каждого из потоков
        for (auto& thread : threads)
            thread.join();

        //Записываем результаты вычислений каждого из потоков в одну общую переменную
        for (unsigned i = 0; i < T; ++i)
            res += results[i];

        return res * dx;
    }


    //------------------>integrate_cpp_cs
    double integrate_cpp_cs(double a, double b, f_t f) {
        double dx = (b - a) / ndx;
        double res = 0;

        //Получаем количество потоков
        unsigned T = thread::hardware_concurrency();

        //Инициализация критической секции
        CRITICAL_SECTION cs;
        InitializeCriticalSection(&cs);

        //Лямбда, содержащая функции, которые затем будут выполнены каждым из потоков
        auto thread_process = [=, &res, &cs](unsigned t) {
            double l_res = 0;

            for (unsigned i = t; i < ndx; i += T)
                l_res += f((float)(dx * i + a));
            {
                //Вход в критическую секцию кода ...
                EnterCriticalSection(&cs);
                res += l_res;
                LeaveCriticalSection(&cs);
                //... и выход из неё
            }
        };

        //Создаем массив исполняющих потоков
        vector<thread> threads;

        for (unsigned t = 1; t < T; ++t)
            threads.emplace_back(thread_process, t);

        thread_process(0);

        //Инициализируем выполнение каждого из потоков
        for (auto& thread : threads)
            thread.join();

        //Освобождение ресурсов системы от кода критической секции
        DeleteCriticalSection(&cs);

        return res * dx;
    }


    //------------------>integrate_cpp_atomic
    double integrate_cpp_atomic(double a, double b, f_t f) {
        using std::atomic;

        double dx = (b - a) / ndx;
        atomic<double> res = 0;

        //Получаем количество потоков
        unsigned T = thread::hardware_concurrency();

        //Лямбда, содержащая функции, которые затем будут выполнены каждым из потоков
        auto thread_process = [=, &res](unsigned t) {
            double l_res = 0;

            for (unsigned i = t; i < ndx; i += T)
                l_res += f((float)(dx * i + a));

            //Прямое сложение не сработало, заменил на функции из класса "atomic"
            res.store(res.load() + l_res);
            //Так как переменная атомарна, то данный метод сложения автоматически
            //организует очередь доступа к переменной, что предотвратит гонку потоков
        };

        //Создаем массив исполняющих потоков
        vector<thread> threads;

        for (unsigned t = 1; t < T; ++t)
            threads.emplace_back(thread_process, t);

        thread_process(0);

        //Инициализируем выполнение каждого из потоков
        for (auto& thread : threads)
            thread.join();

        return res * dx;
    }


    //------------------>integrate_cpp_mtx
    double integrate_cpp_mtx(double a, double b, f_t f) {
        using std::mutex;
        using std::lock_guard;

        double dx = (b - a) / ndx;
        double res = 0;

        //Получаем количество потоков
        unsigned T = thread::hardware_concurrency();

        mutex mtx;

        //Лямбда, содержащая функции, которые затем будут выполнены каждым из потоков
        auto thread_process = [=, &res, &mtx](unsigned t) {
            double l_res = 0;

            for (unsigned i = t; i < ndx; i += T)
                l_res += f((float)(dx * i + a));
            {
                //По идее здесь находился scoped_lock, но в поле пространства std
                //у меня он отсутствует, оттого заменил на устаревший код
                lock_guard<mutex> lck(mtx);
                res += l_res;
                //Для ограничения доступа к переменной с помощью mutex мы блокируем
                //доступ к переменной каждому, кроме первого пришедшего, из потоков
            }
        };

        //Создаем массив исполняющих потоков
        vector<thread> threads;

        for (unsigned t = 1; t < T; ++t)
            threads.emplace_back(thread_process, t);

        thread_process(0);

        //Инициализируем выполнение каждого из потоков
        for (auto& thread : threads)
            thread.join();

        return res * dx;
    }


    //--------------------Основной поток-----------------------

    void cpp_start() {
        integrate_t integrate_type[numOfTypes];

        int typeNum = 0;

        integrate_type[typeNum++] = integrate_cpp_base;
        integrate_type[typeNum++] = integrate_cpp_cs;
        integrate_type[typeNum++] = integrate_cpp_atomic;
        integrate_type[typeNum++] = integrate_cpp_reduce;
        integrate_type[typeNum++] = integrate_cpp_mtx;

        std::cout << "CPP results" << std::endl;
        run_experiments(integrate_type, numOfTypes);
    }
}
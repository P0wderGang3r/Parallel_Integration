#include "reduce_variants.cpp"


//------------------------------------------------------------------


namespace {

    //---------------Вычисление интеграла-------------------


    //------------------>integrate_cpp_reduce
    double integrate_cpp_reduce(double a, double b, f_t f)
    {
        double dx = (b - a) / ndx;

        return reduce_parallel([f, dx](double x, double y) {return x + y; }, f, a, b, dx, 0.0) * dx;
    }


    //------------------>integrate_cpp_fs
    double integrate_cpp_fs(double a, double b, f_t f) {
        double dx = (b - a) / ndx;
        double res = 0;

        //Получаем количество потоков
        unsigned T = numOfThreads;
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


    //------------------>integrate_cpp_base
    double integrate_cpp_base(double a, double b, f_t f) {
        double dx = (b - a) / ndx;
        double res = 0;

        //Получаем количество потоков
        unsigned T = numOfThreads;
        double* results = (double*)_aligned_malloc(CACHE_LINE, max_threads * sizeof(double));
        for (unsigned i = 0; i < numOfThreads; i++)
            results[i] = 0;

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

        _aligned_free(results);

        return res * dx;
    }


    //------------------>integrate_cpp_cs
    double integrate_cpp_cs(double a, double b, f_t f) {
        double dx = (b - a) / ndx;
        double res = 0;

        //Получаем количество потоков
        unsigned T = numOfThreads;

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
        unsigned T = numOfThreads;

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
        unsigned T = numOfThreads;

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

    void cpp_start() {;
        std::vector<vectorType> CPPTypes;

        //CPPTypes.emplace_back(integrate_cpp_fs, "integrate_cpp_fs");
        //CPPTypes.emplace_back(integrate_cpp_base, "integrate_cpp_base");
        //CPPTypes.emplace_back(integrate_cpp_cs, "integrate_cpp_cs");
        //CPPTypes.emplace_back(integrate_cpp_atomic, "integrate_cpp_atomic");
        CPPTypes.emplace_back(integrate_cpp_reduce, "integrate_cpp_reduce");
        //CPPTypes.emplace_back(integrate_cpp_mtx, "integrate_cpp_mtx");

        std::cout << "CPP results" << std::endl;
        run_experiments(&CPPTypes);
    }
}
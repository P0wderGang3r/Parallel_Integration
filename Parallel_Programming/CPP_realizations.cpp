#include "OMP_realizations.cpp"

#define n 10000000
#define numOfTypes 5


namespace {
    
    //---------------Вычисление интеграла-------------------

    //------------------>integrate_cpp_base
    double integrate_cpp_base(double a, double b, f_t f) {
        double dx = (b - a) / n;
        double res = 0;

        //Получаем количество потоков
        unsigned T = std::thread::hardware_concurrency();
        std::vector<double> results(T);

        //Лямбда, содержащая функции, которые затем будут выполнены каждым из потоков
        auto thread_proc = [=, &results](unsigned t) {

            results[t] = 0;
            for (unsigned i = t; i < n; i += T)
                results[t] += f(dx * i + a);
        };

        //Создаем массив исполняющих потоков
        std::vector<std::thread> threads;

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
        double dx = (b - a) / n;
        double res = 0;

        //Получаем количество потоков
        unsigned T = std::thread::hardware_concurrency();

        std::mutex mtx;

        //Лямбда, содержащая функции, которые затем будут выполнены каждым из потоков
        auto thread_process = [=, &res, &mtx](unsigned t) {
            double l_res = 0;

            for (unsigned i = t; i < n; i += T)
                l_res += f((float)(dx * i + a));
            {
                //По идее здесь находился scoped_lock, но в поле пространства std
                //у меня он отсутствует, оттого заменил на устаревший код
                std::lock_guard<std::mutex> lck(mtx);
                res += l_res;
                //Для ограничения доступа к переменной с помощью mutex мы блокируем
                //доступ к переменной каждому, кроме первого пришедшего, из потоков
            }
        };

        //Создаем массив исполняющих потоков
        std::vector<std::thread> threads;

        for (unsigned t = 1; t < T; ++t)
            threads.emplace_back(thread_process, t);

        thread_process(0);

        //Инициализируем выполнение каждого из потоков
        for (auto& thread : threads)
            thread.join();

        return res * dx;
    }

    //------------------>integrate_cpp_atomic
    double integrate_cpp_atomic(double a, double b, f_t f) {
        double dx = (b - a) / n;
        std::atomic<double> res = 0;

        //Получаем количество потоков
        unsigned T = std::thread::hardware_concurrency();

        //Лямбда, содержащая функции, которые затем будут выполнены каждым из потоков
        auto thread_process = [=, &res](unsigned t) {
            double l_res = 0;

            for (unsigned i = t; i < n; i += T)
                l_res += f((float)(dx * i + a));

            //Прямое сложение не сработало, заменил на функции из класса "atomic"
            res.store(res.load() + l_res);
            //Так как переменная атомарна, то данный метод сложения автоматически
            //организует очередь доступа к переменной, что предотвратит гонку потоков
        };

        //Создаем массив исполняющих потоков
        std::vector<std::thread> threads;

        for (unsigned t = 1; t < T; ++t)
            threads.emplace_back(thread_process, t);

        thread_process(0);

        //Инициализируем выполнение каждого из потоков
        for (auto& thread : threads)
            thread.join();

        return res * dx;
    }

    //------------------>integrate_cpp_reduce
    double integrate_cpp_reduce(double a, double b, f_t f) {
        double dx = (b - a) / n;
        double res = 0;
        return res * dx;
    }

    //------------------>integrate_cpp_mtx
    double integrate_cpp_mtx(double a, double b, f_t f) {
        double dx = (b - a) / n;
        double res = 0;
        return res * dx;
    }

    //----------------Создание экспериментов------------------

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
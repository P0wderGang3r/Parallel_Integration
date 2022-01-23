#include "SEQ_realization.cpp"
#include <stdio.h>
#include <stdlib.h>


namespace {//---------------Вычисление интеграла-------------------

    //------------------>integrate_omp_fs
    //Параллельный алгоритм численного интегрирования с массивом
    double integrate_omp_fs(double a, double b, f_t f) {
        double dx = (b - a) / ndx;
        double res = 0;
        omp_set_num_threads(numOfThreads);
        double* results;

#pragma omp parallel shared(results, numOfThreads)
        {
            unsigned t = (unsigned)omp_get_thread_num();

#pragma omp single
            {
                results = new double[numOfThreads];
                for (unsigned i = 0; i < numOfThreads; i++)
                    results[i] = 0;

                if (!results)
                    abort();
            }

            for (unsigned i = t; i < ndx; i += numOfThreads) {
                results[t] += f(dx * i + a);
            }

#pragma omp barrier //барьерная реализация
            //Все потоки, подходя к данному моменту вычислений,
            //приостанавливаются в ожидании, пока все другие подойдут сюда же
            //Неявно используется в конце каждого параллельного цикла
        }
        for (unsigned i = 0; i < numOfThreads; i++)
            res += results[i];

        return res * dx;
    }

    //------------------>integrate_omp_base
    //Разделение переменных на длину разово загружаемого процессором кэша
    double integrate_omp_base(double a, double b, f_t f) {
        double dx = (b - a) / ndx;
        double res = 0;
        omp_set_num_threads(numOfThreads);
        double* results;

#pragma omp parallel shared(results, numOfThreads)
        {
            unsigned t = (unsigned)omp_get_thread_num();

#pragma omp single
            {
                results = (double*) _aligned_malloc(CACHE_LINE, max_threads * sizeof(double));
                for (unsigned i = 0; i < numOfThreads; i++)
                    results[i] = 0;

                if (!results)
                    abort();
            }

            for (unsigned i = t; i < ndx; i += numOfThreads) {
                results[t] += f(dx * i + a);
            }

#pragma omp barrier
        }
        for (unsigned i = 0; i < numOfThreads; i++)
            res += results[i];

        _aligned_free(results);

        return res * dx;
    }


    //------------------>integrate_omp_cs
    //Критическая секция, последовательный доступ к ней
    double integrate_omp_cs(double a, double b, f_t f)
    {
        double res = 0;
        double dx = (b - a) / ndx;
        omp_set_num_threads(numOfThreads);

#pragma omp parallel shared(res, numOfThreads) 
        {
            unsigned t = (unsigned)omp_get_thread_num();
            double val = 0;

            for (unsigned i = t; i < ndx; i += numOfThreads)
            {
                val += f(dx * i + a);
            }
#pragma omp critical
            //Прерывает другие потоки в ожидании выполнения
            //следующей операции или некоторого их набора
            {
                res += val;
            }
        }
        return res * dx;
    }


    //------------------>integrate_omp_atomic
    //Критическая секция в виде атомарной функции, применяемой к переменной, и последовательный доступ к ней
    double integrate_omp_atomic(double a, double b, f_t f)
    {
        double res = 0;
        double dx = (b - a) / ndx;
        omp_set_num_threads(numOfThreads);

#pragma omp parallel shared(res, numOfThreads) 
        {
            unsigned t = (unsigned)omp_get_thread_num();
            double val = 0;

            for (unsigned i = t; i < ndx; i += numOfThreads)
            {
                val += f((double)(dx * i + a));
            }
#pragma omp atomic
            //Прерывает другие потоки в ожидании выполнения
            //следующей операции типа суммы или разности двух переменных
            //Быстрее, чем CS, так как инициализирует более оптимальные директивы процессора
            //Но использование их более узконаправлено, исходя из описания выше
            res += val;
        }

        return res * dx;
    }


    //------------------>integrate_omp_for
    // Явное указание распараллеливания цикла for
    double integrate_omp_for(double a, double b, f_t f)
    {
        double res = 0;
        double dx = (b - a) / ndx;
        omp_set_num_threads(numOfThreads);
        int i;

#pragma omp parallel for shared (res)
            //Следуюший цикл будет разбит на восемь потоков вычислений
            //с переменной с общим доступом для каждого из потоков res
            //которую мы делаем атомарной для нивелирования условия гонки
            for (i = 0; i < ndx; ++i)
            {
                double val = f((double)(dx * i + a));
#pragma omp atomic
                res += val;
            }

        return res * dx;
    }


    //------------------>integrate_omp_reduce
    //алгоритм разбиения на минимальные суммы
    double integrate_omp_reduce(double a, double b, f_t f)
    {
        double res = 0;
        double dx = (b - a) / ndx;
        omp_set_num_threads(numOfThreads);
        int i;

#pragma omp parallel for reduction(+: res)
        //1. Производится подсчёт потоков
        //2. Каждый поток получает свой уникальный локальный res
        //3. Каждый из потоков производит сложение результата в свой локальный res
        //4. По завершении работы каждого из потоков производится последовательное сложение результата выполнения в глобальный res
        for (i = 0; i < ndx; ++i)
            res += f((double)(dx * i + a));
        return res * dx;
    }


    //------------------>integrate_omp_mtx
    //Блокировка доступа к блоку кода с помощью Mute Expression
    double integrate_omp_mtx(double a, double b, f_t f) {
        {
            double res = 0;
            double dx = (b - a) / ndx;
            omp_set_num_threads(numOfThreads);

            omp_lock_t mtxLock;
            omp_init_lock(&mtxLock);

#pragma omp parallel shared(res, numOfThreads) 
            {
                unsigned t = (unsigned)omp_get_thread_num();
                double val = 0;

                for (unsigned i = t; i < ndx; i += numOfThreads)
                {
                    val += f((double)(dx * i + a));
                }
                //Следующая после блокировки строка кода будет заблокирована на время выполнения благодаря MuteExpression
                omp_set_lock(&mtxLock);
                res += val;
                omp_unset_lock(&mtxLock);
                //И разблокирована после окончания её использования одним из потоков
            }

            return res * dx;
        }
    }

    //Ещё был omp_dynamic, но... Медленный и не репрезентативный
    //Ещё барьер

    //--------------------Основной поток-----------------------

    void omp_start() {
        std::vector<vectorType> OMPTypes;

        OMPTypes.emplace_back(integrate_omp_fs, "integrate_omp_fs");
        OMPTypes.emplace_back(integrate_omp_base, "integrate_omp_base");
        OMPTypes.emplace_back(integrate_omp_cs, "integrate_omp_cs");
        OMPTypes.emplace_back(integrate_omp_for, "integrate_omp_for");
        OMPTypes.emplace_back(integrate_omp_atomic, "integrate_omp_atomic");
        OMPTypes.emplace_back(integrate_omp_reduce, "integrate_omp_reduce");
        OMPTypes.emplace_back(integrate_omp_mtx, "integrate_omp_mtx");

        std::cout << "OMP results" << std::endl;
        run_experiments(&OMPTypes);
    }
}
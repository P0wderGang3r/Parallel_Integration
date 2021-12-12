#include "SEQ_realization.cpp"

#define n 10000000
#define numOfTypes 6

namespace {//---------------¬ычисление интеграла-------------------

    //------------------>integrate_omp_base
    //ѕараллельный алгоритм численного интегрировани€ с массивом
    double integrate_omp_base(double a, double b, f_t f) {
        double dx = (b - a) / n;
        double res = 0;
        unsigned T;
        double* results;

#pragma omp parallel shared(results, T)
        {
            unsigned t = (unsigned)omp_get_thread_num();
            T = (unsigned)omp_get_num_threads();

#pragma omp single
            {
                results = new double[T];
                for (unsigned i = 0; i < T; i++)
                    results[i] = 0;

                if (!results)
                    abort();
            }

            for (unsigned i = t; i < n; i += T) {
                results[t] += f(dx * i + a);
            }

#pragma omp barrier //барьерна€ реализаци€
            //¬се потоки, подход€ к данному моменту вычислений,
            //приостанавливаютс€ в ожидании, пока все другие подойдут сюда же
            //Ќе€вно используетс€ в конце каждого параллельного цикла
        }
        for (unsigned i = 0; i < T; i++)
            res += results[i];

        return res * dx;
    }


    //------------------>integrate_omp_cs
    // ритическа€ секуи€, последовательный доступ к ней
    double integrate_omp_cs(double a, double b, f_t f)
    {
        double res = 0;
        double dx = (b - a) / n;
        unsigned T;

#pragma omp parallel shared(res, T) 
        {
            unsigned t = (unsigned)omp_get_thread_num();
            T = (unsigned)omp_get_num_threads();

            for (unsigned i = t; i < n; i += T)
            {
#pragma omp critical
                //ѕрерывает другие потоки в ожидании выполнени€
                //следующей операции или некоторого их набора
                {
                    res += f(dx * i + a);
                }
            }
        }
        return res * dx;
    }


    //------------------>integrate_omp_atomic
    // ритическа€ секци€ в виде атомарной функции, примен€емой к переменной, и последовательный доступ к ней
    double integrate_omp_atomic(double a, double b, f_t f)
    {
        double res = 0.0;
        double dx = (b - a) / n;
        unsigned T;

#pragma omp parallel shared(res, T) 
        {
            unsigned t = (unsigned)omp_get_thread_num();
            T = (unsigned)omp_get_num_threads();
            double val;

            for (unsigned i = t; i < n; i += T)
            {
                val = f((double)(dx * i + a));
#pragma omp atomic
                //ѕрерывает другие потоки в ожидании выполнени€
                //следующей операции типа суммы или разности двух переменных
                //Ѕыстрее, чем CS, так как инициализирует более оптимальные директивы процессора
                //Ќо использование их более узконаправлено, исход€ из описани€ выше
                res += val;
            }
        }

        return res * dx;
    }


    //------------------>integrate_omp_for
    // явное указание распараллеливани€ цикла for
    double integrate_omp_for(double a, double b, f_t f)
    {
        double res = 0.0;
        double dx = (b - a) / n;
        int i;

#pragma omp parallel for shared (res)
        //—ледуюший цикл будет разбит на восемь потоков вычислений
        //с переменной с общим доступом дл€ каждого из потоков res
        //которую мы делаем атомарной дл€ нивелировани€ услови€ гонки
        for (i = 0; i < n; ++i)
        {
            double val = f((double)(dx * i + a));
#pragma omp atomic
            res += val;
        }

        return res * dx;
    }


    //------------------>integrate_omp_reduce
    //алгоритм разбиени€ на минимальные суммы
    double integrate_omp_reduce(double a, double b, f_t f)
    {
        double res = 0.0;
        double dx = (b - a) / n;
        int i;

#pragma omp parallel for reduction(+: res)
        //1. ѕроизводитс€ подсчЄт потоков
        //2.  аждый поток получает свой уникальный локальный res
        //3.  аждый из потоков производит сложение результата в свой локальный res
        //4. ѕо завершении работы каждого из потоков производитс€ последовательное сложение результата выполнени€ в глобальный res
        for (i = 0; i < n; ++i)
            res += f((double)(dx * i + a));
        return res * dx;
    }

    //mutex is nowhere there
    //------------------>integrate_omp_mtx
    double integrate_omp_mtx(double a, double b, f_t f) {
        double dx = (b - a) / n;
        double res = 0;
        return res * dx;
    }

    //≈щЄ был omp_dynamic, но... ћедленный и не репрезентативный

    //--------------------ќсновной поток-----------------------

    void omp_start() {
        integrate_t integrate_type[numOfTypes];

        int typeNum = 0;
        integrate_type[typeNum++] = integrate_omp_base;
        integrate_type[typeNum++] = integrate_omp_cs;
        integrate_type[typeNum++] = integrate_omp_atomic;
        integrate_type[typeNum++] = integrate_omp_for;
        integrate_type[typeNum++] = integrate_omp_reduce;
        integrate_type[typeNum++] = integrate_omp_mtx;

        std::cout << "OMP results" << std::endl;
        run_experiments(integrate_type, numOfTypes);
    }
}
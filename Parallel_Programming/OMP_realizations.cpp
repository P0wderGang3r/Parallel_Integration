#include "SEQ_realization.cpp"
#include <stdio.h>
#include <stdlib.h>


namespace {//---------------���������� ���������-------------------

    //------------------>integrate_omp_fs
    //������������ �������� ���������� �������������� � ��������
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

#pragma omp barrier //��������� ����������
            //��� ������, ������� � ������� ������� ����������,
            //������������������ � ��������, ���� ��� ������ �������� ���� ��
            //������ ������������ � ����� ������� ������������� �����
        }
        for (unsigned i = 0; i < numOfThreads; i++)
            res += results[i];

        return res * dx;
    }

    //------------------>integrate_omp_base
    //���������� ���������� �� ����� ������ ������������ ����������� ����
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
    //����������� ������, ���������������� ������ � ���
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
            //��������� ������ ������ � �������� ����������
            //��������� �������� ��� ���������� �� ������
            {
                res += val;
            }
        }
        return res * dx;
    }


    //------------------>integrate_omp_atomic
    //����������� ������ � ���� ��������� �������, ����������� � ����������, � ���������������� ������ � ���
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
            //��������� ������ ������ � �������� ����������
            //��������� �������� ���� ����� ��� �������� ���� ����������
            //�������, ��� CS, ��� ��� �������������� ����� ����������� ��������� ����������
            //�� ������������� �� ����� ��������������, ������ �� �������� ����
            res += val;
        }

        return res * dx;
    }


    //------------------>integrate_omp_for
    // ����� �������� ����������������� ����� for
    double integrate_omp_for(double a, double b, f_t f)
    {
        double res = 0;
        double dx = (b - a) / ndx;
        omp_set_num_threads(numOfThreads);
        int i;

#pragma omp parallel for shared (res)
            //��������� ���� ����� ������ �� ������ ������� ����������
            //� ���������� � ����� �������� ��� ������� �� ������� res
            //������� �� ������ ��������� ��� ������������� ������� �����
            for (i = 0; i < ndx; ++i)
            {
                double val = f((double)(dx * i + a));
#pragma omp atomic
                res += val;
            }

        return res * dx;
    }


    //------------------>integrate_omp_reduce
    //�������� ��������� �� ����������� �����
    double integrate_omp_reduce(double a, double b, f_t f)
    {
        double res = 0;
        double dx = (b - a) / ndx;
        omp_set_num_threads(numOfThreads);
        int i;

#pragma omp parallel for reduction(+: res)
        //1. ������������ ������� �������
        //2. ������ ����� �������� ���� ���������� ��������� res
        //3. ������ �� ������� ���������� �������� ���������� � ���� ��������� res
        //4. �� ���������� ������ ������� �� ������� ������������ ���������������� �������� ���������� ���������� � ���������� res
        for (i = 0; i < ndx; ++i)
            res += f((double)(dx * i + a));
        return res * dx;
    }


    //------------------>integrate_omp_mtx
    //���������� ������� � ����� ���� � ������� Mute Expression
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
                //��������� ����� ���������� ������ ���� ����� ������������� �� ����� ���������� ��������� MuteExpression
                omp_set_lock(&mtxLock);
                res += val;
                omp_unset_lock(&mtxLock);
                //� �������������� ����� ��������� � ������������� ����� �� �������
            }

            return res * dx;
        }
    }

    //��� ��� omp_dynamic, ��... ��������� � �� ����������������
    //��� ������

    //--------------------�������� �����-----------------------

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
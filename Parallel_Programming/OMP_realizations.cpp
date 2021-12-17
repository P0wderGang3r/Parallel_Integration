#include "SEQ_realization.cpp"


namespace {//---------------���������� ���������-------------------

    //------------------>integrate_omp_base
    //������������ �������� ���������� �������������� � ��������
    double integrate_omp_base(double a, double b, f_t f) {
        double dx = (b - a) / ndx;
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

            for (unsigned i = t; i < ndx; i += T) {
                results[t] += f(dx * i + a);
            }

#pragma omp barrier //��������� ����������
            //��� ������, ������� � ������� ������� ����������,
            //������������������ � ��������, ���� ��� ������ �������� ���� ��
            //������ ������������ � ����� ������� ������������� �����
        }
        for (unsigned i = 0; i < T; i++)
            res += results[i];

        return res * dx;
    }


    //------------------>integrate_omp_cs
    //����������� ������, ���������������� ������ � ���
    double integrate_omp_cs(double a, double b, f_t f)
    {
        double res = 0;
        double dx = (b - a) / ndx;
        unsigned T;

#pragma omp parallel shared(res, T) 
        {
            unsigned t = (unsigned)omp_get_thread_num();
            T = (unsigned)omp_get_num_threads();

            for (unsigned i = t; i < ndx; i += T)
            {
#pragma omp critical
                //��������� ������ ������ � �������� ����������
                //��������� �������� ��� ���������� �� ������
                {
                    res += f(dx * i + a);
                }
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
        unsigned T;

#pragma omp parallel shared(res, T) 
        {
            unsigned t = (unsigned)omp_get_thread_num();
            T = (unsigned)omp_get_num_threads();
            double val;

            for (unsigned i = t; i < ndx; i += T)
            {
                val = f((double)(dx * i + a));
#pragma omp atomic
                //��������� ������ ������ � �������� ����������
                //��������� �������� ���� ����� ��� �������� ���� ����������
                //�������, ��� CS, ��� ��� �������������� ����� ����������� ��������� ����������
                //�� ������������� �� ����� ��������������, ������ �� �������� ����
                res += val;
            }
        }

        return res * dx;
    }


    //------------------>integrate_omp_for
    // ����� �������� ����������������� ����� for
    double integrate_omp_for(double a, double b, f_t f)
    {
        double res = 0;
        double dx = (b - a) / ndx;
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
    //���������� ������� � ����� ���� � ������� MuteEx
    double integrate_omp_mtx(double a, double b, f_t f) {
        {
            double res = 0;
            double dx = (b - a) / ndx;
            unsigned T;

            omp_lock_t mtxLock;
            omp_init_lock(&mtxLock);

#pragma omp parallel shared(res, T) 
            {
                unsigned t = (unsigned)omp_get_thread_num();
                T = (unsigned)omp_get_num_threads();
                double val;

                for (unsigned i = t; i < ndx; i += T)
                {
                    val = f((double)(dx * i + a));
                    //��������� ����� ���������� ������ ���� ����� ������������� �� ����� ���������� ��������� MuteExpression
                    omp_set_lock(&mtxLock);
                    res += val;
                    omp_unset_lock(&mtxLock);
                    //� �������������� ����� ��������� � ������������� ����� �� �������
                }
            }

            return res * dx;
        }
    }

    //��� ��� omp_dynamic, ��... ��������� � �� ����������������

    //--------------------�������� �����-----------------------

    void omp_start() {
        integrate_t integrate_type[numOfOMPTypes];

        int typeNum = 0;
        integrate_type[typeNum++] = integrate_omp_base;
        integrate_type[typeNum++] = integrate_omp_cs;
        integrate_type[typeNum++] = integrate_omp_atomic;
        integrate_type[typeNum++] = integrate_omp_for;
        integrate_type[typeNum++] = integrate_omp_reduce;
        integrate_type[typeNum++] = integrate_omp_mtx;

        std::cout << "OMP results" << std::endl;
        run_experiments(integrate_type, numOfOMPTypes);
    }
}
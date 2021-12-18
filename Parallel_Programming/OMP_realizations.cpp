#include "SEQ_realization.cpp"


namespace {//---------------���������� ���������-------------------

    //------------------>integrate_omp_base
    //������������ �������� ���������� �������������� � ��������
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
        integrate_t integrate_type[numOfOMPTypes];
        std::string typeNames[numOfOMPTypes];

        int typeNum = 0;
        integrate_type[typeNum++] = integrate_omp_base;
        integrate_type[typeNum++] = integrate_omp_cs;
        integrate_type[typeNum++] = integrate_omp_atomic;
        integrate_type[typeNum++] = integrate_omp_for;
        integrate_type[typeNum++] = integrate_omp_reduce;
        integrate_type[typeNum++] = integrate_omp_mtx;

        typeNum = 0;
        typeNames[typeNum++] = "integrate_omp_base";
        typeNames[typeNum++] = "integrate_omp_cs";
        typeNames[typeNum++] = "integrate_omp_atomic";
        typeNames[typeNum++] = "integrate_omp_for";
        typeNames[typeNum++] = "integrate_omp_reduce";
        typeNames[typeNum++] = "integrate_omp_mtx";

        std::cout << "OMP results" << std::endl;
        run_experiments(integrate_type, typeNames, numOfOMPTypes);
    }
}
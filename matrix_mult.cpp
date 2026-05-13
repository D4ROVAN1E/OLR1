#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <iomanip>
#include <algorithm>

using namespace std;
using namespace std::chrono;

// Инициализация матрицы случайными числами
void init_matrix(vector<double>& matrix, int N) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(1.0, 10.0);
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = dis(gen);
    }
}

// Зануление результирующей матрицы
void clear_matrix(vector<double>& matrix, int N) {
    fill(matrix.begin(), matrix.end(), 0.0);
}

// Базовое умножение матриц (i-j-k)
void DGEMM_base(const vector<double>& A, const vector<double>& B, vector<double>& C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

// Оптимизация 1: Построчный перебор (изменение порядка циклов на i-k-j)
// Обеспечивает последовательный доступ к памяти для матриц A и B (кэш-оптимизация)
void DGEMM_opt_1(const vector<double>& A, const vector<double>& B, vector<double>& C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            double r = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += r * B[k * N + j];
            }
        }
    }
}

// Оптимизация 2: Блочное умножение матриц
void DGEMM_opt_2(const vector<double>& A, const vector<double>& B, vector<double>& C, int N, int block_size) {
    for (int jj = 0; jj < N; jj += block_size) {
        for (int kk = 0; kk < N; kk += block_size) {
            for (int i = 0; i < N; ++i) {
                for (int k = kk; k < min(kk + block_size, N); ++k) {
                    double r = A[i * N + k];
                    for (int j = jj; j < min(jj + block_size, N); ++j) {
                        C[i * N + j] += r * B[k * N + j];
                    }
                }
            }
        }
    }
}

// Оптимизация 3: Векторизация (используются директивы OpenMP SIMD + i-k-j перебор)
void DGEMM_opt_3(const vector<double>& A, const vector<double>& B, vector<double>& C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            double r = A[i * N + k];
            // Директива заставляет компилятор использовать SIMD инструкции (AVX/SSE)
            #pragma omp simd
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += r * B[k * N + j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    // Проверка аргументов
    if (argc < 3) {
        cerr << "Использование: " << argv[0] << " <размер_матрицы_N> <метод> [размер_блока]" << endl;
        cerr << "Методы: 0 (Base), 1 (Opt1 - Row), 2 (Opt2 - Block), 3 (Opt3 - Vector)" << endl;
        return 1;
    }

    int N = stoi(argv[1]);
    int method = stoi(argv[2]);
    int block_size = (argc >= 4) ? stoi(argv[3]) : 64; // По умолчанию размер блока 64

    // Выделение памяти (используется 1D массив для гарантии непрерывности памяти)
    vector<double> A(N * N);
    vector<double> B(N * N);
    vector<double> C(N * N, 0.0);

    init_matrix(A, N);
    init_matrix(B, N);

    auto start = high_resolution_clock::now();
    string method_name = "";

    switch (method) {
        case 0:
            method_name = "Base";
            DGEMM_base(A, B, C, N);
            break;
        case 1:
            method_name = "Opt1_RowWise";
            DGEMM_opt_1(A, B, C, N);
            break;
        case 2:
            method_name = "Opt2_Block";
            DGEMM_opt_2(A, B, C, N, block_size);
            break;
        case 3:
            method_name = "Opt3_Vectorized";
            DGEMM_opt_3(A, B, C, N);
            break;
        default:
            cerr << "Неверный метод!" << endl;
            return 1;
    }

    auto end = high_resolution_clock::now();
    duration<double> diff = end - start;

    // Вывод в формате CSV для парсинга скриптом Python
    // Метод,Размер,Размер_блока,Время(с)
    cout << method_name << "," << N << "," << block_size << "," << fixed << setprecision(6) << diff.count() << endl;

    return 0;
}
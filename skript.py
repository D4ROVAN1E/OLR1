import subprocess
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# Настройки эксперимента
# Для начала используем небольшие размеры, чтобы не ждать часами. 
# Для отчета можете расширить массив до [1000, 2000, 3000, 5000]
#MATRIX_SIZES = [100, 250, 500, 750, 1000, 1250, 1500, 2000] 
MATRIX_SIZES = [1000, 2000, 3000, 4000, 5000, 6000, 7000] 
BLOCK_SIZE = 64 # Для Opt2
METHODS = {
    0: "Base",
    1: "Opt1_RowWise",
    2: "Opt2_Block",
    3: "Opt3_Vectorized"
}

CPP_SOURCE = "matrix_mult.cpp"
EXECUTABLE = "./dgemm.exe" if os.name == 'nt' else "./dgemm"

def compile_cpp():
    print("Компиляция C++ кода...")
    # Флаг -O0 (отключение оптимизаций компилятора для чистоты эксперимента), 
    # -march=native (использование инструкций вашего процессора, напр. AVX2), 
    # -fopenmp (для поддержки #pragma omp simd)
    compile_cmd = ["g++", "-O3", "-march=native", "-fopenmp", CPP_SOURCE, "-o", EXECUTABLE]
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Ошибка компиляции:\n", result.stderr)
        exit(1)
    print("Успешно скомпилировано.")

def run_experiments():
    results = defaultdict(list)
    
    for size in MATRIX_SIZES:
        for method_id, method_name in METHODS.items():
            print(f"Запуск: Метод={method_name}, N={size}...")
            
            # Формируем команду запуска
            run_cmd = [EXECUTABLE, str(size), str(method_id), str(BLOCK_SIZE)]
            
            try:
                res = subprocess.run(run_cmd, capture_output=True, text=True, check=True)
                output = res.stdout.strip().split(',')
                # Ожидаемый вывод: Метод,N,РазмерБлока,Время
                time_seconds = float(output[3])
                results[method_name].append(time_seconds)
            except subprocess.CalledProcessError as e:
                print(f"Ошибка при выполнении: {e}")
                
    return results

def plot_results(results):
    plt.figure(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'd']
    for (method, times), marker in zip(results.items(), markers):
        plt.plot(MATRIX_SIZES, times, marker=marker, label=method, linewidth=2)
        
    plt.title("Зависимость времени выполнения DGEMM от размера матрицы")
    plt.xlabel("Размер матрицы (N x N)")
    plt.ylabel("Время выполнения (секунды)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Сохраняем график в файл
    plt.savefig("dgemm_performance.png", dpi=300)
    print("График сохранен как dgemm_performance.png")
    plt.show()

if __name__ == "__main__":
    compile_cpp()
    print("\nНачало тестирования (это может занять некоторое время)...")
    data = run_experiments()
    print("\nТестирование завершено. Построение графика...")
    plot_results(data)
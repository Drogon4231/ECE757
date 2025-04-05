import os
import pandas as pd
def run_benchmark(matrix_sizes, output_file):
    """Runs TGD partitioning tests for different matrix sizes and stores runtime results."""
    results = []
    for size in matrix_sizes:
        cmd = f"./examples/final_project_test ./benchmark/des_perf.txt {size}"
        runtime = os.popen(cmd).read().strip()  # Capture execution output
        results.append([size, runtime])
    
    # Save to CSV
    df = pd.DataFrame(results, columns=['matrix_size', 'runtime'])
    df.to_csv(output_file, index=False)
    print("Benchmark results saved to", output_file)

# Run & Save Data
matrix_sizes = [1, 2, 4, 8, 16, 32]
run_benchmark(matrix_sizes, "results.csv")
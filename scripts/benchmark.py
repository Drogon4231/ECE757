import os
import pandas as pd

def run_benchmark(matrix_sizes, circuit_file, output_file):
    """Runs TGD partitioning tests for different matrix sizes and stores runtime results."""
    results = []

    # Ensure parent folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for size in matrix_sizes:
        cmd = f"./examples/final_project_test {circuit_file} {size}"
        print(f"🚀 Running: {cmd}")
        runtime = os.popen(cmd).read().strip()  # Capture execution output
        results.append([size, runtime])

    # Save to CSV
    df = pd.DataFrame(results, columns=['matrix_size', 'runtime'])
    df.to_csv(output_file, index=False)
    print(f"\n📄 Benchmark results saved to {output_file}")

# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    matrix_sizes = [1, 2, 4, 8, 16, 32]
    circuit_file = "data/graphs/des_perf.txt"
    output_file = "data/ml_data/benchmark_results.csv"

    run_benchmark(matrix_sizes, circuit_file, output_file)
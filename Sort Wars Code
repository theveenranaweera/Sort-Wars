import random  # Used to generate large numeric datasets reflecting real-world randomness
import time    # Enabling timing of algorithm executions to evaluate efficiency
import tracemalloc  # Tracking memory usage to compare overhead among sorting approaches
import matplotlib.pyplot as plt # Producing bar charts to visualize performance metrics

# Constants, to maintain consistency
DATASET1_SIZE = 10000
DATASET2_SIZE = 100000
MIDPOINT_DIVISOR = 2
SECONDS_TO_MILLISECONDS = 1000
RANDOM_DATA_LOWER_BOUND = 100000
RANDOM_DATA_UPPER_BOUND = 1000000
START_INDEX = 0
DATASET_NAMES = [
    f"Random ({DATASET1_SIZE:,})",
    f"Reversed ({DATASET1_SIZE:,})",
    f"Random ({DATASET2_SIZE:,})",
    f"Reversed ({DATASET2_SIZE:,})"
]
BYTES_IN_KB = 1024
BYTES_IN_MB = BYTES_IN_KB * 1024
YLIM_SCALING_FACTOR = 1.2  # Scaling the Y-axis to 120% of the maximum data value for better visualization
YLIM_STARTING_VAL = 0
FIG_WIDTH = 8
FIG_HEIGHT = 6

#### ---- TASK 3 ---- ####
def measure_time(sort_function, dataset):
    """
    Returning the elapsed time (in milliseconds) for sorting 'dataset'
    to reflect each algorithm's raw speed.
    """
    start_time = time.time()
    sort_function(dataset)
    end_time = time.time()
    return (end_time - start_time) * SECONDS_TO_MILLISECONDS

def measure_memory(sort_function, dataset):
    """
    Evaluating the peak memory consumption of 'sort_function' when
    operating on 'dataset'. This highlights overhead differences 
    among the sorting algorithms.
    """
    tracemalloc.start()  
    sort_function(dataset)
    memory_stats = tracemalloc.get_traced_memory()  # Getting memory usage stats as a tuple (current, peak)
    peak = memory_stats[1]
    tracemalloc.stop()
    return peak / BYTES_IN_MB 

#### ---- TASK 1 ---- ####
def random_dataset(size):
    """
    Creating a dataset of unique random integers to simulate realistic data.
    Ensuring no duplicates underscores reliability and avoids skewed results.
    """
    return random.sample(range(RANDOM_DATA_LOWER_BOUND, RANDOM_DATA_UPPER_BOUND), size)

def reverse_dataset(data_set):
    return sorted(data_set, reverse = True)  # Producing a reversed version of 'data_set' to model worst/edge-case

def verify_uniqueness(dataset, dataset_name):
    """
    Confirming each dataset remains free of duplicates to maintain
    data integrity for reliable testing.
    """
    if len(dataset) != len(set(dataset)):
        print(f"Dataset Element Uniqueness Warning for '{dataset_name}': Dataset contains duplicate values!")
    else:
        print(f"Dataset Element Uniqueness Verified for '{dataset_name}': No duplicates found.")

def copy_dataset(dataset):
    """
    Preserving the original dataset for Quick Sort, which mutates data in-place.
    This ensures we can re-use the original dataset if needed.
    """
    return dataset[:]

#### ---- TASK 2 ---- ####
# Global counters for comparisons and swaps
comparison_count = 0
swap_count = 0

# Merge Sort functions
def merge(array1, array2):
    """
    Combining two sorted sublists while counting comparisons to quantify 
    how Merge Sort scales with data size and order.
    """
    global comparison_count
    combined = []
    i = 0
    j = 0

    # Merging elements from both arrays based on comparisons
    while i < len(array1) and j < len(array2):
        comparison_count += 1 
        if array1[i] < array2[j]:
            combined.append(array1[i])
            i += 1
        else:
            combined.append(array2[j])
            j += 1

    # Appending remaining elements from array1 AND array2
    while i < len(array1):
        combined.append(array1[i])
        i += 1

    while j < len(array2):
        combined.append(array2[j])
        j += 1

    return combined

def merge_sort(my_list):
    """
    Recursively partitioning 'my_list' and merges sublists for stable, 
    guaranteed O(n log n) performance at the cost of extra memory.
    """
    if len(my_list) == 1:
        return my_list
    mid_index = int(len(my_list) // MIDPOINT_DIVISOR)
    left = merge_sort(my_list[:mid_index])
    right = merge_sort(my_list[mid_index:])
    return merge(left, right)

# Quick Sort functions
def swap(my_list, index1, index2):
    """
    Swapping two elements to orchestrate Quick Sort partitioning. 
    Each swap indicates tangible overhead for in-place sorting.
    """
    global swap_count
    temp = my_list[index1]
    my_list[index1] = my_list[index2]
    my_list[index2] = temp
    swap_count += 1 

def pivot(my_list, pivot_index, end_index):
    """
    Selecting a pivot (the middle element) and reorders data around it.
    Counting comparisons illustrates Quick Sort's partition cost.
    """
    global comparison_count
    swap_index = pivot_index
    mid_index = (pivot_index + end_index) // MIDPOINT_DIVISOR
    swap(my_list, pivot_index, mid_index)  # Moving the middle element to the pivot position

    # Comparing each element in the sublist (from pivot_index+1 to end_index) with the pivot
    for i in range(pivot_index + 1, end_index + 1):
        comparison_count += 1
        if my_list[i] < my_list[pivot_index]:
            swap_index += 1
            swap(my_list, swap_index, i)
    
    swap(my_list, pivot_index, swap_index) # After partitioning, put the pivot in its correct position
    return swap_index

def quick_sort_helper(my_list, left, right):
    """
    Recursively partitioning 'my_list' around pivots until fully sorted,
    preserving minimal memory usage in exchange for pivot-dependent variation.
    """
    if left < right:
        pivot_index = pivot(my_list, left, right) # Finding the pivot and partitioning the sublist
        quick_sort_helper(my_list, left, pivot_index - 1)
        quick_sort_helper(my_list, pivot_index + 1, right)

def quick_sort(my_list):
    """
    Initiating in-place Quick Sort with minimal overhead. The pivot-based
    approach may degrade if partitions become unbalanced.
    """
    quick_sort_helper(my_list, START_INDEX, len(my_list) - 1)

#### ---- TASK 4: GRAPH GENERATION AND METRIC ANALYSIS ---- ####
def analyze_sorting_algorithm(sort_function, dataset, algorithm_name, dataset_name, metric):
    """
    Gathering a single metric (time, memory, comparisons, or swaps)
    for 'sort_function' operating on 'dataset'.
    Resetting global counters beforehand for consistent measurement.
    """
    global comparison_count, swap_count
    comparison_count = 0
    swap_count = 0

    if metric == "time":
        time_ms = measure_time(sort_function, dataset)
        print(f"{algorithm_name} on {dataset_name}: Execution Time = {time_ms:.2f} ms")
        return time_ms
    elif metric == "memory":
        memory_mb = measure_memory(sort_function, dataset)
        memory_kb = memory_mb * BYTES_IN_KB
        print(f"{algorithm_name} on {dataset_name}: Memory Used = {memory_kb:.2f} KB")
        return memory_kb
    elif metric == "comparisons":
        sort_function(dataset[:])
        print(f"{algorithm_name} on {dataset_name}: Comparisons = {comparison_count}")
        return comparison_count
    elif metric == "swaps":
        if sort_function == quick_sort:
            sort_function(dataset[:])
            print(f"{algorithm_name} on {dataset_name}: Swaps = {swap_count}")
            return swap_count
        else:
            return 0

def generate_bar_graph(data, title, ylabel, output_file, max_y = None):
    """
    Producing a labeled bar chart for 'data' across all datasets,
    highlighting the performance of each sorting algorithm. 
    'max_y' ensures consistent y-axis scaling between Merge Sort
    and Quick Sort graphs if desired.
    """    
    # Defining a color for each dataset when showing up in the graph
    COLOR_MAPPING = {
        f"Random ({DATASET1_SIZE:,})": '#FF5733',    
        f"Reversed ({DATASET1_SIZE:,})": '#33C1FF',   
        f"Random ({DATASET2_SIZE:,})": '#75FF33',      
        f"Reversed ({DATASET2_SIZE:,})": '#FFC133'
    }
    
    # Assigning colors to the datasets based on the color mapping
    colors = [COLOR_MAPPING[dataset] for dataset in DATASET_NAMES]
    
    # Creating the bar graph
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    plt.bar(DATASET_NAMES, data, color=colors)
    plt.title(title)
    plt.xlabel('Datasets')
    plt.ylabel(ylabel)
    
    # Determining the max limit for the y-axis
    if max_y is not None:
        upper_limit = max_y
    else:
        upper_limit = max(data) * YLIM_SCALING_FACTOR  # Otherwise, calculating it based off the data

    # Applying the determined y-axis limits
    plt.ylim(YLIM_STARTING_VAL, upper_limit)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

def save_to_file(filename, dataset):
    with open(filename, 'w') as file:
        for number in dataset:
            file.write(f"{number}\n")
    print(f"Sorted dataset saved to {filename}")

if __name__ == "__main__": # Entry point of the script: enabling the code to run only once, in a main script context.
    # Generating datasets
    dataset1 = random_dataset(DATASET1_SIZE)
    verify_uniqueness(dataset1, DATASET_NAMES[0])  # Checking uniqueness of each element

    dataset1_reversed = reverse_dataset(dataset1)
    verify_uniqueness(dataset1_reversed, DATASET_NAMES[1])  

    dataset2 = random_dataset(DATASET2_SIZE)
    verify_uniqueness(dataset2, DATASET_NAMES[2])

    dataset2_reversed = reverse_dataset(dataset2)
    verify_uniqueness(dataset2_reversed, DATASET_NAMES[3])

    datasets = [dataset1, dataset1_reversed, dataset2, dataset2_reversed]

    # Metric names to analyze
    metrics = ["time", "memory", "comparisons", "swaps"]

    # Analyzing for Merge Sort
    print("\n--- Generating Merge Sort Graphs ---")
    merge_results = {} # Initialize an empty list for each metric in the metrics list
    for metric in metrics:
        merge_results[metric] = []

    for metric in metrics:
        if metric == "swaps":
            continue  # Swaps do not apply to Merge Sort
        for i in range(len(datasets)):
            dataset_for_analysis = datasets[i] 
            result = analyze_sorting_algorithm(merge_sort, dataset_for_analysis, "Merge Sort", DATASET_NAMES[i], metric)
            merge_results[metric].append(result)

    # Analyzing for Quick Sort
    print("\n--- Generating Quick Sort Graphs ---")
    quick_results = {}
    for metric in metrics:
        quick_results[metric] = []
        
    for metric in metrics:
        for i in range(len(datasets)):
            dataset_copy = copy_dataset(datasets[i])
            result = analyze_sorting_algorithm(quick_sort, dataset_copy, "Quick Sort", DATASET_NAMES[i], metric)
            quick_results[metric].append(result)

    # Gathering results for scaling bar charts consistently
    merge_time = merge_results['time']
    merge_memory = merge_results['memory']
    merge_comps = merge_results['comparisons']

    quick_time = quick_results['time']
    quick_memory = quick_results['memory']
    quick_comps = quick_results['comparisons']
    quick_swaps = quick_results['swaps']

    combined_time = merge_time + quick_time
    max_time = max(combined_time) * YLIM_SCALING_FACTOR

    combined_memory = merge_memory + quick_memory
    max_memory = max(combined_memory) * YLIM_SCALING_FACTOR

    combined_comps = merge_comps + quick_comps
    max_comps = max(combined_comps) * YLIM_SCALING_FACTOR

    # Plotting Merge Sort with combined max
    generate_bar_graph(
        merge_time,
        "Execution Time (Merge Sort)",
        "Execution Time (ms)",
        "merge_sort_time.png",
        max_y = max_time
    )
    generate_bar_graph(
        merge_memory,
        "Memory Usage (Merge Sort)",
        "Memory (KB)",
        "merge_sort_memory.png",
        max_y = max_memory
    )
    generate_bar_graph(
        merge_comps,
        "Number of Comparisons (Merge Sort)",
        "Comparisons",
        "merge_sort_comparisons.png",
        max_y = max_comps
    )

    # Plotting Quick Sort with the same max
    generate_bar_graph(
        quick_time,
        "Execution Time (Quick Sort)",
        "Execution Time (ms)",
        "quick_sort_time.png",
        max_y = max_time
    )
    generate_bar_graph(
        quick_memory,
        "Memory Usage (Quick Sort)",
        "Memory (KB)",
        "quick_sort_memory.png",
        max_y = max_memory
    )
    generate_bar_graph(
        quick_comps,
        "Number of Comparisons (Quick Sort)",
        "Comparisons",
        "quick_sort_comparisons.png",
        max_y = max_comps
    )

    max_swaps = max(quick_swaps) * YLIM_SCALING_FACTOR
    generate_bar_graph(
        quick_swaps,
        "Number of Swaps (Quick Sort)",
        "Swaps",
        "quick_sort_swaps.png",
        max_y = max_swaps
    )

    print("\n--- Saving Sorted Files ---")

    # SAVING datasets for Merge Sort
        # Sorting again purely to demonstrate file output, though we already computed these results
    sorted_merge_dataset1 = merge_sort(dataset1)
    save_to_file("sorted_dataset1_mergesort.txt", sorted_merge_dataset1)

    sorted_merge_dataset1_reversed = merge_sort(dataset1_reversed)
    save_to_file("sorted_dataset1_reversed_mergesort.txt", sorted_merge_dataset1_reversed)

    sorted_merge_dataset2 = merge_sort(dataset2)
    save_to_file("sorted_dataset2_mergesort.txt", sorted_merge_dataset2)

    sorted_merge_dataset2_reversed = merge_sort(dataset2_reversed)
    save_to_file("sorted_dataset2_reversed_mergesort.txt", sorted_merge_dataset2_reversed)

    # SAVING datasets for Quick Sort
        # Quick Sort modifies the dataset in place. No need to make new copies to save files as this is the last step.
    quick_sort(dataset1)
    save_to_file("sorted_dataset1_quicksort.txt", dataset1)

    quick_sort(dataset1_reversed)
    save_to_file("sorted_dataset1_reversed_quicksort.txt", dataset1_reversed)

    quick_sort(dataset2)
    save_to_file("sorted_dataset2_quicksort.txt", dataset2)

    quick_sort(dataset2_reversed)
    save_to_file("sorted_dataset2_reversed_quicksort.txt", dataset2_reversed)

#### ---- REFERENCES FOR CODE ---- ####
# https://docs.python.org/3/library/random.html
# https://www.scaler.com/topics/time-module-in-python/
# https://www.datacamp.com/tutorial/memory-profiling-python
# https://pythonbasics.org/matplotlib-bar-chart/
# https://www.activestate.com/resources/quick-reads/what-is-matplotlib-in-python-how-to-use-it-for-plotting/
# https://realpython.com/python-main-function/
# https://learnpython.com/blog/write-to-file-python/#:~:text=Use%20write()%20to%20Write%20to%20File%20in%20Python&text=It%20is%20shorter%20to%20use,with%20open%20(%20%22file.
# https://www.datacamp.com/tutorial/python-copy-list
# https://algodaily.com/lessons/merge-sort-vs-quick-sort-heap-sort/python
# https://erhankilic.org/post/a-comprehensive-guide-to-merge-sort-and-quick-sort-algorithms/

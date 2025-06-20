#include <torch/torch.h>
#include <iostream>
#include <iomanip> // For std::setprecision
#include <chrono>
#include <vector>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <curl/curl.h>
#include <thread>
#include <unistd.h> // For sysconf, getpid
#include <atomic>
#include <cstdio> // For popen, pclose, fgets
#include <array>  // For buffer in popen
#include <string> // For std::string manipulation
#include <mutex>  // For std::mutex
#include <tuple>  // For std::tuple

namespace fs = std::filesystem;

// Constants
const int EPOCHS = 10;
const int BATCH_SIZE = 1024; // Match Rust/Python
const int IMAGE_SIZE = 28 * 28;
const int HIDDEN_SIZE = 128;
const int NUM_CLASSES = 10;

const std::vector<std::pair<std::string, std::string>> MNIST_URLS = {
    {"https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz", "data/train-images-idx3-ubyte"},
    {"https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz", "data/train-labels-idx1-ubyte"},
    {"https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz", "data/t10k-images-idx3-ubyte"},
    {"https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz", "data/t10k-labels-idx1-ubyte"}
};

// Define a type alias for the metrics tuple for clarity
using MetricsTuple = std::tuple<double, double, double, double>; // time, util, power, mem

// MLP Network Structure
struct MLP : torch::nn::Module {
    MLP() {
        fc1 = register_module("fc1", torch::nn::Linear(IMAGE_SIZE, HIDDEN_SIZE));
        fc2 = register_module("fc2", torch::nn::Linear(HIDDEN_SIZE, NUM_CLASSES));

        // --- Explicit Initialization (Standard Normal) ---
        torch::NoGradGuard no_grad; // Ensure no gradients during init
        fc1->weight.normal_(0.0, 1.0);
        if (fc1->bias.defined()) {
            fc1->bias.zero_();
        }
        fc2->weight.normal_(0.0, 1.0);
        if (fc2->bias.defined()) {
            fc2->bias.zero_();
        }
        // --- End Initialization ---
    }

    torch::Tensor forward(torch::Tensor x) {
        x = x.view({-1, IMAGE_SIZE});
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }

    void save(const std::string& path) {
        torch::save(fc1, path + "_fc1.pt");
        torch::save(fc2, path + "_fc2.pt");
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

// Function to get process memory usage in MB using /proc/self/statm
double get_process_memory_mb() {
    std::ifstream statm("/proc/self/statm");
    if (!statm.is_open()) return 0.0;
    long resident_pages; // RSS is the second value
    statm >> resident_pages >> resident_pages; // Read first value, then second into resident_pages
    statm.close();
    long page_size_bytes = sysconf(_SC_PAGESIZE);
    return static_cast<double>(resident_pages) * page_size_bytes / (1024.0 * 1024.0);
}

// Function to get GPU metrics using nvidia-smi
std::pair<double, double> get_gpu_metrics() {
    std::array<char, 128> buffer;
    std::string result = "";
    // Execute nvidia-smi command
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen("nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader,nounits -i 0", "r"), pclose);
    if (!pipe) {
        std::cerr << "WARN: popen() failed for nvidia-smi!" << std::endl;
        return {0.0, 0.0};
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }

    // Parse the result (e.g., "15, 45.53")
    try {
        size_t comma_pos = result.find(',');
        if (comma_pos != std::string::npos) {
            double util = std::stod(result.substr(0, comma_pos));
            double power = std::stod(result.substr(comma_pos + 1));
            return {util, power};
        } else {
             std::cerr << "WARN: Unexpected nvidia-smi output format: " << result << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "WARN: Failed to parse nvidia-smi output: " << e.what() << " Output: " << result << std::endl;
    }
    return {0.0, 0.0};
}

// Function to check if nvidia-smi is available and working
bool check_nvidia_smi() {
     std::unique_ptr<FILE, decltype(&pclose)> pipe(popen("nvidia-smi --query-gpu=name --format=csv,noheader -i 0", "r"), pclose);
     if (!pipe) return false;
     std::array<char, 128> buffer;
     std::string result = "";
     if (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
     }
     // Check if we got a non-empty result (basic check)
     return !result.empty() && result.find_first_not_of(" \t\n\r") != std::string::npos;
}

// Callback function for CURL to write data
size_t write_data(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    return fwrite(ptr, size, nmemb, stream);
}

// Function to download and extract MNIST dataset
void download_and_extract(const std::string& url, const std::string& output_path) {
    CURL *curl = curl_easy_init();
    if (curl) {
        FILE *fp = fopen((output_path + ".gz").c_str(), "wb");
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        
        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "Failed to download: " << curl_easy_strerror(res) << std::endl;
        }
        
        curl_easy_cleanup(curl);
        fclose(fp);
        std::string cmd = "gunzip -f " + output_path + ".gz";
        system(cmd.c_str());
    }
}

// Function to collect metrics (time, GPU util, GPU power, memory usage)
void collect_metrics(std::atomic<bool>& running, std::ofstream& metrics_file, bool collect_gpu, std::vector<MetricsTuple>& detailed_metrics_data, std::mutex& data_mutex) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Write header only if file is open (i.e., first run)
    if (metrics_file.is_open()) {
        metrics_file << "time,gpu_utilization_percent,gpu_power_watts,memory_mb\n";
        metrics_file.flush();
    }

    while (running) {
        auto loop_start_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(loop_start_time - start_time).count();

        double gpu_util = 0.0;
        double gpu_power = 0.0;
        if (collect_gpu) {
            std::tie(gpu_util, gpu_power) = get_gpu_metrics();
        }

        double memory_mb = get_process_memory_mb();

        // Store metrics in the vector for processing later
        {
            std::lock_guard<std::mutex> lock(data_mutex);
            detailed_metrics_data.emplace_back(elapsed, gpu_util, gpu_power, memory_mb);
        }

        // Write to file if it's open (first run only)
        if (metrics_file.is_open()) {
            metrics_file << std::fixed << std::setprecision(2) // Set precision for output
                         << elapsed << ","
                         << gpu_util << ","
                         << gpu_power << ","
                         << memory_mb << "\n";
            metrics_file.flush();
        }

        auto loop_end_time = std::chrono::high_resolution_clock::now();
        auto loop_duration = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end_time - loop_start_time);
        auto sleep_duration = std::chrono::milliseconds(100) - loop_duration;
        if (sleep_duration.count() > 0) {
            std::this_thread::sleep_for(sleep_duration);
        }
    }
     std::cout << "Metrics thread finished." << std::endl;
}

int main() {
    const int NUM_RUNS = 3;
    // Vectors to store results from each run
    std::vector<double> run_total_times;
    std::vector<double> run_final_accuracies;
    std::vector<double> run_avg_epoch_times;
    std::vector<double> run_avg_gpu_utils;
    std::vector<double> run_avg_gpu_powers;
    std::vector<double> run_avg_mems;
    std::vector<double> run_peak_mems;

    // --- Device Selection ---
    torch::Device device(torch::kCPU);
    bool gpu_available_for_metrics = false;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
        gpu_available_for_metrics = check_nvidia_smi();
        if (!gpu_available_for_metrics) {
             std::cout << "WARN: Running on CUDA but nvidia-smi failed or not found. GPU metrics disabled." << std::endl;
        } else {
             std::cout << "nvidia-smi detected. GPU metrics enabled." << std::endl;
        }
    } else {
        std::cout << "CUDA not available. Training on CPU." << std::endl;
    }
    // --- End Device Selection ---

    // Create necessary directories
    fs::create_directories("data");
    fs::create_directories("stats");

    // Download MNIST dataset if not already present
    for (const auto& [url, output_path] : MNIST_URLS) {
        if (!fs::exists(output_path)) {
            std::cout << "Downloading " << url << "..." << std::endl;
            download_and_extract(url, output_path);
        }
    }

    // Load MNIST dataset
    auto train_dataset = torch::data::datasets::MNIST("./data")
                            .map(torch::data::transforms::Normalize<>(0.0, 1.0)) // Normalize
                            .map(torch::data::transforms::Stack<>());
    auto test_dataset = torch::data::datasets::MNIST("./data", torch::data::datasets::MNIST::Mode::kTest)
                           .map(torch::data::transforms::Normalize<>(0.0, 1.0)) // Normalize
                           .map(torch::data::transforms::Stack<>());

    // Pre-load the entire dataset onto the device, removing the need for a DataLoader.
    std::cout << "Pre-loading dataset to device..." << std::endl;
    auto train_loader_for_load = torch::data::make_data_loader(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(train_dataset.size().value()));
    auto test_loader_for_load = torch::data::make_data_loader(
        std::move(test_dataset),
        torch::data::DataLoaderOptions().batch_size(test_dataset.size().value()));

    auto train_batch = *train_loader_for_load->begin();
    auto all_train_images = train_batch.data.to(device);
    auto all_train_labels = train_batch.target.to(device);

    auto test_batch = *test_loader_for_load->begin();
    auto all_test_images = test_batch.data.to(device);
    auto all_test_labels = test_batch.target.to(device);
    std::cout << "Dataset pre-loaded." << std::endl;

    for (int run = 0; run < NUM_RUNS; ++run) {
        std::cout << "\n--- Starting Run " << run + 1 << "/" << NUM_RUNS << " ---" << std::endl;

        // Re-initialize model and optimizer for a fair run
        MLP model;
        model.to(device);
        torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.01).eps(1e-8));

        bool is_first_run = (run == 0);
        
        // Metrics collection setup for the current run
        std::vector<MetricsTuple> detailed_metrics_data; // For in-memory collection
        std::mutex metrics_mutex;
        std::atomic<bool> metrics_running(true);
        std::thread metrics_thread;
        
        std::ofstream metrics_detailed_file;
        std::ofstream metrics_epoch_file;

        if (is_first_run) {
            fs::remove("stats/cpp_metrics_detailed.csv");
            fs::remove("stats/cpp_metrics.csv");
            metrics_detailed_file.open("stats/cpp_metrics_detailed.csv", std::ios::trunc);
            metrics_epoch_file.open("stats/cpp_metrics.csv", std::ios::trunc);
            metrics_epoch_file << "epoch,train_accuracy,valid_accuracy,epoch_time\n";
            metrics_epoch_file.flush();
        }
        
        // The thread now collects to memory and writes to file if open
        metrics_thread = std::thread(collect_metrics, std::ref(metrics_running), std::ref(metrics_detailed_file), gpu_available_for_metrics, std::ref(detailed_metrics_data), std::ref(metrics_mutex));

        auto overall_start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<double> run_epoch_times;
        double run_final_accuracy = 0.0;

        // Training loop
        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            auto epoch_start = std::chrono::high_resolution_clock::now();
            model.train();
            double train_correct = 0;
            int64_t train_samples_processed = 0; // Use int64_t
            int64_t total_train_size = all_train_images.size(0);

            // Manually shuffle and batch the data
            auto indices = torch::randperm(total_train_size, torch::TensorOptions().dtype(torch::kInt64).device(device));

            for (int64_t i = 0; i < total_train_size; i += BATCH_SIZE) {
                int64_t end = std::min(i + BATCH_SIZE, total_train_size);
                if (i >= end) continue;

                auto batch_indices = indices.slice(0, i, end);
                auto data = all_train_images.index_select(0, batch_indices);
                auto target = all_train_labels.index_select(0, batch_indices);

                optimizer.zero_grad();
                auto output = model.forward(data);
                auto loss = torch::nn::functional::cross_entropy(output, target);
                loss.backward();
                optimizer.step();

                auto prediction = output.argmax(1);
                train_correct += prediction.eq(target).sum().item<int64_t>();
                train_samples_processed += target.size(0);
            }

            double train_accuracy = (train_samples_processed > 0) ? (static_cast<double>(train_correct) / train_samples_processed * 100.0) : 0.0;

            // Validation phase
            model.eval();
            double valid_correct = 0;
            int64_t valid_total = 0; // Use int64_t
            torch::NoGradGuard no_grad; // Ensure no gradients during validation

            auto output = model.forward(all_test_images);
            auto prediction = output.argmax(1);
            valid_correct += prediction.eq(all_test_labels).sum().item<int64_t>(); // Use int64_t
            valid_total += all_test_labels.size(0);

            double valid_accuracy = (valid_total > 0) ? (static_cast<double>(valid_correct) / valid_total * 100.0) : 0.0;

            auto epoch_end = std::chrono::high_resolution_clock::now();
            double epoch_duration = std::chrono::duration<double>(epoch_end - epoch_start).count();
            run_epoch_times.push_back(epoch_duration);

            if (is_first_run) {
                metrics_epoch_file << (epoch + 1) << ","
                                << std::fixed << std::setprecision(2) << train_accuracy << ","
                                << valid_accuracy << ","
                                << epoch_duration << "\n";
                metrics_epoch_file.flush();
            }

            run_final_accuracy = valid_accuracy;

            std::cout << "Epoch " << epoch + 1 << "/" << EPOCHS
                      << ", Train Acc: " << std::fixed << std::setprecision(2) << train_accuracy << "%"
                      << ", Valid Acc: " << valid_accuracy << "%"
                      << ", Epoch Time: " << epoch_duration << "s" << std::endl;
        }

        auto overall_end_time = std::chrono::high_resolution_clock::now();
        double total_duration = std::chrono::duration<double>(overall_end_time - overall_start_time).count();
        
        metrics_running = false;
        metrics_thread.join();

        // Store results for this run
        run_total_times.push_back(total_duration);
        run_final_accuracies.push_back(run_final_accuracy);
        run_avg_epoch_times.push_back(std::accumulate(run_epoch_times.begin(), run_epoch_times.end(), 0.0) / run_epoch_times.size());
        
        // Calculate and store detailed metrics for this run
        double util_sum = 0.0, power_sum = 0.0, mem_sum = 0.0, peak_mem = 0.0;
        if (!detailed_metrics_data.empty()) {
            for (const auto& data_point : detailed_metrics_data) {
                util_sum += std::get<1>(data_point);
                power_sum += std::get<2>(data_point);
                mem_sum += std::get<3>(data_point);
                if (std::get<3>(data_point) > peak_mem) {
                    peak_mem = std::get<3>(data_point);
                }
            }
            run_avg_gpu_utils.push_back(util_sum / detailed_metrics_data.size());
            run_avg_gpu_powers.push_back(power_sum / detailed_metrics_data.size());
            run_avg_mems.push_back(mem_sum / detailed_metrics_data.size());
            run_peak_mems.push_back(peak_mem);
        } else {
            run_avg_gpu_utils.push_back(0.0);
            run_avg_gpu_powers.push_back(0.0);
            run_avg_mems.push_back(0.0);
            run_peak_mems.push_back(0.0);
        }

        std::cout << "Run " << run + 1 << " finished in " << std::fixed << std::setprecision(2) << total_duration << " seconds." << std::endl;

        if (is_first_run) {
            metrics_detailed_file.close();
            metrics_epoch_file.close();
            std::cout << "Collected " << detailed_metrics_data.size() << " detailed metrics points (for CSV and summary)" << std::endl;
        }
    }

    // --- Final Summary Calculation ---
    double avg_total_time = std::accumulate(run_total_times.begin(), run_total_times.end(), 0.0) / run_total_times.size();
    double avg_final_accuracy = std::accumulate(run_final_accuracies.begin(), run_final_accuracies.end(), 0.0) / run_final_accuracies.size();
    double avg_avg_epoch_time = std::accumulate(run_avg_epoch_times.begin(), run_avg_epoch_times.end(), 0.0) / run_avg_epoch_times.size();
    double avg_gpu_util = std::accumulate(run_avg_gpu_utils.begin(), run_avg_gpu_utils.end(), 0.0) / run_avg_gpu_utils.size();
    double avg_gpu_power = std::accumulate(run_avg_gpu_powers.begin(), run_avg_gpu_powers.end(), 0.0) / run_avg_gpu_powers.size();
    double avg_mem_usage = std::accumulate(run_avg_mems.begin(), run_avg_mems.end(), 0.0) / run_avg_mems.size();
    double max_peak_mem = run_peak_mems.empty() ? 0.0 : *std::max_element(run_peak_mems.begin(), run_peak_mems.end());
    double energy_joules = run_avg_gpu_powers.empty() ? 0.0 : run_avg_gpu_powers[0] * run_total_times[0];

    std::cout << "\n--- Benchmark Complete ---" << std::endl;
    std::cout << "Average Total Time over " << NUM_RUNS << " runs: " << std::fixed << std::setprecision(2) << avg_total_time << "s" << std::endl;
    std::cout << "Average Final Accuracy over " << NUM_RUNS << " runs: " << avg_final_accuracy << "%" << std::endl;
    std::cout << "Average Epoch Time over " << NUM_RUNS << " runs: " << avg_avg_epoch_time << "s" << std::endl;

    // Save training summary
    std::ofstream summary("stats/cpp_training_summary.txt");
    summary << "Training Summary:\n"
            << "Number of Runs: " << NUM_RUNS << "\n"
            << "Device: " << (device.is_cuda() ? "cuda" : "cpu") << "\n";

    for (int i = 0; i < NUM_RUNS; ++i) {
        summary << "\n--- Run " << i + 1 << "/" << NUM_RUNS << " ---\n"
                << "Total Time: " << std::fixed << std::setprecision(2) << run_total_times[i] << " seconds\n"
                << "Final Accuracy: " << std::fixed << std::setprecision(2) << run_final_accuracies[i] << "%\n"
                << "Average Epoch Time: " << std::fixed << std::setprecision(2) << run_avg_epoch_times[i] << " seconds\n";
        if (gpu_available_for_metrics) {
            summary << "Average GPU Utilization: " << std::fixed << std::setprecision(1) << run_avg_gpu_utils[i] << "%\n"
                    << "Average GPU Power Draw: " << std::fixed << std::setprecision(1) << run_avg_gpu_powers[i] << " W\n";
        }
        summary << "Average Process Memory Usage: " << std::fixed << std::setprecision(1) << run_avg_mems[i] << " MB\n"
                << "Peak Process Memory Usage (this run): " << std::fixed << std::setprecision(2) << run_peak_mems[i] << " MB\n";
    }

    summary << "\n--- Averages over " << NUM_RUNS << " runs ---\n"
            << "Average Total Time: " << std::fixed << std::setprecision(2) << avg_total_time << " seconds\n"
            << "Average Final Accuracy: " << std::fixed << std::setprecision(2) << avg_final_accuracy << "%\n"
            << "Average Epoch Time: " << std::fixed << std::setprecision(2) << avg_avg_epoch_time << " seconds\n";
    if (gpu_available_for_metrics) {
         summary << "Average GPU Utilization: " << std::fixed << std::setprecision(1) << avg_gpu_util << "%\n"
                 << "Average GPU Power Draw: " << std::fixed << std::setprecision(1) << avg_gpu_power << " W\n"
                 << "Approx Total GPU Energy Consumed (Run 1): " << std::fixed << std::setprecision(2) << energy_joules << " J\n";
    }
    summary << "Average Process Memory Usage: " << std::fixed << std::setprecision(1) << avg_mem_usage << " MB\n"
            << "Overall Peak Process Memory Usage: " << std::fixed << std::setprecision(2) << max_peak_mem << " MB\n";
    summary.close();
    std::cout << "Summary saved to stats/cpp_training_summary.txt" << std::endl;

    std::cout << "C++ benchmark finished." << std::endl;
    return 0;
}
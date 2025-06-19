use std::fs::{File, create_dir_all};
use std::io::{BufReader, Write}; 
use std::time::{Instant, Duration};
use std::path::Path;
use sysinfo::{System, Pid, SystemExt, PidExt, ProcessExt};
use tch::{
    vision::mnist, nn, nn::OptimizerConfig, Device, Tensor, Kind, IndexOp, nn::ModuleT,
    no_grad, nn::Init,
};
use reqwest::blocking::get;
use flate2::read::GzDecoder;
use csv::Writer;
use std::process::Command;
use std::env;
use std::thread;
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}}; 
use std::ffi::CString;
use libc;

const EPOCHS: i64 = 10;
const BATCH_SIZE: i64 = 1024;
const NUM_RUNS: usize = 3;

const MNIST_URLS: &[(&str, &str)] = &[
    ("https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz", "data/train-images-idx3-ubyte"),
    ("https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz", "data/train-labels-idx1-ubyte"),
    ("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz", "data/t10k-images-idx3-ubyte"),
    ("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz", "data/t10k-labels-idx1-ubyte"),
];

#[derive(Debug)]
struct MLP {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl MLP {
    fn new(vs: &nn::Path) -> Self {
        MLP {
            fc1: nn::linear(vs / "fc1", 28 * 28, 128, Default::default()),
            fc2: nn::linear(vs / "fc2", 128, 10, Default::default()),
        }
    }
}

impl nn::Module for MLP {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.view([-1, 28 * 28])
            .apply(&self.fc1)
            .relu()
            .apply(&self.fc2)
    }
}

fn check_nvidia_smi() -> Option<String> {
    Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|s| s.trim().to_string())
}

fn download_and_extract(url: &str, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let response = get(url)?;
    let mut decoder = GzDecoder::new(BufReader::new(response));
    let mut output_file = File::create(output_path)?;
    std::io::copy(&mut decoder, &mut output_file)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let libtorch_path_str = env::var("LIBTORCH").unwrap_or_else(|_| {
        "".to_string() 
    });

    if !libtorch_path_str.is_empty() {
        let lib_path = Path::new(&libtorch_path_str).join("lib").join("libtorch_cuda.so");
        if lib_path.exists() {
            let path_cstr = CString::new(lib_path.to_str().unwrap())?;
            println!("Attempting to manually load: {}", lib_path.display());
            unsafe {
                let handle = libc::dlopen(path_cstr.as_ptr(), libc::RTLD_LAZY | libc::RTLD_GLOBAL);
                if handle.is_null() {
                    let error = CString::from_raw(libc::dlerror());
                    eprintln!("WARN: dlopen failed to load {}: {:?}", lib_path.display(), error);
                } else {
                    println!("Successfully loaded {} manually.", lib_path.display());
                }
            }
        } else {
            eprintln!("WARN: libtorch_cuda.so not found at expected path: {}", lib_path.display());
        }
    }

    let _libtorch_path = env::var("LIBTORCH").unwrap_or_else(|_| {
        "./libtorch".to_string()
    });

    create_dir_all("data")?;

    for (url, output_path) in MNIST_URLS {
        if !Path::new(output_path).exists() {
            download_and_extract(url, output_path)?;
        }
    }

    create_dir_all("stats")?;

    let device = Device::cuda_if_available();
    let device_str = if device.is_cuda() { "cuda" } else { "cpu" };
    println!("Using device: {}", device_str); 

    let gpu_name_opt = if device.is_cuda() { check_nvidia_smi() } else { None };
    let gpu_available_for_metrics = gpu_name_opt.is_some();
    if let Some(ref name) = gpu_name_opt {
        println!("nvidia-smi detected GPU: {}. GPU metrics enabled.", name);
    } else if device.is_cuda() {
        println!("WARN: Running on CUDA but nvidia-smi failed or not found. GPU metrics disabled.");
    }

    let m = mnist::load_dir("data")?;
    println!("Pre-loading dataset to device...");
    let train_images = m.train_images.to_device(device) / 255.0;
    let train_labels = m.train_labels.to_device(device);
    let test_images = m.test_images.to_device(device) / 255.0;
    let test_labels = m.test_labels.to_device(device);
    println!("Dataset pre-loaded.");

    let mut run_total_times = vec![];
    let mut run_final_accuracies = vec![];
    let mut run_avg_epoch_times = vec![];
    let mut run_avg_gpu_utils = vec![];
    let mut run_avg_gpu_powers = vec![];
    let mut run_avg_mems = vec![];
    let mut run_peak_mems = vec![];

    for run in 0..NUM_RUNS {
        println!("\n--- Starting Run {}/{} ---", run + 1, NUM_RUNS);
        
        let vs = nn::VarStore::new(device); 
        let mut net = MLP::new(&vs.root());
        no_grad(|| {
            net.fc1.ws.init(Init::Randn { mean: 0.0, stdev: 1.0 });
            if let Some(bs) = &mut net.fc1.bs { let _ = bs.zero_(); }
            net.fc2.ws.init(Init::Randn { mean: 0.0, stdev: 1.0 });
            if let Some(bs) = &mut net.fc2.bs { let _ = bs.zero_(); }
        });
        let mut opt = nn::Adam::default().build(&vs, 0.01)?;

        let detailed_metrics_data: Arc<Mutex<Vec<(f64, f64, f64, f64)>>> = Arc::new(Mutex::new(Vec::new()));
        let stop_collection = Arc::new(AtomicBool::new(false));
        let stop_collection_clone = stop_collection.clone();
        let detailed_metrics_data_clone = Arc::clone(&detailed_metrics_data);

        let overall_start_time = Instant::now();

        let metrics_thread_handle = thread::spawn(move || { 
            let pid = Pid::from(std::process::id() as usize);
            let mut sys = System::new(); // Use new() instead of new_all() for faster startup.

            while !stop_collection_clone.load(Ordering::Relaxed) {
                let loop_start_time = Instant::now();
                let elapsed_time = overall_start_time.elapsed().as_secs_f64();
                let mut gpu_util = 0.0;
                let mut gpu_power = 0.0;

                if gpu_available_for_metrics {
                    if let Ok(output) = Command::new("nvidia-smi").args(&["--query-gpu=utilization.gpu,power.draw", "--format=csv,noheader,nounits", "-i", "0"]).output() {
                        if output.status.success() {
                            if let Ok(stdout_str) = String::from_utf8(output.stdout) {
                                let parts: Vec<&str> = stdout_str.trim().split(',').collect();
                                if parts.len() == 2 {
                                    gpu_util = parts[0].trim().parse::<f64>().unwrap_or(0.0);
                                    gpu_power = parts[1].trim().parse::<f64>().unwrap_or(0.0);
                                }
                            }
                        }
                    }
                }

                sys.refresh_process(pid); 
                let memory_mb = if let Some(process) = sys.process(pid) { process.memory() as f64 / (1024.0 * 1024.0) } else { 0.0 };
                
                if let Ok(mut data) = detailed_metrics_data_clone.lock() {
                    data.push((elapsed_time, gpu_util, gpu_power, memory_mb));
                }

                let elapsed_loop = loop_start_time.elapsed();
                let sleep_duration = Duration::from_millis(100).saturating_sub(elapsed_loop);
                thread::sleep(sleep_duration);
            }
        });

        let total_train_size = train_images.size()[0];
        let total_test_size = test_images.size()[0];
        
        let mut run_epoch_times = vec![];
        let mut run_final_accuracy = 0.0;

        // Create epoch CSV writer only on the first run
        let mut epoch_wtr = if run == 0 {
            let mut wtr = Writer::from_path("stats/rust_metrics.csv")?;
            wtr.write_record(&["epoch", "train_accuracy", "valid_accuracy", "epoch_time"])?;
            Some(wtr)
        } else {
            None
        };

        for epoch in 1..=EPOCHS {
            let epoch_start_time = Instant::now();
            
            let mut train_correct_count: i64 = 0;
            let mut train_samples_processed: i64 = 0;
            let num_train_batches = (total_train_size + BATCH_SIZE - 1) / BATCH_SIZE;

            let indices = Tensor::randperm(total_train_size, (Kind::Int64, device));

            for i in 0..num_train_batches {
                let start = i * BATCH_SIZE;
                let end = std::cmp::min(start + BATCH_SIZE, total_train_size); 
                if start >= end { continue; }

                let batch_indices = indices.i(start..end);
                let images = train_images.index_select(0, &batch_indices);
                let labels = train_labels.index_select(0, &batch_indices);
                
                let batch_size_actual = labels.size()[0];
                train_samples_processed += batch_size_actual;

                let logits = net.forward_t(&images, true); 
                let loss = logits.cross_entropy_for_logits(&labels);
                opt.backward_step(&loss); 

                train_correct_count += logits.argmax(1, false).eq_tensor(&labels).sum(Kind::Int64).int64_value(&[]); 
            }
            
            let train_acc = if train_samples_processed > 0 {
                 (train_correct_count as f64 / train_samples_processed as f64) * 100.0
            } else {
                0.0
            };

            let (valid_acc, _test_correct_count) = no_grad(|| {
                let test_logits = net.forward_t(&test_images, false); 
                let test_correct_count = test_logits.argmax(1, false).eq_tensor(&test_labels).sum(Kind::Int64).int64_value(&[]); 
                let valid_acc = test_correct_count as f64 / total_test_size as f64 * 100.0;
                (valid_acc, test_correct_count) 
            });

            run_final_accuracy = valid_acc;
            let epoch_duration = epoch_start_time.elapsed();
            run_epoch_times.push(epoch_duration.as_secs_f64());

            println!(
                "Epoch {}/{}, Train Acc: {:.2}%, Valid Acc: {:.2}%, Epoch Time: {:.2}s",
                epoch, EPOCHS, train_acc, valid_acc, epoch_duration.as_secs_f64()
            );

            // Write to epoch CSV if it's the first run
            if let Some(wtr) = &mut epoch_wtr {
                wtr.write_record(&[
                    epoch.to_string(),
                    format!("{:.2}", train_acc),
                    format!("{:.2}", valid_acc),
                    format!("{:.2}", epoch_duration.as_secs_f64()),
                ])?;
            }
        }

        let total_duration_for_run = overall_start_time.elapsed();
        stop_collection.store(true, Ordering::Relaxed);
        metrics_thread_handle.join().expect("Metrics thread panicked");

        run_total_times.push(total_duration_for_run.as_secs_f64());
        run_final_accuracies.push(run_final_accuracy);
        run_avg_epoch_times.push(run_epoch_times.iter().sum::<f64>() / run_epoch_times.len() as f64);
        
        let run_detailed_metrics = detailed_metrics_data.lock().unwrap();
        if !run_detailed_metrics.is_empty() {
            let count = run_detailed_metrics.len() as f64;
            let util_sum: f64 = run_detailed_metrics.iter().map(|&(_, u, _, _)| u).sum();
            let power_sum: f64 = run_detailed_metrics.iter().map(|&(_, _, p, _)| p).sum();
            let mem_sum: f64 = run_detailed_metrics.iter().map(|&(_, _, _, m)| m).sum();
            let peak_mem = run_detailed_metrics.iter().fold(0.0, |max, &(_, _, _, m)| f64::max(max, m));
            
            run_avg_gpu_utils.push(util_sum / count);
            run_avg_gpu_powers.push(power_sum / count);
            run_avg_mems.push(mem_sum / count);
            run_peak_mems.push(peak_mem);
        } else {
            run_avg_gpu_utils.push(0.0);
            run_avg_gpu_powers.push(0.0);
            run_avg_mems.push(0.0);
            run_peak_mems.push(0.0);
        }
        
        println!("Run {} finished in {:.2} seconds.", run + 1, total_duration_for_run.as_secs_f64());

        if run == 0 {
            // Flush the epoch writer
            if let Some(mut wtr) = epoch_wtr {
                wtr.flush()?;
                println!("Epoch metrics saved to stats/rust_metrics.csv");
            }

            println!("\nCollected {} detailed metrics points (for CSV and summary)", run_detailed_metrics.len());
            let mut detailed_wtr = Writer::from_path("stats/rust_metrics_detailed.csv")?;
            detailed_wtr.write_record(&["time", "gpu_utilization_percent", "gpu_power_watts", "memory_mb"])?;
            for (time, util, power, mem) in run_detailed_metrics.iter() {
                detailed_wtr.write_record(&[format!("{:.2}", time), format!("{:.2}", util), format!("{:.2}", power), format!("{:.2}", mem)])?;
            }
            detailed_wtr.flush()?;
            println!("Detailed metrics saved to stats/rust_metrics_detailed.csv");
        }
    }

    let avg_total_time = run_total_times.iter().sum::<f64>() / run_total_times.len() as f64;
    let avg_final_accuracy = run_final_accuracies.iter().sum::<f64>() / run_final_accuracies.len() as f64;
    let avg_avg_epoch_time = run_avg_epoch_times.iter().sum::<f64>() / run_avg_epoch_times.len() as f64;
    let avg_gpu_util = run_avg_gpu_utils.iter().sum::<f64>() / run_avg_gpu_utils.len() as f64;
    let avg_gpu_power = run_avg_gpu_powers.iter().sum::<f64>() / run_avg_gpu_powers.len() as f64;
    let avg_mem_usage = run_avg_mems.iter().sum::<f64>() / run_avg_mems.len() as f64;
    let max_peak_mem = run_peak_mems.iter().fold(0.0, |max, &val| f64::max(max, val));
    let energy_joules = run_avg_gpu_powers.get(0).unwrap_or(&0.0) * run_total_times.get(0).unwrap_or(&0.0);

    println!("\n--- Benchmark Complete ---");
    println!("Average Total Time over {} runs: {:.2}s", NUM_RUNS, avg_total_time);
    println!("Average Final Accuracy over {} runs: {:.2}%", NUM_RUNS, avg_final_accuracy);
    println!("Average Epoch Time over {} runs: {:.2}s", NUM_RUNS, avg_avg_epoch_time);

    let mut summary_file = File::create("stats/rust_training_summary.txt")?;
    writeln!(summary_file, "Training Summary:")?;
    writeln!(summary_file, "Number of Runs: {}", NUM_RUNS)?;
    writeln!(summary_file, "Device: {}", if device.is_cuda() { "cuda" } else { "cpu" })?;

    for i in 0..NUM_RUNS {
        writeln!(summary_file, "\n--- Run {}/{} ---", i + 1, NUM_RUNS)?;
        writeln!(summary_file, "Total Time: {:.2} seconds", run_total_times[i])?;
        writeln!(summary_file, "Final Accuracy: {:.2}%", run_final_accuracies[i])?;
        writeln!(summary_file, "Average Epoch Time: {:.2} seconds", run_avg_epoch_times[i])?;
        if gpu_available_for_metrics {
            writeln!(summary_file, "Average GPU Utilization: {:.1}%", run_avg_gpu_utils[i])?;
            writeln!(summary_file, "Average GPU Power Draw: {:.1} W", run_avg_gpu_powers[i])?;
        }
        writeln!(summary_file, "Average Process Memory Usage: {:.1} MB", run_avg_mems[i])?;
        writeln!(summary_file, "Peak Process Memory Usage (this run): {:.2} MB", run_peak_mems[i])?;
    }

    writeln!(summary_file, "\n--- Averages over {} runs ---", NUM_RUNS)?;
    writeln!(summary_file, "Average Total Time: {:.2} seconds", avg_total_time)?;
    writeln!(summary_file, "Average Final Accuracy: {:.2}%", avg_final_accuracy)?;
    writeln!(summary_file, "Average Epoch Time: {:.2} seconds", avg_avg_epoch_time)?;
    if gpu_available_for_metrics {
        writeln!(summary_file, "Average GPU Utilization: {:.1}%", avg_gpu_util)?;
        writeln!(summary_file, "Average GPU Power Draw: {:.1} W", avg_gpu_power)?;
        writeln!(summary_file, "Approx Total GPU Energy Consumed (Run 1): {:.2} J", energy_joules)?;
    }
    writeln!(summary_file, "Average Process Memory Usage: {:.1} MB", avg_mem_usage)?;
    writeln!(summary_file, "Overall Peak Process Memory Usage: {:.2} MB", max_peak_mem)?;
    
    println!("Summary saved to stats/rust_training_summary.txt");
    println!("Rust benchmark finished.");
    Ok(())
}

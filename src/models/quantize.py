import torch
import torch.nn as nn
import time
import os

def quantize_model(model):
    """
    Applies Post-Training Dynamic Quantization to the model.
    Converts FP32 Linear layers to INT8 for faster CPU inference.
    """
    model.eval()
    model.to("cpu") # Quantization in PyTorch is currently CPU-centric
    
    # Apply dynamic quantization to nn.Linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear}, 
        dtype=torch.qint8
    )
    
    return quantized_model

def get_model_size_mb(model):
    """
    Calculates the size of the model in Megabytes by saving it to a temporary file.
    """
    torch.save(model.state_dict(), "temp_size.p")
    size_mb = os.path.getsize("temp_size.p") / 1e6
    os.remove("temp_size.p")
    return size_mb

def compare_model_sizes(fp32_model, int8_model):
    """
    Compares and prints the size reduction between full-precision and quantized models.
    """
    fp32_size = get_model_size_mb(fp32_model)
    int8_size = get_model_size_mb(int8_model)
    reduction = (1 - (int8_size / fp32_size)) * 100
    
    print("\\n--- Model Size Comparison (Edge Deployment) ---")
    print(f"FP32 Model Size : {fp32_size:.2f} MB")
    print(f"INT8 Model Size : {int8_size:.2f} MB")
    print(f"Size Reduction  : {reduction:.2f}%")
    
    return fp32_size, int8_size, reduction

def benchmark_inference(model, data_loader, num_batches=50, device="cpu"):
    """
    Measures the average inference latency (milliseconds per batch).
    """
    model.eval()
    model.to(device)
    
    latencies = []
    
    with torch.no_grad():
        # Warmup
        for i, (data, _) in enumerate(data_loader):
            data = data.to(device)
            _ = model(data)
            if i >= 5:
                break
                
        # Benchmark
        for i, (data, _) in enumerate(data_loader):
            data = data.to(device)
            
            start_time = time.perf_counter()
            _ = model(data)
            end_time = time.perf_counter()
            
            latencies.append((end_time - start_time) * 1000) # milliseconds
            
            if i >= num_batches:
                break
                
    mean_latency = sum(latencies) / len(latencies)
    p99_latency = sorted(latencies)[int(len(latencies)*0.99)]
    
    print(f"--- Inference Latency (Batch Size: {data_loader.batch_size}) ---")
    print(f"Mean Latency: {mean_latency:.2f} ms")
    print(f"P99 Latency : {p99_latency:.2f} ms")
    
    return mean_latency

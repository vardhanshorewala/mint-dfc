import torch
import os
import sys

# Add src directory to path to ensure imports work
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.eval import get_model_from_run
from src.samplers import get_data_sampler
from src.tasks import get_task_sampler

# Path to the trained model
run_path = "models/mixed_function/13a33f04-ec13-44b6-8635-7e2334022910"

# Load the model and configuration
model, conf = get_model_from_run(run_path)

# Extract model dimensions and batch size
n_dims = conf.model.n_dims
batch_size = conf.training.batch_size

# Create data sampler from configuration
data_sampler = get_data_sampler(conf.training.data, n_dims)

# Create task sampler from configuration
task_sampler = get_task_sampler(
    conf.training.task,
    n_dims,
    batch_size,
    **conf.training.task_kwargs
)

print("Model loaded successfully!")
print(f"Model name: {model.name}")
print(f"Model dimensions: {n_dims}")
print(f"Batch size: {batch_size}")
print(f"Data sampler: {conf.training.data}")
print(f"Task: {conf.training.task}")

# Example of how to use the model for inference
# Note: This is a basic example and may need adjustment based on your task
def run_inference(num_samples=5, n_points=10):
    # Sample data points
    xs = data_sampler.sample_xs(n_points, num_samples)
    
    # Get task instance
    task = task_sampler()
    
    # Get ground truth
    ys = task.evaluate(xs)
    
    # Get model predictions
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        predictions = model(xs.to(device), ys.to(device)).cpu()
    
    print(f"\nInference results (first {min(3, num_samples)} samples):")
    for i in range(min(3, num_samples)):
        print(f"Sample {i+1}:")
        print(f"  Ground truth: {ys[i, -1].item():.4f}")
        print(f"  Prediction: {predictions[i, -1].item():.4f}")
    
    # Calculate error
    metric = task.get_metric()
    error = metric(predictions, ys).mean().item()
    print(f"\nAverage error: {error:.4f}")

if __name__ == "__main__":
    run_inference() 
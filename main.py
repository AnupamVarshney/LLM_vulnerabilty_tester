import argparse
import logging
from utils.logger import Logger
from utils.storage import StorageManager
from models.loader import ModelLoader
from experiments.builder import ExperimentBuilder
from metrics.accuracy import AccuracyEvaluator
from metrics.robustness import AttackSuccessRate
from metrics.latency import LatencyEvaluator

def main():
    """
    Entry point for running LLM security experiments.
    Parses arguments, loads models/datasets, applies attacks, and evaluates results.
    """
    # Initialize logger
    Logger.setup_logger()

    # Argument parser for CLI execution
    parser = argparse.ArgumentParser(description="Run LLM Security Experiments")
    
    parser.add_argument("--model", type=str, required=True, help="Pretrained model name (e.g., meta-llama/Llama-2-7b)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., imdb, sst2)")
    parser.add_argument("--attack", type=str, required=True, help="Attack type (e.g., prompt_injection, adversarial, gradient_based, backdoor)")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization method (e.g., GPTQ, AWQ, SmoothQuant, BitsAndBytes-8bit)")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of examples to attack")

    args = parser.parse_args()

    logging.info(f"Running experiment: {args.attack} on {args.model} with {args.quantization or 'No Quantization'}")

    # Run experiment
    experiment = ExperimentBuilder(args.model, args.dataset, args.attack, args.quantization)
    results = experiment.run_experiment()

    if results["status"] != "success":
        logging.error("Experiment failed. Exiting.")
        return

    # Extract attack results
    adversarial_data = results["adversarial_data"]

    # Evaluate attack impact
    original_labels = [d["label"] for d in adversarial_data]
    perturbed_labels = [d.get("perturbed_label", d["label"]) for d in adversarial_data]  # Use perturbed labels if available

    accuracy_before = AccuracyEvaluator.calculate_accuracy(original_labels, original_labels)
    accuracy_after = AccuracyEvaluator.calculate_accuracy(original_labels, perturbed_labels)
    asr = AttackSuccessRate.calculate_asr(original_labels, perturbed_labels)

    # Measure latency impact
    latency_before = LatencyEvaluator.measure_latency(experiment.model, [d["text"] for d in adversarial_data], experiment.tokenizer)
    latency_after = LatencyEvaluator.measure_latency(experiment.model, [d["text"] for d in adversarial_data], experiment.tokenizer)
    # Store experiment results
    final_results = {
        "model": args.model,
        "quantization": args.quantization,
        "attack": args.attack,
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "accuracy_drop": round(accuracy_before - accuracy_after, 2),
        "attack_success_rate": asr,
        "latency_before_ms": latency_before,
        "latency_after_ms": latency_after,
        "latency_increase_ms": round(latency_after - latency_before, 2),
    }

    logging.info(f"Experiment Results: {final_results}")
    StorageManager.save_results_json(final_results)
    StorageManager.save_results_csv([final_results])

    logging.info("Experiment completed successfully!")

if __name__ == "__main__":
    main()
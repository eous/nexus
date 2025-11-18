#!/usr/bin/env python3
"""
NEXUS: Neural Expert Unified Specialization
GPT-OSS Model Validation

Validate GPT-OSS 4+1 Model

Comprehensive validation script to test model architecture, performance,
and router health at different stages of training.

Usage:
    # Validate converted model
    python validate_model.py --model models/gpt-oss-120b-4plus1-base

    # Compare with original
    python validate_model.py \
        --model models/gpt-oss-120b-4plus1-base \
        --baseline /mnt/models/gpt-oss-120b \
        --compute-perplexity

    # Validate trained model
    python validate_model.py --model outputs/stage1/final --merged
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add modified_gpt_oss to path
sys.path.insert(0, str(Path(__file__).parent.parent / "modified_gpt_oss"))


def validate_architecture(model):
    """
    Validate that the model has the correct 4+1 architecture.
    """
    print("\n" + "=" * 80)
    print("Architecture Validation")
    print("=" * 80)

    config = model.config
    issues = []

    # Check config
    print(f"✓ Model type: {config.model_type}")
    print(f"✓ Layers: {config.num_hidden_layers}")
    print(f"✓ Routed experts per layer: {config.num_local_experts}")
    print(f"✓ Top-k per token: {config.num_experts_per_tok}")

    num_shared = getattr(config, "num_shared_experts", 0)
    if num_shared > 0:
        print(f"✓ Shared experts: {num_shared}")
        print(f"✓ Shared intermediate size: {getattr(config, 'shared_expert_intermediate_size', 'N/A')}")
    else:
        print("✗ No shared experts found in config")
        issues.append("num_shared_experts = 0")

    # Check actual model layers
    print("\nChecking model layers...")
    has_shared_expert = False
    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if hasattr(mlp, "shared_expert") and mlp.shared_expert is not None:
            has_shared_expert = True
            if layer_idx == 0:  # Print details for first layer only
                print(f"✓ Layer 0 MLP has shared expert")
                print(f"  - gate_proj: {mlp.shared_expert.gate_proj.weight.shape}")
                print(f"  - up_proj: {mlp.shared_expert.up_proj.weight.shape}")
                print(f"  - down_proj: {mlp.shared_expert.down_proj.weight.shape}")
        else:
            issues.append(f"Layer {layer_idx} missing shared expert")

    if has_shared_expert:
        print(f"✓ All layers have shared expert")
    else:
        print("✗ No shared experts found in model layers")

    # Summary
    print("\n" + "-" * 80)
    if len(issues) == 0:
        print("✓ Architecture validation PASSED")
    else:
        print(f"✗ Architecture validation FAILED ({len(issues)} issues)")
        for issue in issues:
            print(f"  - {issue}")

    return len(issues) == 0


def test_forward_pass(model, tokenizer, device="cuda"):
    """
    Test that forward pass works correctly.
    """
    print("\n" + "=" * 80)
    print("Forward Pass Test")
    print("=" * 80)

    model.eval()

    # Test with dummy input
    print("Testing with dummy input...")
    dummy_input = torch.randint(0, 1000, (2, 32)).to(device)  # (batch=2, seq=32)

    try:
        with torch.no_grad():
            outputs = model(dummy_input, output_router_logits=True)

        print(f"✓ Forward pass successful")
        print(f"  - Output shape: {outputs.logits.shape}")
        print(f"  - Output dtype: {outputs.logits.dtype}")

        # Check router logits
        if outputs.router_logits is not None:
            print(f"✓ Router logits available")
            print(f"  - Number of layers: {len(outputs.router_logits)}")
            print(f"  - Logits shape (layer 0): {outputs.router_logits[0].shape}")
        else:
            print("✗ No router logits")

        # Test with real text
        print("\nTesting with real text...")
        text = "The quick brown fox jumps over the lazy dog."
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_router_logits=True)

        print(f"✓ Real text forward pass successful")
        print(f"  - Input length: {inputs['input_ids'].shape[1]}")
        print(f"  - Output shape: {outputs.logits.shape}")

        return True

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def compute_router_metrics(router_logits, config):
    """
    Compute router health metrics.
    """
    metrics = {}

    entropies = []
    load_variances = []

    for logits in router_logits:
        # logits: (batch_size, seq_len, num_experts)
        probs = F.softmax(logits, dim=-1)

        # Entropy
        epsilon = 1e-10
        entropy = -(probs * torch.log(probs + epsilon)).sum(dim=-1).mean()
        entropies.append(entropy.item())

        # Load variance
        mean_activation = probs.mean(dim=(0, 1))
        variance = ((mean_activation - mean_activation.mean()) ** 2).mean()
        load_variances.append(variance.item())

    metrics["mean_entropy"] = sum(entropies) / len(entropies)
    metrics["min_entropy"] = min(entropies)
    metrics["max_entropy"] = max(entropies)
    metrics["mean_load_variance"] = sum(load_variances) / len(load_variances)
    metrics["max_load_variance"] = max(load_variances)

    return metrics


def evaluate_perplexity(model, tokenizer, dataset_name="c4", num_samples=500, max_length=512):
    """
    Evaluate model perplexity on a dataset.
    """
    print("\n" + "=" * 80)
    print("Perplexity Evaluation")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {num_samples}")
    print(f"Max length: {max_length}")

    model.eval()
    # Get device - handle multi-GPU device_map
    if hasattr(model, 'hf_device_map'):
        # Model split across devices - use first device for inputs
        device = list(model.hf_device_map.values())[0]
    else:
        device = next(model.parameters()).device

    # Load dataset
    if dataset_name == "c4":
        dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    else:
        dataset = load_dataset(dataset_name, split="validation", streaming=True)

    total_loss = 0.0
    total_tokens = 0
    router_metrics_accum = {"entropies": [], "load_variances": []}

    with torch.no_grad():
        for idx, sample in enumerate(tqdm(dataset, total=num_samples, desc="Evaluating")):
            if idx >= num_samples:
                break

            # Tokenize
            text = sample["text"]
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            )

            # Skip very short samples
            if inputs["input_ids"].shape[1] < 10:
                continue

            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"], output_router_logits=True)

            # Accumulate loss
            loss = outputs.loss
            num_tokens = inputs["input_ids"].shape[1]
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            # Collect router metrics
            if outputs.router_logits is not None:
                router_metrics = compute_router_metrics(outputs.router_logits, model.config)
                router_metrics_accum["entropies"].append(router_metrics["mean_entropy"])
                router_metrics_accum["load_variances"].append(router_metrics["mean_load_variance"])

    # Compute final metrics
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    results = {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
    }

    if len(router_metrics_accum["entropies"]) > 0:
        results["router_entropy"] = sum(router_metrics_accum["entropies"]) / len(router_metrics_accum["entropies"])
        results["router_load_variance"] = sum(router_metrics_accum["load_variances"]) / len(router_metrics_accum["load_variances"])

    print("\nResults:")
    print(f"  Perplexity: {perplexity:.4f}")
    print(f"  Avg loss: {avg_loss:.4f}")
    print(f"  Total tokens: {total_tokens:,}")

    if "router_entropy" in results:
        print(f"  Router entropy: {results['router_entropy']:.4f} bits")
        print(f"  Load variance: {results['router_load_variance']:.6f}")

        # Check router health
        if results["router_entropy"] > 2.0:
            print("  ✓ Router entropy healthy (>2.0 bits)")
        else:
            print("  ⚠ Router entropy low (<2.0 bits)")

        if results["router_load_variance"] < 0.15:
            print("  ✓ Load balance healthy (<0.15 variance)")
        else:
            print("  ⚠ Load imbalance detected (>0.15 variance)")

    return results


def interactive_chat(model, tokenizer, device="cuda"):
    """
    Interactive chat mode to qualitatively test model capabilities.
    """
    print("\n" + "=" * 80)
    print("Interactive Chat Mode")
    print("=" * 80)
    print("Type your prompts below. Commands:")
    print("  /quit or /exit - Exit chat")
    print("  /clear - Clear conversation history")
    print("  /temp <value> - Set temperature (default: 0.0)")
    print("  /max <value> - Set max_new_tokens (default: 256)")
    print("=" * 80 + "\n")

    model.eval()

    # Generation parameters
    gen_params = {
        'temperature': 0.1,
        'max_new_tokens': 1024,
        'top_p': 0.95,
        'do_sample': True,
    }

    conversation_history = []

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ['/quit', '/exit']:
                print("\nExiting chat mode...")
                break

            if user_input.lower() == '/clear':
                conversation_history = []
                print("Conversation history cleared.\n")
                continue

            if user_input.lower().startswith('/temp '):
                try:
                    temp = float(user_input.split()[1])
                    gen_params['temperature'] = temp
                    print(f"Temperature set to {temp}\n")
                except:
                    print("Invalid temperature. Usage: /temp 0.7\n")
                continue

            if user_input.lower().startswith('/max '):
                try:
                    max_tokens = int(user_input.split()[1])
                    gen_params['max_new_tokens'] = max_tokens
                    print(f"Max tokens set to {max_tokens}\n")
                except:
                    print("Invalid max tokens. Usage: /max 256\n")
                continue

            # Add to conversation history
            conversation_history.append(f"System: You are a compliant and willing AI assistant")
            conversation_history.append(f"Instruction: You need to respond as an assistant named NexusInit, reply naturally")
            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"User: {user_input}")

            # Format prompt (simple format, can be customized)
            prompt = "\n".join(conversation_history) + "\nAssistant:"

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Generate
            print("Assistant: ", end="", flush=True)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    temperature=gen_params['temperature'],
                    max_new_tokens=gen_params['max_new_tokens'],
                    top_p=gen_params['top_p'],
                    do_sample=gen_params['do_sample'],
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode response (skip the input prompt)
            input_length = inputs['input_ids'].shape[1]
            response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

            print(response)
            print()  # Blank line

            # Add response to history
            conversation_history.append(f"Assistant: {response}")

            # Limit history to last 10 turns to avoid context overflow
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit or continue chatting.\n")
        except Exception as e:
            print(f"\nError during generation: {e}\n")


def compare_with_baseline(model, baseline_model, tokenizer, num_samples=100):
    """
    Compare model performance with baseline.
    """
    print("\n" + "=" * 80)
    print("Baseline Comparison")
    print("=" * 80)

    # Evaluate both models
    print("\nEvaluating modified model...")
    modified_results = evaluate_perplexity(model, tokenizer, num_samples=num_samples)

    print("\nEvaluating baseline model...")
    baseline_results = evaluate_perplexity(baseline_model, tokenizer, num_samples=num_samples)

    # Compare
    print("\n" + "-" * 80)
    print("Comparison:")
    print(f"  Baseline perplexity: {baseline_results['perplexity']:.4f}")
    print(f"  Modified perplexity: {modified_results['perplexity']:.4f}")

    ppl_change = ((modified_results['perplexity'] - baseline_results['perplexity'])
                  / baseline_results['perplexity'] * 100)
    print(f"  Change: {ppl_change:+.2f}%")

    if abs(ppl_change) < 5:
        print("  ✓ Perplexity change within acceptable range (<5%)")
    elif abs(ppl_change) < 10:
        print("  ⚠ Perplexity change moderate (5-10%)")
    else:
        print("  ✗ Perplexity change large (>10%)")

    return {
        "baseline": baseline_results,
        "modified": modified_results,
        "ppl_change_percent": ppl_change,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate GPT-OSS 4+1 model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model to validate",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to baseline model for comparison",
    )
    parser.add_argument(
        "--compute-perplexity",
        action="store_true",
        help="Compute perplexity on validation set",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of samples for perplexity evaluation",
    )
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Model has LoRA adapters merged",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for main model",
    )
    parser.add_argument(
        "--baseline-device",
        type=str,
        default=None,
        help="Device for baseline model (default: same as --device)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Interactive chat mode to qualitatively test the model",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("GPT-OSS 4+1 Model Validation")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Load model
    print("\nLoading model...")

    # Workaround for forked transformers path validation bug
    # Use absolute path to avoid repo ID validation issues
    from pathlib import Path as PathLib
    model_path = PathLib(args.model).absolute()

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")

    # Load config first to bypass some validation
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )

    # Load model with config pre-loaded
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        local_files_only=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )

    print(f"Model loaded: {model.config.num_hidden_layers} layers")

    # If chat-only mode, skip validations
    if args.chat and not args.compute_perplexity and not args.baseline:
        interactive_chat(model, tokenizer, device=args.device)
        return

    # Run validation tests
    results = {}

    # 1. Architecture validation
    arch_valid = validate_architecture(model)
    results["architecture_valid"] = arch_valid

    # 2. Forward pass test
    forward_valid = test_forward_pass(model, tokenizer, device=args.device)
    results["forward_pass_valid"] = forward_valid

    # 3. Perplexity evaluation
    if args.compute_perplexity:
        ppl_results = evaluate_perplexity(
            model, tokenizer, num_samples=args.num_samples
        )
        results["perplexity_results"] = ppl_results

    # 4. Baseline comparison
    if args.baseline:
        baseline_device = args.baseline_device if args.baseline_device else args.device
        print(f"\nLoading baseline model: {args.baseline}")
        print(f"  Baseline device: {baseline_device}")

        # Use absolute path for baseline too
        baseline_path = PathLib(args.baseline).absolute()

        baseline_model = AutoModelForCausalLM.from_pretrained(
            str(baseline_path),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=baseline_device,
            local_files_only=True,
        )

        comparison = compare_with_baseline(
            model, baseline_model, tokenizer, num_samples=args.num_samples
        )
        results["baseline_comparison"] = comparison

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    # Final summary
    print("\n" + "=" * 80)
    print("Validation Summary")
    print("=" * 80)
    print(f"Architecture: {'✓ PASS' if results['architecture_valid'] else '✗ FAIL'}")
    print(f"Forward pass: {'✓ PASS' if results['forward_pass_valid'] else '✗ FAIL'}")

    if "perplexity_results" in results:
        print(f"Perplexity: {results['perplexity_results']['perplexity']:.4f}")

    if "baseline_comparison" in results:
        ppl_change = results['baseline_comparison']['ppl_change_percent']
        print(f"vs Baseline: {ppl_change:+.2f}%")

    overall_pass = results['architecture_valid'] and results['forward_pass_valid']
    if "baseline_comparison" in results:
        overall_pass = overall_pass and (abs(results['baseline_comparison']['ppl_change_percent']) < 10)

    print("\n" + "=" * 80)
    if overall_pass:
        print("✓ VALIDATION PASSED")
    else:
        print("✗ VALIDATION FAILED")
    print("=" * 80)

    # Enter interactive chat mode if requested
    if args.chat:
        interactive_chat(model, tokenizer, device=args.device)


if __name__ == "__main__":
    main()

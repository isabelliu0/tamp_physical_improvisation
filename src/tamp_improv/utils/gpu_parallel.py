"""GPU parallel training manager for multiple policies."""

import os
from multiprocessing.pool import Pool
from typing import Any, Callable

import torch
import torch.multiprocessing as mp


class GPUParallelTrainer:
    """Manages parallel training of multiple policies across available GPUs."""

    def __init__(
        self,
        num_workers: int | None = None,
        gpu_ids: list[int] | None = None,
        use_cuda: bool = True,
    ):
        """Initialize parallel trainer."""
        self.use_cuda = use_cuda and torch.cuda.is_available()

        if self.use_cuda:
            self.num_gpus = torch.cuda.device_count()
            if gpu_ids is None:
                self.gpu_ids = list(range(self.num_gpus))
            else:
                self.gpu_ids = gpu_ids
                self.num_gpus = len(self.gpu_ids)
        else:
            self.num_gpus = 0
            self.gpu_ids = []

        if num_workers is None:
            self.num_workers = max(1, self.num_gpus)
        else:
            self.num_workers = num_workers

        print(f"Parallel Trainer initialized with {self.num_workers} workers")
        if self.use_cuda:
            print(f"Using {len(self.gpu_ids)} GPUs: {self.gpu_ids}")
        else:
            print("Running on CPU")

        self.pool: Pool | None = None
        # Initialize process pool if using multiprocessing
        if self.num_workers > 1 and self.use_cuda:
            # Set start method only if not already set
            try:
                mp.set_start_method("spawn")
            except RuntimeError:
                pass
            self.pool = mp.Pool(self.num_workers)

    def train_policies(
        self,
        policies_to_train: dict[str, tuple[Any, Any, Any]],
        train_function: Callable,
        **train_kwargs,
    ) -> dict[str, Any]:
        """Train multiple policies in parallel."""
        results = {}

        if self.num_workers <= 1 or len(policies_to_train) <= 1 or not self.use_cuda:
            # Sequential training (single worker or CPU)
            for policy_key, (policy, env, train_data) in policies_to_train.items():
                device = (
                    f"cuda:{self.gpu_ids[0]}"
                    if self.use_cuda and self.gpu_ids
                    else "cpu"
                )
                print(f"Training policy {policy_key} on {device}")

                # Set policy device
                if hasattr(policy, "config"):
                    policy.config.device = device

                # Train the policy
                result = train_function(
                    policy, env, train_data, policy_key=policy_key, **train_kwargs
                )
                results[policy_key] = result
        else:
            # Parallel training
            tasks = []
            for i, (policy_key, (policy, env, train_data)) in enumerate(
                policies_to_train.items()
            ):
                gpu_idx = self.gpu_ids[i % len(self.gpu_ids)]
                device = f"cuda:{gpu_idx}"
                if hasattr(policy, "config"):
                    policy.config.device = device
                tasks.append(
                    (policy_key, policy, env, train_data, train_kwargs, device)
                )

            # Create process pool and train in parallel
            if self.pool:
                async_results = []
                for task in tasks:
                    async_result = self.pool.apply_async(
                        _train_policy_process, (train_function, *task)
                    )
                    async_results.append((task[0], async_result))

                # Collect results
                for policy_key, async_result in async_results:
                    try:
                        results[policy_key] = async_result.get()
                    except Exception as e:
                        print(f"Error training policy {policy_key}: {e}")
                        results[policy_key] = None

        return results

    def close(self) -> None:
        """Clean up resources."""
        if self.pool:
            self.pool.close()
            self.pool.join()


def _train_policy_process(
    train_function, policy_key, policy, env, train_data, train_kwargs, device
):
    """Process function for training a policy in a separate process."""
    try:
        # Set CUDA device for this process
        if "cuda" in device:
            gpu_idx = device.split(":")[1]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            policy.config.device = "cuda:0"
            torch.cuda.set_device(0)

        # Train the policy
        print(f"Process for {policy_key} starting training on {device}")
        result = train_function(
            policy, env, train_data, policy_key=policy_key, **train_kwargs
        )

        # Save the model directly (don't use the policy.save method)
        save_path = None
        if "save_dir" in train_kwargs:
            safe_key = "".join(
                c if c.isalnum() or c in "-_" else "_" for c in policy_key
            )
            save_path = os.path.join(train_kwargs["save_dir"], f"policy_{safe_key}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if hasattr(policy, "model") and policy.model is not None:
                policy.model.save(save_path)
                return {"success": True, "saved_path": save_path}

        print(f"Process for {policy_key} completed training but no model was saved")
        return {"success": result}
    except Exception as e:
        print(f"Error in training process for {policy_key}: {e}")
        raise

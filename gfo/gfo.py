import torch
from torch.utils.data import DataLoader, Subset

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    top_k_accuracy_score,
)
import matplotlib.pyplot as plt


from pymoo.core.problem import Problem
from pymoo.core.callback import Callback

import jax
import jax.numpy as jnp
from jax import vmap, jit


class GFOProblem(Problem):
    def __init__(
        self,
        n_var=None,
        model=None,
        sample_size=None,
        dataset=None,
        block=False,
        codebook=None,
        orig_dims=None,
        set_model_state=None,
        device=None,
        criterion=None,
        test_loader=None,
        train_loader=None,
    ):
        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_ieq_constr=0,
            xl=-5.0,
            xu=5.0,
            vtype=float,
        )
        self.model = model
        self.sample_size = sample_size
        self.block = block  # Enable / disable block
        self.dataset = dataset
        self.test_loader = test_loader
        self.set_model_state = set_model_state
        self.device = device
        self.codebook = codebook
        self.orig_dims = orig_dims
        self.criterion = criterion
        if criterion is None:
            self.fitness_func = self.f1score_func
        elif criterion == "crossentropy":
            self.fitness_func = self.crossentropy_func
        elif criterion == "f1":
            self.fitness_func = self.f1score_func
        elif criterion == "top1":
            self.fitness_func = self.top1_func

        if train_loader is None:
            self.data_loader = self.data_sampler()
        else:
            self.data_loader = train_loader

    def data_sampler(self):
        # random_indices = np.random.uniform(low=0, high=len(self.dataset), size=self.sample_size).astype(int)
        random_indices = np.random.choice(
            np.arange(len(self.dataset)), size=self.sample_size, replace=False
        )
        random_dataset = Subset(self.dataset, random_indices)
        return DataLoader(random_dataset, batch_size=128, shuffle=True)

    def unblocker(self, blocked_params):

        unblocked_params = np.ones(self.orig_dims)
        for block_idx, indices in self.codebook.items():
            unblocked_params[indices] *= blocked_params[block_idx]

        return unblocked_params

    def unblocker_jax(self, blocked_params):

        # # Initialize B with zeros
        # B = jnp.zeros(10)

        # # Define a function to update B based on A and codebook
        # def update_B(b, a, indices):
        #     b = b.at[indices].set(a)
        #     return b

        # # Vectorize the function using vmap
        # vectorized_update_B = vmap(update_B, in_axes=(None, 0, 0))

        # # Prepare the indices and values for vmap
        # indices = [self.codebook[i] for i in range(len(blocked_params))]
        # values = blocked_params

        # # Use jit to compile the function
        # compiled_update_B = jit(vectorized_update_B)
        # Define the indices to be updated
        # Function to distribute A values into B according to the codebook
        # B = jnp.zeros(self.orig_dims)

        def copy_values(A, codebook):
            B = jnp.zeros(self.orig_dims)  # Initialize B with zeros
            for key, indices in codebook.items():
                B = B.at[indices].set(A)
            return B

        # Use vmap to efficiently apply the function across multiple copies of A
        parallel_copy = vmap(lambda x: copy_values(x, self.codebook))

        # Apply the vectorized function
        unblocked_params = parallel_copy(blocked_params)
        unblocked_params_np = jax.device_get(unblocked_params)
        print(unblocked_params_np.shape)

        return unblocked_params_np

    def crossentropy_func(self, model, data_loader, device):
        model.eval()
        fitness = 0

        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            fitness += torch.nn.functional.cross_entropy(output, target).item()

        return fitness / len(data_loader)

    def f1score_func(self, model, data_loader, device):
        model.eval()
        fitness = 0

        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=False)

            fitness += f1_score(
                y_true=target.cpu().numpy(), y_pred=pred.cpu().numpy(), average="macro"
            )

        return 1 - (fitness / len(data_loader))

    def top1_func(self, model, data_loader, device):
        model.eval()
        fitness = 0

        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=False)

            fitness += top_k_accuracy_score(
                y_true=target.cpu().numpy(), y_pred=pred.cpu().numpy(), k=1
            )

        return 1 - (fitness / len(data_loader))

    def test_func(self, X):
        uxi = X.copy()
        if self.block:
            uxi = self.unblocker(uxi)

        self.set_model_state(model=self.model, parameters=uxi)
        self.model.eval()

        fitness = 0
        with torch.no_grad():
            for idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)

                fitness += f1_score(
                    y_true=target.cpu().numpy(),
                    y_pred=pred.cpu().numpy(),
                    average="macro",
                )

        return fitness / len(self.test_loader)

    def _calc_pareto_front(self):
        return 0

    def _calc_pareto_set(self):
        return np.full(self.n_var, 0)

    def _evaluate(self, X, out, reset=True, *args, **kwargs):
        NP = X.shape[0]
        fout = np.zeros(NP)

        for i in range(NP):
            xi = X[i]
            uxi = xi.copy()
            if self.block:
                uxi = self.unblocker(uxi)

            self.set_model_state(model=self.model, parameters=uxi)

            fitness = self.fitness_func(
                model=self.model, data_loader=self.data_loader, device=self.device
            )
            fout[i] = fitness

        out["F"] = fout

    def scipy_fitness_func(self, X):

        uxi = X.copy()
        if self.block:
            uxi = self.unblocker(uxi)

        self.set_model_state(model=self.model, parameters=uxi)

        fitness = self.fitness_func(
            model=self.model, data_loader=self.data_loader, device=self.device
        )

        return fitness


class SOCallback(Callback):

    def __init__(self, k_steps=10, problem=None, csv_path=None, plt_path=None) -> None:
        super().__init__()
        self.k_steps = k_steps
        self.csv_path = csv_path
        self.plt_path = plt_path
        self.problem = problem

        self.data["opt_F"] = []
        self.data["pop_F"] = []
        self.data["n_evals"] = []

    def notify(self, algorithm):
        self.data["opt_F"].append(algorithm.opt.get("F")[0][0])
        self.data["pop_F"].append(algorithm.pop.get("F"))
        self.data["n_evals"].append(algorithm.evaluator.n_eval)

        if algorithm.n_iter % self.k_steps == 0:
            best_X = algorithm.opt.get("X")[0]

            NP = len(algorithm.pop)
            # algorithm.evaluator.n_eval += NP

            df = pd.read_csv(self.csv_path)
            # Define the new row as a dictionary
            new_row = {
                "n_step": algorithm.n_iter,
                "f_best": algorithm.opt.get("F")[0][0],
                "f_avg": algorithm.pop.get("F").mean(),
                "f_std": algorithm.pop.get("F").std(),
                "test_f1_best": algorithm.problem.test_func(best_X),
            }
            # Append the new row to the DataFrame
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            # Save the DataFrame back to the CSV file
            df.to_csv(self.csv_path, index=False)

            plt.plot(df["n_step"], df["f_best"], label="train")
            if algorithm.problem.criterion != "crossentropy":
                plt.plot(df["n_step"], 1 - df["test_f1_best"], label="test")
            plt.xlabel("Steps")
            # if self.criterion != "crossentropy":
            plt.ylabel("Error")
            plt.title(f"{algorithm.__class__.__name__}, {algorithm.problem.criterion}")
            plt.legend()
            plt.savefig(self.plt_path)
            plt.close()

    def scipy_func(self, intermediate_result):

        if intermediate_result.nit % self.k_steps == 0:
            best_X, best_F = intermediate_result.x, intermediate_result.fun
            df = pd.read_csv(self.csv_path)
            # Define the new row as a dictionary
            new_row = {
                "n_step": intermediate_result.nit,
                "f_best": best_F,
                "f_avg": intermediate_result.population_energies.mean(),
                "f_std": intermediate_result.population_energies.std(),
                "test_f1_best": self.problem.test_func(best_X),
            }
            # Append the new row to the DataFrame
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            # Save the DataFrame back to the CSV file
            df.to_csv(self.csv_path, index=False)

            plt.plot(df["n_step"], df["f_best"], label="train")
            if self.problem.criterion != "crossentropy":
                plt.plot(df["n_step"], 1 - df["test_f1_best"], label="test")
            plt.xlabel("Steps")
            # if self.criterion != "crossentropy":
            plt.ylabel("Error")
            plt.title(f"DE, {self.problem.criterion}")
            plt.legend()
            plt.savefig(self.plt_path)
            plt.close()

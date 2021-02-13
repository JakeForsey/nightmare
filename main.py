from concurrent import futures
from PIL import Image
import time

import nevergrad as ng
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")


def errors_to_image(a):
    out = (a - a.min()) / (a.max() - a.min()) * 255
    return out.astype(np.int)


class ImageSimilarityObjective:

    def __init__(self):
        target = Image.open("imgs\\tyrion.png")
        lowry = Image.open("imgs\\lowry.jpg").resize(target.size)
        scream = Image.open("imgs\\scream.jpg").resize(target.size)

        self.seed_array = np.concatenate([np.array(scream), np.array(lowry)], axis=1)
        self.target_array = np.array(target)
        self.patch_size = 30
        self.n_patches = 350
        self.iteration = 0
        self.loss_history = []
        self.timer = time.time()

    def initialise_parameters(self):
        h, w, c = self.seed_array.shape
        seed_x = ng.p.Array(
            init=np.random.randint(0, w - self.patch_size, self.n_patches),
            mutable_sigma=True,
        ).set_bounds(0, w - self.patch_size).set_integer_casting().set_mutation(sigma=100)
        seed_y = ng.p.Array(
            init=np.random.randint(0, h - self.patch_size, self.n_patches),
            mutable_sigma=True,
        ).set_bounds(0, h - self.patch_size).set_integer_casting().set_mutation(sigma=100)

        h, w, c = self.target_array.shape
        target_x = ng.p.Array(
            init=np.random.randint(0, w - self.patch_size, self.n_patches),
            mutable_sigma=True,
        ).set_bounds(0, w - self.patch_size).set_integer_casting().set_mutation(sigma=100)
        target_y = ng.p.Array(
            init=np.random.randint(0, h - self.patch_size, self.n_patches),
            mutable_sigma=True,
        ).set_bounds(0, h - self.patch_size).set_integer_casting().set_mutation(sigma=100)

        return ng.p.Tuple(seed_x, seed_y, target_x, target_y)

    def score(self, x):
        out = self.resolve(x)
        errors = self.errors(out)
        squared_errors = errors ** 2
        return squared_errors.mean()

    def errors(self, out):
        return np.abs(out.astype(np.float) - self.target_array.astype(np.float))

    def resolve(self, x):
        total = np.zeros(self.target_array.shape)
        count = np.zeros(self.target_array.shape)
        for x, y, xx, yy in np.stack(x).T.astype(np.int):
            total[yy: yy + self.patch_size, xx: xx + self.patch_size] += \
                self.seed_array[y: y + self.patch_size, x: x + self.patch_size]
            count[yy: yy + self.patch_size, xx: xx + self.patch_size] += 1

        total[total == 0] = np.nan
        count[count == 0] = np.nan

        avg = total / count
        return np.nan_to_num(avg).astype(np.int)

    def log(self, optimizer, candidate, value):
        recommendation = optimizer.current_bests["optimistic"].parameter
        self.loss_history.append(recommendation.loss)

        if self.iteration % 50 == 0:
            iterations_per_second = 50 / (time.time() - self.timer)
            self.timer = time.time()
            print(f"{self.iteration=}, {recommendation.loss=}, {iterations_per_second}")
            x = recommendation.value
            out = self.resolve(x)
            errors = self.errors(out)
            fig, axes = plt.subplots(3, 1, figsize=(16, 16))
            axes[0].imshow(np.concatenate([self.seed_array, self.target_array], axis=1))
            axes[0].axis("off")
            axes[1].imshow(np.concatenate([out, errors_to_image(errors)], axis=1))
            axes[1].axis("off")
            axes[2].plot(self.loss_history)
            axes[2].set_ylabel("Loss")
            axes[2].set_xlabel("Steps")
            try:
                plt.savefig(f"plots/out-{self.iteration}.png")
            except OSError:
                pass
            plt.close(fig)

        self.iteration += 1


def main():
    objective = ImageSimilarityObjective()
    parameters = objective.initialise_parameters()

    optimizer = ng.optimizers.NGOpt(parametrization=parameters, budget=50_000, num_workers=14)
    optimizer.register_callback("tell", objective.log)

    with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
        recommendation = optimizer.minimize(objective.score, executor=executor, batch_mode=False)


if __name__ == "__main__":
    main()

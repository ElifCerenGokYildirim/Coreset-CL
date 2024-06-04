import numpy as np
from .coresetmethod import CoresetMethod


class Random(CoresetMethod):
    def __init__(self, network, dst_train, args, fraction=0.5, random_seed=None, device=None, task_id = None, epochs=200, specific_model=None, balance=True,  **kwargs):
        super().__init__(network, dst_train, args, fraction, random_seed, device, task_id, **kwargs)
        self._device = device
        self.balance = balance
        self.replace = False
        self.n_train = len(dst_train)

    def select_balance(self):
        """The same sampling proportions were used in each class separately."""
        np.random.seed(self.random_seed)
        self.index = np.array([], dtype=np.int64)
        all_index = np.arange(self.n_train)
        for c in range(self.num_classes):
            c_index = (self.dst_train.labels == c)
            self.index = np.append(self.index,
                                   np.random.choice(all_index[c_index], round(self.fraction * c_index.sum().item()),
                                                    replace=self.replace))
        return self.index

    def select_no_balance(self):
        np.random.seed(self.random_seed)
        self.index = np.random.choice(np.arange(self.n_train), round(self.n_train * self.fraction),
                                      replace=self.replace)

        return  self.index

    def select(self, **kwargs):
        return {"indices": self.select_balance() if self.balance else self.select_no_balance()}

import copy

from .earlytrain import EarlyTrain
import torch
import numpy as np
from .selection_utils import euclidean
from .selection_utils.euclidean import euclidean_dist
#from ..nets.nets_utils import MyDataParallel


class Herding(EarlyTrain):
    def __init__(self, network, dst_train, args, fraction=0.5, random_seed=None, device=None, task_id = None, epochs=200,
                 specific_model="ResNet18", balance: bool = False, metric="euclidean", **kwargs):
        super().__init__(network, dst_train, args, fraction, random_seed, device, task_id, epochs=epochs, **kwargs)

        self._device = device
        self.task_id = task_id

        if metric == "euclidean":
            self.metric = euclidean_dist
        elif callable(metric):
            self.metric = metric
        else:
            self.metric = euclidean_dist
            self.run = lambda: self.finish_run()

            def _construct_matrix(index=None):
                data_loader = torch.utils.data.DataLoader(
                    self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                    batch_size=self.n_train if index is None else len(index), num_workers=self.args.workers)
                inputs, _ = next(iter(data_loader))
                return inputs.flatten(1).requires_grad_(False).to(self.args.device)

            self.construct_matrix = _construct_matrix

        self.balance = balance

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args["print_freq"] == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))

    def construct_matrix(self, index=None):
        #self._model = copy.deepcopy(self.model)
        self._model.eval()
        self._model.no_grad = True
        with torch.no_grad():
            with self._model.embedding_recorder:
                sample_num = self.n_train if index is None else len(index)
                matrix = torch.zeros([sample_num, self.emb_dim], requires_grad=False).to(self._device)

                data_loader = torch.utils.data.DataLoader(self.dst_train if index is None else
                                            torch.utils.data.Subset(self.dst_train, index),
                                            batch_size=self.args["selection_batch"],
                                            num_workers=self.args["workers"])

                for i, (_, inputs, _) in enumerate(data_loader):
                    inputs = inputs.to(self._device)
                    self._model(inputs)
                    matrix[i * self.args["selection_batch"]:min((i + 1) * self.args["selection_batch"], sample_num)] = self._model.embedding_recorder.embedding["features"]

        self._model.no_grad = False
        return matrix

    def before_run(self):
        self._model = copy.deepcopy(self.model)
        self.emb_dim = self._model.get_last_layer().in_features

    def herding(self, matrix, budget: int, index=None):

        sample_num = matrix.shape[0]

        if budget < 0:
            raise ValueError("Illegal budget size.")
        elif budget > sample_num:
            budget = sample_num

        indices = np.arange(sample_num)
        with torch.no_grad():
            mu = torch.mean(matrix, dim=0)
            select_result = np.zeros(sample_num, dtype=bool)

            for i in range(budget):
                #if i % self.args["print_freq"] == 0:
                    #print("| Selecting [%3d/%3d]" % (i + 1, budget))
                dist = self.metric(((i + 1) * mu - torch.sum(matrix[select_result], dim=0)).view(1, -1),
                                   matrix[~select_result])
                p = torch.argmax(dist).item()
                p = indices[~select_result][p]
                select_result[p] = True
        if index is None:
            index = indices
        return index[select_result]

    def finish_run(self):

        if self.balance:
            selection_result = np.array([], dtype=np.int32)
            start_label = self.task_id * 2
            end_label = (self.task_id + 1) * 2
            for c in range(start_label, end_label):
                class_index = np.arange(self.n_train)[self.dst_train.labels == c]

                selection_result = np.append(selection_result, self.herding(self.construct_matrix(class_index),
                        budget=round(self.fraction * len(class_index)), index=class_index))
        else:
            selection_result = self.herding(self.construct_matrix(), budget=self.coreset_size)
        return {"indices": selection_result}

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result


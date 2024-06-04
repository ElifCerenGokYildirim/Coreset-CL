import os
import numpy as np
import torch
import  json
from enum import Enum

acc_matrix = []

#acc = np.array([[89.8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                #[68.8, 75.4, 0, 0, 0, 0, 0, 0, 0, 0],
                #[63.1, 56.1, 84.2, 0, 0, 0, 0, 0, 0, 0],
                #[58.4, 49.2, 70.3, 76.0, 0, 0, 0, 0, 0, 0],
                #[58.1, 47.3, 68.5, 63.8, 75.4, 0, 0, 0, 0, 0],
                #[55.0, 45.2, 64.9, 57.5, 68.6, 64.8, 0, 0, 0, 0],
                #[57.6, 43.2, 64.8, 52.8, 65.3, 56.7, 67.3, 0, 0, 0],
                #[56.7, 42.6, 61.2, 52.0, 60.7, 50.9, 61.6, 59.6, 0, 0],
                #[56.7, 42.1, 61.4, 49.3, 59.9, 48.5, 58.8, 53.0, 54.6, 0],
                #[55.9, 40.7, 60.5, 48.4, 57.6, 49.1, 58.1, 49.1, 49.9, 54.6]])

class ConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)


def clear_bwt():
    acc_matrix.clear()


def create_bwt_matrix():
    bwt = np.zeros(10)
    acc_array = np.zeros((10,10), dtype=int)
    acc_array[np.tril_indices(10, k=0)] = acc_matrix

    for task_id in range(1, 10):
        for t in range(task_id):
            bwt[task_id] += (acc_array[task_id, t] - acc_array[t, t]) / task_id

    print("BWT: ", bwt[-1])

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy_cnn(y_pred, y_true, nb_old, increment=2):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )
    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
        acc_matrix.append(all_acc[label])

    print("acc", acc_matrix)


    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc

def accuracy_cnn_foster(y_pred, y_true, nb_old, increment=2):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )
    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
        
    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc


def accuracy_nme(y_pred, y_true, nb_old, increment=2):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
        #amk_acc.append(all_acc[label])
        #print(amk_acc)

        #for j in range(1,10):
            #for i in range(j):
                #x = all_acc[label][i]
            #acc[j-1][i] = x


        #print("ACC", acc)

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc



def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

def save_fc(args, model):
    _path = os.path.join(args['logfilename'], "fc.pt")
    if len(args['device']) > 1: 
        fc_weight = model._network.fc.weight.data    
    else:
        fc_weight = model._network.fc.weight.data.cpu()
    torch.save(fc_weight, _path)

    _save_dir = os.path.join(f"./results/fc_weights/{args['prefix']}")
    os.makedirs(_save_dir, exist_ok=True)
    _save_path = os.path.join(_save_dir, f"{args['csv_name']}.csv")
    with open(_save_path, "a+") as f:
        f.write(f"{args['time_str']},{args['model_name']},{_path} \n")

def save_model(args, model):
    #used in PODNet
    _path = os.path.join(args['logfilename'], "model.pt")
    if len(args['device']) > 1:
        weight = model._network   
    else:
        weight = model._network.cpu()
    torch.save(weight, _path)

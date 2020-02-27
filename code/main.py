"""
Fairness experiments
"""


import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader


N_HIDDEN = 64


class MLP(nn.Module):
    def __init__(self, n_inp, n_hidden, n_out):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(n_inp, n_hidden), nn.Tanh(),)
        self.l2 = nn.Linear(n_hidden, n_out)

    def forward(self, inp):
        rep = self.l1(inp)
        out = self.l2(rep)
        return rep, out


class ScalarFeature(object):
    def __init__(self, name):
        self.name = name

    def __call__(self, value):
        return np.array([float(value)])


class CatFeature(object):
    def __init__(self, name, values):
        self.name = name
        self.values = values

    def __call__(self, value):
        out = np.zeros(len(self.values))
        out[self.values.index(value)] = 1
        return out


def load(fname, hide_features=("sex", "race")):
    n_features = 15
    types = [set() for _ in range(n_features)]

    with open(fname) as reader:
        for line in reader:
            values = line.strip().split(",")
            for i, value in enumerate(values):
                types[i].add(value.strip())

    features = [
        ScalarFeature("age"),
        CatFeature("workclass", sorted(types[1])),
        ScalarFeature("fnlwgt"),
        CatFeature("education", sorted(types[3])),
        ScalarFeature("education-num"),
        CatFeature("marital-status", sorted(types[5])),
        CatFeature("occupation", sorted(types[6])),
        CatFeature("relationship", sorted(types[7])),
        CatFeature("race", sorted(types[8])),
        CatFeature("sex", sorted(types[9])),
        ScalarFeature("capital-gain"),
        ScalarFeature("capital-loss"),
        ScalarFeature("hours-per-week"),
        CatFeature("native-country", sorted(types[13])),
    ]

    rows = []
    ys = []
    with open(fname) as reader:
        for line in reader:
            values = line.strip().split(",")
            if len(values) < n_features:
                continue
            rows.append(
                np.concatenate(
                    [
                        feature(val.strip())
                        for feature, val in zip(features, values[:-1])
                        if feature.name not in hide_features
                    ]
                )
            )
            ys.append(0 if values[-1].strip() == "<=50K" else 1)
    xs = np.array(rows, dtype=np.float32)
    ys = np.array(ys)
    mean = np.mean(xs, axis=0, keepdims=True)
    std = np.std(xs, axis=0, keepdims=True)
    xs = (xs - mean) / std
    return xs, mean, std, ys, features


def train(args):
    xs, mean, std, ys, features = load(args.train_data)
    loader = DataLoader(list(zip(xs, ys)), batch_size=100, shuffle=True)

    model = MLP(xs.shape[1], N_HIDDEN, 2)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.0001)

    for i in range(10):
        epoch_loss = 0
        epoch_acc = 0
        batch_count = 0
        for batch_xs, batch_ys in loader:
            _, pred = model(batch_xs)
            loss = loss_fn(pred, batch_ys)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            acc = (pred.argmax(dim=1) == batch_ys).float().mean().item()
            epoch_acc += acc
            batch_count += 1
        print(epoch_loss / batch_count)
        print(epoch_acc / batch_count)
        print()

    torch.save(model.state_dict(), args.save_model)


def names(features):
    out = []
    for feature in features:
        if isinstance(feature, ScalarFeature):
            out.append(feature.name)
        elif isinstance(feature, CatFeature):
            for value in feature.values:
                out.append(feature.name + ":" + value)
        else:
            assert False
    return out


def compute_derived_features(names, masks):
    n_comb = masks.shape[1] * (masks.shape[1] - 1) // 2
    conjunctions = np.zeros((masks.shape[0], n_comb))
    disjunctions = np.zeros((masks.shape[0], n_comb))
    conj_names = []
    disj_names = []
    counter = 0
    for i in range(masks.shape[1]):
        for j in range(i + 1, masks.shape[1]):
            conjunctions[:, counter] = np.minimum(masks[:, i], masks[:, j])
            disjunctions[:, counter] = np.maximum(masks[:, i], masks[:, j])
            conj_names.append(names[i] + " AND " + names[j])
            disj_names.append(names[i] + " OR " + names[j])
            counter += 1
    return disj_names, disjunctions


def analyze(args):
    xs, _, _, ys, features = load(args.train_data)
    n_ref = 2000
    ref_xs = xs[:n_ref, :]
    ref_ys = ys[:n_ref]

    feature_names = names(features)
    feature_masks = ref_xs > 0
    derived_feature_names, derived_feature_masks = compute_derived_features(
        feature_names, feature_masks,
    )

    all_feature_names = feature_names + derived_feature_names
    all_feature_masks = np.concatenate((feature_masks, derived_feature_masks), axis=1)
    # all_feature_names = feature_names
    # all_feature_masks = feature_masks

    loader = DataLoader(list(zip(ref_xs, ref_ys)), batch_size=100)

    model = MLP(xs.shape[1], N_HIDDEN, 2)
    model.load_state_dict(torch.load(args.save_model))
    neuron_masks = np.zeros((n_ref, N_HIDDEN), dtype=np.byte)
    for i_batch, (batch_xs, batch_ys) in enumerate(loader):
        with torch.no_grad():
            reps, preds = model(batch_xs)
        reps = reps.cpu().numpy()
        pred_labels = preds.argmax(axis=1)
        acc = (pred_labels == batch_ys).float().mean().item()
        active = reps > 0
        neuron_masks[i_batch * 100 : (i_batch + 1) * 100, :] = active

    out_weights = list(model.parameters())[-2].detach().numpy()
    assert out_weights.shape == (2, N_HIDDEN)
    explain(list(range(N_HIDDEN)), neuron_masks, all_feature_masks, all_feature_names)


def explain(neurons, neuron_masks, all_feature_masks, all_feature_names):
    for i_neuron in neurons:
        neuron_mask = neuron_masks[:, i_neuron]
        intersection = np.minimum(neuron_mask[:, np.newaxis], all_feature_masks).sum(
            axis=0
        )
        union = np.maximum(neuron_mask[:, np.newaxis], all_feature_masks).sum(axis=0)
        iou = intersection / union
        best = np.argmax(iou)
        if all_feature_masks[:, best].mean() in (0, 1):
            continue
        if iou[best] < 0.5:
            continue
        print(i_neuron, best, iou[best])
        print(neuron_mask.mean())
        print(all_feature_masks[:, best].mean())
        print(all_feature_names[best])
        print()


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("mode", choices=["train", "analyze"])
    parser.add_argument("--save_model", default="models/model.pt")
    parser.add_argument("--train_data", default="data/train.csv")
    parser.add_argument("--test_data", default="data/test.csv")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "analyze":
        analyze(args)

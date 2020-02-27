"""
Fairness experiments
"""

import os

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from tqdm import trange


class MLP(nn.Module):
    def __init__(self, n_inp, n_hidden, n_out):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(n_inp, n_hidden), nn.Tanh(),)
        self.l2 = nn.Linear(n_hidden, n_out)

    def forward(self, inp):
        rep = self.l1(inp)
        out = self.l2(rep).squeeze(1)
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

    xs = []
    xs_hidden = []
    ys = []
    with open(fname) as reader:
        for line in reader:
            values = line.strip().split(",")
            if len(values) < n_features:
                continue
            xs_hidden.append(
                np.concatenate(
                    [
                        feature(val.strip())
                        for feature, val in zip(features, values[:-1])
                        if feature.name not in hide_features
                    ]
                )
            )
            xs.append(
                np.concatenate(
                    [
                        feature(val.strip())
                        for feature, val in zip(features, values[:-1])
                    ]
                )
            )
            ys.append(0 if values[-1].strip() == "<=50K" else 1)

    xs = np.array(xs, dtype=np.float32)
    xs_hidden = np.array(xs_hidden, dtype=np.float32)
    ys = np.array(ys)

    mean = np.mean(xs, axis=0, keepdims=True)
    std = np.std(xs, axis=0, keepdims=True)
    xs = (xs - mean) / std

    mean_hidden = np.mean(xs_hidden, axis=0, keepdims=True)
    std_hidden = np.std(xs_hidden, axis=0, keepdims=True)
    xs_hidden = (xs_hidden - mean_hidden) / std_hidden

    return xs, xs_hidden, ys, features


def train(args):
    xs, xs_hidden, ys, features = load(args.train_data)
    loader = DataLoader(
        list(zip(xs_hidden, ys)), batch_size=args.batch_size, shuffle=True
    )

    model = MLP(xs_hidden.shape[1], args.n_hidden, 1)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=0.0001)

    ranger = trange(args.epochs, desc=f"epoch 0")
    for i in ranger:
        epoch_loss = 0
        epoch_acc = 0
        batch_count = 0
        for batch_xs, batch_ys in loader:
            _, pred = model(batch_xs)
            loss = loss_fn(pred, batch_ys.float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            acc = ((pred > 0) == batch_ys).float().mean().item()
            epoch_acc += acc
            batch_count += 1

        avg_loss = epoch_loss / batch_count
        avg_acc = epoch_acc / batch_count
        ranger.set_description(f"epoch {i} loss: {avg_loss:.3f}, acc: {avg_acc:.3f}")

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


def compute_features(base_names, base_masks, max_length):
    names = base_names[:]
    masks = [base_masks[:, i] for i in range(len(names))]

    for i in range(max_length - 1):
        masks_to_add = []
        names_to_add = []

        for name, mask in zip(names, masks):
            for j, name_to_add in enumerate(base_names):
                if name_to_add == name:
                    continue
                mask_to_add = base_masks[:, j]
                disj_mask = mask | mask_to_add
                disj_name = f"({name}) OR ({name_to_add})"

                conj_mask = mask & mask_to_add
                conj_name = f"({name}) AND ({name_to_add})"

                masks_to_add.append(disj_mask)
                names_to_add.append(disj_name)

        names.extend(names_to_add)
        masks.extend(masks_to_add)

    masks = np.stack(masks, axis=1)
    return names, masks


def mask_threshold(feats, threshold):
    thresholds = np.quantile(feats, 1 - threshold, axis=0, keepdims=True)
    return feats > thresholds


def analyze(args):
    xs, xs_hidden, ys, features = load(args.train_data)

    ref_xs = xs[: args.n_explain, :]
    ref_xs_hidden = xs_hidden[: args.n_explain, :]
    ref_ys = ys[: args.n_explain]

    feature_names = names(features)
    #  feature_masks = mask_threshold(ref_xs, args.feature_threshold)
    feature_masks = ref_xs > 0
    feature_names, feature_masks = compute_features(
        feature_names, feature_masks, args.max_formula_length
    )

    loader = DataLoader(list(zip(ref_xs_hidden, ref_ys)), batch_size=args.batch_size)

    model = MLP(xs_hidden.shape[1], args.n_hidden, 1)
    model.load_state_dict(torch.load(args.save_model))

    accs = []

    # Get neuron activations
    neuron_acts = np.zeros((args.n_explain, args.n_hidden), dtype=np.float32)
    for i_batch, (batch_xs, batch_ys) in enumerate(loader):
        with torch.no_grad():
            reps, preds = model(batch_xs)
        reps = reps.cpu().numpy()
        pred_labels = preds > 0
        acc = (pred_labels == batch_ys).float().mean().item()
        accs.append(acc)

        start = i_batch * args.batch_size
        end = start + args.batch_size
        neuron_acts[start:end] = reps

    print(f"accuracy: {np.mean(accs):.3f}")
    neuron_masks = mask_threshold(neuron_acts, args.neuron_threshold)

    final_weight = model.l2.weight.detach().numpy()
    return explain(
        list(range(args.n_hidden)),
        neuron_masks,
        feature_masks,
        feature_names,
        final_weight,
    )


def explain(neurons, neuron_masks, feature_masks, feature_names, final_weight):
    records = []
    for i_neuron in neurons:
        neuron_mask = neuron_masks[:, i_neuron]
        intersection = (feature_masks & neuron_mask[:, np.newaxis]).sum(axis=0)
        union = (feature_masks | neuron_mask[:, np.newaxis]).sum(axis=0)

        iou = intersection / union
        best = np.argmax(iou)

        if feature_masks[:, best].mean() in {0, 1}:
            continue

        # Compute final weight
        weight = final_weight[0, i_neuron]
        records.append(
            {
                "neuron": i_neuron,
                "feature": feature_names[best],
                "iou": iou[best],
                "weight": weight,
            }
        )
        print(
            f"neuron {i_neuron:03d}: feature {feature_names[best]} (iou {iou[best]:.3f}, weight {weight:.3f})"
        )

    return pd.DataFrame(records)


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("mode", choices=["train", "analyze"])
    parser.add_argument("--n_hidden", default=64, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--n_explain", default=2000, type=int)
    parser.add_argument("--neuron_threshold", default=0.5, type=float)
    parser.add_argument("--feature_threshold", default=0.5, type=float)
    parser.add_argument("--max_formula_length", default=2, type=int)
    parser.add_argument("--save_model", default="models/model.pt")
    parser.add_argument("--save_analysis", default="analysis/neurons.csv")
    parser.add_argument("--train_data", default="data/train.csv")
    parser.add_argument("--test_data", default="data/test.csv")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "analyze":
        df = analyze(args)
        os.makedirs(os.path.split(args.save_analysis)[0], exist_ok=True)
        df.to_csv(args.save_analysis, index=False)

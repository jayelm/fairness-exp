"""
Fairness experiments
"""

import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from tqdm import trange
import formula as F


class Adversary(nn.Module):
    def __init__(self, n_inp, n_hidden, n_out):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_inp, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_out)
        )

    def forward(self, inp):
        return self.mlp(inp)


class MLP(nn.Module):
    def __init__(self, n_inp, n_hidden, n_out):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(n_inp, n_hidden), nn.ReLU(),)
        self.l2 = nn.Linear(n_hidden, n_out)

    def forward(self, inp):
        rep = self.l1(inp)
        out = self.l2(rep).squeeze(1)
        return rep, out


class ScalarFeature(object):
    def __init__(self, name):
        self.name = name
        self.type = "scalar"

    def __call__(self, value):
        return np.array([float(value)])


class CatFeature(object):
    def __init__(self, name, values):
        self.name = name
        self.values = values
        self.type = "categorical"

    def __call__(self, value):
        out = np.zeros(len(self.values))
        out[self.values.index(value)] = 1
        return out


def load(
    fname,
    hide_features=(),
    remove_features=("fnlwgt", "education-num"),
    protected_features=("sex")
    #  fname, hide_features=(), remove_features=("fnlwgt", "education-num")
):
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
    xs_heldout = []
    xs_prot = []
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
                        and feature.name not in remove_features
                    ]
                )
            )
            if hide_features:
                xs_heldout.append(
                    np.concatenate(
                        [
                            feature(val.strip())
                            for feature, val in zip(features, values[:-1])
                            if feature.name in hide_features
                            and feature.name not in remove_features
                        ]
                    )
                )
            else:
                xs_heldout.append([])
            if protected_features:
                xs_prot.append(
                    np.concatenate(
                        [
                            feature(val.strip())
                            for feature, val in zip(features, values[:-1])
                            if feature.name in protected_features
                            and feature.name not in remove_features
                        ]
                    )
                )
            else:
                xs_prot.append([])
            xs.append(
                np.concatenate(
                    [
                        feature(val.strip())
                        for feature, val in zip(features, values[:-1])
                        if feature.name not in remove_features
                    ]
                )
            )
            ys.append(0 if values[-1].strip() == "<=50K" else 1)

    features_cleaned = [f for f in features if f.name not in remove_features]

    xs = np.array(xs, dtype=np.float32)
    xs_hidden = np.array(xs_hidden, dtype=np.float32)
    xs_prot = np.array(xs_prot, dtype=np.float32)
    xs_heldout = np.array(xs_heldout, dtype=np.float32)
    ys = np.array(ys)

    mean = np.mean(xs, axis=0, keepdims=True)
    std = np.std(xs, axis=0, keepdims=True)
    xs = (xs - mean) / std

    mean_hidden = np.mean(xs_hidden, axis=0, keepdims=True)
    std_hidden = np.std(xs_hidden, axis=0, keepdims=True)
    xs_hidden = (xs_hidden - mean_hidden) / std_hidden

    #  mean_heldout = np.mean(xs_heldout, axis=0, keepdims=True)
    #  std_heldout = np.std(xs_heldout, axis=0, keepdims=True)
    #  xs_heldout = (xs_heldout - mean_heldout) / std_heldout

    return xs, xs_hidden, xs_heldout, xs_prot, ys, features_cleaned


def train(args):
    xs, xs_hidden, xs_heldout, xs_prot, ys, features = load(args.train_data)
    feature_names = names(features)
    loader = DataLoader(
        list(zip(xs_hidden, ys, xs_heldout, xs_prot)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = MLP(xs_hidden.shape[1], args.n_hidden, 1)
    adversary = Adversary(2, 2, xs_prot.shape[1])
    a_loss_fn = nn.BCEWithLogitsLoss()
    a_opt = optim.Adam(list(adversary.parameters()), lr=0.0001)
    a_params = list(adversary.named_parameters())

    loss_fn = nn.BCEWithLogitsLoss()
    params = list(model.named_parameters())
    opt = optim.Adam(list(model.parameters()), lr=0.001)
    scheduler = optim.lr_scheduler.LambdaLR(opt, lambda epoch: 1 / (epoch + 1))

    ranger = trange(args.epochs, desc=f"epoch 0")
    for i in ranger:
        epoch_loss = 0
        epoch_acc = 0
        adversary_acc = 0
        adversary_loss = 0
        batch_count = 0
        all_preds = []
        all_ys = []
        all_xs = []
        for batch_i, (batch_xs, batch_ys, batch_xs_heldout, batch_xs_prot) in enumerate(
            loader
        ):
            _, pred = model(batch_xs)

            loss = loss_fn(pred, batch_ys.float())

            a_inp = torch.stack((pred, batch_ys.float()), 1)
            a_pred = adversary(a_inp)
            # Unwrap
            protected = batch_xs_prot.view(-1)
            a_pred = a_pred.view(-1)
            a_loss = a_loss_fn(a_pred, protected)
            a_acc = ((a_pred > 0) == protected).float().mean().item()

            all_preds.append((pred > 0).cpu().numpy())
            all_ys.append((batch_ys).numpy())
            all_xs.append(batch_xs.numpy())

            a_loss.backward(retain_graph=True)

            protect_grad = {name: p.grad.clone() for (name, p) in params}

            # Update the model
            opt.zero_grad()
            a_opt.zero_grad()
            loss.backward(retain_graph=True)
            epoch_loss += loss.item()

            for name, p in params:
                unit_protect = protect_grad[name] / (
                    torch.norm(protect_grad[name]) + np.finfo(np.float32).tiny
                )
                # Modify graadients
                if args.debias:
                    p.grad -= (p.grad * unit_protect).sum() * unit_protect
                    alpha = np.sqrt(i + 1)
                    p.grad -= alpha * protect_grad[name]

            opt.step()

            # Update the adversary
            a_opt.zero_grad()
            opt.zero_grad()
            a_loss.backward()
            a_opt.step()

            adversary_acc += a_acc
            adversary_loss += a_loss.item()

            acc = ((pred > 0) == batch_ys).float().mean().item()
            epoch_acc += acc
            batch_count += 1

        all_preds = np.concatenate(all_preds)
        all_xs = np.concatenate(all_xs)
        all_ys = np.concatenate(all_ys)
        report_bias_stats(all_xs, all_ys, all_preds, feature_names)

        avg_loss = epoch_loss / batch_count
        avg_a_loss = adversary_loss / batch_count
        avg_acc = epoch_acc / batch_count
        avg_a_acc = adversary_acc / batch_count
        desc = f"epoch {i} loss: {avg_loss:.3f}, acc: {avg_acc:.3f}, adv loss {avg_a_loss:.3f}, adv acc: {avg_a_acc:.3f}"
        ranger.set_description(desc)

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


def mask_threshold(feats, threshold):
    thresholds = np.quantile(feats, 1 - threshold, axis=0, keepdims=True)
    return feats > thresholds


def report_bias_stats(xs, ys, preds, feature_names):
    # TPs/FNs
    fi = feature_names.index("sex:Female")
    mi = feature_names.index("sex:Male")

    is_f = xs[:, fi] > 0
    is_m = xs[:, mi] > 0
    n_female = is_f.sum()
    n_male = is_m.sum()
    print(f"n female: {n_female} n male: {n_male}")

    print(f"female acc: {(ys[is_f] == preds[is_f]).mean()}")
    print(f"male acc: {(ys[is_m] == preds[is_m]).mean()}")

    cm = confusion_matrix(ys[is_f], preds[is_f])
    tn, fp, fn, tp = cm.ravel()
    fp = fp / n_female
    fn = fn / n_female
    print(f"female error rates: FP {fp:.3f} FN {fn:.3f}")

    cm = confusion_matrix(ys[is_m], preds[is_m])
    tn, fp, fn, tp = cm.ravel()
    fp = fp / n_male
    fn = fn / n_male
    print(f"male error rates: FP {fp:.3f} FN {fn:.3f}")


def analyze(args):
    xs, xs_hidden, xs_heldout, xs_prot, ys, features = load(args.train_data)

    ref_xs = xs[: args.n_explain, :]
    ref_xs_hidden = xs_hidden[: args.n_explain, :]
    ref_ys = ys[: args.n_explain]
    feature_names = names(features)
    assert len(feature_names) == xs.shape[1]
    feature_masks = mask_threshold(ref_xs, args.feature_threshold)
    #  feature_masks = ref_xs > 0
    #  divf = feature_names.index('marital-status:Divorced')

    loader = DataLoader(
        list(zip(ref_xs_hidden, ref_ys)), batch_size=args.batch_size, shuffle=False
    )

    model = MLP(xs_hidden.shape[1], args.n_hidden, 1)
    model.load_state_dict(torch.load(args.save_model))

    accs = []
    preds = []

    # Get neuron activations
    neuron_acts = np.zeros((len(ref_xs), args.n_hidden), dtype=np.float32)
    for i_batch, (batch_xs, batch_ys) in enumerate(loader):
        with torch.no_grad():
            reps, preds = model(batch_xs)
        reps = reps.cpu().numpy()
        pred_labels = preds > 0
        preds.append(pred_labels.cpu().numpy())
        acc = (pred_labels == batch_ys).float().mean().item()
        accs.append(acc)

        start = i_batch * args.batch_size
        end = start + args.batch_size
        neuron_acts[start:end] = reps
    preds = np.concatenate(preds)

    report_bias_stats(ref_xs, ref_ys, preds, feature_names)

    print(f"accuracy: {np.mean(accs):.3f}")
    neuron_masks = neuron_acts > 0

    final_weight = model.l2.weight.detach().numpy()
    comp_df, prim_df = explain(
        list(range(args.n_hidden)),
        neuron_masks,
        feature_masks,
        feature_names,
        final_weight,
        args,
    )
    os.makedirs(os.path.split(args.save_analysis)[0], exist_ok=True)
    comp_df.to_csv(args.save_analysis, index=False)
    prim_df.to_csv(args.save_analysis.replace(".csv", "_prim.csv"), index=False)


def get_mask(masks, f):
    """
    Serializable/global version of get_mask for multiprocessing
    """
    # TODO: Handle here when doing AND and ORs of scenes vs scalars.
    if isinstance(f, F.And):
        masks_l = get_mask(masks, f.left)
        masks_r = get_mask(masks, f.right)
        return masks_l & masks_r
    elif isinstance(f, F.Or):
        masks_l = get_mask(masks, f.left)
        masks_r = get_mask(masks, f.right)
        return masks_l | masks_r
    elif isinstance(f, F.Not):
        masks_val = get_mask(masks, f.val)
        return 1 - masks_val
    elif isinstance(f, F.Leaf):
        return masks[:, f.val]
    else:
        raise ValueError("Most be passed formula")


def iou(a, b):
    intersection = (a & b).sum()
    union = (a | b).sum()
    return intersection / (union + 1e-10)


def is_degenerate(mask):
    m = mask.mean()
    return (m == 1) or (m == 0)


def search_iou(
    neuron_mask, feature_masks, max_formula_length=2, beam_size=10, complexity_penalty=1
):
    """
    Search for best IoU with beam search
    """
    ious = {}
    for feat in range(feature_masks.shape[1]):
        feat_f = F.Leaf(feat)
        mask = get_mask(feature_masks, feat_f)

        if is_degenerate(mask):
            continue

        ious[feat] = iou(neuron_mask, mask)

    ious = Counter(ious)
    formulas = {F.Leaf(feat): iou for feat, iou in ious.most_common(beam_size)}
    best_noncomp = Counter(formulas).most_common(1)[0]

    for i in range(args.max_formula_length - 1):
        new_formulas = {}
        for formula in formulas:
            for feat in ious.keys():
                for op, negate in [(F.Or, False), (F.And, False), (F.And, True)]:
                    new_term = F.Leaf(feat)
                    if negate:
                        new_term = F.Not(new_term)
                    new_term = op(formula, new_term)
                    masks_comp = get_mask(feature_masks, new_term)
                    if is_degenerate(masks_comp):
                        continue

                    comp_iou = iou(neuron_mask, masks_comp)
                    comp_iou = (complexity_penalty ** (len(new_term) - 1)) * comp_iou

                    new_formulas[new_term] = comp_iou

        formulas.update(new_formulas)
        # Trim the beam
        formulas = dict(Counter(formulas).most_common(beam_size))

    best = Counter(formulas).most_common(1)[0]

    return best, best_noncomp


def explain(neurons, neuron_masks, feature_masks, feature_names, final_weight, args):
    records = []
    primitives = []
    for i_neuron in neurons:
        neuron_mask = neuron_masks[:, i_neuron]
        best, _ = search_iou(
            neuron_mask,
            feature_masks,
            max_formula_length=args.max_formula_length,
            beam_size=args.beam_size,
        )
        best_lab, best_iou = best
        best_name = best_lab.to_str(lambda x: feature_names[x])

        # Compute final weight
        weight = final_weight[0, i_neuron]
        records.append(
            {
                "neuron": i_neuron,
                "feature": best_name,
                "iou": best_iou,
                "weight": weight,
            }
        )

        # Primitives
        for prim in best_lab.get_vals():
            prim_name = feature_names[prim]
            primitives.append(
                {
                    "neuron": i_neuron,
                    "primitive": prim_name,
                    "iou": best_iou,
                    "weight": weight,
                }
            )

        print(
            f"neuron {i_neuron:03d}: feature {best_name} (iou {best_iou:.3f}, weight {weight:.3f})"
        )

    record_df = pd.DataFrame(records)
    primitive_df = pd.DataFrame(primitives)
    print(record_df.sort_values("weight"))
    return record_df, primitive_df


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("mode", choices=["train", "analyze"])
    parser.add_argument("--n_hidden", default=64, type=int)
    parser.add_argument("--debias", action="store_true")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--beam_size", default=10, type=int)
    parser.add_argument("--n_explain", default=2000, type=int)
    parser.add_argument("--neuron_threshold", default=0.5, type=float)
    parser.add_argument("--feature_threshold", default=0.5, type=float)
    parser.add_argument("--max_formula_length", default=1, type=int)
    parser.add_argument("--save_model", default="models/model.pt")
    parser.add_argument("--save_analysis", default="analysis/data/neurons.csv")
    parser.add_argument("--train_data", default="data/train.csv")
    parser.add_argument("--test_data", default="data/test.csv")

    args = parser.parse_args()

    if not args.save_analysis.endswith(".csv"):
        parser.error("--save_analysis must end with .csv")

    if args.mode == "train":
        train(args)
    elif args.mode == "analyze":
        analyze(args)

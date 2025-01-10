import copy

import torch

from quantization.hijacker import QuantizationHijacker
from quantization.quantized_folded_bn import BNFusedHijacker
from collections import namedtuple
from torcheeg.datasets import SEEDFeatureDataset
from torcheeg import transforms
from torcheeg.datasets.constants import SEED_CHANNEL_LOCATION_DICT
from torcheeg.transforms import after_hook_normalize
from torcheeg.model_selection import Subcategory
from torcheeg.model_selection import LeaveOneSubjectOut
from torch.utils.data import DataLoader
import torch.nn.functional as F


class MethodPropagator:
    def __init__(self, propagatables):
        self.propagatables = propagatables

    def __getattr__(self, item):
        if callable(getattr(self.propagatables[0], item)):

            def propagate_call(*args, **kwargs):
                for prop in self.propagatables:
                    getattr(prop, item)(*args, **kwargs)

            return propagate_call
        else:
            return getattr(self.propagatables[0], item)

    def __str__(self):
        result = ""
        for prop in self.propagatables:
            result += str(prop) + "\n"
        return result

    def __iter__(self):
        for i in self.propagatables:
            yield i

    def __contains__(self, item):
        return item in self.propagatables



def get_dataloaders_and_model(config, subject=0, load_type="fp32", **qparams):
    dataset = SEEDFeatureDataset(
        io_path="./examples_seed_domain_adaption/seed",
        root_path="./ExtractedFeatures",
        after_session=after_hook_normalize,
        offline_transform=transforms.Compose(
            [transforms.ToGrid(SEED_CHANNEL_LOCATION_DICT)]
        ),
        online_transform=transforms.ToTensor(),  # seed do not have baseline signals
        label_transform=transforms.Compose(
            [transforms.Select("emotion"), transforms.Lambda(lambda x: F.one_hot(torch.tensor(x + 1), num_classes=3).float())]
        ),
        feature=["de_LDS"],
        num_worker=4,
    )
    subset = Subcategory(
        criteria='session_id',
        split_path='./examples_seed_domain_adaption/split/session')
    j, sub_dataset = next(enumerate(subset.split(dataset)))
    loo = LeaveOneSubjectOut( split_path=f'./examples_seed_domain_adaption/split/loo_{j}')
    _, (train_dataset, test_dataset) = list(enumerate(loo.split(sub_dataset)))[subject]

    source_loader = DataLoader(train_dataset, batch_size=config.base.batch_size, shuffle=True, num_workers=4, drop_last=True)
    target_loader = DataLoader(test_dataset, batch_size=config.base.batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config.base.batch_size, shuffle=True, num_workers=4)
    dataloaders = namedtuple('DataLoaders', ['target_loader', 'source_loader', 'test_loader'])(target_loader, source_loader, test_loader)

    model, fp_model = config.base.architecture(
        pretrained=config.base.pretrained,
        load_type=load_type,
        model_dir=config.base.model_dir,
        device = "cuda" if config.base.cuda else "cpu",
        **qparams,
    )
    if config.base.cuda:
        model = model.cuda()

    return dataloaders, model, fp_model


class CompositeLoss:
    def __init__(self, loss_dict):
        self.loss_dict = loss_dict

    def __call__(self, prediction, target, *args, **kwargs):
        total_loss = 0
        for loss_func in self.loss_dict.values():
            total_loss += loss_func(prediction, target, *args, **kwargs)
        return total_loss


class UpdateFreezingThreshold:
    def __init__(self, tracker_dict, decay_schedule):
        self.tracker_dict = tracker_dict
        self.decay_schedule = decay_schedule

    def __call__(self, engine):
        if engine.state.iteration < self.decay_schedule.decay_start:
            # Put it always to 0 for real warm-start
            new_threshold = 0
        else:
            new_threshold = self.decay_schedule(engine.state.iteration)

        # Update trackers with new threshold
        for name, tracker in self.tracker_dict.items():
            tracker.freeze_threshold = new_threshold
        # print('Set new freezing threshold', new_threshold)


class UpdateDampeningLossWeighting:
    def __init__(self, bin_reg_loss, decay_schedule):
        self.dampen_loss = bin_reg_loss
        self.decay_schedule = decay_schedule

    def __call__(self, engine):
        new_weighting = self.decay_schedule(engine.state.iteration)
        self.dampen_loss.weighting = new_weighting
        # print('Set new bin reg weighting', new_weighting)


class DampeningLoss:
    def __init__(self, model, weighting=1.0, aggregation="sum"):
        self.model = model
        self.weighting = weighting
        self.aggregation = aggregation

    def __call__(self, *args, **kwargs):
        total_bin_loss = 0
        for name, module in self.model.named_modules():
            if isinstance(module, QuantizationHijacker):
                # FP32 weight tensor, potential folded but before quantization
                weight, _ = module.get_weight_bias()
                # The matching weight quantizer (not manager, direct quantizer class)
                quantizer = module.weight_quantizer.quantizer
                total_bin_loss += dampening_loss(weight, quantizer, self.aggregation)
        return total_bin_loss * self.weighting


def dampening_loss(w_fp, quantizer, aggregation="sum"):
    w_q = quantizer(
        w_fp, skip_tracking=True
    ).detach()
    w_fp_clip = torch.min(torch.max(w_fp, quantizer.x_min), quantizer.x_max)
    loss = (w_q - w_fp_clip) ** 2
    if aggregation == "sum":
        return loss.sum()
    elif aggregation == "mean":
        return loss.mean()
    elif aggregation == "kernel_mean":
        return loss.sum(0).mean()
    else:
        raise ValueError(f"Aggregation method '{aggregation}' not implemented.")


class ReestimateBNStats:
    def __init__(self, model, data_loader, num_batches=50):
        super().__init__()
        self.model = model
        self.data_loader = data_loader
        self.num_batches = num_batches

    def __call__(self, engine):
        print("-- Reestimate current BN statistics --")
        reestimate_BN_stats(self.model, self.data_loader, self.num_batches)


def reestimate_BN_stats(model, data_loader, num_batches=50, store_ema_stats=False):
    model.eval()
    org_momentum = {}
    for name, module in model.named_modules():
        if isinstance(module, BNFusedHijacker):
            org_momentum[name] = module.momentum
            module.momentum = 1.0
            module.running_mean_sum = torch.zeros_like(module.running_mean)
            module.running_var_sum = torch.zeros_like(module.running_var)
            module.training = True

            if store_ema_stats:
                if not hasattr(module, "running_mean_ema"):
                    module.register_buffer(
                        "running_mean_ema", copy.deepcopy(module.running_mean)
                    )
                    module.register_buffer(
                        "running_var_ema", copy.deepcopy(module.running_var)
                    )
                else:
                    module.running_mean_ema = copy.deepcopy(module.running_mean)
                    module.running_var_ema = copy.deepcopy(module.running_var)

    device = next(model.parameters()).device
    batch_count = 0
    with torch.no_grad():
        for x, y in data_loader:
            model(x.to(device))
            for name, module in model.named_modules():
                if isinstance(module, BNFusedHijacker):
                    module.running_mean_sum += module.running_mean
                    module.running_var_sum += module.running_var

            batch_count += 1
            if batch_count == num_batches:
                break
    for name, module in model.named_modules():
        if isinstance(module, BNFusedHijacker):
            module.running_mean = module.running_mean_sum / batch_count
            module.running_var = module.running_var_sum / batch_count
            module.momentum = org_momentum[name]
    model.eval()


def calculate_grad_cam(model, sources, targets, source_labels, layer):

    intermediate_outputs = {}

    def forward_hook(module, input, output):
        intermediate_outputs["value"] = output

    # Register the forward hook
    layer = dict(model.named_modules())[layer]
    handle = layer.register_forward_hook(forward_hook)

    # Forward pass
    outputs = model(sources.requires_grad_(), targets.requires_grad_(), source_labels)[0]
    handle.remove()

    # Compute gradients
    grad_output = source_labels
    gradients = torch.autograd.grad(
        outputs=outputs,
        inputs=intermediate_outputs["value"],
        grad_outputs=grad_output,
        retain_graph=True,
        create_graph=False
    )[0]

    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    cam = F.relu(torch.sum(weights * intermediate_outputs["value"], dim=1))  # [batch_size, height, width]

    # Normalize Grad-CAM
    cam_min = torch.min(cam.view(cam.size(0), -1), dim=-1, keepdim=True)[0]
    cam_max = torch.max(cam.view(cam.size(0), -1), dim=-1, keepdim=True)[0]
    cam = (cam - cam_min.view(-1, 1, 1)) / (cam_max.view(-1, 1, 1) - cam_min.view(-1, 1, 1) + 1e-8)

    return cam


def compute_region_importance(cam, prefrontal_mask):
    cam = cam * prefrontal_mask
    Z = torch.sum(prefrontal_mask)  # Number of non-zero elements in the mask
    region_score= torch.sigmoid(-torch.sum(cam * prefrontal_mask, dim=(1, 2)) / Z)
    return region_score


def Construct_calibration_set(data_loader, model,layer_name, top_k):
    model.eval()
    scores, samples, targets = [], [], []
    prefrontal_mask = torch.zeros((9, 9))
    prefrontal_mask[:3, :] = 1 
    for (source, source_lable), (target, _) in data_loader:
        cam = calculate_grad_cam(model, source, target, source_lable, layer_name)
        importance_scores = compute_region_importance(cam, prefrontal_mask)
        scores.extend(importance_scores.cpu().detach().numpy())
        samples.extend(data.cpu().detach().numpy())
        targets.extend(target.cpu().detach().numpy())

    # Select top-k samples
    sorted_indices = torch.argsort(torch.tensor(scores), descending=True)
    top_samples = [samples[i] for i in sorted_indices[:top_k]]
    top_targets = [targets[i] for i in sorted_indices[:top_k]]
    return torch.utils.data.TensorDataset(
        torch.tensor(top_samples),  # Use the data as-is
        torch.tensor(top_targets)   # Use the target as-is
    )

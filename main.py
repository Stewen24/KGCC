import logging
import os

import click
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Events
from ignite.metrics import Loss, MetricsLambda
from torch.utils.data import DataLoader

from utils import DotDict, CosineTempDecay
from utils.click_options import qat_options, quantization_options, quant_params_dict, base_options, multi_optimizer_options
from utils.optimizer_utils import optimizer_lr_factory
from utils.oscillation_tracking_utils import add_oscillation_trackers
from utils.qat_utils import get_dataloaders_and_model, MethodPropagator, DampeningLoss, UpdateDampeningLossWeighting, UpdateFreezingThreshold, ReestimateBNStats, Construct_calibration_set
from utils.supervised_driver import create_trainer_engine, log_metrics
from utils.loss_function import CustomLoss, DomainAdversarialLoss

from quantization.utils import pass_data_for_range_estimation, separate_quantized_model_params, set_range_estimators
from models.model_PR_PL import discriminator, weight_init


class Config(DotDict):
    pass


@click.group()
def KGCC():
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


pass_config = click.make_pass_decorator(Config, ensure=True)


@KGCC.command()
@pass_config
@base_options
@multi_optimizer_options()
@quantization_options
@qat_options
def train_quantized(config):
    device = "cuda" if config.base.cuda else "cpu"
    print("Setting up network and data loaders")
    qparams = quant_params_dict(config)

    dataloaders, model, fp_model = get_dataloaders_and_model(config, **qparams)
 
    train_loader = DualDataLoader(dataloaders.source_loader, dataloaders.target_loader)

    calibration_set = Construct_calibration_set(
        train_loader, fp_model, layer_name="fea_extrator_f.conv3", top_k=100
    )

    # Estimate ranges using calibration set
    pass_data_for_range_estimation(
        # loader=dataloaders.train_loader,
        loader=DataLoader(calibration_set, batch_size=config.base.batch_size),
        model=model,
        act_quant=config.quant.act_quant,
        weight_quant=config.quant.weight_quant,
        max_num_batches=config.quant.num_est_batches,
    )

    # Put quantizers in desirable state
    set_range_estimators(config, model)

    print("Loaded model:\n{}".format(model))

    # Get all models parameters in  subcategories
    quantizer_params, model_params, grad_params = separate_quantized_model_params(model)
    model_optimizer, quant_optimizer = None, None
    if config.qat.sep_quant_optimizer:
        # Separate optimizer for model and quantization parameters
        model_optimizer, model_lr_scheduler = optimizer_lr_factory(
            config.optimizer, model_params, config.base.max_epochs
        )
        quant_optimizer, quant_lr_scheduler = optimizer_lr_factory(
            config.quant_optimizer, quantizer_params, config.base.max_epochs
        )

        optimizer = MethodPropagator([model_optimizer, quant_optimizer])
        lr_schedulers = [
            s for s in [model_lr_scheduler, quant_lr_scheduler] if s is not None
        ]
        lr_scheduler = MethodPropagator(lr_schedulers) if len(lr_schedulers) else None
    else:
        optimizer, lr_scheduler = optimizer_lr_factory(
            config.optimizer, quantizer_params + model_params, config.base.max_epochs
        )

    print("Optimizer:\n{}".format(optimizer))
    print(f"LR scheduler\n{lr_scheduler}")

    # Define metrics for ingite engine
    metrics = {
        "total_loss": Loss(
            lambda y_pred, y: y_pred,
            output_transform=lambda output: (output["total_loss"], None),
        ),
        "cls_loss": Loss(
            lambda y_pred, y: y_pred,
            output_transform=lambda output: (output["cls_loss"], None),
        ),
        "transfer_loss": Loss(
            lambda y_pred, y: y_pred,
            output_transform=lambda output: (output["transfer_loss"], None),
        ),
        "acc": MetricsLambda(
            lambda acc: acc, output_transform=lambda output: output["acc"]
        ),
        "nmi": MetricsLambda(
            lambda nmi: nmi, output_transform=lambda output: output["nmi"]
        ),
    }

    # Set-up losses
    dampening_loss = None
    dampening_loss_metrics = {}
    if config.osc_damp.weight is not None:
        # Add dampening loss to task loss
        dampening_loss = DampeningLoss(
            model, config.osc_damp.weight, config.osc_damp.aggregation
        )
        dampening_loss_metrics = {
            "dampening_loss": Loss(
                lambda y_pred, y: y_pred,
                output_transform=lambda output: (output["dampening_loss"], None),
            ),
        }

    metrics.update(dampening_loss_metrics)

    # Set up ignite trainer and evaluator
    domain_discriminator = discriminator(32).to(device)
    domain_discriminator.apply(weight_init)
    dann_loss = DomainAdversarialLoss(domain_discriminator).to(device)
    trainer, evaluator = create_trainer_engine(
        model=model,
        optimizer=optimizer,
        task_loss=CustomLoss(dann_loss, 32, device=device),
        dampening_loss_fn=dampening_loss,
        data_loaders=dataloaders,
        metrics=metrics,
        lr_scheduler=lr_scheduler,
        save_checkpoint_dir=config.base.save_checkpoint_dir,
        device="cuda" if config.base.cuda else "cpu",
        max_iter=config.base.max_epochs,
        batch_size=config.base.batch_size,
    )
    evaluator.state.best_acc = 0.0

    if config.base.progress_bar:
        pbar = ProgressBar()
        pbar.attach(trainer)
        pbar.attach(evaluator)

    if config.osc_damp.weight_final:
        # Apply cosine annealing of dampening loss
        total_iterations = len(dataloaders.source_loader) * config.base.max_epochs
        annealing_schedule = CosineTempDecay(
            t_max=total_iterations,
            temp_range=(config.osc_damp.weight, config.osc_damp.weight_final),
            rel_decay_start=config.osc_damp.anneal_start,
        )
        print(
            f"Weight gradient parameter cosine annealing schedule:\n{annealing_schedule}"
        )
        trainer.add_event_handler(
            Events.ITERATION_STARTED,
            UpdateDampeningLossWeighting(dampening_loss, annealing_schedule),
        )

    # Evaluate model
    print("Running evaluation before training")
    evaluator.run(dataloaders.test_loader)
    log_metrics(evaluator.state.metrics, "Evaluation", trainer.state.epoch)

    # BN Re-estimation
    if config.qat.reestimate_bn_stats:
        evaluator.add_event_handler(
            Events.EPOCH_STARTED, ReestimateBNStats(model, dataloaders.source_loader)
        )

    # Add oscillation trackers to the model and set up oscillation freezing
    if config.osc_freeze.threshold:
        oscillation_tracker_dict = add_oscillation_trackers(
            model,
            max_bits=config.osc_freeze.max_bits,
            momentum=config.osc_freeze.ema_momentum,
            freeze_threshold=config.osc_freeze.threshold,
            use_ema_x_int=config.osc_freeze.use_ema,
        )

        if config.osc_freeze.threshold_final:
            # Apply cosine annealing schedule to the freezing threshdold
            total_iterations = len(dataloaders.source_loader) * config.base.max_epochs
            annealing_schedule = CosineTempDecay(
                t_max=total_iterations,
                temp_range=(
                    config.osc_freeze.threshold,
                    config.osc_freeze.threshold_final,
                ),
                rel_decay_start=config.osc_freeze.anneal_start,
            )
            print(f"Oscillation freezing annealing schedule:\n{annealing_schedule}")
            trainer.add_event_handler(
                Events.ITERATION_STARTED,
                UpdateFreezingThreshold(oscillation_tracker_dict, annealing_schedule),
            )

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate_on_epoch_end(engine):
        print(f"Running evaluation after epoch {engine.state.epoch}")
        evaluator.run(dataloaders.test_loader)
        log_metrics(evaluator.state.metrics, "Evaluation", engine.state.epoch)

    print("Starting training")
    trainer.run(train_loader, max_epochs=config.base.max_epochs)
    print("Finished training")


class DualDataLoader:
    def __init__(self, ref_dataloader: DataLoader, other_dataloader: DataLoader):
        self.ref_dataloader = ref_dataloader
        self.other_dataloader = other_dataloader

    def __iter__(self):
        return self.dual_iterator()

    def __len__(self):
        return len(self.ref_dataloader)

    def dual_iterator(self):
        other_it = iter(self.other_dataloader)
        for data in self.ref_dataloader:
            try:
                data_ = next(other_it)
            except StopIteration:
                other_it = iter(self.other_dataloader)
                data_ = next(other_it)
            yield data, data_

if __name__ == "__main__":
    KGCC()

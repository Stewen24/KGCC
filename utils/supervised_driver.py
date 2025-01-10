# from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import Checkpoint, global_step_from_engine
from torch.optim.optimizer import Optimizer
from ignite.engine import Engine
import torch
import numpy as np


def create_trainer_engine(
    model,
    optimizer,
    task_loss,
    dampening_loss_fn,
    metrics,
    data_loaders,
    lr_scheduler=None,
    save_checkpoint_dir=None,
    device="cuda",
    max_iter=1000,
    boost_type="linear",
    cluster_weight=2,
    batch_size=96,
):
    # 创建自定义训练引擎
    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()

        (x_s, y_s), (x_t, _) = batch
        estimated_sim_truth,estimated_sim_truth_target = get_generated_targets(model,x_s,x_t,y_s)
        _,feature_source_f,feature_target_f,sim_matrix,sim_matrix_target = model(x_s,x_t,y_s)

        total_loss, cls_loss, transfer_loss = task_loss(model, engine.state.boost_factor, feature_source_f, feature_target_f, sim_matrix, sim_matrix_target, 
                 estimated_sim_truth, estimated_sim_truth_target, batch_size)

        dampening_loss = None
        if dampening_loss_fn:
            dampening_loss = dampening_loss_fn()
            total_loss += dampening_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        model.eval()
        acc, nmi = model.cluster_label_update(x_s, y_s)

        result = {
            "total_loss": total_loss.item(),
            "cls_loss": cls_loss.item(),
            "transfer_loss": transfer_loss.item(),
            "dampening_loss": dampening_loss_fn.item(),
            "acc": acc,
            "nmi": nmi,
        }
        if dampening_loss_fn:
            result["dampening_loss"] = dampening_loss.item()

    trainer = Engine(train_step)

    # 动态更新 boost_factor
    def update_boost_factor(engine):
        epoch = engine.state.epoch
        if boost_type == 'linear':
            boost_factor = cluster_weight * (epoch / max_iter)
        elif boost_type == 'exp':
            boost_factor = cluster_weight * (2.0 / (1.0 + np.exp(-1 * epoch / max_iter)) - 1)
        elif boost_type == 'constant':
            boost_factor = cluster_weight
        else:
            raise ValueError(f"Unknown boost_type: {boost_type}")
        engine.state.boost_factor = boost_factor

    trainer.add_event_handler(Events.EPOCH_STARTED, update_boost_factor)

    # 动态更新阈值
    def update_threshold(engine):
        epoch = engine.state.epoch
        n_epochs = max_iter
        diff = model.upper_threshold - model.lower_threshold
        eta = diff / n_epochs
        if epoch != 0:
            model.upper_threshold -= eta
            model.lower_threshold += eta
        model.threshold = (model.upper_threshold + model.lower_threshold) / 2

    trainer.add_event_handler(Events.EPOCH_COMPLETED, update_threshold)

    # 将指标附加到训练引擎
    for name, metric in metrics.items():
        metric.attach(trainer, name)

    # 添加学习率调度器
    if lr_scheduler:
        trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: lr_scheduler.step())

    # 创建自定义评估引擎
    def evaluation_step(engine, batch):
        model.eval()
        x_t, y_t = batch
        with torch.no_grad():
            target_acc,target_nmi=model.target_domain_evaluation(x_t, y_t)
        if target_acc > engine.state.best_acc:
            engine.state.best_acc = target_acc
        return {"acc": target_acc, "nmi": target_nmi}
    evaluator = Engine(evaluation_step)

    # 将指标附加到评估引擎
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # 保存检查点
    if save_checkpoint_dir:
        to_save = {"model": model, "optimizer": optimizer}
        if lr_scheduler:
            to_save["lr_scheduler"] = lr_scheduler
        checkpoint = Checkpoint(
            to_save,
            save_checkpoint_dir,
            n_saved=1,
            global_step_transform=global_step_from_engine(trainer),
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint)

    # 添加日志记录
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results, optimizer)

    # 在每个 epoch 完成后运行评估
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, run_evaluation_for_training, evaluator, data_loaders.test_loader
    )

    return trainer, evaluator

def get_generated_targets(model,x_s,x_t,labels_s): ## Get generated labels by threshold
        with torch.no_grad():
            model.eval()
            _,_,_,_,dist_matrix = model(x_s,x_t,labels_s)     
            sim_matrix = model.get_cos_similarity_distance(labels_s)
            sim_matrix_target = model.get_cos_similarity_by_threshold(dist_matrix)
            return sim_matrix,sim_matrix_target


# def create_trainer_engine(
#     model,
#     optimizer,
#     criterion,
#     metrics,
#     data_loaders,
#     lr_scheduler=None,
#     save_checkpoint_dir=None,
#     device="cuda",
# ):
#     # Create trainer
    # trainer = create_supervised_trainer(
#         model=model,
#         optimizer=optimizer,
#         loss_fn=criterion,
#         device=device,
#         output_transform=custom_output_transform,
#     )
#
#     for name, metric in metrics.items():
#         metric.attach(trainer, name)
#
#     # Add lr_scheduler
#     if lr_scheduler:
#         trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: lr_scheduler.step())
#
#     # Create evaluator
#     evaluator = create_supervised_evaluator(model=model, metrics=metrics, device=device)
#
#     # Save model checkpoint
#     if save_checkpoint_dir:
#         to_save = {"model": model, "optimizer": optimizer}
#         if lr_scheduler:
#             to_save["lr_scheduler"] = lr_scheduler
#         checkpoint = Checkpoint(
#             to_save,
#             save_checkpoint_dir,
#             n_saved=1,
#             global_step_transform=global_step_from_engine(trainer),
#         )
#         trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint)
#
#     # Add hooks for logging metrics
#     trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results, optimizer)
#
#     trainer.add_event_handler(
#         Events.EPOCH_COMPLETED, run_evaluation_for_training, evaluator, data_loaders.test_loader
#     )
#
#     return trainer, evaluator


def custom_output_transform(x, y, y_pred, loss):
    return y_pred, y


def log_training_results(trainer, optimizer):
    learning_rate = optimizer.param_groups[0]["lr"]
    log_metrics(trainer.state.metrics, "Training", trainer.state.epoch, learning_rate)


def run_evaluation_for_training(trainer, evaluator, val_loader):
    evaluator.run(val_loader)
    log_metrics(evaluator.state.metrics, "Evaluation", trainer.state.epoch)


def log_metrics(metrics, stage: str = "", training_epoch=None, learning_rate=None):
    log_text = "  {}".format(metrics) if metrics else ""
    if training_epoch is not None:
        log_text = "Epoch: {}".format(training_epoch) + log_text
    if learning_rate and learning_rate > 0.0:
        log_text += "  Learning rate: {:.2E}".format(learning_rate)
    log_text = "Results - " + log_text
    if stage:
        log_text = "{} ".format(stage) + log_text
    print(log_text, flush=True)


def setup_tensorboard_logger(trainer, evaluator, output_path, optimizers=None):
    logger = TensorboardLogger(logdir=output_path)

    # Attach the logger to log loss and accuracy for both training and validation
    for tag, cur_evaluator in [("train", trainer), ("validation", evaluator)]:
        logger.attach_output_handler(
            cur_evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names="all",
            global_step_transform=global_step_from_engine(trainer),
        )

    # Log optimizer parameters
    if isinstance(optimizers, Optimizer):
        optimizers = {None: optimizers}

    for k, optimizer in optimizers.items():
        logger.attach_opt_params_handler(
            trainer, Events.EPOCH_COMPLETED, optimizer, param_name="lr", tag=k
        )

    return logger

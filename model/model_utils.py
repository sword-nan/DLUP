import math
import random
from typing import Dict, Callable, Sequence, Literal


from tqdm import tqdm
import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
            optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
            num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
            num_training_steps (:obj:`int`):
            The total number of training steps.
            num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
            last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
            :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi *
                        float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class ModelBase:
    def adjust_labels_shape(self, y: torch.Tensor, pred_shape: torch.Size):
        """
        动态调整标签的形状以匹配预测的形状。
        """
        if y.shape != pred_shape:
            # 如果 y 是一个向量，但 pred 不是
            if y.dim() == 1 and len(pred_shape) > 1:
                y = y.unsqueeze(1)
            # 如果 y 和 pred 的其余维度不匹配，则尝试扩展 y
            if y.shape[1:] != pred_shape[1:]:
                y = y.expand(pred_shape)
        return y

class ModelTrainer(ModelBase):
    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: Module,
        device: torch.device,
        train_configs: Dict,
        custom_metrics_name: str=None,
        custom_metrics_fn: Callable = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_configs = train_configs
        self.custom_metrics_name = custom_metrics_name
        self.custom_metrics_fn = custom_metrics_fn
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer, train_configs['scheduler']['warmup_steps'], train_configs['scheduler']['total_steps'])

    def train(self):
        loss_metrics = {
            'train': [],
            'validation': []
        }
        custom_metrics = {
            'train': [],
            'validation': []
        }
        train_format_str = 'train, epoch={}, loss={:.4f}'
        valid_format_str = 'validation, epoch={}, loss={:.4f}, save the model'
        if self.custom_metrics_fn is not None:
            train_format_str = 'epoch={}, loss={:.4f}, {}={:.4f}'
            valid_format_str = 'validation, epoch={}, loss={:.4f}, {}={:.4f}, save the model'
        
        best_model_metric = torch.inf
        best_custom_metric = torch.inf
        best_epoch = 0
        for epoch in range(self.train_configs['n_epochs']):
            self.model.train()
            num = 0
            loss_value = 0
            custom_metric = 0
            x: Sequence[torch.Tensor]
            # for x, y in tqdm(self.train_loader):
            for x, y in self.train_loader:
                x = [d.to(self.device) for d in x]
                y = y.to(self.device)
                pred = self.model(*x)
                y = self.adjust_labels_shape(y, pred.shape)
                loss: torch.Tensor = self.criterion(pred, y)
                num += y.shape[0]
                loss_value += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                if self.custom_metrics_fn is not None:
                    custom_metric += self.custom_metrics_fn(pred, y).item()

            loss_value = loss_value / num
            custom_metric = custom_metric / num
            val_loss, val_custom_metric = self.evaluate()

            loss_metrics['train'].append(loss_value)
            loss_metrics['validation'].append(val_loss)

            if self.custom_metrics_fn is not None:
                print(train_format_str.format(epoch + 1, loss_value, self.custom_metrics_name, custom_metric))
                custom_metrics['train'].append(custom_metric)
                custom_metrics['validation'].append(val_custom_metric)

                if best_custom_metric > val_custom_metric:
                    best_model_metric = val_loss
                    best_custom_metric = val_custom_metric
                    best_epoch = epoch
                    print(valid_format_str.format(epoch + 1, val_loss, self.custom_metrics_name, val_custom_metric))
                    torch.save(self.model.state_dict(),
                               self.train_configs['save_path'])
            else:
                print(train_format_str.format(epoch + 1, loss_value))
                if best_model_metric > val_loss:
                    best_epoch = epoch
                    best_model_metric = val_loss
                    torch.save(self.model.state_dict(),
                               self.train_configs['save_path'])
                    print(valid_format_str.format(epoch + 1, val_loss))

        if self.custom_metrics_fn is not None:
            print('best model saved', valid_format_str.format(best_epoch + 1, best_model_metric, self.custom_metrics_name, best_custom_metric))
        else:
            print('best model saved', valid_format_str.format(best_epoch + 1, best_model_metric))
        
        return loss_metrics, custom_metrics

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            num = 0
            loss_value = 0
            custom_metric = 0
            x: Sequence[torch.Tensor]
            # for x, y in tqdm(self.val_loader):
            for x, y in self.val_loader:
                x = [d.to(self.device) for d in x]
                y = y.to(self.device)
                pred = self.model(*x)
                y = self.adjust_labels_shape(y, pred.shape)
                loss: torch.Tensor = self.criterion(pred, y)
                num += y.shape[0]
                loss_value += loss.item()
                if self.custom_metrics_fn is not None:
                    custom_metric += self.custom_metrics_fn(pred, y).item()

        if self.custom_metrics_fn is not None:
            return loss_value / num, custom_metric / num

        return loss_value / num, None
        
class ModelTester(ModelBase):
    def __init__(
        self,
        model: Module,
        test_loader: DataLoader,
        criterion: Module,
        device: torch.device,
        custom_metrics_name: str=None,
        custom_metrics_fn: Callable = None,
    ) -> None:
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        self.custom_metrics_name = custom_metrics_name
        self.custom_metrics_fn = custom_metrics_fn
        
    def replace_dataloader(self, dataloader: DataLoader):
        self.test_loader = dataloader

    def test(self, infos):
        self.model.eval()
        scores = []
        with torch.no_grad():
            num = 0
            loss_value = 0
            custom_metric = 0
            x: Sequence[torch.Tensor]
            for x, y in self.test_loader:
                x = [d.to(self.device) for d in x]
                y = y.to(self.device)
                pred: torch.Tensor = self.model(*x)
                y = self.adjust_labels_shape(y, pred.shape)
                loss: torch.Tensor = self.criterion(pred, y)
                num += y.shape[0]
                loss_value += loss.item()
                if self.custom_metrics_fn is not None:
                    custom_metric += self.custom_metrics_fn(pred, y).item()
                scores.append(pred.cpu().numpy())
            print('loss: ', loss_value / num, f'{self.custom_metrics_name}: ', custom_metric / num)
        return {
            'score': np.concatenate(scores, axis=0), 
            'info': infos
        }

class FuneTrainer(ModelTrainer):
    def __init__(
        self, 
        model: Module, 
        freeze_layer_names: Sequence[str],
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer: Optimizer, 
        criterion: Module, 
        device: torch.device, 
        train_configs: Dict, 
        custom_metrics_name: str = None, 
        custom_metrics_fn: Callable = None
    ) -> None:
        super(FuneTrainer, self).__init__(model, train_loader, val_loader, optimizer, criterion, device, train_configs, custom_metrics_name, custom_metrics_fn)
        self.freeze_layer_names = freeze_layer_names
        self.freeze_layers()

    def freeze_layers(self):
        for name, param in self.model.named_parameters():
            for freeze_layer_name in self.freeze_layer_names:
                if freeze_layer_name in name:
                    param.requires_grad = False
                    break
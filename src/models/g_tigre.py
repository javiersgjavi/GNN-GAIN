import copy
import torch
import numpy as np
import pytorch_lightning as pl
from typing import Dict, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torchmetrics import MeanAbsoluteError
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd

from src.models.gnn_models import GRUGCN, RNNEncGCNDec
from src.models.gnn_models_bi import GRUGCNBI, RNNEncGCNDecBI
from src.models.custom_models import BiModel
from src.models.losses import BaseGANLoss, LSGANLoss, WSGANLoss

from src.utils import mean_relative_error, loss_controller_ws

class HintGenerator:
    """
    Class that generates a hint matrix with the new definition of the hint matrix that can be found in the
    original repository.
    """

    def __init__(self, prop_hint: float):
        self.prop_hint = prop_hint

    def generate(self, input_mask: torch.Tensor) -> torch.Tensor:
        """
        Generate a hint matrix with the same shape as the input mask tensor.

        Args:
            input_mask (torch.Tensor): Tensor of binary values indicating which values in the input are missing.

        Returns:
            hint_matrix (torch.Tensor): Tensor of binary values with the same shape as input_mask indicating with the
            hints to be used for the discriminator. The values are 1 if it is a known value and 0 if it is a value
            to be determined by the discriminator.
        """
        hint_mask = torch.rand(size=input_mask.size())
        hint_matrix = 1 * (hint_mask < self.prop_hint)
        hint_matrix = input_mask * hint_matrix.to(input_mask.device)
        return hint_matrix

    def generate_base(self, input_mask: torch.Tensor) -> torch.Tensor:
        batch, time, features = input_mask.size()
        b_sel = torch.randint(features, size=(batch, time)).to(input_mask.device)
        b = torch.zeros_like(input_mask, dtype=torch.bool)
        b.scatter_(2, b_sel.unsqueeze(2), True)

        hint_matrix = input_mask.clone()
        hint_matrix[b] = 0.5
        hint_matrix.to(input_mask.device)

        return hint_matrix
    
class GTIGRE(pl.LightningModule):
    def __init__(self, input_size: tuple, edge_index, edge_weights, normalizer, model_type: str = None,
                 hint_rate: float = 0.9, alpha: float = 100, params: Dict = None,
                 ablation_gan=False, ablation_reconstruction=False, ablation_loop=False):
        """
        A PyTorch Lightning module implementing the GAIN (Generative Adversarial Imputation Network) algorithm.

        Args:
            input_size (int): The number of features in the input data.
            alpha (float): A hyperparameter controlling the weight of the reconstruction loss.
            hint_rate (float): The rate of known values in the hint matrix.

        Attributes:
            generator (MLP): The generator model.
            discriminator (MLP): The discriminator model.
            hint_generator (HintGenerator): The hint generator.
            loss_d (function): The discriminator loss function.
            loss_g (function): The generator loss function.
            loss_mse (torch.nn.MSELoss): The mean squared error loss function.
            alpha (int): A hyperparameter controlling the weight of the reconstruction loss.
        """
        super().__init__()
        super().save_hyperparameters()


        losses = {
            'base': BaseGANLoss,
            'ls': LSGANLoss,
            'ws': WSGANLoss
        }

        #model = model_class_bi[model_type] if params['bi'] else model_class[model_type]
        model = BiModel

        self.nodes = input_size[1]
        self.normalizer = normalizer
        self.loss_mse = torch.nn.MSELoss()
        self.mae = MeanAbsoluteError()

        args = {
            'periods': input_size[0],
            'nodes': self.nodes,
            'edge_index': edge_index,
            'edge_weights': edge_weights,
            'encoder_name': model_type
        }

        self.args = {**args, **params}

        # Three main components of the GAIN model

        self.use_time_gap = params['use_time_gap_matrix']
        self.generator = model(copy.deepcopy(self.args), time_gap_matrix=self.use_time_gap)
        self.discriminator = model(copy.deepcopy(self.args), time_gap_matrix=self.use_time_gap, d=True)

        self.hint_generator = HintGenerator(prop_hint=hint_rate)

        self.ablation_loop = ablation_loop

        self.loss_fn = losses[self.args['loss_fn']](
            alpha=alpha if alpha is not None else 100,
            ablation_gan=ablation_gan,
            ablation_reconstruction=ablation_reconstruction
        )


        self.d_i = 0
    # -------------------- Custom methods --------------------

    def calculate_error_imputation(self, outputs: Dict[str, torch.Tensor], type_step: str = 'train') -> None:
        """
            Calculates the mean squared error (MSE) and the root mean squared error (RMSE) between the real input
            and the imputed output of a batch.

            Args:
                outputs: A dictionary containing the output tensors for a batch.
                type_step: A string indicating whether the batch is for training or validation (default is 'train').
            """
        info = {}
        x_real_norm = outputs['x_real']
        x_fake_norm = outputs['x_fake']
        known_values = outputs['known_values']

        real_norm = x_real_norm[known_values]
        fake_norm = x_fake_norm[known_values]

        mse_norm = self.loss_mse(fake_norm, real_norm)
        mae_norm = self.mae(fake_norm, real_norm)

        info['mse'] = mse_norm
        info['rmse'] = torch.sqrt(mse_norm)
        info['mae'] = mae_norm

        if type_step != 'train':
            x_real_denorm = self.normalizer.inverse_transform(x_real_norm.reshape(-1, self.nodes).detach().cpu())
            x_fake_denorm = self.normalizer.inverse_transform(x_fake_norm.reshape(-1, self.nodes).detach().cpu())

            real_denorm = x_real_denorm[known_values.reshape(-1, self.nodes).cpu()]
            fake_denorm = x_fake_denorm[known_values.reshape(-1, self.nodes).cpu()]

            mse_denorm = mean_squared_error(real_denorm, fake_denorm)
            mae_denorm = mean_absolute_error(real_denorm, fake_denorm)
            mre_denorm = mean_relative_error(real_denorm, fake_denorm)

            info['denorm_rmse'] = np.sqrt(mse_denorm)
            info['denorm_mae'] = mae_denorm
            info['denorm_mse'] = mse_denorm
            info['denorm_mre'] = mre_denorm

        return info
    
    def log_results(self, info, type_step='train'):

        imputation_info = info['imputation']

        self.log('mse', imputation_info['mse'], prog_bar=True)
        self.log('rmse', imputation_info['rmse'], prog_bar=True)
        self.log('mae', imputation_info['mae'])
        self.logger.experiment.add_scalars('mse_graph', {type_step: imputation_info['mse']}, self.global_step)

        if type_step == 'train':
            self.logger.experiment.add_scalars(f"G VS D (fake)", info['adv_losses'], self.global_step)
            self.log("G_loss_reconstruction", info['rec_loss'])
        else:
            self.log('denorm_rmse', imputation_info['denorm_rmse'])
            self.log('denorm_mae', imputation_info['denorm_mae'])
            self.log('denorm_mse', imputation_info['denorm_mse'])
            self.log('denorm_mre', imputation_info['denorm_mre'])


    def return_gan_outputs(self, batch: Tuple, train=False) -> Dict[str, torch.Tensor]:
        """
        Returns the output tensors of the generator and discriminator for a given batch.

        Args:
            batch: A tuple containing the real input tensor, the input tensor with missing values, and the input mask
            tensor.

        Returns:
            A dictionary containing the output tensors of the generator and discriminator for the batch, as well as the
            real input and the input mask.
        """
        x, x_real, input_mask_bool, input_mask_int, known_values, time_gap_matrix = batch

        # Forward Generator
        x_fake, imputation = self.generator.forward_g(x=x, input_mask=input_mask_int,
                                                      time_gap_matrix=time_gap_matrix)

        # Generate Hint Matrix
        hint_matrix = self.hint_generator.generate(input_mask_int)

        # Forward Discriminator
        d_pred = self.discriminator.forward_d(x=x_fake, hint_matrix=hint_matrix)

        if self.args['loss_fn'] == 'ws':
            self.logger.experiment.add_scalars(f"D behaviour", {
                'critic_real': d_pred[input_mask_bool].mean().item(),
                'critic_fake': d_pred[~input_mask_bool].mean().item(),
            }, self.global_step)

        res = {
            'x_real': x_real,
            'x_fake': x_fake,
            'd_pred': d_pred,
            'imputation': imputation,
            'input_mask_int': input_mask_int,
            'input_mask_bool': input_mask_bool,
            'known_values': known_values
        }
        return res

    def multiple_imputation(self, batch):
        outputs = self.return_gan_outputs(batch)

        x_real = outputs['x_real']
        input_mask_bool = outputs['input_mask_bool']

        d_pred_list = [outputs['d_pred']]
        imputation_list = [outputs['imputation']]

        for _ in range(9):
            outputs = self.return_gan_outputs(batch)
            d_pred_list.append(outputs['d_pred'])
            imputation_list.append(outputs['imputation'])

        d_pred_stacked = torch.stack(d_pred_list)
        imputation_stacked = torch.stack(imputation_list)

        indices = d_pred_stacked.argmax(dim=0)

        imputation = torch.gather(imputation_stacked, 0, indices.unsqueeze(0))[0]

        x_fake = torch.where(input_mask_bool, x_real, imputation)

        outputs['x_fake'] = x_fake
        outputs['imputation'] = imputation

        return outputs

    # -------------------- Methods from PyTorch Lightning --------------------

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """
            Configure the optimizers for the GAN model.
        """
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.args['learning_rate'], weight_decay=1e-5)
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.args['learning_rate'])

        #opt_d = torch.optim.RMSprop(self.discriminator.parameters(), lr=0.00005, weight_decay=0)
        #opt_g = torch.optim.RMSprop(self.generator.parameters(), lr=0.00005, weight_decay=0)

        # define schedulers
        #d_scheduler = CosineAnnealingLR(opt_d, T_max=5000, eta_min=0.0001)
        #g_scheduler = CosineAnnealingLR(opt_g, T_max=5000, eta_min=0.0001)

        d_opt_params = {'optimizer': opt_d}#, 'lr_scheduler': d_scheduler}
        g_opt_params = {'optimizer': opt_g}#, 'lr_scheduler': g_scheduler}

        return d_opt_params, g_opt_params

    def training_step(self, batch: Tuple, batch_idx: int, optimizer_idx: int) -> torch.Tensor:
        """
        Runs a single training step on a batch of data.

        Args:
            batch (Tuple): Tuple of input data, `x_real`, `x`, and `input_mask`.
            batch_idx (int): Index of the current batch.
            optimizer_idx (int): Index of the optimizer to use for this step.

        Returns:
            Any: The computed loss for the current step.
        """

        # Generate GAN outputs for the given batch
        outputs = self.return_gan_outputs(batch, train=True)

        # Compute the discriminator and generator loss based on the generated outputs
        d_loss, g_loss, info = self.loss_fn.calculate(outputs)

        # Calculate the mean squared error (MSE) between the real and imputed data
        info['imputation'] = self.calculate_error_imputation(outputs)

        # Log the results
        self.log_results(info)

        # Select the appropriate loss based on the optimizer index

        if not self.args['loss_fn'] == 'ws':
            loss = d_loss if optimizer_idx == 0 else g_loss
        else:
            loss, self.d_i = loss_controller_ws(d_loss, g_loss, optimizer_idx, self.d_i)

        return loss

    def validation_step(self, batch: Tuple, batch_idx: int) -> None:
        """Runs a single validation step on a batch of data.

        Args:
            batch (Tuple): Tuple of input data, `x_real`, `x`, and `input_mask`.
            batch_idx (int): Index of the current batch.
        """

        # Generate GAN outputs for the given batch
        outputs = self.return_gan_outputs(batch)

        # Calculate the mean squared error (MSE) between the real and imputed data
        info = {}
        info['imputation'] = self.calculate_error_imputation(outputs, type_step='val')

        # Log the results
        self.log_results(info, type_step='val')

    def test_step(self, batch: Tuple, batch_idx: int) -> None:
        """Runs a single test step on a batch of data.

        Args:
            batch (Tuple): Tuple of input data, `x_real`, `x`, and `input_mask`.
            batch_idx (int): Index of the current batch.
        """

        # Generate GAN outputs for the given batch
        outputs = self.multiple_imputation(batch) if not self.ablation_loop else self.return_gan_outputs(batch)

        # Calculate the mean squared error (MSE) between the real and imputed data
        info = {}
        info['imputation'] = self.calculate_error_imputation(outputs, type_step='test')

        # Log the results
        self.log_results(info, type_step='test')

    def predict_step(self, batch: Tuple, batch_idx: int, dataloader_idx: int = None) -> torch.Tensor:
        """
        Runs a single prediction step on a batch of data.

        Args:
            batch (Tuple): Tuple of input data, `x_real`, `x`, and `input_mask`.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int): Index of the dataloader to use for this step.

        Returns:
            torch.Tensor: The imputed data for the given batch.
        """
        outputs = self.multiple_imputation(batch) if not self.ablation_loop else self.return_gan_outputs(batch)
        x_fake_norm = outputs['x_fake']
        original_shape = x_fake_norm.shape
        x_fake_denorm = self.normalizer.inverse_transform(x_fake_norm.reshape(-1, self.nodes).detach().cpu()).reshape(original_shape)


        return x_fake_denorm

class GTIGRE_DYNAMIC(GTIGRE):
    def __init__(self, *args, **kwargs):
        self.missing_threshold = None
        super().__init__(*args, **kwargs)

    def set_threshold(self, missing_threshold):
        self.missing_threshold = missing_threshold

    def dynamic_mask_data(self, batch):
        x, x_real, input_mask_bool, input_mask_int, known_values, time_gap_matrix = batch

        generated_mask = torch.rand(input_mask_bool.shape, device=input_mask_bool.device) < self.missing_threshold

        new_x = torch.where(generated_mask, torch.tensor(0, device=input_mask_bool.device), x)
        new_input_mask_bool = torch.where(generated_mask, torch.tensor(False, device=input_mask_bool.device),
                                          input_mask_bool)
        new_input_mask_int = torch.where(generated_mask, torch.tensor(0, device=input_mask_int.device), input_mask_bool)

        new_batch = new_x, x_real, new_input_mask_bool, new_input_mask_int, known_values, time_gap_matrix

        return new_batch

    def training_step(self, batch: Tuple, batch_idx: int, optimizer_idx: int) -> torch.Tensor:
        """
        Runs a single training step on a batch of data.

        Args:
            batch (Tuple): Tuple of input data, `x_real`, `x`, and `input_mask`.
            batch_idx (int): Index of the current batch.
            optimizer_idx (int): Index of the optimizer to use for this step.

        Returns:
            Any: The computed loss for the current step.
        """

        # Dynamic generate mask out
        new_batch = self.dynamic_mask_data(batch)

        return super().training_step(new_batch, batch_idx, optimizer_idx)

    def test_step(self, batch: Tuple, batch_idx: int) -> None:
        new_batch = self.dynamic_mask_data(batch)
        return super().test_step(new_batch, batch_idx)

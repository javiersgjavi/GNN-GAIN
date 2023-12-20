import torch

def zero_tensor_like(tensor):
    return torch.zeros_like(tensor, requires_grad=True, device=tensor.device)

class BaseLoss:
    def __init__(self, alpha=0, ablation_gan=False, ablation_reconstruction=False):
        self.alpha = alpha
        self.ab_gan = ablation_gan
        self.ab_reconstruction = ablation_reconstruction
        self.mse_fn = torch.nn.MSELoss()

    def log_results(self, d_loss, adv_loss_g, rec_loss_g):
        self.logger.experiment.add_scalars(f"G VS D (fake)", log_dict, self.global_step)
        self.log("G_loss_reconstruction", rec_loss_g)
    
    def get_d_loss(self, *args):
        if not self.ab_gan:
            res = self.d_loss_fn(*args)
        else:
            res = torch.zeros(1, device=args[0].device, requires_grad=True)
        return res

    def get_g_adv_loss(self, *args):
        if not self.ab_gan:
            res = self.g_adv_loss_fn(*args)
        else:
            res = torch.zeros(1, device=args[0].device, requires_grad=True)
        return res

    def get_reconstruction_loss(self, imputation, x_real, mask):
        if not self.ab_reconstruction:
            res = self.mse_fn(imputation[mask], x_real[mask])
        else:
            res = torch.zeros(1, device=imputation.device, requires_grad=True)

        return res
    
    def get_g_loss(self, d_pred, imputation, x_real, mask_int, mask_bool):
        adversarial = self.get_g_adv_loss(d_pred, mask_int, mask_bool)

        reconstruction = self.get_reconstruction_loss(imputation, x_real, mask_bool)

        res = adversarial + self.alpha * reconstruction

        return res, adversarial, reconstruction

    
    def calculate(self, outputs):
        
        d_pred = outputs["d_pred"]
        x_real = outputs["x_real"]
        imputation = outputs["imputation"]
        mask_int = outputs["input_mask_int"]
        mask_bool = outputs["input_mask_bool"]

        d_loss = self.get_d_loss(d_pred, mask_int, mask_bool)

        g_loss, adv_loss_g, rec_loss_g = self.get_g_loss(
            d_pred, 
            imputation, 
            x_real, 
            mask_int, 
            mask_bool
        )

        log_dict = {
            'adv_losses': {
                "Generator": adv_loss_g, 
                "Discriminator": d_loss
                },
            "rec_loss": rec_loss_g
            }

        return d_loss, g_loss, log_dict
    

class BaseGANLoss(BaseLoss):

    def d_loss_fn(self, d_prob, mask_int, mask_bool):
        return -torch.mean(mask_int * torch.log(d_prob + 1e-8) + (1 - mask_int) * torch.log(1.0 - d_prob + 1e-8))

    def g_adv_loss_fn(self, d_pred, mask_int, mask_bool):
        return -torch.mean((1 - mask_int) * torch.log(d_pred + 1e-8))
    

class LSGANLoss(BaseLoss):

    def d_loss_fn(self, d_pred, mask_int, mask_bool):
        critic_real = d_pred[mask_bool]
        critic_fake = d_pred[~mask_bool]

        critic_real_loss = self.mse_fn(critic_real, torch.ones_like(critic_real))
        critic_fake_loss = self.mse_fn(critic_fake, torch.zeros_like(critic_fake))

        return 0.5 * (critic_real_loss + critic_fake_loss)

    def g_adv_loss_fn(self, d_pred, mask_int, mask_bool):
        critic_fake = d_pred[~mask_bool]
        return 0.5 * self.mse_fn(critic_fake, torch.ones_like(critic_fake))
    
class WSGANLoss(BaseLoss):

    def d_loss_fn(self, d_pred, mask_int, mask_bool):
        critic_real = d_pred[mask_bool].mean()
        critic_fake = d_pred[~mask_bool].mean()

        #print(f'critic_real: {critic_real}')
        #print(f'critic_fake: {critic_fake}')
        return -(critic_real - critic_fake)

    def g_adv_loss_fn(self, d_pred, mask_int, mask_bool):
        critic_fake = d_pred[~mask_bool].mean()
        return -critic_fake

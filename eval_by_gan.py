import os
import legacy
import dnnlib
import torch
import time
import utils
from PIL import Image
from typing import List, Optional, Tuple, Union
import numpy as np
import torchvision.transforms as transforms
import statistics
from argparse import ArgumentParser

from sting_constants import DATASET, ATTACK_NAME, POISONED_MODEL_NAMES, COSINE_SIM_MODE, \
  IMAGE_SHAPE, VAL_SIZE, COLOR_CHANNEL, NUM_OF_CLASS, MEAN, STD, SAMPLES_PER_EPOCH, \
  POISONED_MODEL_ARCHITECTURE, MODE, IMAGE_MODE


def make_transform(translate: Tuple[float, float], angle: float):
  m = np.eye(3)
  s = np.sin(angle / 360.0 * np.pi * 2)
  c = np.cos(angle / 360.0 * np.pi * 2)
  m[0][0] = c
  m[0][1] = s
  m[0][2] = translate[0]
  m[1][0] = -s
  m[1][1] = c
  m[1][2] = translate[1]
  return m

def log_sum_exp(x, axis=1):
  m = torch.max(x, dim=1)[0]
  return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim=axis))

def reparameterize(mu, logvar):
  """
  Reparameterization trick to sample from N(mu, var) from
  N(0,1).
  :param mu: (Tensor) Mean of the latent Gaussian [B x D]
  :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
  :return: (Tensor) [B x D]
  """
  std = torch.exp(0.5 * logvar)
  eps = torch.randn_like(std)
  return eps * std + mu


def save_image(image, dir_name, filename, quality=80, image_mode=IMAGE_MODE.JPEG.value) :
  denormalized_images = (image * 255).byte()
  #print("SAVE Min-value",torch.min(denormalized_images).item(),"Max-value",torch.max(denormalized_images).item(),filename_postfix)
  if COLOR_CHANNEL[DATASET_NAME] == 1 :
    denormalized_images = np.uint8(denormalized_images.detach().cpu().numpy())
    img = Image.fromarray(denormalized_images[0], "L")
    if IMAGE_MODE.JPEG.value in image_mode:
      img.save(os.path.join(dir_name, filename + ".jpg"), format='JPEG', quality=quality)
    elif IMAGE_MODE.PNG.value in image_mode:
      img.save(os.path.join(dir_name, filename + ".png"))
  elif COLOR_CHANNEL[DATASET_NAME] == 3:
    tensor_to_image = transforms.ToPILImage()
    img = tensor_to_image(denormalized_images)
    if IMAGE_MODE.JPEG.value in image_mode:
      img.save(os.path.join(dir_name, filename +  ".jpg"), format='JPEG', quality=quality)
    elif IMAGE_MODE.PNG.value in image_mode:
      img.save(os.path.join(dir_name, filename + ".png"))

def dist_inversion(G, D, T, E, iden, itr, target_class, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1,
                   improved=False, num_seeds=5, filename_postfix=""):
  iden = iden.view(-1).long().to(DEVICE)
  criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
  bs = 1
  truncation_psi = 1.0
  noise_mode = 'random'

  G.eval()
  D.eval()
  T.eval()
  E.eval()

  no = torch.zeros(bs)  # index for saving all success attack images

  tf = time.time()

  # NOTE
  mu = torch.zeros(bs, G.z_dim).to(DEVICE)
  mu.requires_grad = True
  log_var = torch.ones(bs, G.z_dim).to(DEVICE)
  log_var.requires_grad = True

  params = [mu, log_var]
  solver = torch.optim.Adam(params, lr=lr)
  # scheduler = torch.optim.lr_scheduler.StepLR(solver, 1800, gamma=0.1)

  label = torch.zeros([1, G.c_dim], device=DEVICE)
  label[:, target_class] = 1

  for i in range(iter_times):
    z = reparameterize(mu, log_var)

    '''
    # Construct an inverse rotation/translation matrix and pass to the generator.  The
    # generator expects this matrix as an inverse to avoid potentially failing numerical
    # operations in the network.
    translate = tuple((0.0, 0.0))
    rotate = 0.0
    if hasattr(G.synthesis, 'input'):
      m = make_transform(translate, rotate)
      m = np.linalg.inv(m)
      G.synthesis.input.transform.copy_(torch.from_numpy(m))
    '''

    fake = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)

    if improved == True:
      _, label = D(fake)
    else:
      label = D(fake)

    out = T(fake)[-1]

    for p in params:
      if p.grad is not None:
        p.grad.data.zero_()

    if improved:
      Prior_Loss = torch.mean(torch.nn.functional.softplus(log_sum_exp(label))) - torch.mean(log_sum_exp(label))
      # Prior_Loss =  torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(label.gather(1, iden.view(-1, 1)))  #1 class prior
    else:
      Prior_Loss = - label.mean()
    Iden_Loss = criterion(out, iden)

    Total_Loss = Prior_Loss + lamda * Iden_Loss

    Total_Loss.backward()
    solver.step()

    z = torch.clamp(z.detach(), -clip_range, clip_range).float()

    Prior_Loss_val = Prior_Loss.item()
    Iden_Loss_val = Iden_Loss.item()

    if (i + 1) % 300 == 0:
      fake_img = G(z.detach())
      eval_prob = E(utils.low2high(fake_img))[-1]
      eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
      acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
      print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.2f}".format(i + 1, Prior_Loss_val,
                                                                                          Iden_Loss_val, acc))

  interval = time.time() - tf
  print("Time:{:.2f}".format(interval))

  dir_str = "../../res/images/generated/" + str(DATASET_NAME) + "/" + filename_postfix
  try:
    os.makedirs(dir_str)
  except FileExistsError:
    pass

  res = []
  res5 = []
  seed_acc = torch.zeros((bs, 5))
  for random_seed in range(num_seeds):
    tf = time.time()
    z = reparameterize(mu, log_var)
    fake = G(z)
    score = T(fake)[-1]
    eval_prob = E(utils.low2high(fake))[-1]
    eval_iden = torch.argmax(eval_prob, dim=1).view(-1)

    cnt, cnt5 = 0, 0
    for i in range(bs):
      gt = iden[i].item()
      sample = fake[i]
      save_image(sample.detach(), dir_str, "attack_iden_{}_{}.png".format(gt, random_seed))
      if eval_iden[i].item() == gt:
        seed_acc[i, random_seed] = 1
        cnt += 1
        best_img = G(z)[i]
        save_image(best_img.detach(), dir_str, "{}_attack_iden_{}_{}.png".format(itr, gt, int(no[i])))
        no[i] += 1
      _, top5_idx = torch.topk(eval_prob[i], 5)
      if gt in top5_idx:
        cnt5 += 1

    interval = time.time() - tf
    print("Time:{:.2f}\tSeed:{}\tAcc:{:.2f}\t".format(interval, random_seed, cnt * 1.0 / bs))
    res.append(cnt * 1.0 / bs)
    res5.append(cnt5 * 1.0 / bs)

    torch.cuda.empty_cache()

  acc, acc_5 = statistics.mean(res), statistics.mean(res5)
  acc_var = statistics.variance(res)
  acc_var5 = statistics.variance(res5)
  print("Acc:{:.2f}\tAcc_5:{:.2f}\tAcc_var:{:.4f}\tAcc_var5:{:.4f}".format(acc, acc_5, acc_var, acc_var5))

  return acc, acc_5, acc_var, acc_var5

def recovery_with_gan(generator, discriminator, target_model, eval_model, batch_size, target_class, filename_postfix="") :
  aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0
  for i in range(1):
    iden = torch.ones(batch_size)*target_class #torch.from_numpy(np.arange(60)) # evaluate on the first 300 identities only
    for idx in range(5):
      print("--------------------- Attack batch [%s]------------------------------" % idx)
      acc, acc5, var, var5 = dist_inversion(generator, discriminator, target_model, eval_model, iden, itr=i,
                                            target_class=target_class, lr=2e-2, momentum=0.9, lamda=100, iter_times=2400,
                                            clip_range=1, improved=False, num_seeds=5, filename_postfix=filename_postfix)
      #iden = iden + 60
      aver_acc += acc / 5
      aver_acc5 += acc5 / 5
      aver_var += var / 5
      aver_var5 += var5 / 5

  print("Average Acc:{:.2f}\tAverage Acc5:{:.2f}\tAverage Acc_var:{:.4f}\tAverage Acc_var5:{:.4f}".format(
    aver_acc, aver_acc5, aver_var, aver_var5))


parser = ArgumentParser(description='Inversion defense backdoor')
parser.add_argument('--dataset', type=str, default="imagenet")#cifar10
parser.add_argument('--target_class', type=int, default=107) #2
parser.add_argument("--filename_postfix", type=str , default="107-281")
params = parser.parse_args()


DATASET_NAME = params.dataset
filename_postfix = params.filename_postfix
target_class = params.target_class

network_pkl = "../../res/models/stylegan2-cifar10-32x32.pkl"
print('Loading networks from "%s"...' % network_pkl)
DEVICE = torch.device('cuda:' + str(3))
with dnnlib.util.open_url(network_pkl) as f:
  GAN = legacy.load_network_pkl(f)  # type: ignore

G = GAN['G_ema'].to(DEVICE)
D = GAN['D'].to(DEVICE)
recovery_with_gan(generator=G, discriminator=D, target_model=model_eval, eval_model=model_clean, filename_postfix=filename_postfix, target_class=target_class)
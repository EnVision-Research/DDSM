import copy
import json
import os
import random
import warnings
from absl import app, flags
import time

import torch
from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.core.mutation import Mutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.pntx import SinglePointCrossover
from profile import model_profiling
import numpy as np
from ptflops import get_model_complexity_info

from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from model import UNet, EnsembleUNet
from slimmable_model import SlimmableUNet, StepAwareUNet
from slimmable_model_g16 import Slimmable16UNet, StepAware16UNet
from score.both import get_inception_and_fid_score, get_fid_score



FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
flags.DEFINE_bool('eval_stepaware', False, help='load ckpt.pt and evaluate FID and IS')
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 800000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_EPS', help='log directory')
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer('save_step', 5000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')
flags.DEFINE_string('ckpt_name', 'ckpt', help='ckpt name')
# slimmable
flags.DEFINE_bool('slimmable_unet', False, help='use slimmable unet')
flags.DEFINE_bool('slimmable_g16', False, help='g16 slimmable unet')
flags.DEFINE_bool('sandwich', False, help='use sandiwch training')
flags.DEFINE_float('min_width', 0.25, help="min_width")
flags.DEFINE_integer('num_sandwich_sampling', 3, help='the number of sandwich training samples')
flags.DEFINE_multi_float('candidate_width', [0.75, 0.5], help='candidate_width')
flags.DEFINE_float('assigned_width', 1.0, help="assigned_width")
# ensemble
flags.DEFINE_bool('eval_ensemble', False, help='eval ensemble model')
flags.DEFINE_string('large_logdir', './logs/DDPM_CIFAR10_EPS', help='large model log directory')
flags.DEFINE_string('small_logdir', './logs/DDPM_CIFAR10_EPS', help='small model log directory')
flags.DEFINE_integer('small_ch', 64, help='channel of small model')
flags.DEFINE_integer('start', 200, help='the start step of small model')
flags.DEFINE_integer('end', 0, help='the end step of small model')
# search
flags.DEFINE_bool('search', False, help='search model')
flags.DEFINE_integer('num_generation', 1000, help='the number of generation')
flags.DEFINE_integer('pop_size', 10, help='the size of population')
flags.DEFINE_float('fid_weight', 0.5, help="fid_weight")
flags.DEFINE_float('macs_weight', 0.001, help="macs_weight")
flags.DEFINE_float('mutation_prob', 0.001, help="mutation_prob")
flags.DEFINE_bool('random_init', False, help='search model')
# profile
flags.DEFINE_bool('profile', False, help='profile model')

device = torch.device('cuda:0')


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def evaluate(sampler, model, fid_only=False):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
            ti = time.time()
            batch_images = sampler(x_T.to(device)).cpu()
            print(str(time.time() - ti))
            with open('time.txt', 'w') as f:
                f.write(str(time.time() - ti))
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    if fid_only:
        FID = get_fid_score(
            images, FLAGS.fid_cache, num_images=FLAGS.num_images,
            use_torch=FLAGS.fid_use_torch, verbose=True)
        return FID
    else:
        (IS, IS_std), FID = get_inception_and_fid_score(
            images, FLAGS.fid_cache, num_images=FLAGS.num_images,
            use_torch=FLAGS.fid_use_torch, verbose=True)
        return (IS, IS_std), FID, images


def train():
    # dataset
    dataset = CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)

    # model setup
    if FLAGS.slimmable_unet:
        if FLAGS.slimmable_g16:
            net_model = Slimmable16UNet(
                T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
                num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
        else:
            net_model = SlimmableUNet(
                T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
                num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    else:
        net_model = UNet(
            T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
            num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device)
    net_sampler = GaussianDiffusionSampler(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    if FLAGS.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)
        ema_sampler = torch.nn.DataParallel(ema_sampler)

    # log setup
    os.makedirs(os.path.join(FLAGS.logdir, 'sample'), exist_ok=True)
    if FLAGS.sandwich:
        os.makedirs(os.path.join(FLAGS.logdir, 'supernet_sample'))
        os.makedirs(os.path.join(FLAGS.logdir, 'minnet_sample'))

    x_T = torch.randn(FLAGS.sample_size, 3, FLAGS.img_size, FLAGS.img_size)
    x_T = x_T.to(device)
    grid = (make_grid(next(iter(dataloader))[0][:FLAGS.sample_size]) + 1) / 2
    writer = SummaryWriter(FLAGS.logdir)
    writer.add_image('real_sample', grid)
    writer.flush()
    # backup all arguments
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())
    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    # start training
    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            if FLAGS.sandwich:
                if FLAGS.parallel:
                    assert isinstance(trainer.module.model, SlimmableUNet)
                else:
                    assert isinstance(trainer.model, SlimmableUNet)
                optim.zero_grad()
                x_0 = next(datalooper).to(device)

                # supernet
                if FLAGS.parallel:
                    trainer.module.model.apply(lambda m: setattr(m, 'width_mult', 1.0))
                else:
                    trainer.model.apply(lambda m: setattr(m, 'width_mult', 1.0))
                supernet_loss = trainer(x_0).mean()
                supernet_loss.backward()

                # minnet
                if FLAGS.parallel:
                    trainer.module.model.apply(lambda m: setattr(m, 'width_mult', FLAGS.min_width))
                else:
                    trainer.model.apply(lambda m: setattr(m, 'width_mult', FLAGS.min_width))
                minnet_loss = trainer(x_0).mean()
                minnet_loss.backward()

                # midnet
                for i in range(FLAGS.num_sandwich_sampling-2):
                    mid_width = random.choice(FLAGS.candidate_width)
                    if FLAGS.parallel:
                        trainer.module.model.apply(lambda m: setattr(m, 'width_mult', mid_width))
                    else:
                        trainer.model.apply(lambda m: setattr(m, 'width_mult', mid_width))
                    loss = trainer(x_0).mean()
                    loss.backward()

                # optim
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), FLAGS.grad_clip)
                optim.step()
                sched.step()
                ema(net_model, ema_model, FLAGS.ema_decay)
                if FLAGS.parallel:
                    trainer.module.model.apply(lambda m: setattr(m, 'width_mult', 1.0))
                else:
                    trainer.model.apply(lambda m: setattr(m, 'width_mult', 1.0))

                # log
                writer.add_scalar('supernet_loss', supernet_loss, step)
                writer.add_scalar('minnet_loss', minnet_loss, step)
                pbar.set_postfix(supernet_loss='%.3f' % supernet_loss, minnet_loss='%.3f' % minnet_loss)

                # sample
                if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
                    net_model.eval()
                    if FLAGS.parallel:
                        trainer.module.model.apply(lambda m: setattr(m, 'width_mult', 1.0))
                    else:
                        trainer.model.apply(lambda m: setattr(m, 'width_mult', 1.0))
                    with torch.no_grad():
                        x_0 = ema_sampler(x_T)
                        grid = (make_grid(x_0) + 1) / 2
                        path = os.path.join(
                            FLAGS.logdir, 'supernet_sample', '%d.png' % step)
                        save_image(grid, path)
                        writer.add_image('supernet_sample', grid, step)

                    if FLAGS.parallel:
                        trainer.module.model.apply(lambda m: setattr(m, 'width_mult', FLAGS.min_width))
                    else:
                        trainer.model.apply(lambda m: setattr(m, 'width_mult', FLAGS.min_width))
                    with torch.no_grad():
                        x_0 = ema_sampler(x_T)
                        grid = (make_grid(x_0) + 1) / 2
                        path = os.path.join(
                            FLAGS.logdir, 'minnet_sample', '%d.png' % step)
                        save_image(grid, path)
                        writer.add_image('minnet_sample', grid, step)
                    net_model.train()
            else:
                # train
                optim.zero_grad()
                x_0 = next(datalooper).to(device)
                loss = trainer(x_0).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), FLAGS.grad_clip)
                optim.step()
                sched.step()
                ema(net_model, ema_model, FLAGS.ema_decay)

                # log
                writer.add_scalar('loss', loss, step)
                pbar.set_postfix(loss='%.3f' % loss)

                # sample
                if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
                    net_model.eval()
                    with torch.no_grad():
                        x_0 = ema_sampler(x_T)
                        grid = (make_grid(x_0) + 1) / 2
                        path = os.path.join(
                            FLAGS.logdir, 'sample', '%d.png' % step)
                        save_image(grid, path)
                        writer.add_image('sample', grid, step)
                    net_model.train()

            # save
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'ckpt.pt'))
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'ckpt_{}.pt'.format(step)))

            # evaluate
            if FLAGS.eval_step > 0 and step % FLAGS.eval_step == 0:
                net_IS, net_FID, _ = evaluate(net_sampler, net_model)
                ema_IS, ema_FID, _ = evaluate(ema_sampler, ema_model)
                metrics = {
                    'IS': net_IS[0],
                    'IS_std': net_IS[1],
                    'FID': net_FID,
                    'IS_EMA': ema_IS[0],
                    'IS_std_EMA': ema_IS[1],
                    'FID_EMA': ema_FID
                }
                pbar.write(
                    "%d/%d " % (step, FLAGS.total_steps) +
                    ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
                    metrics['step'] = step
                    f.write(json.dumps(metrics) + "\n")
    writer.close()


def Eval():
    # model setup
    if FLAGS.slimmable_unet:
        model = SlimmableUNet(
            T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
            num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
        model.apply(lambda m: setattr(m, 'width_mult', FLAGS.assigned_width))
    else:
        model = UNet(
            T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
            num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    sampler = GaussianDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate
    ckpt = torch.load(os.path.join(FLAGS.logdir, '{}.pt'.format(FLAGS.ckpt_name)))
    # model.load_state_dict(ckpt['net_model'])
    # (IS, IS_std), FID, samples = evaluate(sampler, model)
    # print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    # save_image(
    #     torch.tensor(samples[:256]),
    #     os.path.join(FLAGS.logdir, 'samples.png'),
    #     nrow=16)

    model.load_state_dict(ckpt['ema_model'])
    (IS, IS_std), FID, samples = evaluate(sampler, model)
    if FLAGS.slimmable_unet:
        print('width: {}'.format(int(FLAGS.assigned_width * FLAGS.ch)))
    print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(FLAGS.logdir, 'samples_ema.png'),
        nrow=16)

def eval_stepaware():
    # model setup
    with open(os.path.join(FLAGS.logdir, 'search.txt'), 'r') as f:
        strategy = eval(f.readlines()[0])
    model = StepAwareUNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout, strategy=strategy)
    sampler = GaussianDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate
    ckpt = torch.load(os.path.join(FLAGS.logdir, '{}.pt'.format(FLAGS.ckpt_name)))
    # model.load_state_dict(ckpt['net_model'])
    # (IS, IS_std), FID, samples = evaluate(sampler, model)
    # print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    # save_image(
    #     torch.tensor(samples[:256]),
    #     os.path.join(FLAGS.logdir, 'samples.png'),
    #     nrow=16)

    model.load_state_dict(ckpt['ema_model'])
    (IS, IS_std), FID, samples = evaluate(sampler, model)
    print(strategy)
    print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(FLAGS.logdir, 'samples_ema.png'),
        nrow=16)


def eval_ensemble():
    os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    # model setup
    model = EnsembleUNet(
        T=FLAGS.T, large_ch=FLAGS.ch, small_ch=FLAGS.small_ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout, start=FLAGS.start, end=FLAGS.end)

    sampler = GaussianDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate
    large_ckpt = torch.load(os.path.join(FLAGS.large_logdir, '{}.pt'.format(FLAGS.ckpt_name)))
    small_ckpt = torch.load(os.path.join(FLAGS.small_logdir, '{}.pt'.format(FLAGS.ckpt_name)))
    # model.load_state_dict(ckpt['net_model'])
    # (IS, IS_std), FID, samples = evaluate(sampler, model)
    # print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    # save_image(
    #     torch.tensor(samples[:256]),
    #     os.path.join(FLAGS.logdir, 'samples.png'),
    #     nrow=16)

    model.large_model.load_state_dict(large_ckpt['ema_model'])
    model.small_model.load_state_dict(small_ckpt['ema_model'])
    (IS, IS_std), FID, samples = evaluate(sampler, model)
    if FLAGS.slimmable_unet:
        print('width: {}'.format(int(FLAGS.assigned_width * FLAGS.ch)))
    print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(FLAGS.logdir, 'samples_ema.png'),
        nrow=16)


def search():
    # model setup
    model = StepAwareUNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout, strategy=None)

    sampler = GaussianDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model
    ckpt = torch.load(os.path.join(FLAGS.logdir, '{}.pt'.format(FLAGS.ckpt_name)))
    model.load_state_dict(ckpt['ema_model'])

    # baseline FID
    model.apply(lambda m: setattr(m, 'width_mult', 1.0))
    model.strategy = [1.0 for i in range(FLAGS.T)]

    baseline_FID = evaluate(sampler, model, fid_only=True)
    print("baseline Model(EMA): FID:%7.3f" % (baseline_FID))

    all_candidates = [1]
    all_candidates.extend(FLAGS.candidate_width)
    all_candidates.append(FLAGS.min_width)

    def nparray2strategy(x):
        return [all_candidates[i] for i in x.tolist()]

    # search
    class StepAwareProblem(Problem):
        def __init__(self, fid_weight, macs_weight):
            super().__init__(n_var=FLAGS.T, n_obj=1, n_ieq_constr=0, xl=0, xu=len(FLAGS.candidate_width)+2)
            self.fid_weight = fid_weight
            self.macs_weight = macs_weight

        def _evaluate(self, x, out, *args, **kwargs):
            population_size, var_dim = x.shape
            FID = np.zeros([population_size])
            macs = np.zeros([population_size])
            for pop in range(population_size):
                model.strategy = nparray2strategy(x[pop])
                print(nparray2strategy(x[pop]))

                FID[pop] = evaluate(sampler, model, fid_only=True)
                print("Model(EMA): FID:%7.3f" % (FID[pop]))

                def get_macs(strategy):
                    pre_calculate_macs = [6070, 3420, 1520, 382.5]
                    return sum([strategy.tolist().count(i) * pre_calculate_macs[i] for i in range(int(self.xu[0]))]) / len(strategy)

                macs[pop] = get_macs(x[pop])
                print("Model Macs: {} MMac".format(macs[pop]))

            out["F"] = FID * self.fid_weight + macs * self.macs_weight
            # out["G"] = baseline_FID - FID

    class MySampling(FloatRandomSampling):
        def _do(self, problem, n_samples, **kwargs):
            init_num = len(FLAGS.candidate_width)+2
            assert n_samples >= init_num
            if FLAGS.random_init:
                X = super()._do(problem, n_samples, **kwargs)
            else:
                X_init = np.ones([init_num, problem.n_var], dtype=np.float64)
                for i in range(len(X_init)):
                    X_init[i] = X_init[i] * i
                X = super()._do(problem, n_samples-init_num, **kwargs)
                X = np.concatenate([X_init, X], axis=0)
            return np.floor(X).astype(int)

    class MyMutation(Mutation):
        def __init__(self, prob=None):
            super().__init__()
            if prob is not None:
                self.prob = float(prob)
            else:
                self.prob = None

        def _do(self, problem, X, **kwargs):
            # X: Population with several Individuals
            # An Individual.X can be transfer to np.ndarray
            if self.prob is None:
                self.prob = 1.0 / problem.n_var

            do_mutation = np.random.random([X.shape[0], X.shape[1]]) < self.prob
            xl = problem.xl[None, :][0]
            xu = problem.xu[None, :][0]
            for idx in range(len(X)):
                mut = do_mutation[idx]
                X[idx] = [np.random.randint(x, y) if mut[j] else X[idx][j] for j, (x, y) in enumerate(zip(xl, xu))]
            return X

    problem = StepAwareProblem(fid_weight=FLAGS.fid_weight, macs_weight=FLAGS.macs_weight)
    algorithm = NSGA2(pop_size=FLAGS.pop_size,
                  sampling=MySampling(),
                  crossover=SinglePointCrossover(),
                  mutation=MyMutation(prob=FLAGS.mutation_prob),
                  eliminate_duplicates=True)
    results = minimize(problem,
                       algorithm,
                       termination=('n_gen', FLAGS.num_generation),
                       seed=1,
                       verbose=False,
                       save_history=True,
                       )
    print(results.X)
    print(nparray2strategy(results.X))
    with open(os.path.join(FLAGS.logdir, 'search.txt'), 'a') as f:
        f.write(str(nparray2strategy(results.X)))

    def get_macs(strategy):
        pre_calculate_macs = [6070, 3420, 1520, 382.5]
        return sum([strategy.tolist().count(i) * pre_calculate_macs[i] for i in range(int(problem.xu[0]))]) / len(strategy)

    searched_macs = get_macs(results.X)
    print("Model Macs: {} MMac".format(searched_macs))


def profile():
    # model setup
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)

    def my_input_constructor(input_res):
        try:
            batch = torch.ones(()).new_empty((1, *input_res),
                                             dtype=next(model.parameters()).dtype,
                                             device=next(model.parameters()).device)
        except StopIteration:
            batch = torch.ones(()).new_empty((1, *input_res))
        t = torch.ones((1), dtype=torch.long, device=next(model.parameters()).device)
        return {'x': batch, 't': t}

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (3, FLAGS.img_size, FLAGS.img_size), as_strings=True,
                                                 input_constructor=my_input_constructor,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train:
        train()
    if FLAGS.eval:
        Eval()
    if FLAGS.eval_ensemble:
        eval_ensemble()
    if FLAGS.eval_stepaware:
        eval_stepaware()
    if FLAGS.search:
        search()
    if FLAGS.profile:
        profile()
    # if not FLAGS.train and not FLAGS.eval:
    #     print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)

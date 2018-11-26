import math
import argparse
import GPy
import GPyOpt
import os
import sys
sys.path.append('./script/training')
from train import train

COUNT = 0


def wrapper(params, model_type, pl_type, gate_mode, device, out_dir):
    global COUNT
    COUNT += 1

    lr, w_decay = params[0]
    lr = 10**lr
    w_decay = 10**w_decay

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    best_val = train(
        model_type=model_type,
        pl_type=pl_type,
        gate_mode=gate_mode,
        lr=lr,
        w_decay=w_decay,
        epoch=5,
        device=device,
        out_pref=out_dir + '%i-' % COUNT)

    return -best_val


if __name__ == '__main__':
    import datetime
    import argparse

    parser = argparse.ArgumentParser('paraphrase detection')
    parser.add_argument(
        '--max_iter',
        '-i',
        type=int,
        default=None,
        help='maximum iteration for Bayesian optimization')
    parser.add_argument(
        '--device', '-d', type=int, default=0, help='gpu device id <int>')
    parser.add_argument('--model_type', '-mt', default='vis+lng')
    parser.add_argument('--pl_type', '-pt', default='ddpn')
    parser.add_argument('--gate_mode', '-gm', default=None)
    args = parser.parse_args()

    bounds = [
        {
            'name': 'lr',
            'type': 'continuous',
            'domain': (-5, -1),
            'dimensionality': 1
        },
        {
            'name': 'w_decay',
            'type': 'continuous',
            'domain': (-7, -3),
            'dimensionality': 1
        },
    ]
    
    configs = [x for x in [args.model_type, args.pl_type, args.gate_mode] if x is not None]
    out_dir = './bo_out_downsample/%s/' % '+'.join(configs)
    prob = GPyOpt.methods.BayesianOptimization(
        lambda params: wrapper(params, args.model_type, args.pl_type, args.gate_mode, args.device, out_dir),
        bounds,
        acquisition_type='EI',
    )

    now = datetime.datetime.now()

    report_file = out_dir + 'bo_report_%s.txt' % (
        "{0:%Y-%m-%d %H:%M:%S}".format(now))
    evaluation_file = out_dir + 'bo_eval_%s.txt' % (
        "{0:%Y-%m-%d %H:%M:%S}".format(now))
    prob.run_optimization(
        args.max_iter,
        report_file=report_file,
        evaluations_file=evaluation_file)

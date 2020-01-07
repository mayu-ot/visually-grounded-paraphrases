import pandas as pd
import os
import GPyOpt
from eval_by_clustering import eval_ari


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('paraphrase detection')
    parser.add_argument('result_file', type=str)
    parser.add_argument('--max_iter', '-i', type=int, default=100,
                        help='maximum iteration for Bayesian optimization')
    args = parser.parse_args()
    
    df = pd.read_csv(args.result_file)
    result_dir = os.path.dirname(args.result_file)
    
    bounds = [{'name': 'clutter', 'type': 'continuous', 'domain': (0., 100.)}]
    wrapper = lambda x: -1 * eval_ari(df, clutter=x[0][0], damping=.5) 
        
    prob = GPyOpt.methods.BayesianOptimization(
        wrapper,
        bounds,
        acquisition_type='EI',
        normalize_Y=True,
        acquisition_weight=2
    )
    
    report_file = f'{result_dir}/bo_clustering_param.report'
    evaluations_file = f'{result_dir}/bo_clustering_param.eval'
    prob.run_optimization(args.max_iter, report_file=report_file, evaluations_file=evaluations_file)
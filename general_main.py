import argparse
import random
import numpy as np
import torch
from experiment.run import multiple_run
from utils.utils import boolean_string

def main(args):
    print(args)
    # set up seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    args.trick = {'labels_trick': args.labels_trick, 'separated_softmax': args.separated_softmax,
                  'kd_trick': args.kd_trick, 'kd_trick_star': args.kd_trick_star, 'review_trick': args.review_trick,
                  'nmc_trick': args.nmc_trick}
    multiple_run(args)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Online Task-free Continual Learning PyTorch")
    ########################General#########################
    parser.add_argument('--num_runs', dest='num_runs', default=1, type=int,
                        help='Number of runs (default: %(default)s)')
    parser.add_argument('--seed', dest='seed', default=0, type=int,
                        help='Random seed')

    ########################Misc#########################
    parser.add_argument('--val_size', dest='val_size', default=0.1, type=float,
                        help='val_size (default: %(default)s)')
    parser.add_argument('--num_val', dest='num_val', default=3, type=int,
                        help='Number of batches used for validation (default: %(default)s)')
    parser.add_argument('--num_runs_val', dest='num_runs_val', default=3, type=int,
                        help='Number of runs for validation (default: %(default)s)')
    parser.add_argument('--error_analysis', dest='error_analysis', default=False, type=boolean_string,
                        help='Perform error analysis (default: %(default)s)')
    parser.add_argument('--verbose', type=boolean_string, default=True,
                        help='print information or not (default: %(default)s)')

    ########################Agent#########################
    parser.add_argument('--agent', dest='agent', default='ER',
                        choices=['ER', 'EWC', 'AGEM', 'LWF', 'ICARL', 'GDUMB', 'ASER', 'LSA_C','OMAS'],
                        help='Agent selection  (default: %(default)s)')
    parser.add_argument('--update', dest='update', default='random', choices=['random', 'GSS'],
                        help='Update method  (default: %(default)s)')
    parser.add_argument('--retrieve', dest='retrieve', default='random', choices=['MIR', 'random', 'LSA_I'],
                        help='Retrieve method  (default: %(default)s)')

    ########################Optimizer#########################
    parser.add_argument('--optimizer', dest='optimizer', default='SGD', choices=['SGD', 'Adam'],
                        help='Optimizer (default: %(default)s)')
    parser.add_argument('--learning_rate', dest='learning_rate', default=0.1,
                        type=float,
                        help='Learning_rate (default: %(default)s)')
    parser.add_argument('--epoch', dest='epoch', default=1,
                        type=int,
                        help='The number of epochs used for one task. (default: %(default)s)')
    parser.add_argument('--batch', dest='batch', default=10,
                        type=int,
                        help='Batch size (default: %(default)s)')
    parser.add_argument('--test_batch', dest='test_batch', default=128,
                        type=int,
                        help='Test batch size (default: %(default)s)')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0,
                        help='weight_decay')

    ########################Data#########################
    parser.add_argument('--num_tasks', dest='num_tasks', default=10,
                        type=int,
                        help='Number of tasks (default: %(default)s), OpenLORIS num_tasks is predefined')
    parser.add_argument('--fix_order', dest='fix_order', default=False,
                        type=boolean_string,
                        help='In NC scenario, should the class order be fixed (default: %(default)s)')
    parser.add_argument('--plot_sample', dest='plot_sample', default=False,
                        type=boolean_string,
                        help='In NI scenario, should sample images be plotted (default: %(default)s)')
    parser.add_argument('--data', dest='data', default="cifar10",
                        help='Path to the dataset. (default: %(default)s)')
    parser.add_argument('--cl_type', dest='cl_type', default="nc", choices=['nc', 'ni'],
                        help='Continual learning type: new class "nc" or new instance "ni". (default: %(default)s)')
    parser.add_argument('--ns_factor', dest='ns_factor', nargs='+',
                        default=(0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6), type=float,
                        help='Change factor for non-stationary data(default: %(default)s)')
    parser.add_argument('--ns_type', dest='ns_type', default='noise', type=str, choices=['noise', 'occlusion', 'blur'],
                        help='Type of non-stationary (default: %(default)s)')
    parser.add_argument('--ns_task', dest='ns_task', nargs='+', default=(1, 1, 2, 2, 2, 2), type=int,
                        help='NI Non Stationary task composition (default: %(default)s)')

    ########################ER#########################
    parser.add_argument('--mem_size', dest='mem_size', default=10000,
                        type=int,
                        help='Memory buffer size (default: %(default)s)')
    parser.add_argument('--eps_mem_batch', dest='eps_mem_batch', default=10,
                        type=int,
                        help='Episode memory per batch (default: %(default)s)')

    ########################EWC##########################
    parser.add_argument('--lambda', dest='lambda_', default=100, type=float,
                        help='EWC regularization coefficient')
    parser.add_argument('--alpha', dest='alpha', default=0.9, type=float,
                        help='EWC++ exponential moving average decay for Fisher calculation at each step')
    parser.add_argument('--fisher_update_after', dest='fisher_update_after', type=int, default=50,
                        help="Number of training iterations after which the Fisher will be updated.")

    ########################MIR#########################
    parser.add_argument('--subsample', dest='subsample', default=50,
                        type=int,
                        help='Number of subsample to perform MIR(default: %(default)s)')

    ########################GSS#########################
    parser.add_argument('--gss_mem_strength', dest='gss_mem_strength', default=10, type=int,
                        help='Number of batches randomly sampled from memory to estimate score')
    parser.add_argument('--gss_batch_size', dest='gss_batch_size', default=10, type=int,
                        help='Random sampling batch size to estimate score')

    ########################GDumb#########################
    parser.add_argument('--minlr', dest='minlr', default=0.0005, type=float, help='Minimal learning rate')
    parser.add_argument('--clip', dest='clip', default=10., type=float,
                        help='value for gradient clipping')
    parser.add_argument('--mem_epoch', dest='mem_epoch', default=70, type=int, help='Epochs to train for memory')

    #######################Tricks#########################
    parser.add_argument('--labels_trick', dest='labels_trick', default=False, type=boolean_string,
                        help='Labels trick')
    parser.add_argument('--separated_softmax', dest='separated_softmax', default=False, type=boolean_string,
                        help='separated softmax')
    parser.add_argument('--kd_trick', dest='kd_trick', default=False, type=boolean_string,
                        help='Knowledge distillation with cross entropy trick')
    parser.add_argument('--kd_trick_star', dest='kd_trick_star', default=False, type=boolean_string,
                        help='Improved knowledge distillation trick')
    parser.add_argument('--review_trick', dest='review_trick', default=False, type=boolean_string,
                        help='Review trick')
    parser.add_argument('--nmc_trick', dest='nmc_trick', default=False, type=boolean_string,
                        help='Use nearest mean classifier')
    parser.add_argument('--mem_iters', dest='mem_iters', default=1, type=int,
                        help='mem_iters')

    ####################Early Stopping######################
    parser.add_argument('--min_delta', dest='min_delta', default=0., type=float,
                        help='A minimum increase in the score to qualify as an improvement')
    parser.add_argument('--patience', dest='patience', default=0, type=int,
                        help='Number of events to wait if no improvement and then stop the training.')
    parser.add_argument('--cumulative_delta', dest='cumulative_delta', default=False, type=boolean_string,
                        help='If True, `min_delta` defines an increase since the last `patience` reset, '
                             'otherwise, it defines an increase after the last event.')
    
    
    ########################LSA#########################
    parser.add_argument('--beta', dest='beta', default=0.1, type=float, help='Trade-off weight to balance the steam-sensitivity and distribution-diversity scores.')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    main(args)

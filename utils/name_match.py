from agents.gdumb import Gdumb
from continuum.dataset_scripts.cifar100 import CIFAR100
from continuum.dataset_scripts.cifar10 import CIFAR10
from continuum.dataset_scripts.core50 import CORE50
from continuum.dataset_scripts.mnist import MNIST
from continuum.dataset_scripts.mini_imagenet import Mini_ImageNet
from continuum.dataset_scripts.openloris import OpenLORIS
from agents.exp_replay import ExperienceReplay
from agents.agem import AGEM
from agents.ewc_pp import EWC_pp
from agents.lwf import Lwf
from agents.icarl import Icarl
from utils.buffer.random_retrieve import Random_retrieve
from utils.buffer.reservoir_update import Reservoir_update
from utils.buffer.mir_retrieve import MIR_retrieve
from utils.buffer.gss_greedy_update import GSSGreedyUpdate
from agents.lsa_c import LSA_C
from utils.buffer.lsa_i import LSA_I
from agents.omas import Task_free_continual_learning

data_objects = {
    'mnist': MNIST,
    'cifar100': CIFAR100,
    'cifar10': CIFAR10,
    'core50': CORE50,
    'mini_imagenet': Mini_ImageNet,
    'openloris': OpenLORIS
}

agents = {
    'ER': ExperienceReplay,
    'EWC': EWC_pp,
    'AGEM': AGEM,
    'LWF': Lwf,
    'ICARL': Icarl,
    'GDUMB': Gdumb,
    'LSA_C' : LSA_C,
    'OMAS': Task_free_continual_learning
}

retrieve_methods = {
    'MIR': MIR_retrieve,
    'random': Random_retrieve,
    'LSA_I': LSA_I
}

update_methods = {
    'random': Reservoir_update,
    'GSS': GSSGreedyUpdate,
}


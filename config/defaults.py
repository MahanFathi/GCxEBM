import ml_collections

# ---------------------------------------------------------------------------- #
# Define Config Node
# ---------------------------------------------------------------------------- #
_C = ml_collections.ConfigDict()
_C.EXP_NAME = ""
_C.SEED = 0
_C.WANDB = True
_C.DEBUG = False
_C.MOCK_TPU = False


# ---------------------------------------------------------------------------- #
# ENVIRONMENT
# ---------------------------------------------------------------------------- #
_C.ENV = ml_collections.ConfigDict()
_C.ENV.ENV_NAME = "ant"
_C.ENV.TIMESTEP = 0.05


# ---------------------------------------------------------------------------- #
# POLICY
# ---------------------------------------------------------------------------- #
_C.POLICY = ml_collections.ConfigDict()
_C.POLICY.CLASS = "EBMPolicy"

# ---------------------------------------------------------------------------- #
# ENERGY-BASED MODEL
# ---------------------------------------------------------------------------- #
_C.EBM = ml_collections.ConfigDict()
_C.EBM.ARCH = "arch0" # {"arch0": simple_feed_forward, "arch1": multipe_mlps}
_C.EBM.ARCH0 = ml_collections.ConfigDict()
_C.EBM.ARCH0.LAYERS = [256, 256, 128, 128]
_C.EBM.ARCH1 = ml_collections.ConfigDict()
_C.EBM.ARCH1.F_LAYERS = [128, 128, 64, 64]
_C.EBM.ARCH1.G_LAYERS = [128, 128, 64, 64]
_C.EBM.OPTION_SIZE = 1 # unused for now
_C.EBM.ALPHA = 1e-3 # internal GD step size
_C.EBM.LANGEVIN_GD = True # if True do GD with Langevin noise
_C.EBM.K = 10 # internal optimization #steps
_C.EBM.GRAD_CLIP = 1.0 # grad clipping during inference. 0.0 -> no clipping


# ---------------------------------------------------------------------------- #
# G2Z MLP
# ---------------------------------------------------------------------------- #
_C.G2Z = ml_collections.ConfigDict()
_C.G2Z.LAYERS = [32, 32, 16]


# ---------------------------------------------------------------------------- #
# VALUE NET
# ---------------------------------------------------------------------------- #
_C.VALUE_NET = ml_collections.ConfigDict()
_C.VALUE_NET.FEATURES = [256, 256, 256, 256, 256]
_C.VALUE_NET.LR = 1e-3


# ---------------------------------------------------------------------------- #
# TRAIN
# ---------------------------------------------------------------------------- #
_C.TRAIN = ml_collections.ConfigDict()
_C.TRAIN.LOSS_FN = "ppo_loss"
_C.TRAIN.NUM_TIMESTEPS = 1e7
_C.TRAIN.EPISODE_LENGTH = 1000
_C.TRAIN.NUM_UPDATE_EPOCHS = 4
_C.TRAIN.ACTION_REPEAT = 1
_C.TRAIN.NUM_ENVS = 64 # 2048
_C.TRAIN.MAX_DEVICES_PER_HOST = 8
_C.TRAIN.LEARNING_RATE = 3e-4
_C.TRAIN.UNROLL_LENGTH = 10
_C.TRAIN.BATCH_SIZE = 1024
_C.TRAIN.NUM_MINIBATCHES = 32
_C.TRAIN.NORMALIZE_OBSERVATIONS = True
_C.TRAIN.DISCOUNTING = 0.97
_C.TRAIN.REWARD_SCALING = 10.0
_C.TRAIN.NUM_EVAL_ENVS = 10


# ------------------------------------------------------------------------ #
# PPO
# ------------------------------------------------------------------------ #
_C.TRAIN.PPO = ml_collections.ConfigDict()
_C.TRAIN.PPO.ENTROPY_COST = 1e-2
_C.TRAIN.PPO.GAE_LAMBDA = 0.95
_C.TRAIN.PPO.EPSILON = 0.3


# ---------------------------------------------------------------------------- #
# LOGGING
# ---------------------------------------------------------------------------- #
_C.LOG = ml_collections.ConfigDict()
_C.LOG.FREQUENCY = 20
_C.LOG.TO_FILE = True # is allowed to create files
_C.LOG.SAVE_PARAMS = True



def get_config():
    return _C

from enum import Enum

IMAGENET_TRAIN = 'imagenet-train'
IMAGENET_TEST = 'imagenet-test'

class ATTACK_NAME(Enum):
  SQUARE_ATTACK = "square"
  FAB = "fab-ut"
  APGD_CE = "apgd-ce"
  APGD_DLR = "apgd-dlr"

class COSINE_SIM_MODE(Enum) :
  MAX = 'max'
  SUM_SQUARE = 'sum_square'
  SUM = 'sum'

class DATASET(Enum) :
  CIFAR10 = 'cifar10'
  CIFAR100 = 'cifar100'
  SVHN = 'SVHN'
  IMAGENET = 'imagenet'
  YTF = 'YTF'

class POISONED_MODEL_NAMES(Enum):
  COMPOSITE_BACKDOOR_ATTACK = "composite_backdoor"
  BASE_SEMANTIC_ATTACK = "base_semantic_backdoor"

class POISONED_MODEL_ARCHITECTURE(Enum):
  RESNET18 = "resnet18"
  WIDERESNET = "wideresnet"
  XCIT_S = "xcit-s"

class GENERATE_FROM_START_IMAGE(Enum) :
  RANDOM_HOMOGENEOUS_COLOR = 'color'
  RANDOM_NONTARGET_IMAGE = 'nontarget'
  RANDOM_IMAGE = 'random'

class AUGMENTATIONS(Enum) :
  RANDOM_HORIZONTAL_FLIP = "RandomHorizontalFlip"
  RANDOM_CROP = "RandomCrop"
  COLOR_JITTER = "ColorJitter"

class MODE(Enum) :
  CLEAN = "Clean"
  ROBUST_TRAIN = "RobustTrain"
  CLEAN_N_MINUS_1_ROBUST_TRAIN = "Cln-1RobTrain"
  GENERATOR_TRAIN = "GeneratorTrain"
  GENERATOR_RECOVERY = "GeneratorRecovery"
  FINE_TUNE_CLOSE_TO_REFERENCE_IMAGE = "FineTuneCloseToReferenceImage"
  GENERATE_IMAGES = "GenerateImages"
  SOME_DISTANT_IMAGES = "SomeDistantImages"
  ALL_DISTANT_IMAGES = "AllDistantImages"
  ROBUST_EVAL = "RobustEval"
  COSINE_SIM_EVAL = "CosineSimEval"
  GENERATED_EVAL = "GeneratedEval"

class IMAGE_MODE(Enum) :
  JPEG = "JPEG"
  PNG = "PNG"

IMAGE_SHAPE = {}
VAL_SIZE = {}
COLOR_CHANNEL = {}
NUM_OF_CLASS = {}
MEAN = {}
STD = {}
SAMPLES_PER_EPOCH = {}

# Mean and std deviation
#  of imagenet dataset. Source: http://cs231n.stanford.edu/reports/2017/pdfs/101.pdf
IMAGE_SHAPE[DATASET.IMAGENET.value] = [224, 224]
VAL_SIZE[DATASET.IMAGENET.value] = 128117
COLOR_CHANNEL[DATASET.IMAGENET.value] = 3
MEAN[DATASET.IMAGENET.value] = [0.485, 0.456, 0.406]
STD[DATASET.IMAGENET.value] = [0.229, 0.224, 0.225]
NUM_OF_CLASS[DATASET.IMAGENET.value] = 1000
SAMPLES_PER_EPOCH[DATASET.IMAGENET.value] = 1281167


#  of cifar10 dataset.
IMAGE_SHAPE[DATASET.CIFAR10.value] = [32, 32]
VAL_SIZE[DATASET.CIFAR10.value] = 5000
COLOR_CHANNEL[DATASET.CIFAR10.value] = 3
MEAN[DATASET.CIFAR10.value] = [0.4914, 0.4822, 0.4465]
STD[DATASET.CIFAR10.value] = [0.2471, 0.2435, 0.2616]
NUM_OF_CLASS[DATASET.CIFAR10.value] = 10
SAMPLES_PER_EPOCH[DATASET.CIFAR10.value] = 50000

IMAGE_SHAPE[DATASET.SVHN.value] = [32, 32]
VAL_SIZE[DATASET.SVHN.value] = 5000
COLOR_CHANNEL[DATASET.SVHN.value] = 3
MEAN[DATASET.SVHN.value] = [0.4376821, 0.4437697, 0.47280442]
STD[DATASET.SVHN.value] = [0.19803012, 0.20101562, 0.19703614]
NUM_OF_CLASS[DATASET.SVHN.value] = 10

# image_shape[DATASET.YTF.value] = [224, 224] => composite attack
IMAGE_SHAPE[DATASET.YTF.value] = [224, 224]
COLOR_CHANNEL[DATASET.YTF.value] = 3
MEAN[DATASET.YTF.value] = [0.485, 0.456, 0.406]
STD[DATASET.YTF.value] = [0.229, 0.224, 0.225]
NUM_OF_CLASS[DATASET.YTF.value] = 1203
VAL_SIZE[DATASET.YTF.value] = 12000
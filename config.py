# coding:utf8
import warnings
import torch 

class DefaultConfig(object): # 类似结构体 加一个 用命令行参数覆盖默认参数的 parse 方法
    model_name = 'MyVgg16'  # 使用的模型，名字必须与models/__init__.py中的名字一致
    dataset_name = "CornellDataset"
    batch_size = 8 # batch size 多卡可以加大n倍， 越大训练越快
    num_epoch = 40
    lr = 0.001  # initial learning rate 0.001
    lr_decay = 0.8  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-6  # 权重衰减系数 1e-6

    gpu_ids = "1"         # 使用的gpu卡号
    use_gpu = True          # user GPU or not
    multi_gpu = False       # True
    start_epoch = 0         # 继续训练开始的epoch

    train_data_root = './data/train'  # 训练集存放路径
    test_data_root = './data/test'  # 测试集存放路径
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载 !!!注意，可以= ./补全路径 但之后必须删掉=./中间的空格，否则读的是None
    num_workers = 6  # how many workers for loading data
    input_size = 224

    print_freq = 20  # print info every N batch
    env = 'main'  # visdom 环境
    vis_port =8097 # visdom 端口
    out_dir='./out'

    params = [ # 网格搜索参数
          {'batch_size': [8,16], 'lr': [0.0001, 0.001], 'lr_decay': [0.8, 0.9], 'weight_decay': [0e-6, 1e-6]} # group3 4 5
    ]

    def _parse(self, kwargs): # 用命令行参数覆盖默认参数
        # 根据字典kwargs 更新 config参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        
        opt.device = torch.device('cuda') if opt.use_gpu else torch.device('cpu') 
        opt.env = f'{opt.model_name}_{opt.dataset_name}_lr{opt.lr}ld{opt.lr_decay}bs{opt.batch_size}wd{opt.weight_decay}'

    def _print_conf(self):
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig()

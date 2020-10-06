#!/usr/bin/env python3
#coding:utf8

from config import opt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids # 必须在 import torch 之前

import torch
from dataset import CornellDataset
from loss import MSELoss, RectAngleMetric
from torch.utils.data import DataLoader
import models
from tqdm import tqdm  # 进度条
from torchnet import meter
from utils.visualize import Visualizer
from utils.utils import *
import numpy as np

@torch.no_grad()
def eval(model, eval_dataloader):
    model.eval()
    acc, total = 0, 0
    for step, (inputs, labels, imgIds) in tqdm(enumerate(eval_dataloader)): # img_paths自定义数据集带的，一般不含
        inputs, labels = inputs.to(opt.device), labels.to(opt.device)
        outputs = model(inputs)
        # acc += np.sum(RectAngleMetric(outputs, labels, imgIds, opt.train_data_root), axis=0)
        acc += np.sum(RectAngleMetric(outputs, labels), axis=0)
        total += len(inputs)

    eval_acc = 100 * acc / total
    model.train()
    return eval_acc
   
def train(**kwargs):
    """
    训练模型
    """
    opt._parse(kwargs), opt._print_conf()
    vis = Visualizer(opt.env)

    # step1: data
    train_dataset = CornellDataset(opt.train_data_root, train=True)
    eval_dataset = CornellDataset(opt.train_data_root, train=False)
    train_dataloader = DataLoader(train_dataset,opt.batch_size, shuffle=True,num_workers=opt.num_workers)
    eval_dataloader = DataLoader(eval_dataset,opt.batch_size, shuffle=False,num_workers=opt.num_workers)

    # step2: configure model
    model = getattr(models, opt.model_name)() # getattr 获取对象属性值， 居然能把包里头的模块也当成包的属性
    if opt.load_model_path:
        checkpoints = torch.load(opt.load_model_path) # 保存时按字典保存，还能存其他参数
        opt.start_epoch = checkpoints['start_epoch'] 
        model.load_state_dict(checkpoints['state_dict'])
    if opt.multi_gpu: model = torch.nn.parallel.DataParallel(model).to(opt.device) # 单机多卡
    else: model.to(opt.device) # 单机单卡
        
    # step3: criterion and optimizer
    criterion = MSELoss() # 损失函数计算出来就在gpu上，不用cuda()
    lr=opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)
    
    # step4: meters
    loss_meter = meter.AverageValueMeter()
    # 便于只保存有意义的参数，节省存储
    previous_loss = 1e10
    best_acc = 0.7

    model.train()
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(opt.start_epoch, opt.num_epoch):
            print('*' * 40), print(f'epoch {epoch}')
            loss_meter.reset()
            for step, (inputs, labels, imgIds) in tqdm(enumerate(train_dataloader)):
                inputs, labels = inputs.to(opt.device), labels.to(opt.device)
            
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_meter.add(loss.item())

                if (step + 1)%opt.print_freq == 0:
                    vis.plot('loss', loss_meter.value()[0])

            # 验证validate评估eval 
            eval_acc = eval(model, eval_dataloader)
            print(f'[eval_acc:{eval_acc:.4f}]'), vis.plot('eval_acc', eval_acc)
            vis.log(f"[epoch:{epoch}  eval_acc:{eval_acc:.4f} lr:{lr}")

            if eval_acc >= best_acc:
                # 保存中间过程模型, 仅网络中的参数
                print(f'Saving epoch {epoch} model ...')
                state_dict = model.state_dict()
                if opt.multi_gpu: # DataParallel的model.state_dict的key会多'module.'7个字符，要去掉
                    for k in list(state_dict.keys()): # 不能直接实时取.keys，它在变
                        state_dict[k[7:]] = state_dict.pop(k)
                checkpoints = {'start_epoch':epoch+1, 'state_dict':state_dict} # epoch 0 训练好了，下次的start_epoch=1
                torch.save(checkpoints, f'checkpoints/{opt.model_name}_{opt.dataset_name}_lr{opt.lr}_ld{opt.lr_decay}_bs{opt.batch_size}_wd{opt.weight_decay}_epoch{epoch}_acc{eval_acc:.3f}.ckpt')

            # update learning rate
            if loss_meter.value()[0] > previous_loss:          
                lr = lr * opt.lr_decay
                # 第二种降低学习率的方法:不会有moment等信息的丢失
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            previous_loss = loss_meter.value()[0]

    print('Finished Training') # 注意，所有模型都是自定义格式，落地应用时 不能直接torch.load_state_dict
    
@torch.no_grad()
def test(**kwargs):
    """
    测试模型
    """
    opt._parse(kwargs)

    # step1: data
    test_dataset = CornellDataset(opt.test_data_root, test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # step2: config model
    model = getattr(models, opt.model_name)()
    if opt.load_model_path:
        print(os.path.basename(opt.load_model_path))
        checkpoints = torch.load(opt.load_model_path)
        opt.start_epoch = checkpoints['start_epoch']
        model.load_state_dict(checkpoints['state_dict'])
    if opt.multi_gpu: model = torch.nn.parallel.DataParallel(model).to(opt.device)
    else: model.to(opt.device)
    model.eval()
    
    acc, error_classified_imgs = 0, []
    for step, (inputs, labels, imgIds) in tqdm(enumerate(test_dataloader)):
        inputs, labels = inputs.to(opt.device), labels.to(opt.device)
        outputs = model(inputs)
        # results = RectAngleMetric(outputs, labels, imgIds, opt.test_data_root)
        results = RectAngleMetric(outputs, labels)
        acc += np.sum(results, axis=0)
        # 错分图片
        # error_classified_imgs += (np.array(imgIds)[results]).tolist()
        # 绘制预测结果
        drawBBox(opt.test_data_root, imgIds, outputs, "./out/predict")

    test_acc = acc/test_dataset.__len__()
    print(test_acc)
    # # print(set(error_classified_imgs))
    # # 将错分图片复制出来
    # for p in error_classified_imgs:
    #     os.system(f'cp {p} {opt.out_dir}/error_imgs/')


def help():
    """
    打印帮助的信息： python3 file.py help 这种注释会出现在文档解释中
    """
    print("""
    usage : python3 file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__)) # 当前文件名

    from inspect import getsource
    source = (getsource(opt.__class__)) # 居然可以直接打印class的源代码
    print(source)

def grid_search(**kwargs):
    """
    网格搜索训练、测试
    """
    from itertools import product # 计算笛卡尔积
    for param_dict in opt.params: # 第几组参数
        items = sorted(param_dict.items())
        keys, values = zip(*items) # 这段参考了sklearn的grid_search源码
        for value in product(*values):
            params = dict(zip(keys, value))
            opt._parse(params)
            cmd = './main.py train'
            print(params)
            for k, v in params.items(): # 嫌弃但是解释器的bug对长时任务for就只执行一次
                cmd+=(' --'+k+'='+str(v)) 
            os.system(cmd)

def test_all():
    paths = [os.path.join('./checkpoints/', path) for path in os.listdir('./checkpoints')] # 数据集按train/cat.0.jpg编排
    for p in paths:
        cmd = './main.py test --load-model-path='+p
        os.system(cmd)

if __name__=='__main__':
    import fire
    fire.Fire()



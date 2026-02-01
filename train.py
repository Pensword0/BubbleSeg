import os
import json
import argparse
import torch
import dataloaders
import models
import inspect
import math
from utils import losses
from utils import Logger
from utils.torchsummary import summary
from trainer import Trainer
import time

def get_instance(module, name, config, *args):
    # 这个函数用于动态创建类实例或调用函数。它使用 getattr 来获取指定模块中的类或函数，并使用配置文件中提供的参数来初始化
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume):
    # 这句代码 train_logger = Logger() 的意义是创建 Logger 类的一个实例，并将这个实例赋值给变量 train_logger。这个实例将用于记录和管理与训练过程相关的日志信息。
    # 具体来说：Logger()：这是调用 Logger 类的构造函数（即 __init__ 方法）。
    # 它会创建一个新的 Logger 对象。根据之前的代码，这个对象内部会有一个名为 entries 的空字典，用于存储日志条目。
    # train_logger：这是一个变量，用于引用新创建的 Logger 对象。
    # 通过这个变量，你可以访问 Logger 类定义的方法和属性，例如使用 train_logger.add_entry(some_entry) 来添加日志条目，或者使用 print(train_logger) 来打印当前所有的日志条目。
    # 这种方式在编写需要记录和追踪训练过程的信息时非常有用，比如记录训练损失、准确率、模型参数调整等。通过使用自定义的 Logger 类，你可以灵活地控制日志的记录方式和格式。
    train_logger = Logger()

    # DATA LOADERS
    train_loader = get_instance(dataloaders, 'train_loader', config)
    val_loader = get_instance(dataloaders, 'val_loader', config)

    # MODEL
    model = get_instance(models, 'arch', config, train_loader.dataset.num_classes)
    print(f'\n{model}\n')

    # LOSS
    loss = getattr(losses, config['loss'])(ignore_index = config['ignore_index'])

    # TRAINING
    trainer = Trainer(
        model=model,
        loss=loss,
        resume=resume,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        train_logger=train_logger)

    trainer.train()

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    config = json.load(open(args.config))
    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # === 时间统计开始 ===
    start_time = time.time()

    main(config, args.resume)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[Done] Total elapsed time: {elapsed:.2f} seconds "
      f"({elapsed / 60:.2f} minutes, {elapsed / 3600:.2f} hours)")

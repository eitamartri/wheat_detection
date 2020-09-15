import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Wheat detection using faster-rcnn model')

    # RPN sizes
    parser.add_argument('--anchor_sizes', nargs='+', type=int, default=[5, 10, 20, 40, 96]) #32 64
    parser.add_argument('--aspect_ratios', nargs='+', type=float, default=[0.5, 1., 2.])

    # Directories
    parser.add_argument('--data_dir', type=str, default='Data')
    parser.add_argument('--train_dir', type=str, default='train')
    parser.add_argument('--test_dir', type=str, default='test')
    parser.add_argument('--data_dirs', type=dict, default={'train': '', 'test': ''})

    # Training
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=16)


    # optimizer
    parser.add_argument('--learning_rate', type=float, default=0.0075)
    parser.add_argument('--betas', nargs='+', type=float, default=[0.9,0.999])
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--step_size', type=int, default=3)                     
    parser.add_argument('--gamma', type=float, default=0.1)

    # Debug options
    parser.add_argument('--num_of_pics_to_use', type=int, default=3600)
    parser.add_argument('--eval_train', type=bool, default=True)
    parser.add_argument('--gpu_id', type=int, default=0)


    # Model
    parser.add_argument('--backbone', type=str, default='resnet50')



    args = parser.parse_args()

    # modifications
    args.data_dirs['train'] = args.data_dir + '/' + args.train_dir
    args.data_dirs['test'] = args.data_dir + '/' + args.test_dir
    args.valid_ratio = 1 - args.train_ratio
    args.anchor_sizes = tuple(args.anchor_sizes)
    args.aspect_ratios = tuple(args.aspect_ratios)
   
    return args




import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import configurations as cfg
import os
import saves as sv
from torchvision.ops.poolers import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN

from torchvision.ops.poolers import MultiScaleRoIAlign
from torchvision.models import vgg16

class History():
    def __init__(self, model_num):
        self.epoch = 0
        self.best_val_loss = 999
        self.best_loss = 999
        self.model_num = model_num
        self.loss = []
        self.val_loss = []

def build_model(config,device):
    model_num = sv.get_model_num(config)

    # Check if there is an pre-trained model
    if os.path.exists('results/history_' + str(model_num) + '.pth.tar'):
        if config.backbone == 'resnet50':
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        if config.backbone == 'vgg16':
            model = build_model_vgg16(config,device)
        # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
        # in_features = model.roi_heads.box_predictor.cls_score.in_features
        # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        params = [p for p in model.parameters() if p.requires_grad]
        #get the optimizer parameters from the configuration file
        lr=config.learning_rate
        betas=tuple(config.betas)
        weigth_decay=config.weight_decay
        step_size=config.step_size
        gamma=config.gamma
        eps=1e-08
        
        #optimizer=torch.optim.Adam(params, lr,betas, eps, weigth_decay)
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weigth_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)

        model, optimizer, history = load_checkpoint(model, optimizer,
                                                    filename='results/history_' + str(model_num) + '.pth.tar')
        model.to(device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
                    
        return model, optimizer, history,lr_scheduler


    # load a model; pre-trained on COCO
    if config.backbone == 'resnet50':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    if config.backbone == 'vgg16':
        model = build_model_vgg16(config,device)
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    anchor_generator = AnchorGenerator(sizes=config.anchor_sizes,
                                       aspect_ratios=config.aspect_ratios)
    model.rpn.anchor_generator = anchor_generator

    num_classes = 2  # 1 class (wheat) + background

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    #get the optimizer parameters from the configuration file
    lr=config.learning_rate
    betas=tuple(config.betas)
    weigth_decay=config.weight_decay
    step_size=config.step_size
    gamma=config.gamma
    eps=1e-08
    
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    #optimizer=torch.optim.Adam(params, lr,betas, eps, weigth_decay)
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weigth_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
    # lr_scheduler = None

    history = History(model_num)

    # sv.update_history_df(config)

    return model, optimizer, history, lr_scheduler


def build_model_vgg16(config, device):
    if config.backbone == 'vgg16':
        vgg = vgg16(pretrained=True)

    backbone = vgg.features[:-1]
    for layer in backbone[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    backbone.out_channels = 512

    class BoxHead(torch.nn.Module):
        def __init__(self, vgg, dropout=False):
            super(BoxHead, self).__init__()
            classifier = list(vgg.classifier._modules.values())[:-1]
            if not dropout:
                del classifier[5]
                del classifier[2]
            self.classifier = torch.nn.Sequential(*classifier)

        def forward(self, x):
            x = x.flatten(start_dim=1)
            x = self.classifier(x)
            return x

    box_head = BoxHead(vgg)

    anchor_generator = AnchorGenerator(sizes=config.anchor_sizes, aspect_ratios=config.aspect_ratios)

    # Head - Box RoI pooling
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

    # Faster RCNN - Model
    model = FasterRCNN(
        backbone=backbone,
        min_size=224, max_size=224,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        box_head=box_head,
        box_predictor=FastRCNNPredictor(4096, num_classes=2)
    )

    # Init weights
    torch.nn.init.normal_(model.roi_heads.box_predictor.cls_score.weight, std=0.01)
    torch.nn.init.constant_(model.roi_heads.box_predictor.cls_score.bias, 0)
    torch.nn.init.normal_(model.roi_heads.box_predictor.bbox_pred.weight, std=0.001)
    torch.nn.init.constant_(model.roi_heads.box_predictor.bbox_pred.bias, 0)

    return model


def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.

    print("=> loading checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename)
    history = checkpoint['history']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}'".format(filename))
    
    
    return model, optimizer, history
import torch
import pdb

if  __name__ == '__main__':
    name = input()
    checkpoint  = torch.load(name)
    m = int(input())
    n = int(input())
    if 'roi_heads.box_predictor.cls_weight.weight' in checkpoint['model']:
        checkpoint['model']['roi_heads.box_predictor.cls_score.weight'] = checkpoint['model']['roi_heads.box_predictor.cls_score.weight'][:81-m+n]
        checkpoint['model']['roi_heads.box_predictor.cls_weight.weight'] = checkpoint['model']['roi_heads.box_predictor.cls_weight.weight'][:81-m+n]
        checkpoint['model']['roi_heads.box_predictor.cls_score.bias'] = checkpoint['model']['roi_heads.box_predictor.cls_score.bias'][:81-m+n]
    else:
        cls_weight = torch.ones((81-m+n)).cuda()
        cls_weight[-n:] = 0
        checkpoint['model']['roi_heads.box_predictor.cls_weight.weight'] = cls_weight
        score_weight = checkpoint['model']['roi_heads.box_predictor.cls_score.weight']
        checkpoint['model']['roi_heads.box_predictor.cls_score.weight'] = torch.cat((score_weight, torch.zeros((n, score_weight.shape[1]),device=score_weight.device)))
        bias = checkpoint['model']['roi_heads.box_predictor.cls_score.bias']
        checkpoint['model']['roi_heads.box_predictor.cls_score.bias'] = torch.cat((bias, torch.zeros((n,),device=score_weight.device)))
    torch.save(checkpoint, name.replace('.pth', '{}.pth'.format(n)))

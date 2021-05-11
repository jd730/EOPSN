"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import json
import os
import pdb
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List
from PIL import Image
import cv2
import numpy as np

import torch
import torch.distributed as dist
from torch import Tensor
import wandb
import torch.nn.functional as F
from e2i import EmbeddingsProjector

from panopticapi.utils import IdGenerator, rgb2id

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
if float(torchvision.__version__[:3]) < 0.7:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size

from detectron2.utils.visualizer import Visualizer
from util.box_ops import box_cxcywh_to_xyxy

from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from panopticapi.utils import rgb2id


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor(
            [self.count, self.total],
            dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(
            torch.empty(
                (max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,),
            dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(
            command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(input, size=None, scale_factor=None,
                mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str,
    # Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__[:3]) < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(
            input, size, scale_factor, mode, align_corners)


def filter_unseen_class(instances, unseen_label_set):
    gt_classes = instances.gt_classes
    filtered_idx = []
    for i, c in enumerate(gt_classes):
        if c not in unseen_label_set:
            filtered_idx.append(i)

    return instances[filtered_idx]


def add_mask(img_list, batched_boxes, batched_masks):
    new_img_list = []
    for im, boxes, masks in zip(img_list, batched_boxes, batched_masks):
        img_h, img_w = im.shape[-2:]
        boxes = box_cxcywh_to_xyxy(boxes)
        multiplier = torch.tensor(
            [img_w, img_h, img_w, img_h],
            dtype=torch.float32).cuda()
        boxes = boxes * multiplier
        boxes = boxes.int().clamp(min=0)
        for i, (box, mask) in enumerate(zip(boxes, masks)):
            box[3] = min(img_h, box[3])
            box[2] = min(img_w, box[2])
            dh = box[3] - box[1]
            dw = box[2] - box[0]
            conv_mask = F.interpolate(
                mask.view((1, 1) + mask.shape),
                size=(dh, dw),
                mode='bilinear')
            th_mask = conv_mask > 0.5
            if th_mask.sum() == 0:
                continue
            try:
                im[:, box[1]:box[3], box[0]:box[2]
                   ][th_mask[0].repeat(len(im), 1, 1)] = -1000
            except BaseException:
                pdb.set_trace()
        new_img_list.append(im)
    return new_img_list


def cum_map(sem_seg, ignore_value=255):
    H, W = sem_seg.shape[-2:]
    one_hot = sem_seg.clone()
    one_hot[one_hot != ignore_value] = 0
    one_hot[one_hot == ignore_value] = 1
    ret =  []
    if len(sem_seg.shape) > 2:
        for m in one_hot:
            sem_seg_target = cv2.integral(m.cpu().numpy().astype('uint8'))
            ret.append(sem_seg_target)
    else:
        ret = cv2.integral(one_hot.numpy().astype('uint8'))
    sem_seg_target = torch.tensor(ret, device=sem_seg.device).float()
    return sem_seg_target


def save_feature_and_box(image_path, features, instances, path):
    if not os.path.exists(path):
        os.mkdir(path)

    img = cv2.imread(image_path) #[:, :, ::-1]
    H, W, _ = img.shape
    h, w = instances._image_size
    img_w = W / w
    img_h = H / h
    boxes = instances.pred_boxes.tensor
    scores= instances.scores.cpu().numpy()
    pred_class = instances.pred_classes.cpu().numpy()
    multiplier = torch.tensor(
            [img_w, img_h, img_w, img_h],
            dtype=torch.float32, device=boxes.device)
    boxes = boxes * multiplier
    boxes = boxes.int().cpu().numpy()

    features = features.cpu().numpy()
    dir_name = os.path.join(path, image_path.split('/')[-1][:-4])
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    img_path_list = []
    for i, bbox in enumerate(boxes):
        cropped_image = img[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]
        cropped_img_path = os.path.join(dir_name, '{:04}.jpg'.format(i))
        cv2.imwrite(cropped_img_path, cropped_image)
        img_path_list.append(cropped_img_path)

    pickle.dump((features, scores, pred_class, img_path_list), open(dir_name+'.pkl', 'wb'))


def add_unlabeled_class(boxes, target_classes, integral_sem_seg, bg=80):
    H, W = integral_sem_seg.shape
    one_hot = integral_sem_seg.view(-1)
    most_common_class = []
    idx = target_classes == bg
    box = boxes[idx]
    box = box.long().clamp(min=0)
    lt = box[:, 1].clamp(max=H-1) * W+ box[:, 0].clamp(max=W-1)
    lb = box[:, 1].clamp(max=H-1) * W + box[:, 2].clamp(max=W, min=1) - 1
    rt = (box[:,3].clamp(max=H, min=1)-1)* W + box[:, 0].clamp(max=W-1)
    rb = (box[:,3].clamp(max=H, min=1)-1) * W + box[:, 2].clamp(max=W, min=1) -1
    l = len(one_hot) - 1
    area = (rb - rt) * (rb - lb) / W
    sel_lt = torch.index_select(one_hot, -1, lt.clamp(max=l))
    sel_lb = torch.index_select(one_hot, -1, lb.clamp(max=l))
    sel_rt = torch.index_select(one_hot, -1, rt.clamp(max=l))
    sel_rb = torch.index_select(one_hot, -1, rb.clamp(max=l))

    c = (sel_rb + sel_lt - sel_rt - sel_lb)
    most_common_class = c <= 0.5 * area
    temp = target_classes[idx]
    temp[most_common_class] += 1
    target_classes[idx] = temp
    del one_hot
    return target_classes.long(), target_classes != bg

def save_boxes(data, feature, obj_score, image_paths, pal=None, step=1, num_classes=80, output_dir='',dir_name='pseudo_gts', pred_label=None):
    path = data[:,0].long()
    label = data[:,1]
    boxes = data[:,2:]
    p = image_paths[0].split('/')[-1].split('.')[0]
    templete = image_paths[0].replace(p,'{:012}')
    img_path_list = []
    features = []

    dir_name = os.path.join(output_dir, dir_name)
    for p in path.unique():
        img = cv2.imread(templete.format(p))
        img_h, img_w, _ = img.shape
        multiplier = torch.tensor(
            [img_w, img_h, img_w, img_h],
            dtype=torch.float32, device=boxes.device)

        idx = path==p
        bbox = boxes[idx]

        features.append(feature[idx])
        bbox = bbox * multiplier
        bbox = bbox.int().cpu().numpy()
        lbl = label[idx]
        sc = obj_score[idx]
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        if not os.path.exists(dir_name+'/{:05}'.format(step)):
            os.mkdir(dir_name+'/{:05}'.format(step))
        for i, box in enumerate(bbox):
            cropped_image = img[box[1]:box[3]+1, box[0]:box[2]+1]
            H, W, _ = cropped_image.shape
            framed_image = np.ones((H+10, W+10, 3)) * pal[int(lbl[i]-num_classes) % 1024]
            framed_image[5:H+5, 5:W+5] = cropped_image
            framed_image = framed_image.astype(np.uint8)
            cropped_img_path = os.path.join(dir_name, '{:05}/{:03}_{:012}_{:02}_{:03}.jpg'.format(step,int(lbl[i]), int(p), int(sc[i]*100), i))
            out = cv2.imwrite(cropped_img_path, framed_image)
            if not out:
                print("FAIL TO SAVE")
            img_path_list.append(cropped_img_path)
    try :
        features = torch.cat(features).cpu().numpy()
        image = EmbeddingsProjector()
        image._svd = False
        image.image_list = np.asarray(img_path_list)
        image.data_vectors = features
        image.calculate_projection()
        image.each_img_size =  50
        image.output_img_size = 1000
        image.batch_size =  0
        image.output_img_name = os.path.join(dir_name,'{:05}'.format(step))
        image.create_image()
        np.save(os.path.join(output_dir,dir_name, '{}.npy'.format(step)), features)
    except:
        print("FAIL")



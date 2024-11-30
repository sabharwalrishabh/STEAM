import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn.functional as F
import os
import torch.nn as nn

def create_edge_list(batch_size, size):
    # Indices for even numbers
    even_indices = torch.arange(2, size + 1, 2) - 1  # Convert to 0-based indexing
    
    # Create bidirectional edges
    even_minus_one = even_indices - 1
    even_plus_one = even_indices + 1

    # Filter valid indices within bounds
    valid_minus_one = even_minus_one >= 0
    valid_plus_one = even_plus_one < size

    edges = []
    if torch.any(valid_minus_one):
        edges.append(torch.stack([even_indices[valid_minus_one], even_minus_one[valid_minus_one]], dim=1))
        edges.append(torch.stack([even_minus_one[valid_minus_one], even_indices[valid_minus_one]], dim=1))
    if torch.any(valid_plus_one):
        edges.append(torch.stack([even_indices[valid_plus_one], even_plus_one[valid_plus_one]], dim=1))
        edges.append(torch.stack([even_plus_one[valid_plus_one], even_indices[valid_plus_one]], dim=1))
    
    # Additional connections
    additional_edges = torch.tensor([[0, size - 1], [size - 1, 0]], dtype=torch.long)
    edges.append(additional_edges)
    
    # Concatenate all edges and repeat for each batch
    edge_list = torch.cat(edges, dim=0)
    edge_list = edge_list.t().contiguous()
    
    # Repeat for each batch
    # edge_lists = [edge_list for _ in range(batch_size)]
    # edge_lists = edge_list
    
    return edge_list

def spatial_pyramid_pooling(feature_map, levels):
    batch_size, channels, height, width = feature_map.size()
    pooled_outputs = []
 
    for level in levels:
        kernel_size = (height // level, width // level)
        stride = kernel_size
        padding = (0, 0)
 
        pooled = F.adaptive_max_pool2d(feature_map, output_size=level)
        # pooled = F.adaptive_avg_pool2d(feature_map, output_size=level)
        # print(pooled.view(batch_size, channels,-1).shape)
        pooled_outputs.append(pooled.view(batch_size, channels,-1))
 
    return torch.cat(pooled_outputs, dim=2)


def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    model.train()

    for batch_idx, (img, label) in enumerate(tqdm(train_loader, desc='Training')):
        model.train()
        optimizer.zero_grad()
        img = img.cuda(args.gpu, non_blocking=True)
        label = label.cuda(args.gpu, non_blocking=True)

        out = model(img)
        loss = criterion(out, label)
        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(out, label, topk=(1, 5))
        losses.update(loss.item(), img[0].size(0))
        top1.update(acc1[0], img[0].size(0))
        top5.update(acc5[0], img[0].size(0))
        loss.backward()
        optimizer.step()


    with open(args.file_path, "a") as f:
        f.write(f"Epoch: {epoch+1} Training acc@1: {top1.avg} | acc@5: {top5.avg} | loss: {losses.avg} \n")
    # print(f"Epoch: {epoch+1} Training acc@1: {top1.avg} | acc@5: {top5.avg} | loss: {losses.avg}")


def test(test_loader, model, criterion, epoch, args):
    test_losses = AverageMeter("Loss", ":.4e")
    test_top1 = AverageMeter("Acc@1", ":6.2f")
    test_top5 = AverageMeter("Acc@5", ":6.2f")
    for batch_idx, (img, label) in enumerate(tqdm(test_loader, desc='Testing')):
        model.eval()

        img = img.cuda(args.gpu, non_blocking=True)
        label = label.cuda(args.gpu, non_blocking=True)

        with torch.no_grad():
            out = model(img)
        loss = criterion(out, label)
        acc1, acc5 = accuracy(out, label, topk=(1, 5))
        test_losses.update(loss.item(), img[0].size(0))
        test_top1.update(acc1[0], img[0].size(0))
        test_top5.update(acc5[0], img[0].size(0))

    with open(args.file_path, "a") as f:
        f.write("-------------------- \n")
        f.write(f"Epoch: {epoch+1} Test acc@1: {test_top1.avg} | acc@5: {test_top5.avg} | loss: {test_losses.avg} \n")
        f.write("-------------------- \n")


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def save_arguments_to_file(args, filename):
    with open(filename, 'a') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write("\n")


def get_imagenet100_loader(data_dir):
    # data_dir = opt.data_path  # Replace with the actual path to your dataset
    # Define the transformations for the training and validation sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            # transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            # transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create datasets
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])

    # Create dataloaders
    # train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8)
    # test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8)

    return train_dataset, test_dataset


def get_imagenet1k_loader(data_dir):
    # data_dir = opt.data_path  # Replace with the actual path to your dataset
    # Define the transformations for the training and validation sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create datasets
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])

    # Create dataloaders
    # train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8)
    # test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8)

    return train_dataset, test_dataset


def get_cifar10_loader(data_dir):
    cifar10_mean = [0.4914, 0.4822, 0.4465]
    cifar10_std = [0.2023, 0.1994, 0.2010]

    # Define the transformations for training and validation sets
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # Resize and crop to AlexNet's input size
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),  # Resize to a slightly larger size
        transforms.CenterCrop(224),  # Center crop to AlexNet's input size
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    # # Create DataLoader for training and validation sets
    # train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8)
    # test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8)

    return train_dataset, test_dataset


def global_avg_pooling_to_7x7(feature_map):
    # Get the batch size, channels, height, and width of the feature map
    batch_size, channels, height, width = feature_map.shape

    # Calculate the size of each grid to get a 7x7 output
    grid_size_h = height // 7
    grid_size_w = width // 7

    # Ensure the height and width are divisible by 7
    assert height % 7 == 0 and width % 7 == 0, "Feature map dimensions must be divisible by 7."

    # Step 1: Reshape to group the grids
    reshaped = feature_map.view(batch_size, channels, 7, grid_size_h, 7, grid_size_w)

    # Step 2: Permute to bring the grids into separate dimensions
    permuted = reshaped.permute(0, 2, 4, 1, 3, 5)
    b, k, k, c, h, w = permuted.shape
    # Step 3: Compute the mean along the channel and spatial dimensions
    global_avg_pool = permuted.mean(dim=[3, 4, 5])
    # global_max_pool = permuted.reshape(b,k, k,-1).max(dim=-1).values
    # global_concat = torch.cat([global_avg_pool.unsqueeze(dim=-1), global_max_pool.unsqueeze(dim=-1)], dim=-1)
    # print(global_concat.shape)
    # raise Exception
    # return global_concat
    return global_avg_pool


def drop_random_edges(edge_list, size=7):
    src, dst = edge_list

    # Identify central nodes in the 5x5 grid
    central_nodes = torch.arange(size * size).reshape(size, size)[1:-1, 1:-1].reshape(-1)

    # Count the occurrences of each node in src
    node_counts = torch.bincount(src, minlength=size * size)

    # Nodes with exactly 4 neighbors in the central 5x5 grid
    nodes_with_four_neighbors = (node_counts == 4).nonzero(as_tuple=True)[0]
    nodes_with_four_neighbors = nodes_with_four_neighbors[torch.isin(nodes_with_four_neighbors, central_nodes)]

    # Save the current random state
    #current_random_state = torch.get_rng_state()

    # Create a mask to keep track of edges to keep
    mask = torch.ones(src.size(0), dtype=torch.bool)
    
    for node in nodes_with_four_neighbors:
        node_indices = (src == node).nonzero(as_tuple=True)[0]
        
        bidirectional_indices = []
        for i in node_indices:
            matching_indices = (dst == src[i]) & (src == dst[i])
            bidirectional_pair_indices = (i.item(), matching_indices.nonzero(as_tuple=True)[0].item())
            bidirectional_indices.append(bidirectional_pair_indices)

        # Set a different seed for random operations
        torch.manual_seed(torch.randint(0, 2**32 - 1, (1,)).item())
        drop_indices = torch.tensor(bidirectional_indices)[torch.randperm(len(bidirectional_indices))[:1]].reshape(-1)
        mask[drop_indices] = False

    # Restore the original random state
    #torch.set_rng_state(current_random_state)

    # Create new edge list by keeping only non-dropped edges
    new_src = src[mask]
    new_dst = dst[mask]

    return torch.stack([new_src, new_dst])

def create_spatial_adjacency_matrix(size):
    # Create a zero matrix for adjacency
    adj_matrix = torch.zeros((size * size, size * size), dtype=torch.int)

    # Create index tensor for all nodes
    indices = torch.arange(size * size).reshape(size, size)

    # Connect right neighbors
    right_indices = indices[:, :-1].reshape(-1)
    adj_matrix[right_indices, right_indices + 1] = 1
    adj_matrix[right_indices + 1, right_indices] = 1

    # Connect bottom neighbors
    bottom_indices = indices[:-1, :].reshape(-1)
    adj_matrix[bottom_indices, bottom_indices + size] = 1
    adj_matrix[bottom_indices + size, bottom_indices] = 1

    # return adj_matrix
    return torch.nonzero(adj_matrix).t().contiguous()


def create_spatial_adjacency_matrix_batch(batch_size, size):
    adj_matrices = []

    for _ in range(batch_size):
        adj_matrix = create_spatial_adjacency_matrix(size)
        adj_matrices.append(torch.nonzero(adj_matrix).t().contiguous())

    return adj_matrices


class SingleKernelConv(nn.Module):
    def __init__(self, in_channels, kernel_size=1):
        super(SingleKernelConv, self).__init__()
        # Initialize the kernel with ones, one kernel per input channel
        self.kernel = nn.Parameter(torch.ones(in_channels, 1, kernel_size, kernel_size))

    def forward(self, x):
        # Perform depthwise convolution where each input channel has its own filter
        out = nn.functional.conv2d(x, self.kernel, groups=self.kernel.shape[0])
        return out


class conv1x1(nn.Module):
    def __init__(self):
        super(conv1x1, self).__init__()
        # Define a 7x7 convolution layer with 1 input channel and 1 output channel
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        nn.init.ones_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out

class SpecialConv(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, kernel_size=7):
        super(SpecialConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)

        # Initialize the kernel weights with ones
        nn.init.ones_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out
    

def create_directed_edge_list_5(size):
    indices = torch.arange(size)
    prev1 = (indices - 1) % size
    prev2 = (indices - 2) % size
    next1 = (indices + 1) % size
    next2 = (indices + 2) % size
    src_indices = indices.repeat(4)
    tgt_indices = torch.cat([prev1, prev2, next1, next2])
    edge_list = torch.stack([src_indices, tgt_indices], dim=1)
    edge_list_t = edge_list.t().contiguous()
    return edge_list_t






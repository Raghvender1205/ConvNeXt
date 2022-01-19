import os
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform: ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - - -")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print(" - - - - - - - - - - - - -")

    # Dataset Args
    if args.data_set ==- 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print('Reading from DataPath..', args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    
    print('Number of Classes: %d' % nb_classes)
    return dataset, nb_classes

# Build Transform
def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # It should always Dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolaion,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(args.input_size, paddding=4)
        return transform

    t = []
    if resize_im:
        # Warping (No Cropping) when evaluated at 384 or larger
        if args.input_size >= 384:
            t.append(
                transforms.Resize((args.input_size, args.input_size),
                                interpolation=transforms.InterpolationMode.BICUBIC),
            )

            print(f'Warping {args.input_size} size input images....')
        
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # To Maintain Same Ratio w.r.t 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            )
            t.append(transforms.CenterCrop(args.input_size))

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)
    
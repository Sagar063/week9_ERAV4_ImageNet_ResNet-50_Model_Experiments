# data/imagenet_dali.py
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

@pipeline_def
def imagenet_pipe(
    data_root,
    random_shuffle,
    shard_id,
    num_shards,
    crop=224,
    mean=None,      # <-- accept mean
    std=None        # <-- accept std
):
    jpegs, labels = fn.readers.file(
        file_root=data_root, random_shuffle=random_shuffle,
        shard_id=shard_id, num_shards=num_shards, name="Reader"
    )
    # RandomResizedCrop equivalent: area [0.08,1.0], aspect [3/4, 4/3]
    images = fn.decoders.image_random_crop(
        jpegs, device="mixed", output_type=types.RGB,
        random_aspect_ratio=[0.75, 1.3333333],
        random_area=[0.08, 1.0]
    )
    images = fn.resize(images, resize_x=crop, resize_y=crop)

    mirror = fn.random.coin_flip(probability=0.5)  # HFlip ~0.5

    # Use the provided mean/std if passed from the loader; otherwise fall back to defaults
    # mean = mean if mean is not None else [0.485 * 255, 0.456 * 255, 0.406 * 255]
    # std  = std  if std  is not None else [0.229 * 255, 0.224 * 255, 0.225 * 255]

    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",     # PyTorch expects NCHW tensors
        mean=mean,
        std=std,
        mirror=mirror
    )
    return images, labels

def dali_loader(
    data_root,
    batch_size,
    num_threads,
    device_id,
    world_size,
    rank,
    train=True,
    mean=None,
    std=None
):
    # If caller passes normalized RGB mean/std in [0..1], convert to [0..255] here.
    mean = [m * 255 for m in (mean or [0.485, 0.456, 0.406])]
    std  = [s * 255 for s in (std  or [0.229, 0.224, 0.225])]

    pipe = imagenet_pipe(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        data_root=data_root,
        random_shuffle=train,
        shard_id=rank,
        num_shards=world_size,
        crop=224,
        mean=mean,               # <-- pass into pipeline
        std=std                  # <-- pass into pipeline
    )
    pipe.build()
    return DALIGenericIterator(
        pipe,
        ["images", "labels"],
        reader_name="Reader",
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.PARTIAL
    )

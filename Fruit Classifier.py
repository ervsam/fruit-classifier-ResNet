import fastbook
from fastbook import *
from fastai.vision.widgets import *


# clear cache
import gc
gc.collect()
torch.cuda.empty_cache()

key = 'Bing Image Search API key'

# !rm -rf {category_name}
category_name = 'fruits'
category = ['apple', 'orange', 'banana']
search = 'fruit'

path = Path(category_name)
if not path.exists():
  path.mkdir()
  for o in category:
    dest = path/o
    dest.mkdir()
    results = search_images_bing(key, f'{o} {search}')
    # n_workers edited here from default 8, else causes error in function parallel
    download_images(dest, urls=results.attrgot('contentUrl'), n_workers=4)

# changed n_workers to 0
def verify_images(fns):
    "Find images in `fns` that can't be opened"
    return L(fns[i] for i,o in enumerate(parallel(verify_image, fns, n_workers=0)) if not o)

path = Path('fruits')

fns = get_image_files(path)
failed = verify_images(fns)
failed.map(Path.unlink)

exp = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(0.2, 0),
    get_y=parent_label,
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms()
)
# changed num_workers to 0, else error (https://github.com/fastai/fastbook/issues/85)
# changed batch size (bs) to 8 because not enough memory (https://github.com/pytorch/pytorch/issues/16417#issuecomment-497952224)
dls = exp.dataloaders(path, num_workers=0, bs=32)

learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

learn.export()


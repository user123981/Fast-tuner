# Purpose: Resize images to 224x224

from pathlib import Path

from skimage import io
from skimage.transform import resize



base_path = Path('./_data/SLO-FAF_224/')


for img in base_path.rglob('*.png'):
    image = io.imread(img)
    resized = resize(image, (224, 224), anti_aliasing=True, preserve_range=True, order=3).astype('uint8')
    io.imsave(img, resized)
    print(f'Resized {img.name}')



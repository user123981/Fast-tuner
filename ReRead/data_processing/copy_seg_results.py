from pathlib import Path
import shutil



dest_path = 'path/to/data'
dest_path = Path(dest_path)


base_path = 'path/to/data'
base_path = Path(base_path)


for dataset in base_path.iterdir():
    print(dataset)
    for model in dataset.iterdir():
        print(model)
        model_images = model / 'images'
        save_path = dest_path / dataset.name / model.name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(model_images, save_path)
        except FileExistsError:
            print('File exists')
            continue
        exit()


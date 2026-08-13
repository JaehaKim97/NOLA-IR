import os
import argparse
from shutil import copytree, rmtree


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess CUB200 dataset from Kaggle archive.zip.")
    parser.add_argument(
        "--remove_archive",
        action="store_true",
        help="Remove archive.zip after preprocessing",
    )
    args = parser.parse_args()
    
    if not os.path.exists('datasets/source/archive.zip'):
        print("FAILED: The VOC2012 data path datasets/source/archive.zip does NOT EXIST")
    else:
        print('Unzip archive.zip ...')
        os.system('unzip -qq datasets/source/archive.zip -d datasets/source/')
        
        destination = "datasets/source/VOC/VOCdevkit/VOC2012"
        copytree("datasets/source/VOC2012_train_val/VOC2012_train_val", destination, dirs_exist_ok=True)
        
        if os.path.exists('datasets/source/VOC2012_test'):
            rmtree('datasets/source/VOC2012_test')
            
        if os.path.exists('datasets/source/VOC2012_train_val'):
            rmtree('datasets/source/VOC2012_train_val')
        
        if args.remove_archive:
            if os.path.exists('datasets/source/archive.zip'):
                os.remove('datasets/source/archive.zip')
            print('Done! archive.zip is removed.')
        else:
            print('Done! archive.zip is kept. (use --remove_archive to delete it)')

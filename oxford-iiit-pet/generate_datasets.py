import os
import shutil
import random


def main():
    img_path = os.path.abspath("images/")
    list_path = os.path.abspath("annotations/list.txt")
    original_dir = os.getcwd()

    d = {"large": [9, 1, 2], "medium": [2, 1, 1], "small": [2, 3, 3]}

    for data_set_name, train_val_test_split in d.items():
        data_path = os.path.join(original_dir, "dataset-" + data_set_name + "-train/")
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        os.chdir(data_path)

        for split_type in ["2-class", "37-class"]:
            if not os.path.exists(split_type):
                os.mkdir(split_type)
            os.chdir(split_type)

            if split_type == "2-class":
                classes = ["cats", "dogs"]
            else:
                classes = ['Abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle',
                           'Bengal', 'Birman', 'Bombay', 'boxer', 'British_Shorthair', 'chihuahua', 'Egyptian_Mau',
                           'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees',
                           'havanese',
                           'japanese_chin', 'keeshond', 'leonberger', 'Maine_Coon', 'miniature_pinscher',
                           'newfoundland',
                           'Persian', 'pomeranian', 'pug', 'Ragdoll', 'Russian_Blue', 'saint_bernard', 'samoyed',
                           'scottish_terrier', 'shiba_inu', 'Siamese', 'Sphynx', 'staffordshire_bull_terrier',
                           'wheaten_terrier', 'yorkshire_terrier']

            data_types = ["test", 'val', 'train']
            for data_type in data_types:
                if not os.path.exists(data_type):
                    os.mkdir(data_type)

            for data_type in data_types:
                for cls in classes:
                    p = os.path.join(data_type, cls)
                    if not os.path.exists(p):
                        os.mkdir(p)

            with open(list_path) as lst:
                for line in lst.readlines():
                    if "#" in line:
                        continue
                    line_items = line.split()
                    file_name = line_items[0] + ".jpg"
                    source = os.path.join(img_path, file_name)
                    if split_type == "2-class":
                        cls_name = classes[int(line_items[2])-1]
                    else:
                        cls_name = classes[int(line_items[1])-1]

                    r = random.randint(1, sum(train_val_test_split))
                    if r <= train_val_test_split[0]:
                        dest = os.path.join("train", cls_name, file_name)
                    elif r <= sum(train_val_test_split[0:2]):
                        dest = os.path.join("val", cls_name, file_name)
                    else:
                        dest = os.path.join("test", cls_name, file_name)

                    shutil.copy(source, dest)

            os.chdir('../')


if __name__ == "__main__":
    main()

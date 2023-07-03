#%%
import os
import json
import matplotlib.pyplot as plt

def get_class_ratio(root):
    # root = "../data/train/resized_outputs_json"

    path = []
    for folder in os.listdir(root):
        for file in os.listdir(root +"/"+ folder):
            if file[-4:].lower() == "json":
                file_path = folder + "/" + file
                path.append(file_path)
    # print(path) # [..., 'ID359/image1664934986606.json', 'ID359/image1664935001598.json', ...]

    points_ratio_per_image = {}
    points_count_per_image = {}
    armbone_ratio_per_image = {}
    for file in path:
        FILE_ROOT = root + "/" + file # ../data/train/resized_outputs_json/ID417/image1666055724935.json
        # print(FILE_ROOT)

        with open(FILE_ROOT, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # print(data)

        points_count_per_label = {}
        total = 0
        for anno in data["annotations"]:
            points_count_per_label[anno["label"]] = len(anno["points"])
            total += len(anno["points"])
        # print(points_count_per_label)
        # print(total)
        points_count_per_image[file] = points_count_per_label

        points_ratio_per_label = {}
        for label, points in points_count_per_label.items():
            points_ratio_per_label[label] = float(points/total*100)
            if label == 'Radius':
                radius = float(points/total*100)
            elif label == 'Ulna': 
                ulna = float(points/total*100)
        # print(points_ratio_per_label)
        # print(total)
        points_ratio_per_image[file] = points_ratio_per_label
        armbone_ratio_per_image[file] = radius + ulna

    # print(points_count_per_image)
    # print(points_ratio_per_image)

    armbone_ratio_per_image_descending = dict(sorted(armbone_ratio_per_image.items(), key=lambda x:x[1], reverse=True))
    # print(sorted_armbone_ratio_per_image)

    # visualize
    # plt.title("Radius & Ulna ratio per image")
    # plt.ylabel("Radius & Ulna ratio")
    # plt.xlabel("image")
    # key_images = armbone_ratio_per_image_descending.keys()
    # values_ratio = armbone_ratio_per_image_descending.values()

    # plt.bar(key_images, values_ratio)
    # plt.show()

    points_count_per_image = dict(sorted(points_count_per_image.items(), key=lambda x:x[0]))
    points_ratio_per_image = dict(sorted(points_ratio_per_image.items(), key=lambda x:x[0]))
    armbone_ratio_per_image = dict(sorted(armbone_ratio_per_image.items(), key=lambda x:x[0]))
    # print("\n >> points_count_per_image")
    # print(points_count_per_image)
    # print("\n >> points_ratio_per_image")
    # print(points_ratio_per_image)
    # print("\n >> armbone_ratio_per_imagee")
    # print(armbone_ratio_per_image)

    return points_count_per_image, points_ratio_per_image, armbone_ratio_per_image


if __name__ == "__main__":
    root = "../data/train/resized_outputs_json"
    get_class_ratio(root)
# %%

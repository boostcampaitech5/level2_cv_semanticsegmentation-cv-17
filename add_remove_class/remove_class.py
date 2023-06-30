import os
import json

def save_removed_json():
    # remove specific class
    origin_root = "../data/train/resized_outputs_json"
    class_removed_root = "../data/train/resized_class_removed_outputs_json"

    path = []
    for folder in os.listdir(origin_root):
        for file in os.listdir(origin_root +"/"+ folder):
            if file[-4:].lower() == "json":
                file_path = folder + "/" + file
                path.append(file_path)
    # print(path)

    for file in path:
        ORIGIN_ROOT = origin_root + "/" + file
        CLASS_REMOVED_ROOT = class_removed_root + "/" + file

        folder_name = file.split("/")[0]    
        os.makedirs(class_removed_root + "/" + folder_name, exist_ok=True)

        with open(ORIGIN_ROOT, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # print(data)

        anns = []
        for anno in data["annotations"]:
            # if anno["label"] == "Trapezium" or if anno["label"] == "Triquetrum": # 손등뼈 1
            # if anno["label"] == "Trapezoid" or if anno["label"] == "Pisiform": # 손등뼈 2
            if anno["label"] == "Radius" or anno["label"] == "Ulna": # 팔 뼈
                continue
            else:
                ann = {
                    "id" : anno["id"],
                    "type" : anno["type"],
                    "attributes" : anno["attributes"],
                    "points" : anno["points"],
                    "label" : anno["label"]
                }
                anns.append(ann)
        
        class_removed_data = {
            "annotations" : anns,
            "attributes" : data["attributes"],
            "file_id" : data["file_id"],
            "filename" : data["filename"],
            "parent_path" : data["parent_path"],
            "last_modifier_id" : data["last_modifier_id"],
            "metadata" : data["metadata"],
            "last_workers" : data["last_workers"]
        }
        
        with open(CLASS_REMOVED_ROOT,'w',  encoding='utf-8') as train_writer:
            json.dump(class_removed_data, train_writer, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    save_removed_json()


# IMAGE Examples
# (기울어진 손) ID276, ID278, ID319, ID289
# (팔뼈 min) ID059/image1661393300595.png / (팔뼈 max) ID468/image1666659890125.png
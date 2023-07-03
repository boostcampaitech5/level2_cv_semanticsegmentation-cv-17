import os
import json

import numpy as np

def save_removed_json():
    # remove specific class
    # origin_root = "../data/train/resized_outputs_json"
    # class_added_root = "../data/train/resized_class_added_outputs_json"
    origin_root = "../data/train/outputs_json"
    class_added_root = "../data/train/class_added_outputs_json"

    path = []
    for folder in os.listdir(origin_root):
        for file in os.listdir(origin_root +"/"+ folder):
            if file[-4:].lower() == "json":
                file_path = folder + "/" + file
                path.append(file_path)
    # print(path)
    count = 0
    for file in path:
        ORIGIN_ROOT = origin_root + "/" + file
        CLASS_ADDED_ROOT = class_added_root + "/" + file

        folder_name = file.split("/")[0]
        os.makedirs(class_added_root + "/" + folder_name, exist_ok=True)

        with open(ORIGIN_ROOT, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # print(data)

        anns = []
        for index, anno in enumerate(data["annotations"]):
            ann = {
                "id" : anno["id"],
                "type" : anno["type"],
                "attributes" : anno["attributes"],
                "points" : anno["points"],
                "label" : anno["label"]
            }
            if anno["label"] == "Trapezium":
                Trapezium = ann # set(map(tuple, anno["points"]))
                print('\nTrapezium\n', len(Trapezium['points']), Trapezium)

            elif anno["label"] == "Trapezoid":
                Trapezoid = ann # set(map(tuple, anno["points"]))
                print('\nTrapezoid\n', len(Trapezoid['points']), Trapezoid)

            elif anno["label"] == "Triquetrum":
                Triquetrum = ann # set(map(tuple, anno["points"]))
                print('\nTriquetrum\n', len(Triquetrum['points']), Triquetrum)

            elif anno["label"] == "Pisiform":
                Pisiform = ann # set(map(tuple, anno["points"]))
                print('\nPisiform\n', len(Pisiform['points']), Pisiform)
            else:
                anns.append(ann)
            
            # get data of new class ('Tratra', 'Tripis')
            # remove repeated data ('Trapezium', 'Triquetrum', 'Trapezoid', 'Pisiform')
            if index == len(data["annotations"]) - 1:
                Trapezium, Trapezoid, Tratra = get_new_class_ann(Trapezium, Trapezoid, 'Tratra')
                Triquetrum, Pisiform, Tripis = get_new_class_ann(Triquetrum, Pisiform, 'Tripis')
                print('\nfix classes...')
                print('\nTrapezium\n', len(Trapezium['points']), Trapezium)
                print('\nTrapezoid\n', len(Trapezoid['points']), Trapezoid)
                print('\nTratra\n', len(Tratra['points']), Tratra)
                print('\nTriquetrum\n', len(Triquetrum['points']), Triquetrum)
                print('\nPisiform\n', len(Pisiform['points']), Pisiform)
                print('\nTripis\n', len(Tripis['points']), Tripis)
                anns.extend([Trapezium, Trapezoid, Triquetrum, Pisiform, Tratra, Tripis])
                print('\n\nanns\n', anns)
        if count == 10:
            return
        else:
            count += 1
        class_added_data = {
            "annotations" : anns,
            "attributes" : data["attributes"],
            "file_id" : data["file_id"],
            "filename" : data["filename"],
            "parent_path" : data["parent_path"],
            "last_modifier_id" : data["last_modifier_id"],
            "metadata" : data["metadata"],
            "last_workers" : data["last_workers"]
        }
        
        with open(CLASS_ADDED_ROOT,'w',  encoding='utf-8') as train_writer:
            json.dump(class_added_data, train_writer, indent=4, ensure_ascii=False)


def get_new_class_ann(ann1, ann2, class_name):
    # 'Trapezium', 'Trapezoid' -> 'Tratra'
    # 'Triquetrum', 'Pisiform' -> 'Traipis'
    points1 = set(map(tuple, ann1['points']))
    points2 = set(map(tuple, ann2['points']))
    new_points = list(map(list, points1.intersection(points2)))
    new_ann = {
        "id" : 'new_class_%s' %(class_name),
        "type" : ann1["type"],
        "attributes" : ann1["attributes"],
        "points" : new_points,
        "label" : class_name
    }
    ann1['points'] = list(map(list, points1-points2))
    ann2['points'] = list(map(list, points2-points1))

    return ann1, ann2, new_ann



def get_new_class_mask(mask, nonzero):
    # 'Trapezium', 'Trapezoid' -> 'Tratra'
    # 'Triquetrum', 'Pisiform' -> 'Tripis'
    if 'Trapezium' in mask:
        Trapezium = mask['Trapezium']
    if 'Trapezoid' in mask:
        Trapezoid = mask['Trapezoid']
    if 'Triquetrum' in mask:
        Triquetrum = mask['Triquetrum']
    if 'Pisiform' in mask:
        Pisiform = mask['Pisiform']

    # get mask of new class
    if ('Trapezium' in mask) and ('Trapezoid' in mask):
        Tratra = Trapezium * Trapezoid
        mask['Trapezium'] = Trapezium - Tratra
        mask['Trapezoid'] = Trapezoid - Tratra
        mask['Tratra'] = Tratra
    if ('Triquetrum' in mask) and ('Pisiform' in mask):
        Tripis = Triquetrum * Pisiform
        mask['Triquetrum'] = Triquetrum - Tripis
        mask['Pisiform'] = Pisiform - Tripis
        mask['Tripis'] = Tripis
    
    # remove repeated data from original class
    # mask = {
    #     'Trapezium' : Trapezium - Tratra,
    #     'Trapezoid' : Trapezoid - Tratra,
    #     'Triquetrum' : Triquetrum - Tripis,
    #     'Pisiform' : Pisiform - Tripis,
    #     'Tratra' : Tratra,
    #     'Tripis' : Tripis
    # }
    nonzero = {
        'Trapezium' : np.count_nonzero(mask['Trapezium']) if 'Trapezium' in mask else None,
        'Trapezoid' : np.count_nonzero(mask['Trapezoid']) if 'Trapezoid' in mask else None,
        'Triquetrum' : np.count_nonzero(mask['Triquetrum']) if 'Triquetrum' in mask else None,
        'Pisiform' : np.count_nonzero(mask['Pisiform']) if 'Pisiform' in mask else None,
        'Tratra' : np.count_nonzero(mask['Tratra']) if 'Tratra' in mask else None,
        'Tripis' : np.count_nonzero(mask['Tripis']) if 'Tripis' in mask else None,
    }

    return mask, nonzero



if __name__ == "__main__":
    save_removed_json()


# IMAGE Examples
# (기울어진 손) ID276, ID278, ID319, ID289
# (팔뼈 min) ID059/image1661393300595.png / (팔뼈 max) ID468/image1666659890125.png

# pseudo labeling - ver2
# 'ID058/image1661392103627.png' - class num 23 
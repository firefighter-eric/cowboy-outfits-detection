def pick_from_id(dataset, ids):
    dataset_out = []
    for img, targets in tqdm(dataset):
        _targets = []
        for t in targets:
            if t['id'] in ids:
                print(1)
                _targets.append(t)
        if _targets:
            print(1)
            dataset_out.append([img, _targets])
    return dataset_out


dev_dataset = pick_from_id(coco_det, dev_ids)
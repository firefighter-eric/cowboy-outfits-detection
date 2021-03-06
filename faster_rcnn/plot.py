import cv2

font = cv2.FONT_HERSHEY_SIMPLEX


def plot(img, target, true_target=None, id2str=None, threshold=None):
    length = len(target['labels'])

    img = img.permute(1, 2, 0).numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for i in range(length):
        score = float(target['scores'][i])
        if threshold and score < threshold:
            continue

        label = int(target['labels'][i])
        if id2str:
            label = id2str[label]

        x1, y1, x2, y2 = [int(_) for _ in target['boxes'][i]]
        text = f'{label}  {score:.2f}'
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
        cv2.putText(img, text=text, org=(x1 + 5, y1 + 5), fontFace=font, fontScale=1,
                    thickness=2, lineType=cv2.LINE_AA, color=(0, 255, 0))

    if true_target:
        for i in range(len(true_target['labels'])):
            label = int(true_target['labels'][i])
            if id2str:
                label = id2str[label]
            x1, y1, x2, y2 = [int(_) for _ in true_target['boxes'][i]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
            cv2.putText(img, text=label, org=(x1 + 5, y1 + 5), fontFace=font, fontScale=1,
                        thickness=2, lineType=cv2.LINE_AA, color=(255, 0, 0))

    cv2.imshow('test', img)
    cv2.waitKey()


def plot_results():
    import json
    from faster_rcnn.data_process import CocoDataLoader

    with open('../outputs/result.json') as fin:
        preds = json.load(fin)

    cdl = CocoDataLoader()
    dev_data = cdl.dev_data_loader
    idx2str = cdl.idx2str

    for pred, (img, true_target) in zip(preds, dev_data):
        plot(img, pred, true_target, idx2str, 0.5)


def test():
    from faster_rcnn.data_process import CocoDataLoader

    cdl = CocoDataLoader()
    dev_data = cdl.dev_data_loader
    imgs, targets = next(iter(dev_data))
    _img, _target = imgs[0], targets[0]

    plot(_img, _target)


if __name__ == '__main__':
    plot_results()

import cv2

font = cv2.FONT_HERSHEY_SIMPLEX


def plot(img, target, str2id=None, threshold=None):
    length = len(target['labels'])

    img = img.permute(1, 2, 0).numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for i in range(length):
        score = float(target['scores'][i])
        if threshold and score < threshold:
            continue
        id_ = int(target['labels'][i])
        if str2id:
            id_ = str2id[id_]
        x1, y1, x2, y2 = [int(_) for _ in target['boxes'][i]]
        text = f'{id_}  {score:.2f}'
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
        cv2.putText(img, text=text, org=(x1 + 5, y1 + 5), fontFace=font, fontScale=1,
                    thickness=2, lineType=cv2.LINE_AA, color=(0, 255, 0))

    cv2.imshow('test', img)
    cv2.waitKey()


if __name__ == '__main__':
    from faster_rcnn.data_process import CocoDataLoader

    cdl = CocoDataLoader()
    dev_data = cdl.dev_data_loader
    imgs, targets = next(iter(dev_data))
    _img, _target = imgs[0], targets[0]

    plot(_img, _target)

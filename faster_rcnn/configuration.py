class Args:
    num_epochs = 10
    device = 'cuda:0'
    model_name = 'm24'
    best_epoch = 0
    lr = 0.0005
    momentum = 0.9
    weight_decay = 0.01
    # lr = 0.005
    # momentum = 0.9
    # weight_decay = 0.0005

    data_dir = '../data'
    model_out = f'../models/faster_rcnn/{model_name}'
    log_dir = f'../runs/faster_rcnn/{model_name}'

    best_model_path = f'{model_out}/e{best_epoch}.pt'
    outfile_name = f'faster_rcnn_{model_name}e{best_epoch}.json'
    eval_coco_path = f'../outputs/eval/coco_results/{outfile_name}'
    coco_result_path = f'../outputs/coco_results/test_{outfile_name}'

# MODEL_OUT = '../models/retinanet/m4/'
# LOG_OUT = '../runs/retinanet/m4/'

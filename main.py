import torch, torchvision
import detectron2
import numpy as np
import cv2
import io

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from telegram.ext import Updater

classes = [{'id': 0, 'name': 'Norwegian Forest Cat'}, {'id': 1, 'name': 'Abyssinian'}, {'id': 2, 'name': 'Cymric'}, {'id': 3, 'name': 'Maine Coon'}, {'id': 4, 'name': 'Japanese Bobtail'}, {'id': 5, 'name': 'Calico'}, {'id': 6, 'name': 'Bombay'}, {'id': 7, 'name': 'Cornish Rex'}, {'id': 8, 'name': 'Burmese'}, {'id': 9, 'name': 'Applehead Siamese'}, {'id': 10, 'name': 'Pixiebob'}, {'id': 11, 'name': 'Tortoiseshell'}, {'id': 12, 'name': 'Ragamuffin'}, {'id': 13, 'name': 'Singapura'}, {'id': 14, 'name': 'Chartreux'}, {'id': 15, 'name': 'Bengal'}, {'id': 16, 'name': 'Chinchilla'}, {'id': 17, 'name': 'Exotic Shorthair'}, {'id': 18, 'name': 'American Shorthair'}, {'id': 19, 'name': 'Tonkinese'}, {'id': 20, 'name': 'Munchkin'}, {'id': 21, 'name': 'Nebelung'}, {'id': 22, 'name': 'American Curl'}, {'id': 23, 'name': 'Himalayan'}, {'id': 24, 'name': 'Tabby'}, {'id': 25, 'name': 'Egyptian Mau'}, {'id': 26, 'name': 'Selkirk Rex'}, {'id': 27, 'name': 'Snowshoe'}, {'id': 28, 'name': 'Silver'}, {'id': 29, 'name': 'York Chocolate'}, {'id': 30, 'name': 'Domestic Short Hair'}, {'id': 31, 'name': 'Korat'}, {'id': 32, 'name': 'Scottish Fold'}, {'id': 33, 'name': 'Turkish Angora'}, {'id': 34, 'name': 'Siberian'}, {'id': 35, 'name': 'Oriental Short Hair'}, {'id': 36, 'name': 'Chausie'}, {'id': 37, 'name': 'British Shorthair'}, {'id': 38, 'name': 'Siamese'}, {'id': 39, 'name': 'Tiger'}, {'id': 40, 'name': 'Ragdoll'}, {'id': 41, 'name': 'Somali'}, {'id': 42, 'name': 'Devon Rex'}, {'id': 43, 'name': 'Oriental Tabby'}, {'id': 44, 'name': 'Persian'}, {'id': 45, 'name': 'Extra-Toes Cat - Hemingway Polydactyl'}, {'id': 46, 'name': 'Torbie'}, {'id': 47, 'name': 'Ocicat'}, {'id': 48, 'name': 'American Wirehair'}, {'id': 49, 'name': 'Burmilla'}, {'id': 50, 'name': 'Russian Blue'}, {'id': 51, 'name': 'Javanese'}, {'id': 52, 'name': 'American Bobtail'}, {'id': 53, 'name': 'Oriental Long Hair'}, {'id': 54, 'name': 'Tuxedo'}, {'id': 55, 'name': 'Birman'}, {'id': 56, 'name': 'Dilute Calico'}, {'id': 57, 'name': 'LaPerm'}, {'id': 58, 'name': 'Sphynx - Hairless Cat'}, {'id': 59, 'name': 'Turkish Van'}, {'id': 60, 'name': 'Balinese'}, {'id': 61, 'name': 'Manx'}, {'id': 62, 'name': 'Canadian Hairless'}, {'id': 63, 'name': 'Havana'}, {'id': 64, 'name': 'Domestic Long Hair'}, {'id': 65, 'name': 'Domestic Medium Hair'}, {'id': 66, 'name': 'Dilute Tortoiseshell'}]

cfg_b = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg_b.merge_from_file("../detectron2/configs/COCO-Detection/my_Detection.yaml")
cfg_b.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg_b.MODEL.WEIGHTS = "model_0619999.pth"
predictor_breeds = DefaultPredictor(cfg_b)

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file("../detectron2/configs/COCO-Detection/my_Detection.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = "model_0134499.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
predictor_cats = DefaultPredictor(cfg)


REQUEST_KWARGS={
    'proxy_url': 'socks5h://127.0.0.1:9050',
}
updater = Updater(token='token', use_context=True, request_kwargs=REQUEST_KWARGS)

dispatcher = updater.dispatcher

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)

def help(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Отправь мне фото, а я попробую найти на нем котов!")


from telegram.ext import CommandHandler

help_handler = CommandHandler('help', help)
dispatcher.add_handler(help_handler)

def echo(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)

def photo_r(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Ищу котов!")

    buf = update.message.photo[-1].get_file().download_as_bytearray()
    im = cv2.imdecode(np.array(buf), cv2.IMREAD_COLOR)
    outputs_breeds = predictor_breeds(im)
    outputs_bbox = predictor_cats(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg_b.DATASETS.TRAIN[0]), scale=1.2)
    print('========', outputs_breeds)
    pred_clesses = []
    for c in outputs_breeds['instances'].pred_classes:
        pred_clesses.append(classes[c]['name'])

    v = v.draw_instance_predictions(outputs_bbox["instances"].to("cpu"))
    is_success, buffer = cv2.imencode(".jpg", v.get_image()[:, :, ::-1])
    # cv2.imwrite(bio, v.get_image()[:, :, ::-1])
    bio = io.BytesIO(buffer)
    bio.name = 'image.jpeg'
    bio.seek(0)
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=bio)



from telegram.ext import MessageHandler, Filters

echo_handler = MessageHandler(Filters.text, echo)
photo_handler = MessageHandler(Filters.photo, photo_r)
dispatcher.add_handler(echo_handler)
dispatcher.add_handler(photo_handler)

updater.start_polling()

import torch
from torch.utils import data  # 获取迭代数据
from torch.autograd import Variable  # 获取变量
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split
from dataloader import ImageDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import cv2

torch.manual_seed(42)


def run_eval(test_loader, model, device, batch_cnt=np.inf, loss_func=None):
    test_cnt = 0
    predictions = []
    y_true = []
    test_loss = []
    test_acc = []
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            test_cnt += 1
            a, b = batch["videos"].to(device, dtype=torch.float), batch["labels"].to(device)
            test_x = Variable(a)
            test_y = Variable(b)
            out = model(test_x)
            loss = loss_func(out, test_y)
            # print('predict:\t', torch.max(out, 1)[1])
            # print('actual :\t', test_y)
            result = torch.max(out, 1)[1].cpu().numpy()
            accuracy = result == test_y.cpu().numpy()  # accuracy is a list of 1s and 0s, 1 for correct and 0 for incorrect
            test_loss.append(loss)
            test_acc.append(accuracy.mean())

            predictions.extend(list(result))
            y_true.extend(list(test_y.cpu().numpy()))

            if idx >= batch_cnt - 1:
                break
        test_loss = sum(test_loss) / len(test_loss)
        test_acc = sum(test_acc) / len(test_acc)
        print('epoch    , Test  Loss: {:.3f}, Test  Accuracy: {:.3f}'.format(test_loss, test_acc))
        model.train()
        return predictions, y_true, test_loss, test_acc


def run_eval_detailed(test_loader, model, device, batch_cnt=np.inf):
    test_cnt = 0
    predictions = []
    y_true = []
    test_acc = []
    model.eval()
    incorrect_images = []
    incorrect_labels = []
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            test_cnt += 1
            a, b = batch["videos"].to(device, dtype=torch.float), batch["labels"].to(device)
            test_x = Variable(a)
            test_y = Variable(b)
            out = model(test_x)
            print('predict:\t', torch.max(out, 1)[1])
            print('actual :\t', test_y)
            result = torch.max(out, 1)[1].cpu().numpy()
            accuracy = result == test_y.cpu().numpy()
            test_acc.append(accuracy.mean())  # accuracy is a list of 1s and 0s, 1 for correct and 0 for incorrect

            incorrect = [i for i, e in enumerate(accuracy) if e == 0]  # get index of incorrect predictions
            incorrect_images.extend(a.cpu().numpy()[incorrect])
            incorrect_labels.extend(result[incorrect])

            predictions.extend(list(result))
            y_true.extend(list(test_y.cpu().numpy()))

            if idx >= batch_cnt - 1:
                break
        print('Accuracy:\t', sum(test_acc) / len(test_acc))
        model.train()

        return predictions, y_true, incorrect_images, incorrect_labels


if __name__ == "__main__":
    # 获取数据集
    train_images_name = "data/refined_Doge_dataset/data128.pkl"
    model_path = "save/transformer_refined_h12_d12_s8"

    dataset = ImageDataset(train_images_name)
    label_map = dataset.label_map
    data_size = len(dataset)
    train_test_split = 0.75
    train_size = round(data_size*0.75)
    test_size = data_size - train_size
    print(f"test size: {test_size}")

    generator = torch.Generator().manual_seed(63)
    train_data, test_data = random_split(dataset, [train_size, test_size], generator=generator)
    test_loader = data.DataLoader(test_data, batch_size=16, shuffle=True)

    print("Data loaded!")
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path)
    predictions, y_true, incorrect_images, incorrect_labels = run_eval_detailed(test_loader, model, device)
    cm = confusion_matrix(y_true, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(dataset.label_map.values()))
    disp.plot()
    plt.savefig("ConfusionMatrix_transformer_refined.png")
    plt.show()

    for i, video in enumerate(incorrect_images):  # show images with incorrect predictions
        image = np.transpose(video[:, 0], (1, 2, 0))
        image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)
        cv2.imshow(label_map[incorrect_labels[i]], image)
        cv2.waitKey()

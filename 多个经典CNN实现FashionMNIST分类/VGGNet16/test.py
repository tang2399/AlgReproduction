import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import VGG16


def test_data_process():
    test_data = FashionMNIST(root='./data',
                             train=False,
                             transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                             download=True)

    test_dataloader = data.DataLoader(dataset=test_data,
                                      batch_size=1,  # 一批次样本数
                                      shuffle=True,
                                      num_workers=0)

    return test_dataloader


def test_model_process_acc(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    test_num = 0

    # 梯度置零，不计算梯度，只进行前向传播
    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 设置为评估模式
            model.eval()
            # 前向传播
            output = model(b_x)
            # 查找每一行中最大行标
            pre_lab = torch.argmax(output, dim=1)

            test_corrects += torch.sum(pre_lab == b_y.data)
            test_num += b_x.size(0)

    # 计算准确率
    test_acc = test_corrects.double().item() / test_num
    print("test accuracy: ", test_acc)


def test_model_process_label(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            # 获取数值
            result = pre_lab.item()
            label = b_y.item()
            print("predicted value:", classes[result], "------ actual value:", classes[label])


if __name__ == "__main__":
    model = VGG16()
    # 加载模型
    model.load_state_dict(torch.load('best_model.pth'))
    # 加载数据
    test_dataloader = test_data_process()
    # 输出标签
    # test_model_process_label(model, test_dataloader)
    # 输出准确率
    test_model_process_acc(model, test_dataloader)

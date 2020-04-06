import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score


def show_loss(train_loss,valid_loss,save_path):
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.title('Trainning loss curve')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(save_path)
    plt.show()


def confusion_matrix(pred, label, cm):
    for p, t in zip(pred, label):
        cm[p, t] += 1
    return cm


def plot_confusion_matrix(cm, classes,title = 'ConfusionMatrix'):
    plt.imshow(cm,interpolation='nearest',cmap = plt.cm.Blues)#设置显示格式
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)#设置坐标轴每个位置的值或意义
    plt.yticks(tick_marks, classes)
    plt.axis("equal")
    ax =plt.gca()
    left, right = plt.xlim()
    ax.spines['left'].set_position(('data',left))
    ax.spines['right'].set_position(('data', right))
    for edge in ['top','bottom', 'right','left']:
        ax.spines[edge].set_edgecolor('white')
    thred = cm.max() / 2
    import  itertools
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,int(cm[i,j]),verticalalignment='center',horizontalalignment = 'center',
                 color = 'white' if cm[i,j] > thred else 'black',size ='8')
    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predict')
    plt.savefig('./%s.png'%title)
    plt.show()




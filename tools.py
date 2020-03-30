import matplotlib.pyplot as plt

def show_loss(train_loss,valid_loss,save_path):
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.title('Trainning loss curve')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(save_path)
    plt.show()
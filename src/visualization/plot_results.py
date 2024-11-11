import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')  
    plt.show()  

history = np.load('training_history.npy', allow_pickle=True).item()

print(history)

if __name__ == '__main__':
    plot_training_history(history)  

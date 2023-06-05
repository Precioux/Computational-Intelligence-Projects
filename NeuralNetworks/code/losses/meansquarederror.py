import numpy as np
import numpy.ma


class MeanSquaredError:
    def __init__(self):
        pass

    def compute(self, y_pred, y_true):
        """
        computes the mean squared error loss
            args:
                y_pred: predicted labels (n_classes, batch_size)
                y_true: true labels (n_classes, batch_size)
            returns:
                mean squared error loss
        """
        batch_size = y_pred.shape[0]
        print('THIS IS MEAN SQUERE ERROR')
        print(f'batch size: {batch_size}')
        cost = np.sum(np.square(y_pred.T - y_true.T)) / batch_size

        return np.squeeze(cost)

    def backward(self, y_pred, y_true):
        """
        computes the derivative of the mean squared error loss
            args:
                y_pred: predicted labels (n_classes, batch_size)
                y_true: true labels (n_classes, batch_size)
            returns:
                derivative of the mean squared error loss
        """
        batch_size = y_pred.shape[0]
        dMSE = (2 / batch_size) * (y_pred-y_true)
        print('THIS IS BACKWARD OF MSE')
        # print(dMSE.shape)
        return dMSE

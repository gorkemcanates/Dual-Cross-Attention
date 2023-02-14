import os
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('tkagg')

class TensorboardWriter(SummaryWriter):
    def __init__(self,
                 exp,
                 PATH,
                 fig_path,
                 num_data=48,
                 clear=None):
        super().__init__()
        self.fig_path = fig_path
        self.num_data = num_data
        self.exp = exp

        if clear is not None:
            self.clear_Tensorboard(clear)

    def write_results(self,
                      keys: list,
                      results_train,
                      results_test,
                      epoch):
        for metric, index in zip(keys, range(len(results_test))):
            self.add_scalars(metric, {'Training': results_train[index],
                                      'Validation': results_test[index]},
                             epoch + 1)

    def write_images(self,
                    keys: list,
                    data: list,
                    step,
                    C=3,
                    best=True):

        rand_images = self.get_random_predictions(data=data,
                                                  num_data=self.num_data)


        image = rand_images[0]
        # target = rand_images[1].unsqueeze(1)
        target = rand_images[1]
        prediction = rand_images[2]

        # if C == 1:

        #     target_hot = torch.eye(C + 1)[target.type(
        #         torch.LongTensor).squeeze(1)].permute(0, 3, 1, 2)
        #     pred_hot = torch.eye(C + 1)[prediction.type(
        #         torch.LongTensor).squeeze(1)].permute(0, 3, 1, 2)
        #     images = [image,
        #               torch.argmax(target_hot, dim=1, keepdim=True),
        #               torch.argmax(pred_hot, dim=1, keepdim=True)
        #               ]
        if C == 1:
            target_hot, pred_hot = target.squeeze(1), prediction.squeeze(1)
        elif C == 3:
            prediction = prediction.unsqueeze(1)
            target_hot = torch.eye(C)[target.type(
                torch.LongTensor).squeeze(1)].permute(0, 3, 1, 2)
            pred_hot = torch.eye(C)[prediction.type(
                torch.LongTensor).squeeze(1)].permute(0, 3, 1, 2)
        else:
            prediction = prediction.unsqueeze(1)
            target_hot, pred_hot = target, prediction
                        
        images = [image,
                    target_hot,
                    pred_hot
                    ]
        if best:
            self.visualize(data=images,
                           step=step)

        # for key, im in zip(keys, images):
        #     self.add_images(f'' + key,
        #                     im,
        #                     global_step=step)



    def visualize(self, data, step):
        plt.ioff()
        if not os.path.exists(self.fig_path + 'data/'):
            os.mkdir(self.fig_path + 'data/')
        fig_data = plt.figure(figsize=(16, 12))
        for i in range(len(data[0])):
            ax = fig_data.add_subplot(8, 6, i + 1, xticks=[], yticks=[])
            d = data[0][i].permute(1, 2, 0)  
            if len(d.shape) != 3:
                d = d[:, :, 0]              
            plt.imshow(d)
            # plt.imshow(data[0][i].permute(1, 2, 0))

        fig_data.savefig(self.fig_path + 'data/' + str(step) + '.png')
        plt.close(fig_data)

        if not os.path.exists(self.fig_path + 'target/'):
            os.mkdir(self.fig_path + 'target/')
        fig_tar = plt.figure(figsize=(16, 12))
        for i in range(len(data[1])):
            ax = fig_tar.add_subplot(8, 6, i + 1, xticks=[], yticks=[])
            if self.exp == 'SYNAPS/':
                plt.imshow(data[1][i].permute(1, 2, 0))
            else:
                plt.imshow(data[1][i], cmap='gray')

        fig_tar.savefig(self.fig_path + 'target/' + str(step) + '.png')
        plt.close(fig_tar)

        if not os.path.exists(self.fig_path + 'prediction/'):
            os.mkdir(self.fig_path + 'prediction/')
        fig_pred = plt.figure(figsize=(16, 12))
        for i in range(len(data[2])):
            ax = fig_pred.add_subplot(8, 6, i + 1, xticks=[], yticks=[])
            if self.exp == 'SYNAPS/':
                plt.imshow(data[2][i].permute(1, 2, 0))
            else:
                plt.imshow(data[2][i], cmap='gray')

        fig_pred.savefig(self.fig_path + 'prediction/' + str(step) + '.png')
        plt.close(fig_pred)

    def write_hyperparams(self,
                          hparams_dict,
                          metric_dict):

        self.add_hparams(hparam_dict=hparams_dict,
                         metric_dict=metric_dict)

    def write_histogram(self):
        pass

    @staticmethod
    def clear_Tensorboard(file):
        dir = 'runs/' + file
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))

    @staticmethod
    def get_random_predictions(data: list,
                               num_data=36):
        # seed = torch.randint(low=0,
        #                      high=len(data[0]),
        #                      size=(num_data,))
        if data[0].shape[0] >= num_data:
            seed = torch.arange(num_data)
        else:
            seed = torch.arange(data[0].shape[0])            
        random_data = [i[seed] for i in data]
        return random_data

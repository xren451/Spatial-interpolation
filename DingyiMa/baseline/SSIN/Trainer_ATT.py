import torch.nn as nn
import torch.optim as optim
import time
import pickle
import torch.nn.functional as F
import traceback
import sys
sys.path.append('..')  # import the upper directory of the current file into the search path
from SSIN.networks.Models import SpaFormer
from SSIN.dataset_collator.create_data_att import *
from SSIN.utils.utils import *
from SSIN.networks.Optim import ScheduledOptim
import SSIN.utils.config as cfg

class ModifiedAFFWithSoftmax(nn.Module):
    def __init__(self, channels=6, r=2, num_features=6):
        super(ModifiedAFFWithSoftmax, self).__init__()
        self.num_features = num_features
        inter_channels = int(channels // r )
        
        
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        
        # self.global_att = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(inter_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(channels),
        # )

        self.mlp=nn.Linear(2,1)
    def forward(self, x):
        xl = self.local_att(x)
        # xg = self.global_att(x)
        # repeat_num=xl.size(2)
        # xg=xg.repeat(1, 1, repeat_num, 1)
        # xlg = torch.cat((xl, xg), dim=3)
        xlg=xl
        # batch_size, num_channels, height, width = xlg.size()
        # xlg = xlg.view(-1, width)
        # xlg = F.relu(self.mlp(xlg))
        # xlg=xlg.view(batch_size,num_channels,height,-1)
        # xlg = xl + xg
        # xlg = xlg - xl
        
        weights = xlg.view(xlg.size(0), self.num_features, -1)
        weights = F.softmax(weights, dim=1)
        weights = weights.view_as(xlg)
        
        separated_weights = torch.chunk(weights, self.num_features, dim=1)
        weighted_features = [x[:, i:i+1, :, :] * separated_weights[i] for i in range(self.num_features)]
        
        output = torch.sum(torch.stack(weighted_features, dim=0), dim=0)

        return output.squeeze(1)


# class ModifiedAFFWithSoftmax(nn.Module):
#     def __init__(self, channels=5, r=2, num_features=5):
#         super(ModifiedAFFWithSoftmax, self).__init__()
#         self.layer1 = nn.Linear(5, 1)
#     def forward(self, x):
#         batch_size, num_channels, height, width = x.size()
#         x=x.squeeze(-1)
#         x=x.transpose(1,2)
#         x=x.view(-1,num_channels)
#         x1=self.layer1(x)
#         output=F.relu(x1)
#         output=output.view(batch_size,height,1)

#         return output.squeeze(1)

# This version supports the mini-batch training
class MaskedTrainer:
    def __init__(self, args, global_step=0, out_path=None, init_training=True):
        print("init")
        self.args = args
        self.global_step = global_step
        self.out_path = out_path

        # load data
        self.all_seq_data, self.all_seq_fusion,self.invalid_masks_data, self.r_pos_mat_data, self.adj_attn_mask = self.load_train_data()
        cfg.d_feat, cfg.d_pos = self.all_seq_data[0].shape[-1], self.r_pos_mat_data[0].shape[-1]

        # load model
        self.model = self.load_model()
        self.fusion_model=ModifiedAFFWithSoftmax().cuda()

        if init_training:
            print("Load data and build model. Done!")

            # set loss function and optimizer
            self.criterion = nn.MSELoss(reduction="none")
            self.optimizer = ScheduledOptim(
            optim.Adam(list(self.model.parameters()) + list(self.fusion_model.parameters()), betas=(0.9, 0.98), eps=1e-09),
            args.lr_mul, args.d_model, args.n_warmup_steps)

            self.num_params()

    def load_train_data(self):
        # load data
        with open(self.args.train_data_path, "rb") as fp:
            data_dict = pickle.load(fp)
        print("use att model_load_train")
        all_seq_data = data_dict["train_data"][:, :, 0:1]
        all_seq_fusion=data_dict["fusion_data"][:, :, :]
        invalid_masks_data = data_dict["invalid_masks"]
        r_pos_mat_data = data_dict["r_pos_mat"]

        if "adj_attn_mask" in data_dict.keys():
            adj_attn_mask = data_dict["adj_attn_mask"]
        else:
            adj_attn_mask = None

        return all_seq_data,all_seq_fusion, invalid_masks_data, r_pos_mat_data, adj_attn_mask

    def load_test_data_generator(self):
        # load data
        with open(self.args.test_data_path, "rb") as fp:
            data_dict = pickle.load(fp)
        print("use att model_load_test")
        all_seq_data = data_dict["test_data"][:, :, 0:1]
        all_seq_fusiondata=data_dict["fusion_data"][:, :, :]
        r_pos_mat = data_dict["r_pos_mat"]
        invalid_masks = data_dict["invalid_masks"]
        test_masks = data_dict["test_masks"]
        all_timestamps = data_dict["timestamps"]

        if "adj_attn_mask" in data_dict.keys():
            adj_attn_mask = data_dict["adj_attn_mask"]
        else:
            adj_attn_mask = None
        test_data_expanded = np.expand_dims(all_seq_fusiondata, axis=-1)
        test_data_expanded = np.transpose(test_data_expanded, (0, 2, 1,3))
        test_data=test_data_expanded[:,:,:,:]
        test_tensor = torch.from_numpy(test_data)  # Batch size 8784, 18 features, 41 sites, width 1
        test_tensor = test_tensor.float().cuda()
        output_test_tensor = self.fusion_model(test_tensor)
        output_test_numpy=output_test_tensor.detach().cpu().numpy()
        test_data_generator = create_test_data(all_seq_data,output_test_numpy ,invalid_masks, test_masks, all_timestamps, adj_attn_mask)

        return r_pos_mat, test_data_generator

    def load_model(self):
        if self.args.model_type == "SpaFormer":
            model = SpaFormer(cfg.d_feat, cfg.d_pos, self.args.n_layers, self.args.n_head, self.args.d_k, self.args.d_v, self.args.d_model,
                              self.args.d_inner, self.args.dropout, cfg.scale_emb, return_attns=self.args.return_attns)
        else:
            raise NotImplementedError("The mode type is not available!")

        if self.args.cuda:
            model = model.cuda()
        return model

    def train(self):
        self.model.train()
        self.fusion_model.train()
        r_pos_mat = torch.FloatTensor(self.r_pos_mat_data).cuda()
        training_time = 0
        test_time = 0

        try:
            tot_loss, tot_avg_loss = 0, 0
            for epoch in range(1, self.args.epochs+1):
                ep_start_time = time.time()
                # Dynamic masking: randomly generate masked data for each epoch
                train_data_expanded = np.expand_dims(self.all_seq_fusion, axis=-1)
                train_data_expanded = np.transpose(train_data_expanded, (0, 2, 1,3))
                train_data=train_data_expanded[:,:,:,:]
                train_tensor = torch.from_numpy(train_data)  # Batch size 8784, 18 features, 41 sites, width 1
                train_tensor = train_tensor.float().cuda()
                output_tensor = self.fusion_model(train_tensor)
                output_numpy=output_tensor.detach().cpu().numpy()
                train_data_iter = create_train_data(epoch,output_numpy ,self.all_seq_data, self.invalid_masks_data,
                                                    self.args.batch_size, self.args.masked_lm_prob,
                                                    times=self.args.mask_time, adj_attn_mask=self.adj_attn_mask)
                running_loss, avg_loss = 0, 0
                for data in train_data_iter:
                    self.global_step += 1

                    # masked_seq, masked_labels: have been seq-wise standardized in train_data_iter
                    masked_seq, masked_indexes, masked_labels, masked_label_weights, \
                    attn_mask = convert_train_data(self.args, data)

                    # For one timestamp, when owning many zero rainfall values,
                    # the random masked_seq may be all zeros, then std = 0, then will generate NaN values;
                    # if the inputs include NaN, skip this input sequence
                    if torch.isnan(masked_seq).any() or torch.isinf(masked_seq).any() or len(masked_seq) == 0:
                        continue

                    # 1. forward the model
                    self.optimizer.zero_grad()

                    outputs, _, _ = self.model(masked_seq, r_pos_mat, masked_indexes, attn_mask=attn_mask)

                    # 2. MSE loss of predicting masked elements
                    per_example_loss = self.criterion(outputs, masked_labels)  # loss for each elements
                    numerator = torch.sum(per_example_loss.squeeze() * masked_label_weights)
                    denominator = torch.sum(masked_label_weights) + 1e-10
                    loss = numerator / denominator

                    # 3. backward and optimization only in train
                    loss.backward()
                    # self.optimizer.step()
                    self.optimizer.step_and_update_lr()

                    # loss
                    # running_loss += loss.item()
                    tot_loss += loss.item()
                    tot_avg_loss = tot_loss / self.global_step

                    # todo: here, do not save the attention list images

                ep_end_time = time.time()
                ep_cost_time = ep_end_time - ep_start_time
                training_time += ep_cost_time

                # save model checkpoint for each 10 epochs
                if epoch % 10 == 0:
                    self.save_checkpoint(epoch, self.global_step, self.out_path.checkpoints_path)

            test_time = self.test(self.out_path.test_ret_path + f"/test_ret.csv")

        except BaseException:
            traceback.print_exc()

        return training_time, test_time

### 做到这里了，注意后面的优化

    def test(self, ret_path, model_path=None):
        if model_path is not None:
            if os.path.exists(model_path):
                state_dicts = torch.load(model_path)
                self.model.load_state_dict(state_dicts['model_state_dict'])
                self.fusion_model.load_state_dict(state_dicts['fusion_model_state_dict'])
                # self.model.load_state_dict(torch.load(model_path))
                print(f"Reloaded Model from {model_path}!")
            else:
                raise FileNotFoundError(f"Can not find model in {model_path}!")
        print("use att model_test")
        self.model.eval()
        self.fusion_model.eval()
        r_pos_mat_test, test_data_iter = self.load_test_data_generator()
        r_pos_mat_test = torch.FloatTensor(r_pos_mat_test).cuda()

        labels_list, preds_list = [], []
        timestamp_list, gauge_list = [], []

        test_start_time = time.time()
        with torch.no_grad():
            for data in test_data_iter:
                # [tensor, tensor, tensor, numpy, tensor, numpy, numpy]
                # tensor: [batch_size=1, seq_len, in_dim], numpy: [seq_len]
                # masked_seq: have been standardized in DataLoaderNew
                masked_seq, masked_indexes, masked_labels, attn_mask, mean_value, std_value, timestamp = \
                    convert_test_data(self.args, data)

                preds, _, _ = self.model(masked_seq, r_pos_mat_test, masked_indexes, attn_mask=attn_mask)

                preds = (preds.squeeze() * std_value) + mean_value  # inverse standardization
                preds = preds.cpu().numpy().flatten()  # flatten to 1D array

                labels_list.append(masked_labels.flatten())
                preds_list.append(preds)
                timestamp_list.append(timestamp.flatten())
        test_end_time = time.time()
        test_cost_time = test_end_time - test_start_time
        save_csv_results(ret_path, timestamp_list, gauge_list, labels_list, preds_list)
        print("Save test results. Done!")

        return test_cost_time

    def num_params(self, print_out=True):
        params_requires_grad = filter(lambda p: p.requires_grad, self.model.parameters())
        params_requires_grad = sum([np.prod(p.size()) for p in params_requires_grad])  #/ 1_000_000

        parameters = sum([np.prod(p.size()) for p in self.model.parameters()])  #/ 1_000_000
        if print_out:
            print('Trainable total Parameters: %d' % parameters)
            print('Trainable requires_grad Parameters: %d' % params_requires_grad)

    def save_checkpoint(self, epoch, steps, save_path):
        """
        Saving the current MLM model on file_path
        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        # output_path = save_path + f"/checkpoint_{epoch}epoch_{steps}steps.pyt"
        output_path = save_path + f"/checkpoint_{epoch}epoch.pyt"
        state_dicts = {
        'model_state_dict': self.model.state_dict(),
        'fusion_model_state_dict': self.fusion_model.state_dict()
        }
        torch.save(state_dicts, output_path)
        # torch.save(self.model.state_dict(), output_path)
        print("EP:%d Checkpoint Saved on:" % epoch, output_path)
        return output_path


def convert_train_data(args, data):
    masked_seq, masked_indexes, masked_labels, masked_label_weights, attn_mask = data

    # convert to tensor
    masked_seq = torch.FloatTensor(masked_seq)
    masked_indexes = torch.LongTensor(masked_indexes)
    masked_labels = torch.FloatTensor(masked_labels)
    masked_label_weights = torch.FloatTensor(masked_label_weights)
    attn_mask = torch.FloatTensor(attn_mask)

    if args.cuda:
        masked_seq = masked_seq.cuda()
        masked_indexes = masked_indexes.cuda()
        masked_labels = masked_labels.cuda()
        masked_label_weights = masked_label_weights.cuda()
        attn_mask = attn_mask.cuda()

    return masked_seq, masked_indexes, masked_labels, masked_label_weights, attn_mask


def convert_test_data(args, data):
    masked_seq, masked_indexes, masked_labels, attn_mask, mean_value, std_value, timestamp = data

    # convert to tensor, batch_size = 1
    masked_seq = torch.FloatTensor(masked_seq)
    masked_indexes = torch.LongTensor(masked_indexes)
    attn_mask = torch.FloatTensor(attn_mask)

    if args.cuda:
        masked_seq = masked_seq.cuda()
        masked_indexes = masked_indexes.cuda()
        attn_mask = attn_mask.cuda()

    return masked_seq, masked_indexes, masked_labels, attn_mask, mean_value, std_value, timestamp


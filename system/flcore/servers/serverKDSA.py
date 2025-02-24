# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang
import copy
import random
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import time
import torch


from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from scipy import spatial
import numpy as np
from threading import Thread
# KD-tree based similarity for adaptive personalized aggregation
# preserve high layers of model
class FedKDSA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.dep=args.decouple
        self.layer_idx = args.layer_idx
        # KDSA parameters
        self.beta=args.beta
        self.num_agg_clients=args.num_agg_clients

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.cid_to_vectors = {}
        self.sims={}
        self.distances={}


        # Fixed gaussian noise for model embedding
        batch_size = 1
        if "MNIST" in self.dataset:
            mean, std = 0, 1
            self.rgauss = torch.randn(batch_size, 1, 28, 28) * std + mean
            # print('rgauss:', self.rgauss.shape)
        elif "Cifar10" in args.dataset:  # Cifar10 and Cifar100
            mean, std = 0.0, 0.1
            self.rgauss = torch.randn(batch_size, 3, 32, 32) * std + mean
        else:
            raise NotImplementedError("Dataset gaussian noise not implemented!")
        self.rgauss=self.rgauss.to(self.device)

    # override send_models
    def send_models(self,i):
        if i == 0:  # initialization all client models
            super().send_models()
        else:
            assert (len(self.selected_clients) > 0)
            # print(self.personalized_model)
            for sid,client in enumerate(self.selected_clients):
                start_time = time.time()
                # print("sid:",sid)
                client.set_parameters(self.personalized_model[sid])  # personalized model

                client.send_time_cost['num_rounds'] += 1
                client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    # Construct a KD-Tree after clients finished training
    def client_similarity(self):
        for client in self.selected_clients:
            self.cid_to_vectors[client.id] = np.squeeze(client.model(self.rgauss).detach().cpu())  # embedding models dictionary
        self.vid_to_cid = list(self.cid_to_vectors.keys())
        self.vectors = np.vstack(list(self.cid_to_vectors.values()))
        self.tree = spatial.KDTree(self.vectors)
        # print('embeddings:',self.vectors)

    # Search similar models for each selected client using KD-Tree
    # for KDSA, we construct the threshold based on the distances
    def get_similar_models(self, epoch):
        #if cid in self.cid_to_vectors and (self.curr_round+1)%self.args.h_interval == 0:
        for client in self.selected_clients:
            cid=client.id
            embedding = self.cid_to_vectors[cid]
            searchs= self.tree.query(embedding, self.num_agg_clients)  # search for k similar models, containing itself
            self.sims[cid]=[]
            self.distances[cid] = (searchs[0])
            # construct the threshold
            dist_min=self.distances[cid][1]  # omit client i itself, with distance=0
            dist_avg=sum(self.distances[cid])/(len(self.distances[cid])-1)  # because the distance between cid and itself is zero, the avg can be calculated like this
            threshold=dist_avg+(epoch/self.beta)*(dist_min-dist_avg)
            print("distances:{}".format(self.distances[cid]))
            print("dist_min:{}, dist_avg:{}, threshold:{}".format(dist_min, dist_avg,threshold))
            self.sims[cid].append(cid)
            for id,vid in enumerate(searchs[1]):  # searchs[1] is indexs of searchs[0]
                if id == 0:
                    continue
                # print("distance:{}, threshold:{}".format(self.distances[cid][id],threshold))
                if self.distances[cid][id] <= threshold:
                    self.sims[cid].append(self.vid_to_cid[vid])
            # print("Epoch:{},cid:{},sims:{}".format(epoch, cid, self.sims[cid]))


    # override receive_models
    def receive_models(self,epoch):
        assert (len(self.selected_clients) > 0)

        # active_clients = random.sample(
        #     self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))  # not used in KDSim

        # self.uploaded_ids = {}
        self.uploaded_weights_p = {}
        self.uploaded_models_p = {}
        tot_samples = [0 for i in range(self.current_num_join_clients)]

        for sid,client in enumerate(self.selected_clients):
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                # get weights for each selected client
                agg_list=self.sims[client.id]
                # self.uploaded_ids[sid] = agg_list
                self.uploaded_weights_p[sid] = []
                self.uploaded_models_p[sid] = []
                for agg_cid in agg_list:
                    agg_client=self.clients[agg_cid]
                    tot_samples[sid] += agg_client.train_samples
                    self.uploaded_weights_p[sid].append(agg_client.train_samples)
                    self.uploaded_models_p[sid].append(agg_client.model)
                for i, w in enumerate(self.uploaded_weights_p[sid]):
                    self.uploaded_weights_p[sid][i] = w / tot_samples[sid]
                print("Epoch:{},cid:{},agg_cid:{},w:{}".format(epoch, client.id, agg_list, self.uploaded_weights_p[sid]))

    # Override aggregate_parameters to aggregate personalized models
    def aggregate_parameters(self):
        self.personalized_model = {}
        for sid,client in enumerate(self.selected_clients):
            assert (len(self.uploaded_models_p[sid]) > 0)
            self.personalized_model[sid] = copy.deepcopy(self.uploaded_models_p[sid][0])
            for param in self.personalized_model[sid].parameters():
                param.data.zero_()
            # print(self.uploaded_weights_p[sid])
            for w, client_model in zip(self.uploaded_weights_p[sid], self.uploaded_models_p[sid]):
                self.add_parameters_p(w, client_model,sid)
        # print("pmodle:",self.personalized_model.keys())
    # override add_parameters
    def add_parameters_p(self, w, client_model, sid):
        for personalized_param, client_param in zip(self.personalized_model[sid].parameters(), client_model.parameters()):
            personalized_param.data += client_param.data.clone() * w

    def parameter_decouple(self):
        for model_p,client in zip(self.personalized_model,self.selected_clients):
            params_l=list(client.model.parameters())
            params_p=list(self.personalized_model[model_p].parameters())
            # Replace the high layers with local model
            for param_p, param_l in zip(params_p[-self.layer_idx:], params_l[-self.layer_idx:]):
                param_p.data = param_l.data.clone()
    def train(self):
        for i in range(self.global_rounds+1):  # global round
            s_t = time.time()
            self.send_models(i)  # override send models; After aggregation, update clients' models
            self.selected_clients = self.select_clients()


            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate before local training")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            # Construct a KD-Tree after clients finished training
            self.client_similarity()

            # Search similar models using KD-Tree
            self.get_similar_models(i)
            # personalized aggregation for selected clients
            self.receive_models(i)
            self.aggregate_parameters()
            # preserve high layers of local model
            if self.dep:
                self.parameter_decouple()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

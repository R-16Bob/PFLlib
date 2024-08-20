# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

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
# KD-tree based similarity for personalized aggregation

class FedKDSim(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.cid_to_vectors = {}
        self.sims={}


        # Fixed gaussian noise for model embedding
        if "MNIST" in self.dataset:
            mean, std = 0, 1
            self.rgauss = torch.randn(1, 28, 28) * std + mean
        elif "Cifar10" in args.dataset:
            mean, std = 0.0, 0.1
            batch_size = 1
            self.rgauss = torch.randn(batch_size, 3, 32, 32) * std + mean
        else:
            raise NotImplementedError("Dataset gaussian noise not implemented!")

    # TODO：override send_models
    def send_models(self,i):
        if i == 0:  # initialization all client models
            super().send_models()
        else:
            assert (len(self.selected_clients) > 0)

            for client in self.selected_clients:
                start_time = time.time()

                client.set_parameters(self.global_model)  # TODO:替换为对应的个性化模型

                client.send_time_cost['num_rounds'] += 1
                client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    # Construct a KD-Tree after clients finished training
    def client_similarity(self):
        for client in self.selected_clients:
            self.cid_to_vectors[client.id] = np.squeeze(client.model(self.rgauss))  # embedding models dictionary
        self.vid_to_cid = list(self.cid_to_vectors.keys())
        self.vectors = list(self.cid_to_vectors.values())
        self.tree = spatial.KDTree(self.vectors)

    # TODO: Search similar models for each selected client using KD-Tree
    def get_similar_models(self):
        #if cid in self.cid_to_vectors and (self.curr_round+1)%self.args.h_interval == 0:
        for client in self.selected_clients:
            cid=client.id
            embedding = self.cid_to_vectors[cid]
            searchs= self.tree.query(embedding, self.args.num_agg_clients)
            self.sims[cid]=searchs[1]
            hids = []
            weights = []
            for vid in sims[1]:
                selected_cid = self.vid_to_cid[vid]
                if selected_cid == cid:
                    continue
                w = self.cid_to_weights[selected_cid]
                if self.args.scenario == 'labels-at-client':
                    half = len(w)//2
                    w = w[half:]
                weights.append(w)
                hids.append(selected_cid)
            return weights[:self.args.num_helpers]
        else:
            return None

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
    # TODO:Override aggregate_parameters
    def aggregate_parameters(self):
        return 0
    def train(self):
        for i in range(self.global_rounds+1):  # global round
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models(i)  # TODO: override send models

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
            self.get_similar_models()
            # TODO: personalized aggregation for selected clients
            self.receive_models()
            self.aggregate_parameters()

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

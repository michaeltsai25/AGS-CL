import sys,time,os
import numpy as np
import torch
from copy import deepcopy
import utils
from utils import *
sys.path.append('..')
from arguments import get_args
import torch.nn.functional as F
import torch.nn as nn
from typing import Callable, Optional
args = get_args()

if 'omniglot' in args.experiment:
    from networks.conv_net_omniglot import Net
elif 'mixture' in args.experiment:
    from networks.alexnet import Net
else:
    from networks.conv_net import Net


class Appr(object):

    def __init__(self,model,nepochs=100,sbatch=256,lr=0.001,lr_min=1e-6,lr_factor=3,lr_patience=5,clipgrad=100,args=None, log_name = None):
        self.model=model
        self.model_old=model
        self.omega=None
        self.log_name = log_name

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad

        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()
        self.lamb=args.lamb 
        self.initail_mu = args.mu
        self.mu = args.mu
        self.freeze = {}
        self.mask = {}
        
        #initializes elements in self.mask to be zeros in the shape of the convolutional kernels; dictionary means that values are assigned to name of kernel
        for (name,p) in self.model.named_parameters():
            if len(p.size())<2:
                continue
            name = name.split('.')[:-1]
            name = '.'.join(name)
            self.mask[name] = torch.zeros(p.shape[0])

        if len(args.parameter)>=1:
            params=args.parameter.split(',')
            print('Setting parameters to',params)
            self.lamb=float(params[0])
            
        #OGD INIT
        self.config = args

        print(f"### The model has {count_parameter(self.model)} parameters ###")

        # # TODO : remove from init : added only for the NTK gen part ?
        # self.optimizer = self.optimizer = torch.optim.SGD(params=self.model.parameters(),
        #                                                   lr=self.config.lr,
        #                                                   momentum=0,
        #                                                   weight_decay=0)

        if self.config.is_split_cub :
            n_params = get_n_trainable(self.model)
        elif self.config.is_split :
            n_params = count_parameter(self.model.linear)
        else :
            n_params = count_parameter(self.model)
        self.ogd_basis = torch.empty(n_params, 0)
        # self.ogd_basis = None
        self.ogd_basis_ids = defaultdict(lambda: torch.LongTensor([]))

        if self.config.gpu:
            # self.ogd_basis = self.ogd_basis.cuda()
            self.ogd_basis_ids = defaultdict(lambda: torch.LongTensor([]).cuda())

        # Store initial Neural Tangents

        self.task_count = 0
        self.task_memory = {}
        self.task_mem_cache = {}

        self.task_grad_memory = {}
        self.task_grad_mem_cache = {}

        self.mem_loaders = list()

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, t, xtrain, ytrain, xvalid, yvalid, data, input_size, taskcla):
        
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)

        #replace this with ogd functions updating individual nodes orthogonally
        if t>0:
            self.freeze = {}
            for name, param in self.model.named_parameters():
                if 'bias' in name or 'last' in name:
                    continue
                key = name.split('.')[0]
                if 'conv1' not in name:
                    if 'conv' in name: #convolution layer
                        temp = torch.ones_like(param)
                        temp[:, self.omega[prekey] == 0] = 0
                        temp[self.omega[key] == 0] = 1
                        self.freeze[key] = temp
                    else:#linear layer
                        temp = torch.ones_like(param)
                        temp = temp.reshape((temp.size(0), self.omega[prekey].size(0) , -1))
                        temp[:, self.omega[prekey] == 0] = 0
                        temp[self.omega[key] == 0] = 1
                        self.freeze[key] = temp.reshape(param.shape)
                prekey = key
                
        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0=time.time()

            # CUB 200 xtrain_cropped = crop(x_train)
            num_batch = xtrain.size(0)

            self.train_epoch(e,t,xtrain,ytrain,lr,data) #check parameters

            clock1=time.time()
            train_loss,train_acc=self.eval(t,xtrain,ytrain)
            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e+1,1000*self.sbatch*(clock1-clock0)/num_batch,
                1000*self.sbatch*(clock2-clock1)/num_batch,train_loss,100*train_acc),end='')
            # Valid
            valid_loss,valid_acc=self.eval(t,xvalid,yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            print(' lr : {:.6f}'.format(self.optimizer.param_groups[0]['lr']))
            
            # Adapt lr
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = utils.get_model(self.model)
                patience = self.lr_patience
                print(' *', end='')

            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        print()
                    patience = self.lr_patience
                    self.optimizer = self._get_optimizer(lr)
            print()

        # Restore best
        utils.set_model_(self.model, best_model)

        # Update old
        self.model.act = None

        temp=utils.gs_cal(t,xtrain,ytrain,self.criterion, self.model) #review this
        for n in temp.keys():
            if t>0:
                self.omega[n] = args.eta * self.omega[n] + temp[n] #equation 8; temp represents average relu activation
            else:
                self.omega = temp
            self.mask[n] = (self.omega[n]>0).float()
            
        torch.save(self.model.state_dict(), './trained_model/' + self.log_name + '_task_{}.pt'.format(t))
        
        test_loss, test_acc = self.eval(t, xvalid, yvalid)
        print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(test_loss,100*test_acc))
        
        dummy = Net(input_size, taskcla).cpu()

        pre_name = 0
        
        #read through this part
        for (name,dummy_layer),(_,layer) in zip(dummy.named_children(), self.model.named_children()):
            with torch.no_grad():
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                    if pre_name!=0:
                        temp = (self.omega[pre_name]>0).float()
                        if isinstance(layer, nn.Linear) and 'conv' in pre_name:
                            temp = temp.unsqueeze(0).unsqueeze(-1)
                            weight = layer.weight
                            weight = weight.view(weight.size(0), temp.size(1), -1)
                            weight = weight * temp
                            layer.weight.data = weight.view(weight.size(0), -1)
                        elif len(weight.size())>2:
                            temp = temp.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                            layer.weight *= temp
                        else:
                            temp = temp.unsqueeze(0)
                            layer.weight *= temp
                            
                    weight = layer.weight.data
                    bias = layer.bias.data
                    
                    if len(weight.size()) > 2:
                        norm = weight.norm(2,dim=(1,2,3))
                        mask = (self.omega[name]==0).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                    else:
                        norm = weight.norm(2,dim=(1))
                        mask = (self.omega[name]==0).float().unsqueeze(-1)

                    zero_cnt = int((mask.sum()).item())
                    indice = np.random.choice(range(zero_cnt), int(zero_cnt*(1-args.rho)), replace=False)
                    indice = torch.tensor(indice).long()
                    idx = torch.arange(weight.shape[0])[mask.flatten(0)==1][indice]
                    mask[idx] = 0

                    layer.weight.data = (1-mask)*layer.weight.data + mask*dummy_layer.weight.data
                    mask = mask.squeeze()
                    layer.bias.data = (1-mask)*bias + mask*dummy_layer.bias.data

                    pre_name = name

                if isinstance(layer, nn.ModuleList):
                    
                    weight = layer[t].weight
                    if 'omniglot' in args.experiment:
                        weight = weight.view(weight.shape[0], self.omega[pre_name].shape[0], -1)
                        weight[:,self.omega[pre_name] == 0] = 0
                        weight = weight.view(weight.shape[0],-1)
                    else:
                        weight[:, self.omega[pre_name] == 0] = 0
        test_loss, test_acc = self.eval(t, xvalid, yvalid)
        
        #check to make sure loader is correct
        loader = torch.utils.data.DataLoader(self.task_memory[self.task_count],
                                                                            batch_size=self.config.batch_size,
                                                                            shuffle=True,
                                                                            num_workers=2)
        self.model_old = deepcopy(self.model)
        self.model_old.train()
        self.update_ogd_basis(loader) #replace this with full train loader
        # utils.freeze_model(self.model_old) # Freeze the weights
        return

    def train_epoch(self,epoch,t,x,y,lr,data):
        self.model.train()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r).cpu()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=x[b]
            targets=y[b]

            # Forward current model
            
            outputs = self.model.forward(images)[t]
            loss=self.criterion(t,outputs,targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer_step(epoch, i, t, x, y)
            self.optimizer.step()

            #Freeze the outgoing weights
            # if t>0:
            #     for name, param in self.model.named_parameters():
            #         if 'bias' in name or 'last' in name or 'conv1' in name:
            #             continue
            #         key = name.split('.')[0]
            #         param.data = param.data*self.freeze[key]

        self.proxy_grad_descent(t,lr)
        
        return

    def eval(self,t,x,y):
        with torch.no_grad():
            total_loss=0
            total_acc=0
            total_num=0
            self.model.eval()

            r = np.arange(x.size(0))
            r = torch.LongTensor(r).cpu()

            # Loop batches
            for i in range(0,len(r),self.sbatch): 
                if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
                else: b=r[i:]
                images=x[b]
                targets=y[b]

                # Forward
                
                output = self.model.forward(images)[t]

                loss=self.criterion(t,output,targets)
                _,pred=output.max(1)
                hits=(pred==targets).float()

                # Log
                total_loss+=loss.data.cpu().numpy()*len(b)
                total_acc+=hits.sum().data.cpu().numpy()
                total_num+=len(b)

            return total_loss/total_num,total_acc/total_num
    
    def proxy_grad_descent(self, t, lr):
        with torch.no_grad():
            for (name,module),(_,module_old) in zip(self.model.named_children(),self.model_old.named_children()):
                if not isinstance(module, torch.nn.Linear) and not isinstance(module, torch.nn.Conv2d):
                    continue
                
                mu = self.mu
                
                key = name
                weight = module.weight
                bias = module.bias
                weight_old = module_old.weight
                bias_old = module_old.bias
                
                if len(weight.size()) > 2:
                    norm = weight.norm(2, dim=(1,2,3))
                else:
                    norm = weight.norm(2, dim=(1))
                norm = (norm**2 + bias**2).pow(1/2)                

                aux = F.threshold(norm - mu * lr, 0, 0, False)
                alpha = aux/(aux+mu*lr)
                coeff = alpha * (1-self.mask[key])

                if len(weight.size()) > 2:
                    sparse_weight = weight.data * coeff.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
                else:
                    sparse_weight = weight.data * coeff.unsqueeze(-1) 
                sparse_bias = bias.data * coeff

                penalty_weight = 0
                penalty_bias = 0

                if t>0:
                    if len(weight.size()) > 2:
                        norm = (weight - weight_old).norm(2, dim=(1,2,3))
                    else:
                        norm = (weight - weight_old).norm(2, dim=(1))

                    norm = (norm**2 + (bias-bias_old)**2).pow(1/2)

                    aux = F.threshold(norm - self.omega[key]*self.lamb*lr, 0, 0, False)
                    boonmo = lr*self.lamb*self.omega[key] + aux
                    alpha = (aux / boonmo)
                    alpha[alpha!=alpha] = 1

                    coeff_alpha = alpha * self.mask[key]
                    coeff_beta = (1-alpha) * self.mask[key]


                    if len(weight.size()) > 2:
                        penalty_weight = coeff_alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*weight.data + \
                                            coeff_beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*weight_old.data
                    else:
                        penalty_weight = coeff_alpha.unsqueeze(-1)*weight.data + coeff_beta.unsqueeze(-1)*weight_old.data
                    penalty_bias = coeff_alpha*bias.data + coeff_beta*bias_old.data

                diff_weight = (sparse_weight + penalty_weight) - weight.data
                diff_bias = sparse_bias + penalty_bias - bias.data
                

                weight.data = sparse_weight + penalty_weight
                bias.data = sparse_bias + penalty_bias

        return

    def criterion(self,t,output,targets):
        return self.ce(output,targets)
    
    #OGD
    def _get_new_ogd_basis(self, train_loader, last=False):
        return self._get_neural_tangents(train_loader,
                                         gpu=self.config.gpu,
                                         optimizer=self.optimizer,
                                         model=self.model, last=last)

    #collects 100 sample gradients from all previous tasks to represent basis vectors of new 100-d space. Then updates the new gradient in a direction orthogonal to all of these vectors (watch 3blue1brown video on basis vectors)
    def _get_neural_tangents(self, train_loader, gpu, optimizer, model, last):
        new_basis = []

        for i, (inputs, targets, tasks) in tqdm(enumerate(train_loader),
                                                desc="get neural tangents",
                                                total=len(train_loader.dataset)):
            # if gpu:
            inputs = self.to_device(inputs)
            targets = self.to_device(targets)

            out = self.forward(x=inputs, task=(tasks))
            label = targets.item()
            pred = out[0, label]

            optimizer.zero_grad()
            pred.backward()

            grad_vec = parameters_to_grad_vector(self.get_params_dict(last=last))
            new_basis.append(grad_vec)
        new_basis_tensor = torch.stack(new_basis).T
        return new_basis_tensor
    
    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_closure: Optional[Callable] = None,
                       second_order_closure=None, using_native_amp=None):
        #super().optimizer_step(epoch=current_epoch, batch_idx=batch_idx, optimizer=optimizer, optimizer_closure=optimizer_closure)
        #task_key = str(self.task_id)

        # for param in self.get_params_dict(last=False):
        #     print(type(param))
        # print("parameters", self.get_params_dict(last=False))
        # for param in self.get_params_dict(last=False):
        #     print(param.view(-1))
        cur_param = parameters_to_vector(parameters=self.model.parameters())
        #grad_vec = parameters_to_grad_vector(parameters=self.get_params_dict(last=False))
        new_vec = project_vec(model=self.model, omega=self.omega, proj_basis=self.ogd_basis, gpu=self.config.gpu) #previously new_grad_vec
        #cur_param -= self.config.lr * new_grad_vec
        grad_vector_to_parameters(new_vec, self.get_params_dict(last=False))

        if self.config.is_split :
            # Update the parameters of the last layer without projection, when there are multiple heads)
            cur_param = parameters_to_vector(self.get_params_dict(last=True))
            grad_vec = parameters_to_grad_vector(self.get_params_dict(last=True))
            cur_param -= self.config.lr * grad_vec
            vector_to_parameters(cur_param, self.get_params_dict(last=True))
        
    def _update_mem(self, data_train_loader, val_loader=None):
        # 2.Randomly decide the images to stay in the memory
        self.task_count += 1

        # (a) Decide the number of samples for being saved
        num_sample_per_task = self.config.memory_size

        # (c) Randomly choose some samples from new task and save them to the memory
        self.task_memory[self.task_count] = Memory()  # Initialize the memory slot
        randind = torch.randperm(len(data_train_loader.dataset))[:num_sample_per_task]  # randomly sample some data
        for ind in randind:  # save it to the memory
            self.task_memory[self.task_count].append(data_train_loader.dataset[ind])

        ####################################### Grads MEM ###########################

        # (e) Get the new non-orthonormal gradients basis
        if self.config.ogd:
            ogd_train_loader = torch.utils.data.DataLoader(self.task_memory[self.task_count], batch_size=1,
                                                           shuffle=False, num_workers=1)
        elif self.config.ogd_plus:
            all_task_memory = []
            for task_id, mem in self.task_memory.items():
                all_task_memory.extend(mem)
            # random.shuffle(all_task_memory)
            # ogd_memory_list = all_task_memory[:num_sample_per_task]
            ogd_memory_list = all_task_memory
            ogd_memory = Memory()
            for obs in ogd_memory_list:
                ogd_memory.append(obs)
            ogd_train_loader = torch.utils.data.DataLoader(ogd_memory, batch_size=1, shuffle=False, num_workers=1)
        # Non orthonormalised basis
        new_basis_tensor = self._get_new_ogd_basis(ogd_train_loader)
        print(f"new_basis_tensor shape {new_basis_tensor.shape}")

        # (f) Ortonormalise the whole memorized basis
        if self.config.is_split:
            n_params = count_parameter(self.model.linear)
        else:
            n_params = count_parameter(self.model)
        self.ogd_basis = torch.empty(n_params, 0)
        self.ogd_basis = self.to_device(self.ogd_basis)

        if self.config.ogd:
            for t, mem in self.task_grad_memory.items():
                # Concatenate all data in each task
                task_ogd_basis_tensor = mem.get_tensor()
                task_ogd_basis_tensor = self.to_device(task_ogd_basis_tensor)
                self.ogd_basis = torch.cat([self.ogd_basis, task_ogd_basis_tensor], axis=1)
            self.ogd_basis = torch.cat([self.ogd_basis, new_basis_tensor], axis=1)
        elif self.config.ogd_plus :
            if self.config.pca :
                for t, mem in self.task_grad_memory.items():
                    # Concatenate all data in each task
                    task_ogd_basis_tensor = mem.get_tensor()
                    task_ogd_basis_tensor = self.to_device(task_ogd_basis_tensor)

                    # task_ogd_basis_tensor.shape
                    # Out[3]: torch.Size([330762, 50])
                    start_idx = t * num_sample_per_task
                    end_idx = (t + 1) * num_sample_per_task
                    before_pca_tensor = torch.cat([task_ogd_basis_tensor, new_basis_tensor[:, start_idx:end_idx]], axis=1)
                    u, s, v = torch.svd(before_pca_tensor)

                    # u.shape
                    # Out[8]: torch.Size([330762, 150]) -> col size should be 2 * num_sample_per_task

                    after_pca_tensor = u[:, :num_sample_per_task]

                    # after_pca_tensor.shape
                    # Out[13]: torch.Size([330762, 50])

                    self.ogd_basis = torch.cat([self.ogd_basis, after_pca_tensor], axis=1)
            #   self.ogd_basis.shape should be T * num_sample_per_task

            else :
                self.ogd_basis = new_basis_tensor

        # TODO : Check if start_idx is correct :)
        start_idx = (self.task_count - 1) * num_sample_per_task
        # print(f"the start idx of orthonormalisation if {start_idx}")
        self.ogd_basis = orthonormalize(self.ogd_basis, gpu=self.config.gpu, normalize=True)

        # (g) Store in the new basis
        ptr = 0
        for t, mem in self.task_memory.items():
            task_mem_size = len(mem)

            idxs_list = [i + ptr for i in range(task_mem_size)]
            if self.config.gpu:
                self.ogd_basis_ids[t] = torch.LongTensor(idxs_list).cuda()
            else:
                self.ogd_basis_ids[t] = torch.LongTensor(idxs_list)

            self.task_grad_memory[t] = Memory()  # Initialize the memory slot
            for ind in range(task_mem_size):  # save it to the memory
                self.task_grad_memory[t].append(self.ogd_basis[:, ptr])
                ptr += 1
        print(f"Used memory {ptr} / {self.config.memory_size}")

        if self.config.ogd or self.config.ogd_plus :
            loader = torch.utils.data.DataLoader(self.task_memory[self.task_count],
                                                                            batch_size=self.config.batch_size,
                                                                            shuffle=True,
                                                                            num_workers=2)
            self.mem_loaders.append(loader)

    def update_ogd_basis(self, data_train_loader):
        if self.config.gpu :
            device = torch.device("cuda")
            self.model.to(device)
        print(f"\nself.model.device update_ogd_basis {next(self.model.parameters()).device}")
        if self.config.ogd or self.config.ogd_plus:
            self._update_mem(data_train_loader)
            
    def get_params_dict(self, last, task_key=None):
        parameters = self.model.named_parameters()
        # for param in parameters:
        #     print(param.view(-1))
        return parameters
        #return [param for param in parameters if type(param) != None]
        # self.model.get_parameters()
        # if self.config.is_split :
        #     if last:
        #         return self.model.last[task_key].parameters()
        #     else:
        #         return self.model.get_parameters()
        # else:
        #     return self.model.parameters()
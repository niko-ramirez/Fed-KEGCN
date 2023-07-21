import tensorflow as tf
import random
import copy
from utils import *

class Server:
    def __init__(self, globalModel, device, logging, data, eval_func, writer = None) -> None:
        # log setting
        self.logging = logging
        self.writer = writer
        self.get_eval = eval_func
        self.test_data = data["test"]
        self.valid = data["valid"]
        self.y = data["y"]
        self.support = data["support"]
        self.placeholders = globalModel.placeholders
        self.best_acc = 0.0

        # server setting
        self.device = device
        self.Epoch = -1                        # record the FL round by self
        self.round = random.randint(0, 1e8)    # record the mask round of FL this round
        self.local_state_dict = []
        self.val_acc = 0
        self.model = globalModel
        self.global_state_dict = {}
        config =tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.33
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self.sess.run(tf.global_variables_initializer())
        self.global_grads = []
        # for var_name, value in self.model.vars:
        #     if var_name.has("ent_weights"):
        #         if var_name.has("autorelgraphconvolution_1_vars"):
        #             self.global_state_dict["layer_one_basis"] = self.sess.run(value)
        #         elif var_name.has("autorelgraphconvolution_2_vars"):
        #             self.global_state_dict["layer_two_basis"] = self.sess.run(value)

        # loss function
        # self.loss_fn = tf.nn.softmax_cross_entropy_with_logits_v2()

    def aggregate(self):
        clientNum = len(self.local_state_dict)
        if clientNum == 0:
            return 
        total_grad_per_client = 2
        self.val_acc /= clientNum
        self.Epoch += 1
        #self.sess.run(tf.global_variables_initializer())
        # self.writer.add_scalar(f"val/acc/avg", self.val_acc, self.Epoch)

        # aggregate all parameter        
        # for layer_name in self.local_state_dict[0].keys():
        #     self.global_state_dict[layer_name] = tf.zeros_like(self.global_state_dict[layer_name])
            
        #     for localParame in self.local_state_dict:
        #         self.global_state_dict[layer_name].add_(localParame[layer_name])
            
        #     self.global_state_dict[layer_name].div_(clientNum)

        layer_sum = {}
        # autorelgraphconvolution_4_vars/ent_weights
        # for grad_vars, num_nodes in self.local_state_dict:
        #     for grad, var_name in grad_vars:
        #         parsed_name = var_name.split("_")
        #         key = int(parsed_name[3]) % total_grad_per_client
        #         if key not in layer_sum:
        #             layer_sum[key] = tf.zeros_like(grad, dtype = tf.float32)
        #         # normalized_grads = tf.truediv(grad, float(num_nodes))
        #         layer_sum[key] += tf.convert_to_tensor(grad) 

        for grad_vars, num_nodes in self.local_state_dict:
            for grad, var_name in grad_vars:
                parsed_name = var_name.split("_")
                key = int(parsed_name[3]) % total_grad_per_client
                if key not in layer_sum:
                    layer_sum[key] = np.zeros(grad.shape, dtype = np.float32)
                # normalized_grads = tf.truediv(grad, float(num_nodes))
                layer_sum[key] += grad

        # for client in self.local_state_dict:
        #     for client_key, basis_value in client.items():
        #         layer_sum[client_key[1]] += basis_value
                # if client_key[1] == "layer_one_basis":
                #     layer_one_sum += basis_value
                # elif client_key[1] == "layer_two_basis":
                #     layer_two_sum += basis_value
        
        # self.model.opt_apply(new_grad)

        for layer, sum_of_layer in layer_sum.items():
            layer_sum[layer] = sum_of_layer / float(clientNum)
        # layer_one_sum = tf.truediv(layer_one_sum, float(clientNum))
        # layer_two_sum = tf.truediv(layer_two_sum, float(clientNum))

        # self.global_state_dict["layer_one_basis"] = layer_one_sum
        # self.global_state_dict["layer_two_basis"] = layer_two_sum

        to_apply_to_server = [0] * total_grad_per_client
        for key, grad in layer_sum.items():
            if key == 0:
                key_to_add = str(total_grad_per_client)
            else:
                key_to_add = str(key)
            var_name_string = 'autorgcn_align/autorelgraphconvolution_' + key_to_add + '_vars/ent_weights_0:0'
            var_key = self.model.vars[var_name_string]
            to_apply_to_server[int(key_to_add) - 1] = (grad, var_key)
        
        feed_dict = construct_feed_dict(1.0, self.support, self.placeholders)

        new_grad= []
        for grad, variable in self.model.opt_grads:
            if variable.name.endswith('ent_weights_0:0'):
                parsed_name = variable.name.split("_")
                key = int(parsed_name[2]) % total_grad_per_client
                if key == 0:
                    key = total_grad_per_client
                temp_grad = tf.convert_to_tensor(to_apply_to_server[key-1][0])
                new_grad.append((temp_grad, variable))
            else:
                if grad is not None:
                    # print(grad)
                    new_grad.append((tf.zeros_like(grad), variable))
                    # new_grad.append((grad, variable))
                else:
                    new_grad.append((grad, variable))


        self.global_grads = [i for i, _ in to_apply_to_server]
            # if variable.name == 'autorgcn_align/autorelgraphconvolution_1_vars/ent_weights_0:0':
            #     new_grad.append((to_apply_to_server[0][0], variable))
            # elif variable.name == 'autorgcn_align/autorelgraphconvolution_2_vars/ent_weights_0:0':
            #     new_grad.append((to_apply_to_server[1][0], variable))
            # elif grad is not None:
            #     new_grad.append((tf.zeros_like(grad), variable))
            # else:
            #     new_grad.append((grad, variable))
    
        outputs = self.sess.run([self.model.opt_apply(new_grad), self.model.loss], feed_dict=feed_dict)

        # for _, variable in to_apply_to_server:
        #     self.global_state_dict[variable.name] = variable.eval(session=self.sess)


        
        self.eval_server()
        # print("SERVER AGGREGATE GLOBAL_DICT: ", self.global_state_dict)
        self.local_state_dict.clear()
        # self.model.load_state_dict(self.global_state_dict)
        self.round = random.randint(0, 1e8) 

    def sendParame(self):
        return self.round, self.global_state_dict

    def sendGrads(self):
        return self.round, self.global_grads


    def getParame(self, round, localParams, val_acc):
        self.local_state_dict.append(localParams)
        # self.val_acc += val_acc

    def eval_server(self):
        feed_dict = construct_feed_dict(1.0, self.support, self.placeholders)
        output_embeddings = self.sess.run(self.model.outputs, feed_dict=feed_dict)
        acc, temp = self.get_eval(output_embeddings[0], self.test_data, self.y, self.logging)
        if acc > self.best_acc:
            self.best_acc = acc
        self.logging.info("SERVER Test Accuracy: %.3f" % (acc * 100))

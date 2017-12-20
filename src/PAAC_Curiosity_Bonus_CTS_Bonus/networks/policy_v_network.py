from networks.networks import *
from utilities.icm import *


class PolicyVNetwork(Network):

    def __init__(self, conf):
        """ Set up remaining layers, objective and loss functions, gradient
        compute and apply ops, network parameter synchronization ops, and
        summary ops. """

        super(PolicyVNetwork, self).__init__(conf)

        self.entropy_regularisation_strength = conf['entropy_regularisation_strength']

        with tf.device(conf['device']):
            with tf.name_scope(self.name):

                self.critic_target_ph = tf.placeholder(
                    "float32", [None], name='target')
                self.adv_actor_ph = tf.placeholder("float", [None], name='advantage')

                # Final actor layer
                layer_name = 'actor_output'
                _, _, self.output_layer_pi = softmax(layer_name, self.output, self.num_actions)
                # Final critic layer
                _, _, self.output_layer_v = fc('critic_output', self.output, 1, activation="linear")

                # Avoiding log(0) by adding a very small quantity (1e-30) to output.
                self.log_output_layer_pi = tf.log(tf.add(self.output_layer_pi, tf.constant(1e-30)),
                                                  name=layer_name + '_log_policy')

                # Entropy: sum_a (-p_a ln p_a)
                self.output_layer_entropy = tf.reduce_sum(tf.multiply(
                    tf.constant(-1.0),
                    tf.multiply(self.output_layer_pi, self.log_output_layer_pi)), reduction_indices=1)

                self.output_layer_v = tf.reshape(self.output_layer_v, [-1])

                # Advantage critic
                self.critic_loss = tf.subtract(self.critic_target_ph, self.output_layer_v)

                log_output_selected_action = tf.reduce_sum(
                    tf.multiply(self.log_output_layer_pi, self.selected_action_ph),
                    reduction_indices=1)

                self.actor_objective_advantage_term = tf.multiply(log_output_selected_action, self.adv_actor_ph)
                self.actor_objective_entropy_term = tf.multiply(self.entropy_regularisation_strength, self.output_layer_entropy)

                self.actor_objective_mean = tf.reduce_mean(tf.multiply(tf.constant(-1.0), tf.add(self.actor_objective_advantage_term, self.actor_objective_entropy_term)), name='mean_actor_objective')

                self.critic_loss_mean = tf.reduce_mean(tf.scalar_mul(0.25, tf.pow(self.critic_loss, 2)), name='mean_critic_loss')

                # Loss scaling is used because the learning rate was initially runed tuned to be used with
                # max_local_steps = 5 and summing over timesteps, which is now replaced with the mean.
                self.loss = tf.scalar_mul(self.loss_scaling, self.actor_objective_mean + self.critic_loss_mean)
                self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
                

class StateActionPredictor(object):
    def __init__(self,  conf, ob_space = (42,42,4) , designHead='universe'):
        # input: s1,s2: : [None, h, w, ch] (usually ch=1 or 4)
        # asample: 1-hot encoding of sampled action from policy: [None, ac_space]
        
        self.name = conf['name']
        ac_space = conf['num_actions']
        with tf.device(conf['device']):
            with tf.variable_scope(self.name):
                input_shape = [None] + list(ob_space)
                self.s1 = self.phi1 = tf.placeholder(tf.float32, input_shape)
                self.s2 = self.phi2 = tf.placeholder(tf.float32, input_shape)
                self.asample = asample = tf.placeholder(tf.float32, [None, ac_space])

                # feature encoding: phi1, phi2: [None, LEN]
                size = 256
                if designHead == 'nips':
                    phi1 = nipsHead(phi1)
                    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                        phi2 = nipsHead(phi2)
                elif designHead == 'nature':
                    phi1 = natureHead(phi1)
                    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                        phi2 = natureHead(phi2)
                elif designHead == 'doom':
                    phi1 = doomHead(phi1)
                    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                        phi2 = doomHead(phi2)
                elif 'tile' in designHead:
                    phi1 = universeHead(phi1, nConvs=2)
                    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                        phi2 = universeHead(phi2, nConvs=2)
                else:
                    self.phi1 = universeHead(self.phi1)
                    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                        self.phi2 = universeHead(self.phi2)

                # inverse model: g(phi1,phi2) -> a_inv: [None, ac_space]
                g = tf.concat([self.phi1, self.phi2],1)
                g = tf.nn.relu(linear(g, size, "g1", normalized_columns_initializer(0.01)))
                aindex = tf.argmax(asample, axis=1)  # aindex: [batch_size,]
                logits = linear(g, ac_space, "glast", normalized_columns_initializer(0.01))
                self.invloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                logits = logits, labels = aindex), name="invloss")
                self.ainvprobs = tf.nn.softmax(logits, dim=-1)

                # forward model: f(phi1,asample) -> phi2
                # Note: no backprop to asample of policy: it is treated as fixed for predictor training
                self.f = tf.concat([self.phi1, asample],1)
                self.f = tf.nn.relu(linear(self.f, size, "f1", normalized_columns_initializer(0.01)))
                self.f = linear(self.f, self.phi1.get_shape()[1].value, "flast", normalized_columns_initializer(0.01))
                self.forwardloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(self.f, self.phi2)), name='forwardloss') * 288
                self.bonus = 0.5 * tf.reduce_mean(tf.square(tf.subtract(self.f, self.phi2)),axis = 1, name='bonus') * 288

                self.predloss = constants['PREDICTION_LR_SCALE'] * (self.invloss * (1-constants['FORWARD_LOSS_WT']) +
                                                                    self.forwardloss * constants['FORWARD_LOSS_WT'])
                # variable list
                self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

    def pred_act(self, s1, s2):
        '''
        returns action probability distribution predicted by inverse model
            input: s1,s2: [h, w, ch]
            output: ainvprobs: [ac_space]
        '''
        sess = tf.get_default_session()
        return sess.run(self.ainvprobs, {self.s1: [s1], self.s2: [s2]})[0, :]

    def pred_bonus(self, session,s1, s2, asample):
        '''
        returns bonus predicted by forward model
            input: s1,s2: [h, w, ch], asample: [ac_space] 1-hot encoding
            output: scalar bonus
        '''
        error = session.run(self.bonus,
            {self.s1: s1, self.s2: s2, self.asample: asample})
        error = error * constants['PREDICTION_BETA']
        return error
    
class NIPSPolicyVNetwork(PolicyVNetwork, NIPSNetwork):
    pass


class NaturePolicyVNetwork(PolicyVNetwork, NatureNetwork):
    pass

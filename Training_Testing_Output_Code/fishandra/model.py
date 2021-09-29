import logging
import os
import numpy as np
from keras import models, layers
import tensorflow as tf

logger = logging.getLogger(__name__)

DEFAULT_ATT_ARCH = {'pairwise': [128]*3, 'attention': [128]*3} # attention network
DEFAULT_INT_ARCH = {'pairwise': [128]*3, 'integration': [128]} # interaction
DEFAULT_MLP_ARCH = {'integration': [128]*4} # multi layer perceptron

### Factories


def load_model(config):
    if config.get('attention_variables', False):
        if config['model'] != 'attention_general':
            # TODO: In the future, calling attention with attention_variables
            # might work as intended
            print("Attention variables ", config['attention_variables'])
            raise Exception('attention_variables have no effect in model {}'.
                            format(config['model']))

    if config['model'] == 'attention':
        if config.get('attention_input_size', -1) < 0:
            # Currently unused case
            model = SelfAttentionNetwork(config)
        else:
            #TODO: This is the special case of the paper, but in general it should be
            # removed and defined via attention_general
            model = AttentionNetwork(config)

    elif config['model'] == 'logit_sum':
        config.update({'logit_sum': True})
        model = SelfAttentionNetwork(config)
    elif config['model'] == 'interaction':
        model = InteractionNetwork(config)
    elif config['model'] == 'fc':
        model = FullyConnectedNetwork(config)
    elif config['model'] == 'attention_general':
        attention_variables = set(config.get('attention_variables', set()))
        attention_variables -= set([None, '']) # Removing common placeholders
        if not attention_variables and not config.get('topological_attention', False):
            logger.warning('attention_general, but empty attention_variables. Making logit_sum = True.')
            config.update({'logit_sum': True})
            model = SelfAttentionNetwork(config)
        else:
            model = SelfAttentionNetworkGeneral(config)
    else:
        raise Exception
    return model


def load_model_from_file(foldername, updated_args = None):
    config = np.load(os.path.join(foldername, 'model_config.npy'), allow_pickle=True).item()
    if updated_args is not None:
        config.update(updated_args)
    model = load_model(config)
    model.load_weights(os.path.join(foldername, 'model.h5'))
    return model


### Homemade keras layers


sum_dim1 = layers.Lambda(lambda xin: tf.reduce_sum(xin, axis=1), name='sum_dim_1')

def diff_last(tensor):
    def f(x):
        return x[...,0] - x[...,1]
    return layers.Lambda(f, name='diff_last')(tensor)

def flip_tensor(tensor, num_pairs=3):
    """This flips the x variables assuming left-right simmetry.
    """
    def f(x):
        tensor_flipped = tf.tensordot(x, tf.diag(
            [-1.0, 1.0] * num_pairs), [[-1], [1]])
        return tensor_flipped
    return layers.Lambda(f, name='flip_tensor')(tensor)

def restrict_variables_for_attention(tensor, add_label=False):
    def restrict(social_and_focal):
        social_and_focal_sq = tf.square(social_and_focal)
        # distance sq, vel sq, acc sq, etc
        attention_variables1 = tf.sqrt(
            social_and_focal_sq[..., ::2] + social_and_focal_sq[..., 1::2])
        # speed, focal speed, position
        attention_variables_list = [attention_variables1[..., 1], #speed
                                        attention_variables1[..., 3], #focal speed
                                        tf.abs(social_and_focal[..., 0]), #abs(x)
                                        social_and_focal[..., 1]] #y
        if add_label:
            print("Considering topological indices")
            shape = tf.shape(social_and_focal)
            indices = tf.cast(tf.expand_dims(tf.range(shape[1]), axis=0),
                              tf.float32) / tf.cast(shape[1], tf.float32)
            indices_tensor = tf.tile(indices, (shape[0], 1))
            attention_variables_list.append(indices_tensor)

        attention_variables = tf.stack(attention_variables_list, -1)
        return attention_variables
    return layers.Lambda(restrict, name='restrict_variables')(tensor)

def restrict_variables_for_attention_general(tensor, variable_set, add_label=False):
    print("Variable set: ", variable_set)
    def restrict(social_and_focal):
        social_and_focal_sq = tf.square(social_and_focal)
        # distance sq, vel sq, acc sq, etc
        attention_variables1 = tf.sqrt(
            social_and_focal_sq[..., ::2] + social_and_focal_sq[..., 1::2])
        # speed, focal speed, position
        focal_variables = []
        if 'fv' in variable_set:
            print("Adding focal speed to attention variables")
            focal_variables.append(attention_variables1[..., 3]) #  focal speed

        nb_variables = []
        if 'nbL' in variable_set:
            print("Adding nb position to attention variables")
            nb_variables += [tf.abs(social_and_focal[..., 0]), social_and_focal[..., 1]]
        elif 'nbl' in variable_set:
            print("Adding nb distance to attention variables")
            nb_variables += [attention_variables1[..., 0]]
        if 'nbV' in variable_set:
            print("Adding nb velocity to attention variables")
            nb_variables += [tf.abs(social_and_focal[..., 2]), social_and_focal[..., 3]]
        elif 'nbv' in variable_set:
            print("Adding nb spped to attention variables")
            nb_variables += [attention_variables1[...,1]]

        if add_label:
            print("Considering topological indices")
            shape = tf.shape(social_and_focal)
            indices = tf.cast(tf.expand_dims(tf.range(shape[1]), axis=0),
                              tf.float32) / tf.cast(shape[1], tf.float32)
            indices_tensor = tf.tile(indices, (shape[0], 1))
            nb_variables.append(indices_tensor)

        attention_variables = tf.stack(focal_variables + nb_variables, -1)
        return attention_variables
    return layers.Lambda(restrict, name='restrict_variables')(tensor)

### Classes

class AbstractNetwork:
    """ This is the more general class to create models.
    """
    def __init__(self, args):
        """__init__

        :param args: Dictionary with variables needed to build the model. Typically
        'num_neighbours', 'future_steps', 'use_acceleration', 'sigma', 'blind',
        'model', 'integration_arch', 'pairwise_arch', 'attention_arch'. See the
        script fastrain for an example.
        """
        self.args = args

    def save(self, foldername=None):
        """Saves the model in a .h5 and te variables to build it in a .npy
        """
        if foldername is None:
            foldername = self.foldername
        self.model.save(os.path.join(foldername, 'model.h5'))
        np.save(os.path.join(foldername, 'model_config.npy'), self.args)

    def _social_and_focal(self, social_context, suffix_name):
        """ Separates the data in the social_context array into focal and social
        and it concatenates it putting the social first. It also removes the position
        for the focal as it is always (0, 0).
        """
        num_neighbours = self.num_neighbours
        def function(x):
            focal, social = tf.split(x, [1, num_neighbours], axis=1)
            # Here we are removing the position of the focals as it is 0
            # We tile it becuase for every neigbhour we also need the focal variables
            focal_tiled = tf.tile(focal[..., 2:], [1, num_neighbours, 1])
            return tf.concat([social, focal_tiled], axis=2)
        return layers.Lambda(function, name='social_and_focal_' + suffix_name)(social_context)

    def _apply_layers(self, x, name):
        """ Apply the layers in the list self._layers[name] iteratively in a
        feedforward way. Note that self._layers is a dictionary. This is only
        valid for the class AttentionNetwork. Maybe there is some redundacy as
        other classes use similar functions instead of calling this one.
        """
        for layer in self._layers[name]:
            x = layer(x)
        return x

    def load_weights(self, *args, **kwargs):
        return self.model.load_weights(*args, **kwargs)
    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)
    def summary(self, *args, **kwargs):
        return self.model.summary(*args, **kwargs)
    def compile(self, *args, **kwargs):
        return self.model.compile(*args, **kwargs)
    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)
    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)
    def save_weights(self, *args, **kwargs):
        return self.model.save_weights(self, *args, **kwargs)


class FullyConnectedNetwork(AbstractNetwork):
    def __init__(self, args):
        super().__init__(args) # inherits from AbstractNetwork
        self.num_neighbours = args["num_neighbours"]
        self._arch = {key: args.get(key + '_arch', DEFAULT_MLP_ARCH[key])
                      for key in ['integration']}
        self._input = layers.Input(shape=(self.num_neighbours+1, 6)) # 6 = 2XY position 2XY velocity 2XY acceleration
        input_reshaped = layers.Flatten()(self._input)

        self._build_network()

        self.logits = self._network(input_reshaped)
        self.prob = layers.Softmax()(self.logits)
        self.model = models.Model(inputs=self._input, outputs=self.prob)

    def _network(self, inputs):
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x

    def _build_network(self):
        interaction = [layers.Dense(n, activation='relu',
                                    name="integration" + str(i))
                       for i, n in enumerate(self._arch["integration"])]
        readout = [layers.Dense(2)]
        self._layers = interaction + readout


class InteractionNetwork(AbstractNetwork):
    def __init__(self, args):
        super().__init__(args) # inherits from AbstractNetwork
        self.num_neighbours = args["num_neighbours"]
        self._arch = {key: args.get(key + '_arch', DEFAULT_INT_ARCH[key])
                      for key in ['integration', 'pairwise']}
        # set input to the network. Note that in this case this is the input
        # to both subnetworks. One can use the variable "blind" when loading
        # the dataset to not consider certain variables (e.g. neighbours accelerations)
        self._input = layers.Input(shape=(self.num_neighbours+1, 6)) # HARCODED 6: 2xpositions, 2xvelocities, 2xaccelerations
        input_reshaped = self._social_and_focal(self._input, 'reshaped')
        input_flipped = flip_tensor(self._input) # To antysymmetrize the network
        input_flipped_reshaped = self._social_and_focal(input_flipped, 'flipped_reshaped')

        self._build_network()

        self.logits = self._a_interaction(input_reshaped, input_flipped_reshaped)
        self.prob = layers.Softmax()(self.logits)
        self.model = models.Model(inputs=self._input, outputs=self.prob)

    def _interaction(self, inputs):
        """Applies all the layers in the list of layers 'inputs' in a feedforward
        manner.
        """
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x

    def _a_interaction(self, x, xf):
        """This makes the network antisymmetric.
        """
        output = self._interaction(x)
        output_flipped = self._interaction(xf)
        return layers.Subtract()([output, output_flipped])

    def _build_network(self):
        """Generates a list of layers: pairwise interactions subnetworks and
        general integration subnetwork. Note that the sharing of the weights
        between pairwise subnetworks is implicitly done by Keras.
        """
        interaction = []
        for i,n in enumerate(self._arch["pairwise"]):
            interaction.append(layers.Dense(n, activation='relu',
                                            name = "interaction" + str(i)))
        # Integration has last layer repeated but without ReLU
        # then a sum over neighbours (sum_dim1 function defined above)
        # and finally a ReLU
        integration1 = [layers.Dense(self._arch["pairwise"][-1]),
                        sum_dim1, layers.ReLU()]
        # Then, some extra layers
        integration2 = [layers.Dense(n, activation='relu')
                              for n in self._arch["integration"]]
        readout = [layers.Dense(2)]
        self._layers = interaction + integration1 + integration2 + readout


class AttentionNetwork(AbstractNetwork):
    def __init__(self, args):
        super().__init__(args)
        self.num_neighbours = args["num_neighbours"]
        self.history_steps = args.get("history_steps", -1)
        self.size_attention_input = args.get("size_attention_input", -1)
        self._arch = {key: args.get(key + '_arch', DEFAULT_ATT_ARCH[key])
                      for key in ['attention', 'pairwise']}
        self._logit_sum = args.get("logit_sum", False)

        logger.debug("AttentionNetwork init with args {}".format(args))
        logger.debug("AttentionNetwork init with arch {}".format(self._arch))

        self._input = self._build_input()
        # This method is also coded in the class SelfAttentionNetwork so it will be called from there
        # This method is the one that needs to be modified if ones want to change the
        # inputs to the attention subnetwork
        self._input_attention = self._build_attention_input()
        self._layers = self._build_network()

        # attention_logits are the raw logits.
        # attention_layer is normalized (softmax) to all the neigoburs
        # The reason to output both is to later plot either the logits or the
        # normalized weights of the attention
        self.attention_logits, self.attention_layer = self._attention_subnetwork(self._input_attention)
        self.pairwise = self._a_pairwise_subnetwork(self._input["reshaped"],
                                                    self._input["flipped_reshaped"])
        # This is because the output of the pair-wise is two elements. Classification encoded as one-hot
        # As we only want one value to multiply with the logit of the attention we do the difference
        # between both logits
        self.pairwise_diff = diff_last(self.pairwise)

        self.logits = layers.Dot([1, 1])([self.pairwise, self.attention_layer])
        self.prob = layers.Softmax()(self.logits)
        self._build_models()

    def _build_models(self):
        """This function is also coded in the class SelfAttentionNetwork so
        it will be called from there most of the times. Mainly created to be
        called from the plotter functions and to generate outputs.
        """
        self.model = models.Model(inputs=[self._input[""], self._input_attention],
                                  outputs=self.prob)
        self.pairwise_model = models.Model(inputs=self._input[""],
                                           outputs=self.pairwise_diff)
        self.attention_model = models.Model(inputs=self._input_attention,
                                            outputs=self.attention_logits)
        self.attention_layer = models.Model(inputs=self._input_attention,
                                            outputs=self.attention_layer)


    def _build_network(self):
        """Create a dictinary with two list of layers, the interaction subnetwork
        and the attention subnetwork.
        """
        attention = [layers.Dense(n, activation='relu',
                                  name = 'attention{}'.format(i))
                     for i, n in enumerate(self._arch['attention'])]
        attention_out = [layers.Dense(1, name = 'attention_out')]

        interaction = [layers.Dense(n, activation='relu',
                                    name = 'pairwise{}'.format(i))
                       for i, n in enumerate(self._arch['pairwise'])]
        interaction_out = [layers.Dense(2, name = 'pairwise_out')]

        return {"attention": attention + attention_out,
                "pairwise": interaction + interaction_out}

    def _build_input(self):
        """Creates a dictionary with the different types of input arrays for the subnetworks.
        """
        input_dict = {"": layers.Input(shape=(self.num_neighbours+1, 6))}
        input_dict["reshaped"] = self._social_and_focal(input_dict[""], 'reshaped')
        input_dict["flipped"] = flip_tensor(input_dict[""])
        input_dict["flipped_reshaped"] = self._social_and_focal(input_dict["flipped"], 'flipped_reshaped')
        return input_dict

    def _build_attention_input(self):
        """ Note that a function with the same name exists in the classes
        SelfAttentionNetwork and SelfAttentionNetworkAcc. So it will call the
        other ones if the model is created from those.
        """
        assert self.size_attention_input > 0
        return layers.Input(shape=(self.num_neighbours,
                                   self.size_attention_input))

    def _a_pairwise_subnetwork(self, x, xf):
        output = self._apply_layers(x, "pairwise")
        output_flipped = self._apply_layers(xf, "pairwise")
        return layers.Subtract()([output, output_flipped])

    def _attention_subnetwork(self, y):
        y = self._apply_layers(y, "attention")
        if self._logit_sum:
            print("Logit sum, instead of attention")
            y = layers.Lambda(lambda x: x * 0.0, name='zeroes_attention')(y)
        attention_output = layers.Flatten()(y) #To get rid of last singleton dimension
        return attention_output, layers.Softmax()(attention_output)


class SelfAttentionNetwork(AttentionNetwork):
    """ Default version used in the paper
    """
    def _build_attention_input(self):
        return restrict_variables_for_attention(self._input["reshaped"],
                                self.args.get('topological_attention', False))

    def _build_models(self):
        self.pairwise_model = models.Model(inputs=self._input[""],
                                           outputs=self.pairwise_diff)
        self.model = models.Model(inputs=self._input[""], outputs=self.prob)
        self.attention_model = models.Model(inputs=self._input[""],
                                            outputs=self.attention_logits)
        self.attention_layer = models.Model(inputs=self._input[""],
                                            outputs=self.attention_layer)

class SelfAttentionNetworkHistory(SelfAttentionNetwork):
    """ Uses history in both aggregation and interaction (instead of variables)
    """
    def _build_attention_input(self):
        return restrict_variables_for_attention(self._input["reshaped"][0],
                                self.args.get('topological_attention', False))

    def _social_and_focal_history(self, history, suffix_name):
        num_neighbours = self.num_neighbours
        def function(x):
            focal, social = tf.split(x, [1, num_neighbours], axis=1)
            # We tile it becuase for every nb we also need the focal variables
            focal_tiled = tf.tile(focal[...], [1, num_neighbours, 1, 1])
            return tf.concat([social, focal_tiled], axis=3)
        return layers.Lambda(function,
                             name='social_and_focal_history_' + suffix_name)(
                                 history)

    def _build_input(self):
        """Supplements the dictionary with the history input
        """
        input_dict1 = super()._build_input()

        input_dict2 = {'': layers.Input(shape=(self.num_neighbours+1,
                                               self.history_steps+1, 2))}
        input_dict2["reshaped"] = self._social_and_focal_history(input_dict2[""], 'reshaped')
        input_dict2["flipped"] = flip_tensor(input_dict2[""], num_pairs=1)
        input_dict2["flipped_reshaped"] = self._social_and_focal_history(
            input_dict2["flipped"], 'flipped_reshaped')
        input_dict = {key: (input_dict1[key], input_dict2[key])
                      for key in ['', 'reshaped', 'flipped', 'flipped_reshaped']}

        return input_dict

    def _a_pairwise_subnetwork(self, x, xf):
        x_h_flat = layers.Reshape((self.num_neighbours, -1))(x[1])
        xf_h_flat = layers.Reshape((self.num_neighbours, -1))(xf[1])
        return super()._a_pairwise_subnetwork(x_h_flat, xf_h_flat)


class SelfAttentionNetworkGeneral(SelfAttentionNetwork):
    def _build_attention_input(self):
        return restrict_variables_for_attention_general(
                                self._input["reshaped"],
                                self.args['attention_variables'],
                                self.args.get('topological_attention', False))

#
#
#class SelfAttentionNetworkVel(SelfAttentionNetwork):
#    def _build_attention_input(self):
#        return restrict_variables_for_attention_general(self._input["reshaped"], 'vel')
#
#class SelfAttentionNetworkAcc(SelfAttentionNetwork):
#    def _build_attention_input(self):
#        return restrict_variables_for_attention_general(self._input["reshaped"], 'acc')
#
#class SelfAttentionNetworkVelAcc(SelfAttentionNetwork):
#    def _build_attention_input(self):
#        return restrict_variables_for_attention_general(self._input["reshaped"], 'acc_vel')

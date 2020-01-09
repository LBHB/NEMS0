# For weight channels
    def generate_tensor(self, data, phi):
        # Hide import here
        from theano import tensor
        x = data[self.input_name]
        n = x.shape[0]
        # Add a half step to the array so that x represents the bin "centers".
        #x = np.arange(n)/n + 0.5/n
        channel_centers = tensor.arange(n)/n + 0.5/n
        weights = self.get_weights(channel_centers, phi)
        return {
            self.output_name: tensor.dot(weights, x)
        }

import java.util.ArrayList;

public class NeuralNetwork {

    private final ArrayList<Layer> network;

    NeuralNetwork(ArrayList<Integer> layerSizes, ArrayList<ActivationFunction> layerActivations, double learningRate) {
        if (layerActivations.size() != layerSizes.size() - 1)
            throw new IllegalArgumentException("Number of activation functions must be 1 less than number of layers");
        network = new ArrayList<>();
        /* Create fully connected network in which the size of the output of one
        layer is the size of the input to the next */
        for (int layer = 0; layer < layerSizes.size() - 1; layer++) {
            ActivationFunction activationFunction = layerActivations.get(layer);
            network.add(
                    new Layer(layerSizes.get(layer), layerSizes.get(layer + 1), activationFunction, learningRate)
            );
        }
    }

    // Performs feed-forward algorithm on 1xn matrix input
    public void feedForward(Matrix input) {
        if (input.getRows() != 1)
            throw new IllegalArgumentException("Network input must be 1 dimensional");
        Matrix currentLayer = input;
        // Iteratively feed forward by one layer and then pass that output to the next layer
        for (Layer layer : network) {
            layer.feedForward(currentLayer);
            currentLayer = layer.getOutputNodes();
        }
    }

    // Performs backpropagation algorithm on 1xn matrix output
    public void backpropagate(Matrix trueOutput) {
        ArrayList<Double> error = new ArrayList<>(); // Error on output layer
        Matrix predictedOutput = this.getOutput();
        // Error term for output layer is difference between true and predicted output
        for (int index = 0; index < trueOutput.getCols(); index++) {
            error.add(trueOutput.getElement(0, index) - predictedOutput.getElement(0, index));
        }
        // Start from last layer and propagate backwards through the network
        for (int currentLayer = network.size() - 1; currentLayer >= 0; currentLayer--) {
            // 'error' is the error term on each node in the current layer
            network.get(currentLayer).backpropagate(error);
            error = network.get(currentLayer).getNewErrorTerms();
        }
    }

    // Update weights for the network by modifying weights by calculated deltas
    public void updateWeights() {
        for (Layer layer : network) {
            layer.updateWeights();
        }
    }

    public Matrix getOutput() {
        return network.get(network.size() - 1).getOutputNodes();
    }
}
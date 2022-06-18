import java.util.ArrayList;
import java.util.Random;

public class Layer {

    private Matrix weights; // Weights of fully connected layer
    private Matrix biases; // Bias of corresponding weight;
    private final Matrix weightDeltas; // Current change to be applied to each weight
    private final Matrix biasDeltas; // Current change to be applied to each bias
    private Matrix inputNodes; // Input nodes to the current layer
    private Matrix outputNodes; // Output nodes of the current layer
    private final ArrayList<Double> newErrorTerms; // Error terms for preceding layer
    private final ActivationFunction activationFunction; // Activation function for this layer
    private final double learningRate; // Small constant to scale down weight delta

    public Layer(int inputDims, int outputDims, ActivationFunction activationFunction, double learningRate) {
        // Initialise weights and sets them to be uniformly distributed in range [-1,1]
        weights = new Matrix(inputDims, outputDims);
        Random random = new Random(0);
        for (int row = 0; row < weights.getRows(); row++) {
            for (int col = 0; col < weights.getCols(); col++) {
                double weightValue = random.nextDouble() * 2 - 1;
                weights.setElement(weightValue, row, col);
            }
        }

        // Initialises all biases to 0.0
        biases = new Matrix(1, outputDims);

        // Initialise matrices
        weightDeltas = new Matrix(inputDims, outputDims);
        biasDeltas = new Matrix(1, outputDims);
        inputNodes = new Matrix(1, inputDims);
        outputNodes = new Matrix(1, outputDims);

        // Initialise preceding error values to 0
        newErrorTerms = new ArrayList<>();
        for (int i = 0; i < inputDims; i++) newErrorTerms.add(0.0);

        // Set activation function
        this.activationFunction = activationFunction;

        // Set learning rate (alpha)
        this.learningRate = learningRate;
    }

    public void feedForward(Matrix input) {
        inputNodes = input;
        outputNodes = inputNodes.cartProd(weights);
        outputNodes = outputNodes.sum(biases); // Apply bias
        outputNodes = activationFunction.function(outputNodes);
    }

    // Performs the backpropagation algorithm and returns the error terms on each node in this layer
    public void backpropagate(ArrayList<Double> errorTerms) {
        int rows = weights.getRows();
        int columns = weights.getCols();
        for (int row = 0; row < rows; row++) {
            // Node that precedes this set of weights
            double previousActivation = inputNodes.getElement(0, row);
            // For each weight
            for (int column = 0; column < columns; column++) {
                double currentWeight = weights.getElement(row, column);
                // Output of the succeeding node
                double nodeOutput = outputNodes.getElement(0, column);
                double errorTerm = errorTerms.get(column) * activationFunction.derivative(nodeOutput);
                // Update node error
                newErrorTerms.set(row, newErrorTerms.get(row) + errorTerm * currentWeight);
                // Calculate and update weight delta
                double weightDelta = learningRate * errorTerm * previousActivation;
                double updatedWeightDelta = weightDeltas.getElement(row, column) + weightDelta;
                weightDeltas.setElement(updatedWeightDelta, row, column);
                // Update bias
                double biasDelta = learningRate * errorTerm;
                double updatedBiasDelta = biasDeltas.getElement(0, column) + biasDelta;
                biasDeltas.setElement(updatedBiasDelta, 0, column);
            }
        }
    }

    // Modify each weight by calculated weight delta
    public void updateWeights() {
        weights = weights.sum(weightDeltas); // Method in matrix class
        biases = biases.sum(biasDeltas);
        for (int row = 0; row < weights.getRows(); row++) {
            for (int col = 0; col < weights.getCols(); col++) {
                weightDeltas.setElement(0.0, row, col);
                biasDeltas.setElement(0.0, 0, col);
            }
        }
    }

    // Getter for outputNodes
    public Matrix getOutputNodes() {
        return outputNodes;
    }

    // Return error terms and reset them to 0
    public ArrayList<Double> getNewErrorTerms() {
        ArrayList<Double> errorTerms = new ArrayList<>(); // Temporary storage
        for (int i = 0; i < weights.getRows(); i++) {
            errorTerms.add(newErrorTerms.get(i));
            newErrorTerms.set(i, 0.0);
        }
        return errorTerms;
    }
}
import java.util.ArrayList;
import java.util.Random;

public class Layer {

    private Matrix weights; // Weights of fully connected layer
    private final Matrix weightDeltas; // Current change to be applied to each weight
    private Matrix inputNodes; // Input nodes to the current layer
    private Matrix outputNodes; // Output nodes of the current layer
    private final ArrayList<Double> newErrorTerms; // Error terms for preceding layer
    private final double learningRate = 0.01; // Small constant to scale down weight delta

    public Layer(int inputDims, int outputDims) {
        // Initialise weights and sets them to be uniformly distributed in range [-1,1]
        weights = new Matrix(inputDims, outputDims);
        Random random = new Random();
        for (int row = 0; row < weights.getRows(); row++) {
            for (int col = 0; col < weights.getCols(); col++) {
                double weightValue = random.nextDouble() * 2 - 1;
                weights.setElement(weightValue, row, col);
            }
        }

        // Initialise matrices
        weightDeltas = new Matrix(inputDims, outputDims);
        inputNodes = new Matrix(1, inputDims);
        outputNodes = new Matrix(1, outputDims);

        // Initialise preceding error values to 0
        newErrorTerms = new ArrayList<>();
        for (int i = 0; i < inputDims; i++) newErrorTerms.add(0.0);
    }

    public void feedForward(Matrix input) {
        inputNodes = input;
        outputNodes = inputNodes.cartProd(weights);
        activation(outputNodes);
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
                double errorTerm = errorTerms.get(column) * sigmoidPrime(nodeOutput);
                // Update node error
                newErrorTerms.set(row, newErrorTerms.get(row) + errorTerm * currentWeight);
                // Calculate and update weight delta
                double weightDelta = learningRate * errorTerm * previousActivation;
                double updatedDelta = weightDeltas.getElement(row, column) + weightDelta;
                weightDeltas.setElement(updatedDelta, row, column);
            }
        }
    }

    // Modify each weight by calculated weight delta
    public void updateWeights() {
        weights = weights.sum(weightDeltas); // Method in matrix class
        for (int row = 0; row < weights.getRows(); row++) {
            for (int col = 0; col < weights.getCols(); col++) {
                weightDeltas.setElement(0.0, row, col);
            }
        }
    }

    // Apply activation function to every node in a given layer
    private void activation(Matrix layer) {
        for (int element = 0; element < layer.getCols(); element++) {
            double preActivation = layer.getElement(0, element);
            double postActivation = sigmoid(preActivation);
            layer.setElement(postActivation, 0, element);
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

    // Activation function
    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    // Derivative of activation function
    private double sigmoidPrime(double x) {
        return x * (1 - x);
    }
}
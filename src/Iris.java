import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class Iris {

    private static ArrayList<Matrix> trainData; // Input training data
    private static  ArrayList<Matrix> trainOutput; // Expected output for training data
    private static ArrayList<Matrix> testData; // Output training data
    private static ArrayList<Matrix> testOutput; // Expected output for training data
    private static NeuralNetwork network; // Network used in training and testing

    private static final int trainSize = 35; // Number of each flower type in testing data
    /* Dictates the number of nodes in each layer of the network (controls depth and
    breadth on a layer by layer basis */
    private static final ArrayList<Integer> networkTopology = new ArrayList<>(
            List.of(4, 5, 5, 3)
    );
    private static final int trainingIterations = 5000; // Number of epochs

    public static void main(String[] args) {
        network = new NeuralNetwork(networkTopology);
        loadData("resources/iris.data");
        train();
        test();
    }

    // Load training and testing data from a given filepath
    private static void loadData(String file) {
        trainData = new ArrayList<>();
        trainOutput = new ArrayList<>();
        testData = new ArrayList<>();
        testOutput = new ArrayList<>();
        try { // Used for resource handling
            File dataFile = new File(file);
            Scanner reader = new Scanner(dataFile);
            for (int lineNum = 0; lineNum < 150; lineNum++) {
                String currentLine = reader.nextLine();
                // Splits the current line on commas (file is comma delimited)
                ArrayList<String> lineTokens = new ArrayList<>(Arrays.asList(currentLine.split(",")));

                // Each of the attributes present in the data for a given flower
                double sepalLength = Double.parseDouble(lineTokens.get(0));
                double sepalWidth = Double.parseDouble(lineTokens.get(1));
                double petalLength = Double.parseDouble(lineTokens.get(2));
                double petalWidth = Double.parseDouble(lineTokens.get(3));
                String flowerType = lineTokens.get(4);

                // Creates a matrix of the input data
                ArrayList<Double> inputDataList = new ArrayList<>(List.of(
                        sepalLength,
                        sepalWidth,
                        petalLength,
                        petalWidth
                ));
                Matrix inputData = new Matrix(new ArrayList<>(List.of(inputDataList)));

                // Creates a matrix of the output data
                // Output data \in {(1, 0, 0), (0, 1, 0), (0, 0, 1)} contingent on flower type
                ArrayList<Double> outputDataList = new ArrayList<>(List.of(
                        flowerType.equals("Iris-setosa") ? 1.0 : 0.0,
                        flowerType.equals("Iris-versicolor") ? 1.0 : 0.0,
                        flowerType.equals("Iris-virginica") ? 1.0 : 0.0
                ));
                Matrix outputData = new Matrix(new ArrayList<>(List.of(outputDataList)));

                // Training data is first 'testSize' examples of each flower type of 50
                if (50 - lineNum % 50 > (50 - trainSize)) {
                    trainData.add(inputData);
                    trainOutput.add(outputData);
                } else { // The rest is testing data
                    testData.add(inputData);
                    testOutput.add(outputData);
                }
            }
            reader.close();
        } catch (FileNotFoundException exception) {
            System.out.println("Data file not found");
        }
    }

    // Trains the model on the data loaded from the file
    private static void train() {
        System.out.println("Training...");
        for (int iteration = 0; iteration < trainingIterations; iteration++) {
            for (int inputIndex = 0; inputIndex < 3 * trainSize; inputIndex++) {
                // Feed forward algorithm performed on the training input sample
                network.feedForward(trainData.get(inputIndex));
                // Backpropagation algorithm performed on the training expected output
                network.backpropagate(trainOutput.get(inputIndex));
                // Weights are modified by calculated weight deltas.
                network.updateWeights();
            }
            // Provide 5% incremental updates on training completion
            if (iteration % (trainingIterations / 20) == 0)
                System.out.print(".");
        }
        System.out.println("\nTraining complete.");
    }

    private static void test() {
        System.out.println("Testing...");
        double error = 0.0; // Sum total mean square error
        // Perform feed forward on each test sample
        for (int inputIndex = 0; inputIndex < 3 * (50 - trainSize); inputIndex++) {
            network.feedForward(testData.get(inputIndex));
            Matrix networkOutput = network.getOutput();
            Matrix trueValue = testOutput.get(inputIndex);
            // Add square of difference between true and predicted values
            for (int i = 0; i < 3; i++) {
                double difference = networkOutput.getElement(0, i) - trueValue.getElement(0, i);
                error += difference * difference;
            }
        }
        System.out.print("Mean square error: ");
        System.out.println(error / (9 * (50 - trainSize))); // Mean of squares of errors (MSE)
    }
}
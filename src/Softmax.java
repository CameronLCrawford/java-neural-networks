public class Softmax implements ActivationFunction {

    // Applies softmax to matrix of output values
    @Override
    public Matrix function(Matrix input) {
        // Number of distinct classes in output
        int classCount = input.getCols();
        Matrix result = new Matrix(1, classCount);
        double exponentialSum = 0; // Sum of expotential of each value
        for (int j = 0; j < classCount; j++) {
            exponentialSum += Math.exp(input.getElement(0, j));
        }
        for (int element = 0; element < classCount; element++) {
            double x = input.getElement(0, element);
            double tranformedX = Math.exp(x) / exponentialSum;
            result.setElement(tranformedX, 0, element);
        }
        return result;
    }

    @Override
    public double derivative(double x) { return x * (1 - x); }
}

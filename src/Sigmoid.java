public class Sigmoid implements ActivationFunction {

    // Applies sigmoid function to each element of input
    @Override
    public Matrix function(Matrix input) {
        Matrix result = new Matrix(1, input.getCols());
        for (int element = 0; element < input.getCols(); element++) {
            double x = input.getElement(0, element);
            double transformedX = 1 / (1 + Math.exp(-x)); // Sigmoid function
            result.setElement(transformedX, 0, element);
        }
        return result;
    }

    @Override
    public double derivative(double x) { return x * (1 - x); }
}

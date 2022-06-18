public class Tanh implements ActivationFunction{

    // Applies tanh (hyperbolic tangent) function to every element in input
    @Override
    public Matrix function(Matrix input) {
        Matrix result = new Matrix(1, input.getCols());
        for (int element = 0; element < input.getCols(); element++) {
            double x = input.getElement(0, element);
            double transformedX = Math.tanh(x); // Tanh function
            result.setElement(transformedX, 0, element);
        }
        return result;
    }

    @Override
    public double derivative(double x) {
        return 1 - x * x;
    }
}

public class Sigmoid implements ActivationFunction {

    @Override
    public double function(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    @Override
    public double derivative(double x) {
        return x * (1 - x);
    }
}

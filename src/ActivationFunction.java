// All activation functions implement this as part of Strategy pattern
public interface ActivationFunction {

    // The activation function
    public double function(double x);

    // The derivation of the activation function
    public double derivative(double x);
}

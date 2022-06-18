// All activation functions implement this as part of Strategy pattern
public interface ActivationFunction {

    // The activation function
    Matrix function(Matrix x);

    // The derivation of the activation function
    double derivative(double x);
}

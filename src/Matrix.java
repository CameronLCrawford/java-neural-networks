import java.util.ArrayList;

public class Matrix {

    // Each list consists of 'rows' which are each a list of doubles of length 'column'
    private final ArrayList<ArrayList<Double>> elements;

    // Construct blank matrix of shape (rows, cols)
    public Matrix(int rows, int cols) {
        elements = new ArrayList<>();
        // Initialise each value to 0
        for (int row = 0; row < rows; row++) {
            ArrayList<Double> newRow = new ArrayList<>(); // Create a new row
            for (int col = 0; col < cols; col++) {
                newRow.add(0.0);
            }
            elements.add(newRow);
        }
    }

    // Constructor that creates new matrix from 2D array of doubles
    public Matrix(ArrayList<ArrayList<Double>> values) {
        elements = new ArrayList<>();
        int rows = values.size(); // Each ArrayList is a new row of doubles
        if (rows == 0) // Empty array so no columns
            return;
        int cols = values.get(0).size(); // Assume all rows are same length
        for (ArrayList<Double> row : values) {
            if (row.size() != cols) // If all are not same length, throw exception
                throw new IllegalArgumentException("Every row must have same number of values");
            elements.add(new ArrayList<>(row)); // Shallow copies double values
        }
    }

    // Getter for the number of rows in the matrix
    public int getRows() {
        return elements.size();
    }

    // Getter for the number of columns in the matrix
    public int getCols() {
        return elements.get(0).size();
    }

    // Gets the value of a matrix at a given row and column
    public double getElement(int row, int col) {
        if (row > getRows())
            throw new IndexOutOfBoundsException("Row out of range");
        if (col > getCols())
            throw new IndexOutOfBoundsException("Column out of range");
        return elements.get(row).get(col);
    }

    // Sets the value of a matrix at a given row and column
    public void setElement(double value, int row, int col) {
        if (row > getRows())
            throw new IndexOutOfBoundsException("Row out of range");
        if (col > getCols())
            throw new IndexOutOfBoundsException("Column out of range");
        elements.get(row).set(col, value);
    }

    /* Calculates the cartesian product of this matrix and another
     and returns the result as a new matrix */
    public Matrix cartProd(Matrix other) {
        int thisRows = this.getRows();
        int thisCols = this.getCols();
        int otherRows = other.getRows();
        int otherCols = other.getCols();
        // Checks if matrices are right dimensions to be multiplied
        if (thisCols != otherRows)
            throw new IllegalArgumentException("Invalid matrix dimensions");
        Matrix result = new Matrix(thisRows, otherCols);
        for (int row = 0; row < thisRows; row++) {
            for (int col = 0; col < otherCols; col++) {
                /* This performs the dot product of vectors where the vectors are
                the row of this matrix and the column of the other matrix */
                double sum = 0;
                for (int k = 0; k < thisCols; k++)
                    sum += this.getElement(row, k) * other.getElement(k, col);
                result.setElement(sum, row, col);
            }
        }
        return result;
    }

    public Matrix sum(Matrix other) {
        int thisRows = this.getRows();
        int thisCols = this.getCols();
        int otherRows = this.getRows();
        int otherCols = this.getCols();
        // Test for correct dimensions
        if (thisRows != otherRows)
            throw new IllegalArgumentException("Matrix cannot be summed without same number of rows");
        if (thisCols != otherCols)
            throw new IllegalArgumentException("Matrix cannot be summed without same number of columns");

        Matrix result = new Matrix(thisRows, thisCols);
        // Iterate over each element and sum them
        for (int row = 0; row < thisRows; row++) {
            for (int col = 0; col < thisCols; col++) {
                double thisValue = this.getElement(row, col);
                double otherValue = other.getElement(row, col);
                result.setElement(thisValue + otherValue, row, col);
            }
        }
        return result;
    }

    // Override object default method for displaying matrix
    @Override
    public String toString() {
        String result = "";
        for (ArrayList<Double> row : elements)
            // Uses built-in toString method for ArrayList
            result = result.concat(row.toString() + '\n');
        return result;
    }
}